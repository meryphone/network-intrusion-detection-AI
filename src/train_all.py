"""
Hybrid Intrusion Detection System - Training Script.

Trains Isolation Forest (Stage 1) + selected Stage 2 model.
IsoForest automatically uses the _benign version of the training dataset.

Usage:
    # Train using dataset registry (recommended)
    python src/train_all.py --dataset netflow --model sgd
    python src/train_all.py --dataset kdd --model all
    python src/train_all.py --dataset cores_iot --model ensemble
    
    # Train with explicit paths (legacy)
    python src/train_all.py --train data/train.csv --test data/test.csv --model sgd
    
    # Train only Stage 2 (skip Isolation Forest training)
    python src/train_all.py --dataset netflow --model mlp --skip-stage1
    python src/train_all.py --dataset netflow --model mlp --stage1-model models/netflow/iso_forest_0.45.pkl
    
    # Custom contamination value
    python src/train_all.py --dataset netflow --model all --contamination 0.3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# =============================================================================
# CONSTANTS
# =============================================================================
BENIGN_SUFFIX = "_benign"
DEFAULT_ISO_CONTAMINATION = 0.45
EVAL_BASE_DIR = Path("evaluation")
MODELS_BASE_DIR = Path("models")

# Import dataset registry
try:
    from datasets.registry import DATASETS, DATASET_NAMES, get_dataset_paths
except ImportError:
    # Fallback if registry not available
    DATASETS = {}
    DATASET_NAMES = []
    get_dataset_paths = None


def get_dataset_type(dataset_path: str) -> str:
    """
    Extract dataset type from path.
    Example: 'datasets/processed/netflow/train.csv' -> 'netflow'
    """
    path = Path(dataset_path)
    # Get the parent folder name as dataset type
    return path.parent.name


def get_benign_path(dataset_path: str) -> Path:
    """
    Convert dataset path to benign-only version.
    Example: 'data/train.csv' -> 'data/train_benign.csv'
    """
    path = Path(dataset_path)
    stem = path.stem
    
    if stem.endswith(BENIGN_SUFFIX):
        return path
    
    return path.parent / f"{stem}{BENIGN_SUFFIX}{path.suffix}"


def get_models_dir(dataset_type: str) -> Path:
    """Get the models directory for a dataset."""
    return MODELS_BASE_DIR / dataset_type


def train_isolation_forest(train_path: str, test_path: str, dataset_type: str, 
                           contamination: float = DEFAULT_ISO_CONTAMINATION) -> Path:
    """Train Isolation Forest (Stage 1) with benign-only data."""
    from train.isoForestTrain import main as iso_main
    
    benign_path = get_benign_path(train_path)
    output_dir = EVAL_BASE_DIR / "isolation_forest" / dataset_type
    models_dir = get_models_dir(dataset_type)
    
    print("\n" + "=" * 70)
    print("STAGE 1: ISOLATION FOREST")
    print("=" * 70)
    
    iso_main(
        train_path=str(benign_path), 
        test_path=test_path, 
        output_dir=str(output_dir),
        models_dir=str(models_dir),
        contamination=contamination
    )
    
    return models_dir / f"iso_forest_{contamination}.pkl"


def train_stage2(model_name: str, train_path: str, test_path: str, stage1_model: Path, dataset_type: str):
    """Train selected Stage 2 model."""
    
    output_dir = EVAL_BASE_DIR / model_name / dataset_type
    models_dir = get_models_dir(dataset_type)
    
    print("\n" + "=" * 70)
    print(f"STAGE 2: {model_name.upper()}")
    print("=" * 70)
    
    if model_name == "sgd":
        from train.sgdTrain import main as train_fn
    elif model_name == "mlp":
        from train.mlpTrain import main as train_fn
    elif model_name == "ensemble":
        from train.ensembleTrain import main as train_fn
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    train_fn(
        train_path=train_path,
        test_path=test_path,
        stage1_model_path=str(stage1_model),
        output_dir=str(output_dir),
        models_dir=str(models_dir)
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train Hybrid IDS: Isolation Forest + Stage 2 model"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=DATASET_NAMES if DATASET_NAMES else ['netflow', 'kdd', 'cores_iot'],
        default=None,
        help="Dataset to use (resolves paths from registry)"
    )
    parser.add_argument(
        "--train",
        type=str,
        default=None,
        help="Path to training CSV (IsoForest will use <name>_benign.csv). Optional if --dataset is provided."
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Path to test CSV (with ANOMALY column). Optional if --dataset is provided."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["sgd", "mlp", "ensemble", "all"],
        help="Stage 2 model to combine with Isolation Forest (use 'all' to train all models)"
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=None,
        help=f"Isolation Forest contamination value (default: {DEFAULT_ISO_CONTAMINATION} or dataset default)"
    )
    parser.add_argument(
        "--skip-stage1",
        action="store_true",
        help="Skip Stage 1 training and use existing Isolation Forest model"
    )
    parser.add_argument(
        "--stage1-model",
        type=str,
        default=None,
        help="Path to existing Stage 1 model (implies --skip-stage1)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths from dataset registry or explicit arguments
    if args.dataset:
        if get_dataset_paths is None:
            print("Error: Dataset registry not available. Use --train and --test instead.")
            sys.exit(1)
        
        paths = get_dataset_paths(args.dataset)
        train_path = str(paths['train'])
        test_path = str(paths['test'])
        dataset_type = args.dataset
        
        # Use dataset-specific default contamination if not specified
        if args.contamination is None:
            args.contamination = DATASETS.get(args.dataset, {}).get('defaults', {}).get(
                'contamination', DEFAULT_ISO_CONTAMINATION
            )
    else:
        if not args.train or not args.test:
            parser.error("Either --dataset or both --train and --test are required")
        train_path = args.train
        test_path = args.test
        dataset_type = get_dataset_type(train_path)
        
        if args.contamination is None:
            args.contamination = DEFAULT_ISO_CONTAMINATION
    
    # Determine if we should skip Stage 1
    skip_stage1 = args.skip_stage1 or args.stage1_model is not None
    
    print("=" * 70)
    print("HYBRID INTRUSION DETECTION SYSTEM")
    print("=" * 70)
    print(f"Dataset: {dataset_type}")
    print(f"Train: {train_path}")
    print(f"Test: {test_path}")
    print(f"Pipeline: IsolationForest + {args.model.upper()}")
    print(f"Contamination: {args.contamination}")
    if skip_stage1:
        print(f"Mode: Stage 2 only (skipping Isolation Forest training)")
    
    # Stage 1: Train or use existing model
    if skip_stage1:
        # Use existing Stage 1 model
        if args.stage1_model:
            stage1_model = Path(args.stage1_model)
        else:
            # Look in dataset-specific models dir first, then fallback to global
            stage1_model = get_models_dir(dataset_type) / f"iso_forest_{args.contamination}.pkl"
            if not stage1_model.exists():
                stage1_model = Path("models") / f"iso_forest_{args.contamination}.pkl"
        
        if not stage1_model.exists():
            print(f"\nError: Stage 1 model not found: {stage1_model}")
            print("Train Isolation Forest first or provide a valid --stage1-model path")
            sys.exit(1)
        
        print(f"\nUsing existing Stage 1 model: {stage1_model}")
    else:
        # Verify benign file exists
        benign_path = get_benign_path(train_path)
        if not benign_path.exists():
            print(f"\nError: Benign dataset not found: {benign_path}")
            sys.exit(1)
        
        # Train Stage 1
        stage1_model = train_isolation_forest(
            train_path, test_path, dataset_type, 
            contamination=args.contamination
        )
        
        if not stage1_model.exists():
            print(f"\nError: Stage 1 model not saved: {stage1_model}")
            sys.exit(1)
    
    # Stage 2
    if args.model == "all":
        # Train all Stage 2 models
        for model_name in ["sgd", "mlp", "ensemble"]:
            train_stage2(model_name, train_path, test_path, stage1_model, dataset_type)
    else:
        train_stage2(args.model, train_path, test_path, stage1_model, dataset_type)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Results: evaluation/<model>/{dataset_type}/")
    print(f"Models: models/{dataset_type}/")


if __name__ == "__main__":
    main()
