"""
Hybrid Intrusion Detection System - Unified Experiment Runner.

Handles the full pipeline: preprocessing, training (Stage 1 + Stage 2),
and cross-dataset comparison. Can run everything or individual phases.

Usage:
    # Full pipeline (preprocess + train + compare) for all datasets
    python src/run_experiments.py

    # Single dataset, single model
    python src/run_experiments.py --dataset netflow --model sgd

    # All models for one dataset
    python src/run_experiments.py --dataset kdd --model all

    # Only preprocess
    python src/run_experiments.py --preprocess-only
    python src/run_experiments.py --preprocess-only --dataset cores_iot

    # Only train (skip preprocessing, assume data is ready)
    python src/run_experiments.py --train-only --dataset netflow --model mlp

    # Only compare existing results
    python src/run_experiments.py --compare-only

    # Training options
    python src/run_experiments.py --dataset netflow --model all --contamination 0.3
    python src/run_experiments.py --dataset netflow --model mlp --skip-stage1
    python src/run_experiments.py --dataset netflow --model ensemble \
                                 --stage1-model models/netflow/iso_forest_0.45.pkl
"""

import argparse
import subprocess
import sys
from pathlib import Path

# =============================================================================
# PATH SETUP
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

# =============================================================================
# CONSTANTS
# =============================================================================
BENIGN_SUFFIX = "_benign"
DEFAULT_ISO_CONTAMINATION = 0.45
EVAL_BASE_DIR = Path("evaluation")
MODELS_BASE_DIR = Path("models")
STAGE2_MODELS = ["sgd", "mlp", "ensemble"]

# =============================================================================
# DATASET REGISTRY
# =============================================================================
from datasets.registry import (
    DATASETS, DATASET_NAMES, get_dataset_paths, get_preprocess_script
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_dataset_type(dataset_path: str) -> str:
    """Extract dataset type from path. E.g. '.../netflow/train.csv' -> 'netflow'"""
    return Path(dataset_path).parent.name


def get_benign_path(dataset_path: str) -> Path:
    """Convert dataset path to benign-only version."""
    path = Path(dataset_path)
    if path.stem.endswith(BENIGN_SUFFIX):
        return path
    return path.parent / f"{path.stem}{BENIGN_SUFFIX}{path.suffix}"


def get_models_dir(dataset_type: str) -> Path:
    return MODELS_BASE_DIR / dataset_type


# =============================================================================
# PREPROCESSING
# =============================================================================
def preprocess_dataset(dataset_name: str) -> bool:
    """Run preprocessing script for a dataset."""
    script_path = get_preprocess_script(dataset_name, PROJECT_ROOT)

    if not script_path.exists():
        print(f"Error: Preprocessing script not found: {script_path}")
        return False

    print(f"\n{'='*70}")
    print(f"PREPROCESSING: {dataset_name.upper()}")
    print(f"{'='*70}")

    try:
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT, check=True, text=True,
        )
        print(f"\n[OK] Preprocessing {dataset_name} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Preprocessing {dataset_name} failed (exit {e.returncode})")
        return False


# =============================================================================
# TRAINING
# =============================================================================
def train_isolation_forest(train_path: str, test_path: str, dataset_type: str,
                           contamination: float = DEFAULT_ISO_CONTAMINATION) -> Path:
    """Train Isolation Forest (Stage 1) with benign-only data."""
    from train.iso_forest_train import main as iso_main

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
        contamination=contamination,
    )

    return models_dir / f"iso_forest_{contamination}.pkl"


def train_stage2(model_name: str, train_path: str, test_path: str,
                 stage1_model: Path, dataset_type: str):
    """Train a single Stage 2 model."""
    output_dir = EVAL_BASE_DIR / model_name / dataset_type
    models_dir = get_models_dir(dataset_type)

    print("\n" + "=" * 70)
    print(f"STAGE 2: {model_name.upper()}")
    print("=" * 70)

    if model_name == "sgd":
        from train.sgd_train import main as train_fn
    elif model_name == "mlp":
        from train.mlp_train import main as train_fn
    elif model_name == "ensemble":
        from train.ensemble_train import main as train_fn
    else:
        raise ValueError(f"Unknown model: {model_name}")

    train_fn(
        train_path=train_path,
        test_path=test_path,
        stage1_model_path=str(stage1_model),
        output_dir=str(output_dir),
        models_dir=str(models_dir),
    )


def train_dataset(dataset_name: str, model: str, contamination: float,
                  skip_stage1: bool = False, stage1_model_path: str = None) -> bool:
    """Run the full training pipeline for a single dataset."""
    paths = get_dataset_paths(dataset_name)
    train_path = str(paths['train'])
    test_path = str(paths['test'])

    print("\n" + "=" * 70)
    print(f"TRAINING: {dataset_name.upper()}")
    print("=" * 70)
    print(f"  Model: {model.upper()}")
    print(f"  Contamination: {contamination}")

    try:
        # Stage 1
        if skip_stage1 or stage1_model_path:
            if stage1_model_path:
                stage1_model = Path(stage1_model_path)
            else:
                stage1_model = get_models_dir(dataset_name) / f"iso_forest_{contamination}.pkl"

            if not stage1_model.exists():
                print(f"\nError: Stage 1 model not found: {stage1_model}")
                return False
            print(f"  Using existing Stage 1 model: {stage1_model}")
        else:
            benign_path = get_benign_path(train_path)
            if not benign_path.exists():
                print(f"\nError: Benign dataset not found: {benign_path}")
                return False
            stage1_model = train_isolation_forest(
                train_path, test_path, dataset_name, contamination
            )
            if not stage1_model.exists():
                print(f"\nError: Stage 1 model not saved: {stage1_model}")
                return False

        # Stage 2
        models_to_train = STAGE2_MODELS if model == "all" else [model]
        for m in models_to_train:
            train_stage2(m, train_path, test_path, stage1_model, dataset_name)

        print(f"\n[OK] Training {dataset_name} completed")
        return True

    except Exception as e:
        print(f"\n[ERROR] Training {dataset_name} failed: {e}")
        return False


# =============================================================================
# COMPARISON
# =============================================================================
def compare_results() -> bool:
    """Run the cross-dataset comparison script."""
    script_path = PROJECT_ROOT / 'src' / 'evaluation' / 'compare_results.py'
    if not script_path.exists():
        print(f"Error: Comparison script not found: {script_path}")
        return False

    print(f"\n{'='*70}")
    print("COMPARING RESULTS")
    print(f"{'='*70}")

    try:
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT, check=True, text=True,
        )
        print("\n[OK] Comparison completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Comparison failed (exit {e.returncode})")
        return False


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Hybrid IDS - Unified Experiment Runner (preprocess + train + compare)"
    )

    # Phase selection
    phase_group = parser.add_mutually_exclusive_group()
    phase_group.add_argument(
        '--preprocess-only', action='store_true',
        help='Only run preprocessing, skip training and comparison',
    )
    phase_group.add_argument(
        '--train-only', action='store_true',
        help='Only run training, skip preprocessing',
    )
    phase_group.add_argument(
        '--compare-only', action='store_true',
        help='Only run comparison of existing results',
    )

    parser.add_argument(
        '--no-compare', action='store_true',
        help='Skip comparison at the end',
    )

    # Dataset & model selection
    parser.add_argument(
        '--dataset', type=str,
        choices=DATASET_NAMES + ['all'], default='all',
        help='Which dataset to process (default: all)',
    )
    parser.add_argument(
        '--model', type=str,
        choices=['sgd', 'mlp', 'ensemble', 'all'], default='all',
        help='Which Stage 2 model to train (default: all)',
    )

    # Training configuration
    parser.add_argument(
        '--contamination', type=float, default=None,
        help=f'Isolation Forest contamination value (default: {DEFAULT_ISO_CONTAMINATION})',
    )
    parser.add_argument(
        '--skip-stage1', action='store_true',
        help='Skip Stage 1 training and use existing Isolation Forest model',
    )
    parser.add_argument(
        '--stage1-model', type=str, default=None,
        help='Path to existing Stage 1 model (implies --skip-stage1)',
    )

    args = parser.parse_args()

    # --- Compare-only mode ---
    if args.compare_only:
        success = compare_results()
        sys.exit(0 if success else 1)

    # --- Determine datasets ---
    datasets_to_process = DATASET_NAMES if args.dataset == 'all' else [args.dataset]

    print("=" * 70)
    print("HYBRID INTRUSION DETECTION SYSTEM - EXPERIMENTS")
    print("=" * 70)
    print(f"Datasets: {', '.join(datasets_to_process)}")
    mode = 'Preprocess only' if args.preprocess_only else \
           'Train only' if args.train_only else 'Full pipeline'
    print(f"Mode: {mode}")
    print(f"Model: {args.model}")

    results = {'preprocess': {}, 'train': {}}

    # --- Preprocessing phase ---
    if not args.train_only:
        print("\n" + "=" * 70)
        print("PHASE 1: PREPROCESSING")
        print("=" * 70)

        for dataset in datasets_to_process:
            success = preprocess_dataset(dataset)
            results['preprocess'][dataset] = success
            if not success:
                print(f"Warning: Preprocessing failed for {dataset}")

    # --- Training phase ---
    if not args.preprocess_only:
        print("\n" + "=" * 70)
        print("PHASE 2: TRAINING")
        print("=" * 70)

        for dataset in datasets_to_process:
            contamination = args.contamination or DATASETS.get(
                dataset, {}
            ).get('defaults', {}).get('contamination', DEFAULT_ISO_CONTAMINATION)

            skip = args.skip_stage1 or args.stage1_model is not None

            success = train_dataset(
                dataset, args.model, contamination,
                skip_stage1=skip,
                stage1_model_path=args.stage1_model,
            )
            results['train'][dataset] = success
            if not success:
                print(f"Warning: Training failed for {dataset}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    if results['preprocess']:
        print("\nPreprocessing:")
        for dataset, success in results['preprocess'].items():
            print(f"  {dataset}: {'OK' if success else 'FAILED'}")

    if results['train']:
        print("\nTraining:")
        for dataset, success in results['train'].items():
            print(f"  {dataset}: {'OK' if success else 'FAILED'}")

    # --- Comparison phase ---
    if not args.no_compare and not args.preprocess_only:
        print("\n" + "=" * 70)
        print("PHASE 3: COMPARISON")
        print("=" * 70)
        compare_success = compare_results()
        results['compare'] = compare_success

    print("\nOutput locations:")
    print("  Processed data: datasets/processed/<dataset>/")
    print("  Models:         models/<dataset>/")
    print("  Evaluation:     evaluation/<model>/<dataset>/")
    print("  Summary:        evaluation/summary.csv")

    all_success = (
        all(results['preprocess'].values()) if results['preprocess'] else True
    ) and (
        all(results['train'].values()) if results['train'] else True
    ) and results.get('compare', True)

    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
