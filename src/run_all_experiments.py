"""
Run All Experiments Script.
Preprocesses all datasets and runs training pipeline on each.
Optionally compares results across datasets at the end.

Run from project root:
    python src/run_all_experiments.py [--preprocess-only] [--train-only] [--dataset DATASET]
    python src/run_all_experiments.py --compare  # Just run comparison
    python src/run_all_experiments.py --all      # Full pipeline + comparison
"""

import argparse
import subprocess
import sys
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Import dataset registry
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
try:
    from datasets.registry import DATASETS, DATASET_NAMES, get_dataset_paths, get_preprocess_script
    USE_REGISTRY = True
except ImportError:
    USE_REGISTRY = False
    DATASET_NAMES = ['netflow', 'kdd', 'cores_iot']
    # Fallback configuration
    DATASETS = {
        'netflow': {
            'preprocess_script': 'src/preprocesssing/data_preprocessing_netflow.py',
            'train_path': 'datasets/processed/netflow/train_processed.csv',
            'test_path': 'datasets/processed/netflow/test_processed.csv',
        },
        'kdd': {
            'preprocess_script': 'src/preprocesssing/data_preprocessing_kdd.py',
            'train_path': 'datasets/processed/kdd/train_processed.csv',
            'test_path': 'datasets/processed/kdd/test_processed.csv',
        },
        'cores_iot': {
            'preprocess_script': 'src/preprocesssing/data_preprocessing_cores_iot.py',
            'train_path': 'datasets/processed/cores_iot/train_processed.csv',
            'test_path': 'datasets/processed/cores_iot/test_processed.csv',
        },
    }


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*70}")
    print(f"RUNNING: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*70)
    
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=True,
            text=True
        )
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ {description} failed: {e}")
        return False


def preprocess_dataset(dataset_name: str) -> bool:
    """Run preprocessing for a single dataset."""
    if USE_REGISTRY:
        script_path = get_preprocess_script(dataset_name, PROJECT_ROOT)
    else:
        config = DATASETS[dataset_name]
        script_path = PROJECT_ROOT / config['preprocess_script']
    
    if not script_path.exists():
        print(f"Error: Preprocessing script not found: {script_path}")
        return False
    
    return run_command(
        [sys.executable, str(script_path)],
        f"Preprocessing {dataset_name.upper()}"
    )


def train_dataset(dataset_name: str, model: str = 'all') -> bool:
    """Run training for a single dataset using --dataset flag."""
    # Use --dataset for cleaner path resolution
    cmd = [
        sys.executable, 
        str(PROJECT_ROOT / 'src' / 'train_all.py'),
        '--dataset', dataset_name,
        '--model', model
    ]
    
    return run_command(
        cmd,
        f"Training {dataset_name.upper()} with {model.upper()} model(s)"
    )


def compare_results() -> bool:
    """Run the cross-dataset comparison script."""
    script_path = PROJECT_ROOT / 'src' / 'evaluation' / 'compare_results.py'
    
    if not script_path.exists():
        print(f"Error: Comparison script not found: {script_path}")
        return False
    
    return run_command(
        [sys.executable, str(script_path)],
        "Comparing results across datasets"
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run preprocessing and training for all datasets"
    )
    parser.add_argument(
        '--preprocess-only',
        action='store_true',
        help='Only run preprocessing, skip training'
    )
    parser.add_argument(
        '--train-only',
        action='store_true',
        help='Only run training, skip preprocessing'
    )
    parser.add_argument(
        '--compare-only',
        action='store_true',
        help='Only run comparison, skip preprocessing and training'
    )
    parser.add_argument(
        '--no-compare',
        action='store_true',
        help='Skip comparison at the end'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=DATASET_NAMES + ['all'],
        default='all',
        help='Which dataset to process (default: all)'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['sgd', 'mlp', 'ensemble', 'all'],
        default='all',
        help='Which Stage 2 model to train (default: all)'
    )
    
    args = parser.parse_args()
    
    # Handle compare-only mode
    if args.compare_only:
        print("=" * 70)
        print("RUNNING COMPARISON ONLY")
        print("=" * 70)
        success = compare_results()
        sys.exit(0 if success else 1)
    
    # Determine which datasets to process
    if args.dataset == 'all':
        datasets_to_process = DATASET_NAMES
    else:
        datasets_to_process = [args.dataset]
    
    print("=" * 70)
    print("MULTI-DATASET INTRUSION DETECTION EXPERIMENTS")
    print("=" * 70)
    print(f"Datasets: {', '.join(datasets_to_process)}")
    print(f"Mode: {'Preprocess only' if args.preprocess_only else 'Train only' if args.train_only else 'Full pipeline'}")
    print(f"Model: {args.model}")
    print(f"Compare: {'No' if args.no_compare else 'Yes'}")
    
    results = {
        'preprocess': {},
        'train': {}
    }
    
    # =========================================================================
    # PREPROCESSING PHASE
    # =========================================================================
    if not args.train_only:
        print("\n" + "=" * 70)
        print("PHASE 1: PREPROCESSING")
        print("=" * 70)
        
        for dataset in datasets_to_process:
            success = preprocess_dataset(dataset)
            results['preprocess'][dataset] = success
            if not success:
                print(f"\nWarning: Preprocessing failed for {dataset}")
    
    # =========================================================================
    # TRAINING PHASE
    # =========================================================================
    if not args.preprocess_only:
        print("\n" + "=" * 70)
        print("PHASE 2: TRAINING")
        print("=" * 70)
        
        for dataset in datasets_to_process:
            success = train_dataset(dataset, args.model)
            results['train'][dataset] = success
            if not success:
                print(f"\nWarning: Training failed for {dataset}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    if not args.train_only:
        print("\nPreprocessing Results:")
        for dataset, success in results['preprocess'].items():
            status = "✓ Success" if success else "✗ Failed"
            print(f"  {dataset}: {status}")
    
    if not args.preprocess_only:
        print("\nTraining Results:")
        for dataset, success in results['train'].items():
            status = "✓ Success" if success else "✗ Failed"
            print(f"  {dataset}: {status}")
    
    # =========================================================================
    # COMPARISON PHASE
    # =========================================================================
    if not args.no_compare and not args.preprocess_only:
        print("\n" + "=" * 70)
        print("PHASE 3: COMPARISON")
        print("=" * 70)
        
        compare_success = compare_results()
        results['compare'] = compare_success
        
        print("\nComparison Results:")
        status = "✓ Success" if compare_success else "✗ Failed"
        print(f"  Cross-dataset comparison: {status}")
    
    print("\nOutput locations:")
    print("  Processed data: datasets/processed/<dataset>/")
    print("  Models: models/<dataset>/")
    print("  Evaluation: evaluation/<model>/<dataset>/")
    print("  Summary: evaluation/summary.csv")
    
    # Return exit code based on success
    all_success = (
        all(results['preprocess'].values()) and 
        all(results['train'].values()) and
        results.get('compare', True)
    )
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
