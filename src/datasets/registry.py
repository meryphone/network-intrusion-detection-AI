"""
Dataset Registry - Central configuration for all datasets.

This module provides a single source of truth for dataset paths, metadata,
and default hyperparameters. Use this registry to avoid hardcoding paths
across the codebase.

Usage:
    from datasets.registry import DATASETS, get_dataset_paths
    
    # Get paths for a dataset
    paths = get_dataset_paths('netflow')
    print(paths['train'])  # Full path to train_processed.csv
"""

from pathlib import Path
from typing import Dict, Any, Optional

# =============================================================================
# DATASET REGISTRY
# =============================================================================
DATASETS: Dict[str, Dict[str, Any]] = {
    'netflow': {
        'name': 'NetFlow',
        'description': 'NetFlow network traffic dataset',
        'raw': {
            'train': 'datasets/raw/netflow/train_net.csv',
            'test': 'datasets/raw/netflow/test_net.csv',
        },
        'processed': {
            'dir': 'datasets/processed/netflow',
            'train': 'train_processed.csv',
            'train_benign': 'train_processed_benign.csv',
            'test': 'test_processed.csv',
        },
        'preprocess_script': 'src/preprocesssing/data_preprocessing_netflow.py',
        'label': {
            'column': 'ANOMALY',
            'positive_values': [1],
        },
        'defaults': {
            'contamination': 0.45,
            'batch_size': 20000,
        },
    },
    'kdd': {
        'name': 'KDD Cup 99',
        'description': 'KDD Cup 1999 intrusion detection dataset',
        'raw': {
            'train': 'datasets/raw/kd/kddcup.data.corrected',
            'test': 'datasets/raw/kd/corrected/corrected',
        },
        'processed': {
            'dir': 'datasets/processed/kdd',
            'train': 'train_processed.csv',
            'train_benign': 'train_processed_benign.csv',
            'test': 'test_processed.csv',
        },
        'preprocess_script': 'src/preprocesssing/data_preprocessing_kdd.py',
        'label': {
            'column': 'ANOMALY',
            'source': 'label',
            'mapping': {'normal': 0, 'attack': 1},
        },
        'defaults': {
            'contamination': 0.45,
            'batch_size': 20000,
        },
    },
    'cores_iot': {
        'name': 'CORES IoT',
        'description': 'CORES IoT network intrusion dataset',
        'raw': {
            'file': 'datasets/raw/cores-iot/cores_iot.csv',
        },
        'processed': {
            'dir': 'datasets/processed/cores_iot',
            'train': 'train_processed.csv',
            'train_benign': 'train_processed_benign.csv',
            'test': 'test_processed.csv',
        },
        'preprocess_script': 'src/preprocesssing/data_preprocessing_cores_iot.py',
        'label': {
            'column': 'ANOMALY',
            'positive_values': [1],
        },
        'defaults': {
            'contamination': 0.45,
            'batch_size': 20000,
        },
    },
}

# List of all available datasets
DATASET_NAMES = list(DATASETS.keys())


def get_project_root() -> Path:
    """Get the project root directory."""
    # Assumes this file is in src/datasets/
    return Path(__file__).resolve().parent.parent.parent


def get_dataset_paths(dataset_name: str, project_root: Optional[Path] = None) -> Dict[str, Path]:
    """
    Get full paths for a dataset's processed files.
    
    Args:
        dataset_name: Name of the dataset (netflow, kdd, cores_iot)
        project_root: Optional project root path. Defaults to auto-detected.
    
    Returns:
        Dictionary with keys: 'train', 'train_benign', 'test', 'dir'
    
    Raises:
        ValueError: If dataset_name is not in the registry
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {DATASET_NAMES}")
    
    root = project_root or get_project_root()
    config = DATASETS[dataset_name]['processed']
    base_dir = root / config['dir']
    
    return {
        'dir': base_dir,
        'train': base_dir / config['train'],
        'train_benign': base_dir / config['train_benign'],
        'test': base_dir / config['test'],
    }


def get_raw_paths(dataset_name: str, project_root: Optional[Path] = None) -> Dict[str, Path]:
    """
    Get full paths for a dataset's raw files.
    
    Args:
        dataset_name: Name of the dataset
        project_root: Optional project root path
    
    Returns:
        Dictionary with raw file paths
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {DATASET_NAMES}")
    
    root = project_root or get_project_root()
    raw_config = DATASETS[dataset_name]['raw']
    
    return {key: root / path for key, path in raw_config.items()}


def get_dataset_defaults(dataset_name: str) -> Dict[str, Any]:
    """
    Get default hyperparameters for a dataset.
    
    Args:
        dataset_name: Name of the dataset
    
    Returns:
        Dictionary with default values (contamination, batch_size, etc.)
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {DATASET_NAMES}")
    
    return DATASETS[dataset_name].get('defaults', {})


def get_preprocess_script(dataset_name: str, project_root: Optional[Path] = None) -> Path:
    """
    Get the preprocessing script path for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        project_root: Optional project root path
    
    Returns:
        Path to the preprocessing script
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {DATASET_NAMES}")
    
    root = project_root or get_project_root()
    return root / DATASETS[dataset_name]['preprocess_script']


def get_evaluation_dir(model_name: str, dataset_name: str, project_root: Optional[Path] = None) -> Path:
    """
    Get the evaluation output directory for a model and dataset.
    
    Args:
        model_name: Name of the model (isolation_forest, sgd, mlp, ensemble)
        dataset_name: Name of the dataset
        project_root: Optional project root path
    
    Returns:
        Path to evaluation directory
    """
    root = project_root or get_project_root()
    return root / 'evaluation' / model_name / dataset_name


def get_model_path(model_name: str, dataset_name: str, 
                   contamination: float = 0.45,
                   project_root: Optional[Path] = None) -> Path:
    """
    Get the model file path.
    
    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        contamination: Contamination value (for isolation forest)
        project_root: Optional project root path
    
    Returns:
        Path to the model file
    """
    root = project_root or get_project_root()
    models_dir = root / 'models' / dataset_name
    
    if model_name == 'isolation_forest':
        return models_dir / f'iso_forest_{contamination}.pkl'
    else:
        return models_dir / f'{model_name}_classifier.pkl'
