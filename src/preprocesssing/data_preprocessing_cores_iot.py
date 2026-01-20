"""
Data Preprocessing Script for CORES-IoT Dataset.
Prepares training and test data for the intrusion detection system.

Run from project root:
    python src/preprocesssing/data_preprocessing_cores_iot.py
"""

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# =============================================================================
# CONFIGURATION
# =============================================================================
TEST_SIZE = 0.2  # 80% train, 20% test
RANDOM_STATE = 42

# Number of features (excluding label)
N_FEATURES = 19

# Generate column names (feature_0, feature_1, ..., feature_18, ANOMALY)
COLUMN_NAMES = [f'feature_{i}' for i in range(N_FEATURES)] + ['ANOMALY']

# Columns to drop (irrelevant features based on feature importance analysis):
# - Zero-variance: feature_11, feature_12, feature_13 (100% constant = 0)
# - Near-zero variance with very low importance (<0.001):
#   feature_0, feature_6, feature_17, feature_18, feature_3
COLS_TO_DROP = [
    # Zero-variance features (100% constant value = 0)
    'feature_11',  # Always 0, importance = 0.0000
    'feature_12',  # Always 0, importance = 0.0000
    'feature_13',  # Always 0, importance = 0.0000
    # Near-zero variance with very low importance
    'feature_6',   # 99.68% = 0, importance = 0.0000
    'feature_0',   # Unique per row (like ID), importance = 0.0001
]

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_PATH = PROJECT_ROOT / 'datasets' / 'raw' / 'cores-iot' / 'cores_iot.csv'
PROCESSED_DIR = PROJECT_ROOT / 'datasets' / 'processed' / 'cores_iot'

OUTPUT_TRAIN_BENIGN = PROCESSED_DIR / 'train_processed_benign.csv'  # Stage 1 (benign only)
OUTPUT_TRAIN = PROCESSED_DIR / 'train_processed.csv'                # Stage 2 (full)
OUTPUT_TEST = PROCESSED_DIR / 'test_processed.csv'


def main():
    """Main preprocessing pipeline."""
    
    print("=" * 60)
    print("DATA PREPROCESSING - CORES-IoT")
    print("=" * 60)
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print("\nLoading data...")
    print(f"Source: {RAW_DATA_PATH}")
    
    df_full = pd.read_csv(RAW_DATA_PATH, names=COLUMN_NAMES, header=None)
    print(f"Total samples: {len(df_full)}")
    print(f"Features: {N_FEATURES}")
    
    # Check class distribution
    class_counts = df_full['ANOMALY'].value_counts()
    print(f"Class distribution:")
    print(f"  Normal (0): {class_counts.get(0, 0)}")
    print(f"  Attack (1): {class_counts.get(1, 0)}")
    
    # =========================================================================
    # TRAIN/TEST SPLIT
    # =========================================================================
    print(f"\n--- Splitting data (Train: {1-TEST_SIZE:.0%}, Test: {TEST_SIZE:.0%}) ---")
    
    df_train, df_test = train_test_split(
        df_full, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=df_full['ANOMALY']  # Maintain class distribution
    )
    
    print(f"Training samples: {len(df_train)}")
    print(f"Test samples: {len(df_test)}")
    
    # =========================================================================
    # DROP IRRELEVANT FEATURES
    # =========================================================================
    print(f"\n--- Dropping {len(COLS_TO_DROP)} irrelevant features ---")
    existing_cols = [c for c in COLS_TO_DROP if c in df_train.columns]
    df_train = df_train.drop(columns=existing_cols)
    df_test = df_test.drop(columns=existing_cols)
    print(f"Dropped: {existing_cols}")
    print(f"Remaining features: {len(df_train.columns) - 1}")  # -1 for ANOMALY
    
    # Separate benign samples for Stage 1
    df_train_benign = df_train[df_train['ANOMALY'] == 0].copy()
    print(f"Benign training samples (Stage 1): {len(df_train_benign)}")
    
    # =========================================================================
    # STAGE 1: BENIGN-ONLY TRAINING DATA
    # =========================================================================
    print("\n--- Processing Stage 1 (Benign only) ---")
    
    X_train = df_train_benign.drop(columns=['ANOMALY'])
    print(f"Features: {X_train.shape[1]}")
    
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=X_train.columns
    )
    
    X_train_scaled.to_csv(OUTPUT_TRAIN_BENIGN, index=False)
    print(f"Saved: {OUTPUT_TRAIN_BENIGN}")
    
    # =========================================================================
    # STAGE 2: FULL TRAINING DATA (WITH ATTACKS)
    # =========================================================================
    print("\n--- Processing Stage 2 (Full dataset) ---")
    
    anomaly_labels = df_train['ANOMALY'].copy()
    X_s2_features = df_train.drop(columns=['ANOMALY'])
    
    X_s2_scaled = pd.DataFrame(
        scaler.transform(X_s2_features),
        columns=X_s2_features.columns
    )
    X_s2_scaled['ANOMALY'] = anomaly_labels.values
    
    X_s2_scaled.to_csv(OUTPUT_TRAIN, index=False)
    print(f"Saved: {OUTPUT_TRAIN}")
    
    # =========================================================================
    # TEST DATA
    # =========================================================================
    print("\n--- Processing Test Data ---")
    
    anomaly_test = df_test['ANOMALY'].copy()
    X_test = df_test.drop(columns=['ANOMALY'])
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    X_test_scaled['ANOMALY'] = anomaly_test.values
    
    X_test_scaled.to_csv(OUTPUT_TEST, index=False)
    print(f"Saved: {OUTPUT_TEST}")
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE - CORES-IoT")
    print("=" * 60)
    print(f"Output directory: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
