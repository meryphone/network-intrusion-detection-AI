"""
Data Preprocessing Script for NetFlow Dataset.
Prepares training and test data for the intrusion detection system.

Run from project root:
    python src/preprocesssing/data_preprocessing_netflow.py
"""

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler

# =============================================================================
# CONFIGURATION
# =============================================================================
SAMPLE_FRAC = 0.8
RANDOM_STATE = 42

# Columns to drop:
# - ID columns: FLOW_ID, ID (unique identifiers, not predictive)
# - Metadata: ALERT, ANALYSIS_TIMESTAMP (not network features)
# - Address/Port: IPV4_SRC_ADDR, IPV4_DST_ADDR, L4_SRC_PORT, L4_DST_PORT 
#   (too specific, would cause overfitting)
# - Zero-variance: MIN_IP_PKT_LEN, MAX_IP_PKT_LEN, TOTAL_PKTS_EXP, TOTAL_BYTES_EXP
#   (constant values, provide no discrimination)
COLS_TO_DROP = [
    # ID columns
    'FLOW_ID', 'ID',
    # Metadata columns
    'ALERT', 'ANALYSIS_TIMESTAMP',
    # Address/Port columns (would cause overfitting)
    'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_SRC_PORT', 'L4_DST_PORT',
    # Zero-variance features (constant values in dataset)
    'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN', 'TOTAL_PKTS_EXP', 'TOTAL_BYTES_EXP',
]

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_TRAIN_PATH = PROJECT_ROOT / 'datasets' / 'raw' / 'netflow' / 'train_net.csv'
RAW_TEST_PATH = PROJECT_ROOT / 'datasets' / 'raw' / 'netflow' / 'test_net.csv'
PROCESSED_DIR = PROJECT_ROOT / 'datasets' / 'processed' / 'netflow'

OUTPUT_TRAIN_BENIGN = PROCESSED_DIR / 'train_processed_benign.csv'  # Stage 1 (benign only)
OUTPUT_TRAIN = PROCESSED_DIR / 'train_processed.csv'                # Stage 2 (full)
OUTPUT_TEST = PROCESSED_DIR / 'test_processed.csv'


def filter_columns(df: pd.DataFrame, cols_to_drop: list) -> pd.DataFrame:
    """Drop columns that exist in the dataframe."""
    existing_cols = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=existing_cols)


def encode_categorical(df: pd.DataFrame, encoders: dict = None, fit: bool = True) -> tuple:
    """Encode categorical columns. Returns (encoded_df, encoders)."""
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if encoders is None:
        encoders = {}
    
    for col in categorical_cols:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        elif col in encoders:
            df[col] = encoders[col].transform(df[col])
    
    return df, encoders


def main():
    """Main preprocessing pipeline."""
    
    print("=" * 60)
    print("DATA PREPROCESSING - NETFLOW")
    print("=" * 60)
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # LOAD AND SAMPLE DATA
    # =========================================================================
    print("\nLoading training data...")
    df_train_full = pd.read_csv(RAW_TRAIN_PATH)
    df_train = df_train_full.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)
    df_train = df_train.dropna(subset=['ANOMALY'])
    print(f"Training samples: {len(df_train)}")
    
    df_train_stage2 = df_train.copy()
    df_train_benign = df_train[df_train['ANOMALY'] == 0].copy()
    print(f"Benign samples (Stage 1): {len(df_train_benign)}")
    
    # =========================================================================
    # STAGE 1: BENIGN-ONLY TRAINING DATA
    # =========================================================================
    print("\n--- Processing Stage 1 (Benign only) ---")
    
    # Drop irrelevant columns and ANOMALY (label)
    X_train = filter_columns(df_train_benign, COLS_TO_DROP + ['ANOMALY'])
    X_train, label_encoders = encode_categorical(X_train, fit=True)
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
    
    X_s2 = filter_columns(df_train_stage2, COLS_TO_DROP)
    
    anomaly_labels = X_s2['ANOMALY'].copy()
    X_s2_features = X_s2.drop(columns=['ANOMALY'])
    
    X_s2_features, _ = encode_categorical(X_s2_features, encoders=label_encoders, fit=False)
    
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
    
    df_test_full = pd.read_csv(RAW_TEST_PATH)
    df_test = df_test_full.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)
    df_test = df_test.dropna(subset=['ANOMALY'])
    print(f"Test samples: {len(df_test)}")
    
    anomaly_test = df_test['ANOMALY'].copy()
    X_test = filter_columns(df_test, COLS_TO_DROP + ['ANOMALY'])
    
    X_test, _ = encode_categorical(X_test, encoders=label_encoders, fit=False)
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    X_test_scaled['ANOMALY'] = anomaly_test.values
    
    X_test_scaled.to_csv(OUTPUT_TEST, index=False)
    print(f"Saved: {OUTPUT_TEST}")
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()