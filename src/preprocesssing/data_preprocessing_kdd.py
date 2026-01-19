"""
Data Preprocessing Script for KDD Cup 1999 Dataset.
Prepares training and test data for the intrusion detection system.

Run from project root:
    python src/preprocesssing/data_preprocessing_kdd.py
"""

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler

# =============================================================================
# CONFIGURATION
# =============================================================================
SAMPLE_FRAC = 0.8
RANDOM_STATE = 42

# Column names from kddcup.names (41 features + label)
COLUMN_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label'
]

# Categorical columns that need encoding
CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_TRAIN_PATH = PROJECT_ROOT / 'datasets' / 'raw' / 'kd' / 'kddcup.data' / 'kddcup.data'
RAW_TEST_PATH = PROJECT_ROOT / 'datasets' / 'raw' / 'kd' / 'corrected' / 'corrected'
PROCESSED_DIR = PROJECT_ROOT / 'datasets' / 'processed' / 'kdd'

OUTPUT_TRAIN_BENIGN = PROCESSED_DIR / 'train_processed_benign.csv'  # Stage 1 (benign only)
OUTPUT_TRAIN = PROCESSED_DIR / 'train_processed.csv'                # Stage 2 (full)
OUTPUT_TEST = PROCESSED_DIR / 'test_processed.csv'


def load_kdd_data(file_path: Path) -> pd.DataFrame:
    """Load KDD data file with column names."""
    df = pd.read_csv(file_path, names=COLUMN_NAMES, header=None)
    return df


def convert_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Convert label column: 'normal.' -> 0, all attacks -> 1."""
    df = df.copy()
    # Remove trailing dot from labels
    df['label'] = df['label'].str.rstrip('.')
    # Convert to binary: normal=0, attack=1
    df['ANOMALY'] = (df['label'] != 'normal').astype(int)
    df = df.drop(columns=['label'])
    return df


def encode_categorical(df: pd.DataFrame, encoders: dict = None, fit: bool = True) -> tuple:
    """Encode categorical columns. Returns (encoded_df, encoders)."""
    df = df.copy()
    
    if encoders is None:
        encoders = {}
    
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
            
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        elif col in encoders:
            # Handle unseen categories by mapping to a default value
            known_classes = set(encoders[col].classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in known_classes else encoders[col].classes_[0]
            )
            df[col] = encoders[col].transform(df[col])
    
    return df, encoders


def main():
    """Main preprocessing pipeline."""
    
    print("=" * 60)
    print("DATA PREPROCESSING - KDD CUP 1999")
    print("=" * 60)
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # LOAD AND SAMPLE TRAINING DATA
    # =========================================================================
    print("\nLoading training data...")
    print(f"Source: {RAW_TRAIN_PATH}")
    df_train_full = load_kdd_data(RAW_TRAIN_PATH)
    print(f"Full training samples: {len(df_train_full)}")
    
    # Sample for faster processing
    df_train = df_train_full.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)
    print(f"Sampled training samples: {len(df_train)}")
    
    # Convert labels
    df_train = convert_labels(df_train)
    
    df_train_stage2 = df_train.copy()
    df_train_benign = df_train[df_train['ANOMALY'] == 0].copy()
    print(f"Benign samples (Stage 1): {len(df_train_benign)}")
    print(f"Attack samples: {len(df_train) - len(df_train_benign)}")
    
    # =========================================================================
    # STAGE 1: BENIGN-ONLY TRAINING DATA
    # =========================================================================
    print("\n--- Processing Stage 1 (Benign only) ---")
    
    X_train = df_train_benign.drop(columns=['ANOMALY'])
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
    
    anomaly_labels = df_train_stage2['ANOMALY'].copy()
    X_s2_features = df_train_stage2.drop(columns=['ANOMALY'])
    
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
    print(f"Source: {RAW_TEST_PATH}")
    
    df_test_full = load_kdd_data(RAW_TEST_PATH)
    print(f"Full test samples: {len(df_test_full)}")
    
    df_test = df_test_full.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)
    print(f"Sampled test samples: {len(df_test)}")
    
    # Convert labels
    df_test = convert_labels(df_test)
    
    anomaly_test = df_test['ANOMALY'].copy()
    X_test = df_test.drop(columns=['ANOMALY'])
    
    X_test, _ = encode_categorical(X_test, encoders=label_encoders, fit=False)
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    X_test_scaled['ANOMALY'] = anomaly_test.values
    
    X_test_scaled.to_csv(OUTPUT_TEST, index=False)
    print(f"Saved: {OUTPUT_TEST}")
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE - KDD")
    print("=" * 60)
    print(f"Output directory: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
