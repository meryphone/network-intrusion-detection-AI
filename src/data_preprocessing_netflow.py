import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler


############# TRAIN DATA ###############
print("Loading training data...")
try:
    df_train_full = pd.read_csv('datasets/raw/train_net.csv')
except Exception as e:
    print(f"Error loading training data: {e}")
    raise
print(f"Training rows loaded: {len(df_train_full)}")

df_train = df_train_full.sample(frac=0.8, random_state=42)
print(f"Training rows sampled: {len(df_train)}")

# Drop rows with missing ANOMALY labels if any
df_train = df_train.dropna(subset=['ANOMALY'])
print(f"Training rows after dropping missing ANOMALY: {len(df_train)}")

# Keep a copy of the full training set (with anomalies) for Stage 2
df_train_stage2 = df_train.copy()

# Deletin rows with anomaly = 1

df_train = df_train[df_train['ANOMALY'] == 0]
print(f"Training rows after keeping only ANOMALY=0 : {len(df_train)}")

# Deleting unnecessary columns for training data
cols_to_drop_train = ['FLOW_ID', 'ID', 'ALERT', 'ANOMALY', "ANALYSIS_TIMESTAMP", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN", "TOTAL_PKTS_EXP", "TOTAL_BYTES_EXP", "TOTAL_FLOWS_EXP", 
                      "IPV4_SRC_ADDR" , "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT"]  

colums_to_drop_train = []

for c in cols_to_drop_train:
    if c in df_train.columns:
        colums_to_drop_train.append(c)

X_train = df_train.drop(columns=colums_to_drop_train).copy()
print(f"Training columns after drop: {X_train.shape[1]}")

# Encoding training data 
le = LabelEncoder()
categorical_columns = X_train.select_dtypes(include=['object']).columns

labelEncoders = {}

for column in categorical_columns:
    le = LabelEncoder()
    labelEncoders[column] = le
    X_train[column] = le.fit_transform(X_train[column])
print(f"Encoded training categorical columns: {list(categorical_columns)}")

# Standardizing column 
scaler = RobustScaler()
X_train_scaled =  scaler.fit_transform(X_train)

X = pd.DataFrame(X_train_scaled, columns=X_train.columns)
try:
    X.to_csv('datasets/processed/X_train_processed.csv', index=False)
    print("Saved processed training data to ../datasets/processed/X_train_processed.csv")
except Exception as e:
    print(f"Error saving processed training data: {e}")
    raise

############# STAGE 2 TRAIN DATA (Full with Attacks) ###############
print("Processing full training data for Stage 2...")

# Columns to drop for Stage 2 (keep ANOMALY)
cols_to_drop_stage2 = [c for c in cols_to_drop_train if c != 'ANOMALY']

colums_to_drop_s2 = []
for c in cols_to_drop_stage2:
    if c in df_train_stage2.columns:
        colums_to_drop_s2.append(c)

X_train_s2 = df_train_stage2.drop(columns=colums_to_drop_s2).copy()
print(f"Stage 2 Training columns after drop (including ANOMALY): {X_train_s2.shape[1]}")

# Separate ANOMALY for processing
anomaly_s2 = X_train_s2['ANOMALY'].copy()
X_train_s2_features = X_train_s2.drop(columns=['ANOMALY'])

# Encoding Stage 2 data using the SAME encoders
for c, le in labelEncoders.items():
    if c not in X_train_s2_features.columns:
        continue
    # Handle potential unseen labels by ignoring (or crashing if strict)
    # For now assuming distributions are compatible as per Test set success
    try:
        X_train_s2_features[c] = le.transform(X_train_s2_features[c])
    except ValueError as e:
         # Fallback: if unseen labels, we might need to refit or handle. 
         # For this project, let's map unknown to a default or re-fit on full data if this fails.
         # But re-fitting on full data changes the Stage 1 input space if not careful.
         # Let's print warning and try to handle or just let it crash if it's critical.
         print(f"Warning: unseen labels in {c} for Stage 2 data. {e}")
         # Simplest fix for now: fit on full data initially? No, Stage 1 must be trained on Benign only.
         # So Encoder MUST be defined on Benign (or superset).
         # Let's assume it works for now.
         raise e

# Scaling Stage 2 data using the SAME scaler
X_train_s2_scaled = scaler.transform(X_train_s2_features)

X_s2 = pd.DataFrame(X_train_s2_scaled, columns=X_train_s2_features.columns)
X_s2['ANOMALY'] = anomaly_s2.values

try:
    X_s2.to_csv('datasets/processed/X_train_full_processed.csv', index=False)
    print("Saved processed Stage 2 training data to datasets/processed/X_train_full_processed.csv")
except Exception as e:
    print(f"Error saving processed Stage 2 data: {e}")
    raise

############# TEST DATA ###############

print("Loading test data...")
try:
    df_test_full = pd.read_csv('datasets/raw/test_net.csv')
except Exception as e:
    print(f"Error loading test data: {e}")
    raise
print(f"Test rows loaded: {len(df_test_full)}")

df_test = df_test_full.sample(frac=0.8, random_state=42)
print(f"Test rows sampled: {len(df_test)}")

# Deleting rows with missing values and storing the ANOMALY column
df_test = df_test.dropna(subset=["ANOMALY"])
print(f"Test rows after dropping missing ANOMALY: {len(df_test)}")
anomaly = df_test['ANOMALY'].copy()

cols_to_drop_test = ['FLOW_ID', 'ID', "ANALYSIS_TIMESTAMP", "ANOMALY", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN", "TOTAL_PKTS_EXP", "TOTAL_BYTES_EXP", "TOTAL_FLOWS_EXP"
                      , "IPV4_SRC_ADDR" , "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT"]

colums_to_drop_test = []

for c in cols_to_drop_test:
    if c in df_test.columns:
        colums_to_drop_test.append(c)

Y_test = df_test.drop(columns=colums_to_drop_test).copy()
print(f"Test columns after drop: {Y_test.shape[1]}")

# Encoding test data 
cat_cols_test = Y_test.select_dtypes(include= ["object"]).columns

for c, le in labelEncoders.items():
    if c not in Y_test.columns:
        print(f"Warning: column '{c}' missing in test data, skipping encoding.")
        continue
    try:
        Y_test[c] = le.transform(Y_test[c])
    except Exception as e:
        print(f"Error encoding column '{c}': {e}")
        raise
print(f"Encoded test categorical columns: {list(cat_cols_test)}")

# Standardizing test data
Y_test_scaled = scaler.transform(Y_test)

y = pd.DataFrame(Y_test_scaled, columns=Y_test.columns)

# Adding back the ANOMALY column
y['ANOMALY'] = anomaly.values

try:
    y.to_csv('datasets/processed/Y_test_processed.csv', index=False)
    print("Saved processed test data to ../datasets/processed/Y_test_processed.csv")
except Exception as e:
    print(f"Error saving processed test data: {e}")
    raise