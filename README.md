# Network Intrusion Detection AI

A **Hybrid Two-Stage Network Intrusion Detection System (IDS)** using Machine Learning. Combines a lightweight static anomaly detector with an adaptive, dynamically updated classifier to detect network attacks efficiently.

## Project Goal

Detect network intrusions (attacks) in real-time network traffic data. Traditional static ML models struggle with evolving attack patterns, while fully dynamic models are computationally expensive. This project proposes a **Hybrid approach**:

1. **Stage 1 (Static Screening)**: A lightweight, unsupervised **Isolation Forest** model filters all traffic. Trained only on benign traffic to learn "normality", it flags deviations as "suspicious".
2. **Stage 2 (Dynamic Detection)**: A supervised classifier (Online Learning) processes *only* the traffic flagged as suspicious by Stage 1. It learns incrementally from new data, adapting to new attack types over time.

## Project Structure

```
intrusion-detection-AI/
├── datasets/
│   ├── raw/                          # Original unprocessed datasets
│   │   └── netflow/                  # NetFlow dataset
│   │       ├── train_net.csv
│   │       └── test_net.csv
│   └── processed/                    # Preprocessed datasets ready for training
│       ├── netflow/
│       │   ├── train_processed.csv        # Full training set (for Stage 2)
│       │   ├── train_processed_benign.csv # Benign-only (for Stage 1)
│       │   └── test_processed.csv         # Test set
│       ├── kdd/                      # KDD Cup 99 processed data
│       │   └── ...
│       └── cores_iot/                # CORES IoT processed data
│           └── ...
│
├── models/                           # Trained model files (.pkl) by dataset
│   ├── netflow/
│   │   ├── iso_forest_0.45.pkl      # Isolation Forest (Stage 1)
│   │   ├── sgd_classifier.pkl       # SGD Classifier (Stage 2)
│   │   ├── mlp_classifier.pkl       # MLP Classifier (Stage 2)
│   │   └── dynamic_ensemble.pkl     # Dynamic Ensemble (Stage 2)
│   ├── kdd/
│   │   └── ...
│   └── cores_iot/
│       └── ...
│
├── evaluation/                       # Evaluation results per model and dataset
│   ├── summary.csv                  # Cross-dataset comparison summary
│   ├── summary.md                   # Markdown comparison report
│   ├── isolation_forest/
│   │   ├── netflow/
│   │   │   └── isolation_forest_evaluation.csv
│   │   ├── kdd/
│   │   └── cores_iot/
│   ├── sgd/
│   │   ├── netflow/
│   │   │   └── sgd_evaluation.csv
│   │   ├── kdd/
│   │   └── cores_iot/
│   ├── mlp/
│   │   └── ...
│   └── ensemble/
│       └── ...
│
├── notebooks/                        # Jupyter notebooks for analysis (by dataset)
│   ├── netflow/                      # NetFlow dataset analysis
│   │   ├── netflow_data_analisys.ipynb      # Raw data exploration
│   │   ├── netflow_processed_analysis.ipynb # Processed data analysis
│   │   ├── iso_analysis.ipynb               # Isolation Forest results
│   │   ├── sgd_results.ipynb                # SGD classifier results
│   │   ├── mlp_results.ipynb                # MLP classifier results
│   │   ├── ensemble_results.ipynb           # Dynamic Ensemble results
│   │   └── comparasing_results.ipynb        # Model comparison
│   ├── kdd/                          # KDD dataset analysis
│   └── cores_iot/                    # CORES IoT dataset analysis
│
├── src/                              # Source code
│   ├── datasets/
│   │   └── registry.py              # Central dataset configuration
│   ├── evaluation/
│   │   └── compare_results.py       # Cross-dataset comparison script
│   ├── preprocesssing/
│   │   ├── data_preprocessing_netflow.py
│   │   ├── data_preprocessing_kdd.py
│   │   └── data_preprocessing_cores_iot.py
│   ├── train/
│   │   ├── isoForestTrain.py        # Isolation Forest training (Stage 1)
│   │   ├── sgdTrain.py              # SGD Classifier training (Stage 2)
│   │   ├── mlpTrain.py              # MLP Classifier training (Stage 2)
│   │   └── ensembleTrain.py         # Dynamic Ensemble training (Stage 2)
│   ├── train_all.py                 # Main training pipeline script
│   └── run_all_experiments.py       # Multi-dataset experiment runner
│
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Supported Datasets

| Dataset | Description | Train Samples | Test Samples | Features (after cleaning) |
|---------|-------------|---------------|--------------|---------------------------|
| **NetFlow** | Network flow traffic data | 1,851,011 | 911,914 | 20 (from 33) |
| **KDD Cup 99** | Classic intrusion detection benchmark | 3,918,745 | 248,823 | 32 (from 41) |
| **CORES IoT** | IoT network intrusion data | 806,998 | 201,750 | 14 (from 19) |

## Data Preprocessing & Cleaning

Each dataset undergoes the following preprocessing pipeline:

### Cleaning Steps Applied

| Step | NetFlow | KDD | CORES-IoT |
|------|---------|-----|-----------|
| **Missing Values** | `dropna(ANOMALY)` | None present | None present |
| **Duplicate Removal** | Not applied* | Not applied* | Not applied* |
| **Feature Removal** | 12 columns dropped | 9 columns dropped | 5 columns dropped |
| **Label Encoding** | Categorical → numeric | 3 categorical columns | N/A (all numeric) |
| **Scaling** | RobustScaler | RobustScaler | RobustScaler |
| **Train/Test Split** | Pre-split files | Pre-split files | 80/20 stratified |

*Duplicates preserved as they may represent valid repeated patterns in network traffic.

### Features Removed (based on importance analysis)

**NetFlow (12 features removed):**
- ID columns: `FLOW_ID`, `ID`
- Metadata: `ALERT`, `ANALYSIS_TIMESTAMP`
- Address/Port (overfitting risk): `IPV4_SRC_ADDR`, `IPV4_DST_ADDR`, `L4_SRC_PORT`, `L4_DST_PORT`
- Zero-variance: `MIN_IP_PKT_LEN`, `MAX_IP_PKT_LEN`, `TOTAL_PKTS_EXP`, `TOTAL_BYTES_EXP`

**KDD (9 features removed):**
- Zero-variance (100% constant): `land`, `num_outbound_cmds`, `is_host_login`
- Near-zero variance + zero importance: `su_attempted`, `wrong_fragment`, `num_failed_logins`, `num_access_files`, `rerror_rate`, `srv_rerror_rate`

**CORES-IoT (5 features removed):**
- Zero-variance: `feature_11`, `feature_12`, `feature_13`
- Near-zero importance: `feature_6` (99.7% = 0), `feature_0` (unique per row, like ID)

### Feature Analysis Notebooks

Comprehensive feature analysis is available in the notebooks:
- [notebooks/kdd/kdd_data_analysis.ipynb](notebooks/kdd/kdd_data_analysis.ipynb) - KDD dataset analysis
- [notebooks/cores_iot/cores_iot_data_analysis.ipynb](notebooks/cores_iot/cores_iot_data_analysis.ipynb) - CORES-IoT analysis
- [notebooks/netflow/netflow_data_analisys.ipynb](notebooks/netflow/netflow_data_analisys.ipynb) - NetFlow analysis

Each notebook includes:
1. Missing values analysis
2. Duplicate detection
3. Class distribution & imbalance
4. Feature correlation analysis
5. Outlier detection (IQR method)
6. Feature importance (Random Forest)

## Architecture & Methods

The system supports multiple strategies for Stage 2, allowing comparison between different levels of complexity:

| Model | Type | Description |
|-------|------|-------------|
| **SGDClassifier** | Linear | Fast, lightweight online learning with stochastic gradient descent |
| **MLPClassifier** | Neural Network | Captures non-linear patterns with online learning (`partial_fit`) |
| **Dynamic Ensemble** | Ensemble | Pool of MLP classifiers using SEA algorithm. Trains new model per batch and prunes worst performer |

### Pipeline Flow

```
Incoming Traffic → Stage 1 (Isolation Forest)
                          │
                          ├─ Normal → Classification: BENIGN (Stop)
                          │
                          └─ Suspicious → Stage 2 (Adaptive Classifier)
                                                │
                                                ├─ Predict: Attack or Benign
                                                │
                                                └─ Update model with true label
                                                   (online learning)
```

## Setup & Installation

### Prerequisites
- Python 3.8+
- Virtual Environment (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd intrusion-detection-AI
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # venv\Scripts\activate   # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start: Run All Experiments

The easiest way to run the full pipeline across all datasets:

```bash
# Full pipeline: preprocess + train + compare all datasets
python src/run_all_experiments.py

# Run for a specific dataset only
python src/run_all_experiments.py --dataset netflow
python src/run_all_experiments.py --dataset kdd
python src/run_all_experiments.py --dataset cores_iot

# Compare existing results only
python src/run_all_experiments.py --compare-only
```

### 1. Data Preprocessing

Prepares raw datasets for training. Handles cleaning, encoding, and scaling.

```bash
# Preprocess individual datasets
python src/preprocesssing/data_preprocessing_netflow.py
python src/preprocesssing/data_preprocessing_kdd.py
python src/preprocesssing/data_preprocessing_cores_iot.py

# Or preprocess all at once
python src/run_all_experiments.py --preprocess-only
```

**Output** (per dataset):
- `datasets/processed/<dataset>/train_processed_benign.csv` - Benign-only for Stage 1
- `datasets/processed/<dataset>/train_processed.csv` - Full training set for Stage 2 warmup
- `datasets/processed/<dataset>/test_processed.csv` - Test set

### 2. Training Pipeline

Train both stages using the unified training script with `--dataset`:

```bash
# Train using dataset name (recommended)
python src/train_all.py --dataset netflow --model sgd
python src/train_all.py --dataset kdd --model mlp
python src/train_all.py --dataset cores_iot --model ensemble

# Train all Stage 2 models for a dataset
python src/train_all.py --dataset netflow --model all

# Custom contamination value for Isolation Forest
python src/train_all.py --dataset netflow --model all --contamination 0.3

# Skip Stage 1 training (use existing Isolation Forest)
python src/train_all.py --dataset netflow --model mlp --skip-stage1

# Use a specific Stage 1 model
python src/train_all.py --dataset netflow --model ensemble \
                        --stage1-model models/netflow/iso_forest_0.45.pkl

# Legacy: Train with explicit paths
python src/train_all.py --train datasets/processed/netflow/train_processed.csv \
                        --test datasets/processed/netflow/test_processed.csv \
                        --model sgd
```

**Options**:
- `--dataset`: Dataset name (`netflow`, `kdd`, `cores_iot`)
- `--model`: Choose `sgd`, `mlp`, `ensemble`, or `all`
- `--contamination`: Isolation Forest contamination value (default: 0.45)
- `--skip-stage1`: Skip Isolation Forest training, use existing model
- `--stage1-model`: Path to a specific Stage 1 model

**Output**:
- Models saved to `models/<dataset>/`
- Evaluation results saved to `evaluation/<model>/<dataset>/`

### 3. Cross-Dataset Comparison

Compare results across all datasets and models:

```bash
# Generate comparison summary
python src/evaluation/compare_results.py

# Output formats
python src/evaluation/compare_results.py --format csv      # CSV only
python src/evaluation/compare_results.py --format markdown # Markdown only
python src/evaluation/compare_results.py --format both     # Both (default)
```

**Output**:
- `evaluation/summary.csv` - Full results table
- `evaluation/summary_best.csv` - Best result per model/dataset
- `evaluation/summary.md` - Markdown summary

### 4. Individual Training Scripts

You can also train models individually:

```bash
# Stage 1: Isolation Forest
python src/train/isoForestTrain.py --train datasets/processed/netflow/train_processed_benign.csv \
                                   --test datasets/processed/netflow/test_processed.csv

# Stage 2: SGD Classifier
python src/train/sgdTrain.py --train datasets/processed/netflow/train_processed.csv \
                             --test datasets/processed/netflow/test_processed.csv \
                             --stage1-model models/iso_forest_0.45.pkl

# Stage 2: MLP Classifier
python src/train/mlpTrain.py --train datasets/processed/netflow/train_processed.csv \
                             --test datasets/processed/netflow/test_processed.csv \
                             --stage1-model models/iso_forest_0.45.pkl

# Stage 2: Dynamic Ensemble
python src/train/ensembleTrain.py --train datasets/processed/netflow/train_processed.csv \
                                  --test datasets/processed/netflow/test_processed.csv \
                                  --stage1-model models/iso_forest_0.45.pkl
```

## Components Description

### Stage 1: Isolation Forest (`isoForestTrain.py`)

- **Purpose**: Unsupervised anomaly detection trained only on benign traffic
- **Method**: Identifies anomalies by how quickly they are isolated in random trees
- **Key Parameter**: `contamination` - Controls the threshold for flagging suspicious traffic
- **Output**: Flags traffic as "Normal" or "Suspicious" for Stage 2

### Stage 2 Models

#### SGDClassifier (`sgdTrain.py`)
- **Type**: Linear classifier with stochastic gradient descent
- **Pros**: Very fast, low memory usage
- **Cons**: Limited to linear decision boundaries
- **Best for**: High-throughput scenarios where speed is critical

#### MLPClassifier (`mlpTrain.py`)
- **Type**: Multi-layer Perceptron neural network
- **Architecture**: Single hidden layer (100 neurons), ReLU activation
- **Pros**: Captures non-linear patterns, online learning via `partial_fit`
- **Best for**: Balance between accuracy and speed

#### Dynamic Ensemble (`ensembleTrain.py`)
- **Type**: Pool of MLP classifiers using SEA (Streaming Ensemble Algorithm)
- **Method**: Trains new model per batch, prunes worst performer
- **Pros**: High accuracy, robust to concept drift
- **Cons**: Higher computational cost
- **Best for**: Maximum accuracy when resources allow

## Evaluation Metrics

Each model evaluation includes:

| Metric | Description |
|--------|-------------|
| `accuracy` | Overall correct predictions |
| `balanced_accuracy` | Average of recall for each class |
| `precision_anomaly` | Precision for attack detection |
| `recall_anomaly` | Attack detection rate (sensitivity) |
| `f1_anomaly` | Harmonic mean of precision and recall |
| `stage2_call_rate` | Percentage of traffic reaching Stage 2 |
| `total_time_sec` | Processing time for the batch size |

## Analysis Notebooks

Located in `notebooks/<dataset>/`:

| Notebook | Description |
|----------|-------------|
| `netflow_data_analisys.ipynb` | Explore raw dataset characteristics |
| `netflow_processed_analysis.ipynb` | Analyze preprocessed data |
| `iso_analysis.ipynb` | Isolation Forest performance across contamination levels |
| `sgd_results.ipynb` | SGD classifier performance vs batch size |
| `mlp_results.ipynb` | MLP classifier performance vs batch size |
| `ensemble_results.ipynb` | Ensemble performance vs batch size |
| `comparasing_results.ipynb` | Compare all Stage 2 models |

## Experimental Results

### Cross-Dataset Comparison Summary

Results from training the hybrid IDS across all three datasets:

| Dataset | Best Model | Accuracy | Balanced Accuracy | F1 Macro | Stage 2 Call Rate |
|---------|------------|----------|-------------------|----------|-------------------|
| **CORES-IoT** | Ensemble | **95.64%** | 95.32% | **95.58%** | 74.66% |
| **KDD** | MLP | 95.15% | **95.52%** | 92.72% | 82.01% |
| **NetFlow** | Ensemble | 94.93% | 95.98% | 92.13% | 54.76% |

### NetFlow Dataset Results

| Model | Accuracy | Balanced Accuracy | F1 Macro | Recall (Attacks) | Stage 2 Call Rate | Time |
|-------|----------|-------------------|----------|------------------|-------------------|------|
| **Ensemble** | 94.93% | **95.98%** | **92.13%** | **97.64%** | 54.76% | 109.95s |
| MLP | 93.00% | 90.23% | 88.65% | 85.88% | 54.76% | 15.77s |
| SGD | 91.74% | 87.38% | 86.41% | 80.55% | 54.76% | **11.51s** |
| Isolation Forest | 62.96% | N/A | 60.01% | 98.99% | N/A | 9.21s |

**Key Findings (NetFlow):**
- **Ensemble** achieves highest accuracy with 95.98% balanced accuracy
- Stage 1 filter reduces Stage 2 load by ~45% (only 54.76% of traffic flagged)
- SGD is fastest but with lower accuracy

### KDD Cup 99 Dataset Results

| Model | Accuracy | Balanced Accuracy | F1 Macro | Recall (Attacks) | Stage 2 Call Rate | Time |
|-------|----------|-------------------|----------|------------------|-------------------|------|
| **MLP** | **95.15%** | **95.52%** | **92.72%** | 94.91% | 82.01% | 34.71s |
| Ensemble | 95.03% | 96.11% | 92.65% | 94.35% | 82.01% | 327.13s |
| SGD | 91.78% | 90.85% | 87.81% | 92.37% | 82.01% | 21.73s |
| Isolation Forest | 91.74% | N/A | 86.43% | **95.78%** | N/A | **3.95s** |

**Key Findings (KDD):**
- **MLP** slightly outperforms Ensemble while being 10x faster
- High Stage 2 call rate (82%) indicates more suspicious traffic patterns
- Isolation Forest performs better on KDD than NetFlow (F1 0.86 vs 0.60)

### CORES-IoT Dataset Results

| Model | Accuracy | Balanced Accuracy | F1 Macro | Recall (Attacks) | Stage 2 Call Rate | Time |
|-------|----------|-------------------|----------|------------------|-------------------|------|
| **Ensemble** | **95.64%** | **95.32%** | **95.58%** | **99.99%** | 74.66% | 51.25s |
| MLP | 90.96% | 90.98% | 90.92% | 90.57% | 74.66% | 6.16s |
| SGD | 89.58% | 89.51% | 89.53% | 90.50% | 74.66% | **4.41s** |
| Isolation Forest | 78.82% | N/A | 77.00% | 100% | N/A | 3.05s |

**Key Findings (CORES-IoT):**
- **Ensemble** achieves highest overall F1 Macro (95.58%) across all datasets
- Near-perfect attack recall (99.99%) - captures almost all attacks
- Smallest dataset leads to fastest training times

### Stage 2 Model Comparison (F1 Macro)

| Model | CORES-IoT | KDD | NetFlow |
|-------|-----------|-----|---------|
| **Ensemble** | **95.58%** | 92.65% | **92.13%** |
| MLP | 90.92% | **92.72%** | 88.65% |
| SGD | 89.53% | 87.81% | 86.41% |

### Stage 2 Model Comparison (Balanced Accuracy)

| Model | CORES-IoT | KDD | NetFlow |
|-------|-----------|-----|---------|
| **Ensemble** | **95.32%** | **96.11%** | **95.98%** |
| MLP | 90.98% | 95.52% | 90.23% |
| SGD | 89.51% | 90.85% | 87.38% |

### Key Insights (Pre-Cleaning)

1. **Ensemble consistently performs best** across datasets, achieving 95%+ accuracy on all three
2. **MLP offers the best speed-accuracy tradeoff**, especially on larger datasets like KDD
3. **Stage 1 filtering is effective**: Reduces Stage 2 load by 18-45% depending on dataset
4. **Dataset characteristics matter**:
   - NetFlow: Lower Stage 2 call rate suggests cleaner benign traffic patterns
   - KDD: Higher call rate indicates more ambiguous traffic
   - CORES-IoT: Highest F1 scores suggest clearer attack signatures in IoT data

---

## Results After Data Cleaning

After applying feature importance analysis and removing irrelevant/zero-variance features (see [Data Preprocessing & Cleaning](#data-preprocessing--cleaning) section for details).

### Cross-Dataset Comparison Summary (Post-Cleaning)

| Dataset | Best Model | Accuracy | Balanced Accuracy | F1 Macro | Stage 2 Call Rate |
|---------|------------|----------|-------------------|----------|-------------------|
| **NetFlow** | Ensemble | **97.52%** | **97.66%** | **95.96%** | 54.64% |
| **CORES-IoT** | Ensemble | **95.64%** | 95.31% | **95.58%** | 74.65% |
| **KDD** | Ensemble | 95.01% | **96.13%** | 92.62% | 85.04% |

### NetFlow Dataset Results (Post-Cleaning)

| Model | Accuracy | Balanced Accuracy | F1 Macro | Recall (Attacks) | Stage 2 Call Rate | Time |
|-------|----------|-------------------|----------|------------------|-------------------|------|
| **Ensemble** | **97.52%** | **97.66%** | **95.96%** | **97.89%** | 54.64% | 130.14s |
| MLP | 95.05% | 93.10% | 91.88% | 90.04% | 54.64% | 14.93s |
| SGD | 91.08% | 83.08% | 84.36% | 70.55% | 54.64% | **11.07s** |
| Isolation Forest | 63.05% | N/A | 60.08% | 98.92% | N/A | 9.24s |

**Key Findings (NetFlow Post-Cleaning):**
- **Accuracy improved from 94.93% to 97.52%** (+2.59% with Ensemble)
- **MLP improved from 93.00% to 95.05%** (+2.05%)
- Removing irrelevant features significantly improved model performance

### KDD Cup 99 Dataset Results (Post-Cleaning)

| Model | Accuracy | Balanced Accuracy | F1 Macro | Recall (Attacks) | Stage 2 Call Rate | Time |
|-------|----------|-------------------|----------|------------------|-------------------|------|
| **Ensemble** | 95.01% | **96.13%** | 92.62% | 94.29% | 85.04% | 372.77s |
| MLP | **94.84%** | 95.42% | **92.31%** | 94.47% | 85.04% | 27.79s |
| SGD | 91.55% | 90.13% | 87.41% | 92.46% | 85.04% | 16.75s |
| Isolation Forest | 90.89% | N/A | 84.01% | **97.13%** | N/A | **3.93s** |

**Key Findings (KDD Post-Cleaning):**
- Performance remained stable after removing 9 irrelevant features
- **Ensemble balanced accuracy improved from 96.11% to 96.13%**
- Training time slightly reduced due to fewer features

### CORES-IoT Dataset Results (Post-Cleaning)

| Model | Accuracy | Balanced Accuracy | F1 Macro | Recall (Attacks) | Stage 2 Call Rate | Time |
|-------|----------|-------------------|----------|------------------|-------------------|------|
| **Ensemble** | **95.64%** | **95.31%** | **95.58%** | **99.98%** | 74.65% | 55.46s |
| SGD | 90.45% | 90.35% | 90.39% | 91.74% | 74.65% | 5.05s |
| MLP | 89.23% | 89.38% | 89.21% | 87.25% | 74.65% | 6.62s |
| Isolation Forest | 78.83% | N/A | 77.01% | 100% | N/A | **3.65s** |

**Key Findings (CORES-IoT Post-Cleaning):**
- **Ensemble maintained 95.64% accuracy** after removing 5 irrelevant features
- **SGD improved from 89.58% to 90.45%** (+0.87%)
- Near-perfect attack recall (99.98%) preserved

### Comparison: Pre-Cleaning vs Post-Cleaning (Best Model per Dataset)

| Dataset | Model | Accuracy (Before) | Accuracy (After) | Change |
|---------|-------|-------------------|------------------|--------|
| **NetFlow** | Ensemble | 94.93% | **97.52%** | **+2.59%** |
| **KDD** | Ensemble | 95.03% | 95.01% | -0.02% |
| **CORES-IoT** | Ensemble | 95.64% | 95.64% | 0.00% |

### Stage 2 Model Comparison - F1 Macro (Post-Cleaning)

| Model | CORES-IoT | KDD | NetFlow |
|-------|-----------|-----|---------|
| **Ensemble** | **95.58%** | 92.62% | **95.96%** |
| MLP | 89.21% | **92.31%** | 91.88% |
| SGD | 90.39% | 87.41% | 84.36% |

### Stage 2 Model Comparison - Balanced Accuracy (Post-Cleaning)

| Model | CORES-IoT | KDD | NetFlow |
|-------|-----------|-----|---------|
| **Ensemble** | **95.31%** | **96.13%** | **97.66%** |
| MLP | 89.38% | 95.42% | 93.10% |
| SGD | 90.35% | 90.13% | 83.08% |

### Key Insights (Post-Cleaning)

1. **Feature cleaning significantly improved NetFlow results** - Accuracy jumped from 94.93% to 97.52% by removing 12 irrelevant features
2. **KDD and CORES-IoT remained stable** - These datasets already had cleaner feature sets
3. **Ensemble remains the best performer** across all datasets post-cleaning
4. **Reduced feature count** improves training efficiency:
   - NetFlow: 33 → 20 features (39% reduction)
   - KDD: 41 → 32 features (22% reduction)
   - CORES-IoT: 19 → 14 features (26% reduction)
