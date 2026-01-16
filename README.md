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
│       └── netflow/
│           ├── train_processed.csv        # Full training set (for Stage 2)
│           ├── train_processed_benign.csv # Benign-only (for Stage 1)
│           └── test_processed.csv         # Test set
│
├── models/                           # Trained model files (.pkl)
│   ├── iso_forest_0.45.pkl          # Isolation Forest (Stage 1)
│   ├── sgd_classifier.pkl           # SGD Classifier (Stage 2)
│   ├── mlp_classifier.pkl           # MLP Classifier (Stage 2)
│   └── dynamic_ensemble.pkl         # Dynamic Ensemble (Stage 2)
│
├── evaluation/                       # Evaluation results per model and dataset
│   ├── isolation_forest/
│   │   └── netflow/
│   │       └── isolation_forest_evaluation.csv
│   ├── sgd/
│   │   └── netflow/
│   │       └── sgd_evaluation.csv
│   ├── mlp/
│   │   └── netflow/
│   │       └── mlp_evaluation.csv
│   └── ensemble/
│       └── netflow/
│           └── ensemble_evaluation.csv
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
│   ├── KDD/                          # KDD dataset analysis (future)
│   └── Iot-23/                       # IoT-23 dataset analysis (future)
│
├── src/                              # Source code
│   ├── preprocessing/
│   │   └── data_preprocessing_netflow.py   # Data preprocessing script
│   ├── train/
│   │   ├── isoForestTrain.py        # Isolation Forest training (Stage 1)
│   │   ├── sgdTrain.py              # SGD Classifier training (Stage 2)
│   │   ├── mlpTrain.py              # MLP Classifier training (Stage 2)
│   │   └── ensembleTrain.py         # Dynamic Ensemble training (Stage 2)
│   └── train_all.py                 # Main training pipeline script
│
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

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

### 1. Data Preprocessing

Prepares raw datasets for training. Handles cleaning, encoding, and scaling.

```bash
python src/preprocessing/data_preprocessing_netflow.py
```

**Input**: `datasets/raw/netflow/train_net.csv`, `datasets/raw/netflow/test_net.csv`

**Output**:
- `datasets/processed/netflow/train_processed_benign.csv` - Benign-only for Stage 1
- `datasets/processed/netflow/train_processed.csv` - Full training set for Stage 2 warmup
- `datasets/processed/netflow/test_processed.csv` - Test set

### 2. Training Pipeline

Train both stages using the unified training script:

```bash
# Train Stage 1 (Isolation Forest) + Stage 2 (single model)
python src/train_all.py --train datasets/processed/netflow/train_processed.csv \
                        --test datasets/processed/netflow/test_processed.csv \
                        --model sgd

# Train all Stage 2 models at once
python src/train_all.py --train datasets/processed/netflow/train_processed.csv \
                        --test datasets/processed/netflow/test_processed.csv \
                        --model all

# Skip Stage 1 training (use existing Isolation Forest)
python src/train_all.py --train datasets/processed/netflow/train_processed.csv \
                        --test datasets/processed/netflow/test_processed.csv \
                        --model mlp \
                        --skip-stage1

# Use a specific Stage 1 model
python src/train_all.py --train datasets/processed/netflow/train_processed.csv \
                        --test datasets/processed/netflow/test_processed.csv \
                        --model ensemble \
                        --stage1-model models/iso_forest_0.45.pkl
```

**Options**:
- `--model`: Choose `sgd`, `mlp`, `ensemble`, or `all`
- `--skip-stage1`: Skip Isolation Forest training, use existing model
- `--stage1-model`: Path to a specific Stage 1 model

**Output**:
- Models saved to `models/`
- Evaluation results saved to `evaluation/<model>/<dataset>/`

### 3. Individual Training Scripts

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

## Experimental Results (NetFlow Dataset)

### Model Comparison (Optimal Batch Sizes)

| Metric | SGD | MLP | Ensemble |
|--------|-----|-----|----------|
| **Balanced Accuracy** | 87.38% | 90.85% | **96.04%** |
| **Recall (Attacks)** | 80.55% | 87.22% | **97.65%** |
| **F1-Score (Attacks)** | 0.7791 | 0.8218 | **0.8761** |
| **Processing Time** | **11.61s** | 17.87s | 139.15s |
| **Optimal Batch Size** | 25000 | 20000 | 25000 |

### Key Findings

- **Ensemble** achieves highest accuracy but requires more processing time
- **MLP** offers good balance between accuracy and speed
- **SGD** is fastest but with lower accuracy
- Stage 1 filter reduces Stage 2 load by ~45% (only suspicious traffic processed)

## Future Work

- [ ] Implement explicit drift detection
- [ ] Add support for more datasets (KDD, IoT-23)
- [ ] Feature engineering improvements
- [ ] Real-time streaming interface
- [ ] Model serving API
