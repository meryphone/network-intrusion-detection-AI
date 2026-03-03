# Network Intrusion Detection AI

A **Hybrid Two-Stage Network Intrusion Detection System (IDS)** using Machine Learning. Combines a lightweight unsupervised anomaly detector (Isolation Forest) with an adaptive, online-learning classifier to detect network attacks across multiple datasets.

## About the Project

### Problem

Traditional signature-based intrusion detection systems fail to keep up with constantly evolving network attacks. Static machine learning models, while effective initially, degrade over time as new attack patterns emerge (concept drift). Fully retraining models on every new data batch is computationally expensive and impractical for real-time traffic analysis.

### Proposed Solution

This project implements a **hybrid two-stage pipeline** that balances efficiency and adaptability:

1. **Stage 1 — Static Screening (Isolation Forest):** A lightweight, unsupervised anomaly detector trained exclusively on benign (normal) network traffic. It learns a model of "normality" and flags any traffic that deviates significantly as *suspicious*. Traffic classified as normal is immediately cleared, reducing the workload for the next stage.

2. **Stage 2 — Adaptive Classification (Online Learning):** Only the suspicious traffic flagged by Stage 1 is forwarded to a supervised classifier that supports **online learning** (`partial_fit`). This model processes data in batches, incrementally updating its parameters with each new batch to adapt to emerging attack types without full retraining.

This two-stage design provides two key advantages:
- **Efficiency**: Stage 1 filters out the majority of benign traffic (55–85% depending on the dataset), so Stage 2 only processes a fraction of the total volume.
- **Adaptability**: Stage 2 continuously learns from new labeled data, keeping the system effective as attack patterns evolve over time.

### Datasets

The system is evaluated across three network intrusion detection datasets with different characteristics:

| Dataset | Description | Train Samples | Test Samples | Features (after cleaning) |
|---------|-------------|---------------|--------------|---------------------------|
| **NetFlow** | Network flow traffic data | 1,851,011 | 911,914 | 20 (from 33) |
| **KDD Cup 99** | Classic intrusion detection benchmark | 3,918,745 | 248,823 | 32 (from 41) |
| **CORES IoT** | IoT network intrusion data | 806,998 | 201,750 | 14 (from 19) |

### Models

| Model | Type | Role | Description |
|-------|------|------|-------------|
| **Isolation Forest** | Unsupervised | Stage 1 | Anomaly detection trained on benign traffic only |
| **SGDClassifier** | Linear | Stage 2 | Fast online learning with stochastic gradient descent |
| **MLPClassifier** | Neural Network | Stage 2 | Non-linear patterns via single hidden layer (100 neurons), online learning with `partial_fit` |
| **Dynamic Ensemble** | Ensemble | Stage 2 | Pool of MLP classifiers using SEA (Streaming Ensemble Algorithm); trains a new model per batch and prunes the worst performer |

### Pipeline Flow

```
Incoming Traffic → Stage 1 (Isolation Forest)
                          │
                          ├─ Normal → BENIGN (no further processing)
                          │
                          └─ Suspicious → Stage 2 (Online Classifier)
                                                │
                                                ├─ Predict: Attack or Benign
                                                └─ Update model with new batch (online learning)
```

## Project Structure

```
intrusion-detection-AI/
├── datasets/
│   ├── raw/                          # Original unprocessed datasets
│   │   ├── netflow/
│   │   ├── kd/
│   │   └── cores-iot/
│   └── processed/                    # Preprocessed datasets ready for training
│       ├── netflow/
│       │   ├── train_processed.csv
│       │   ├── train_processed_benign.csv
│       │   └── test_processed.csv
│       ├── kdd/
│       └── cores_iot/
│
├── models/                           # Trained model files (.pkl) by dataset
│
├── evaluation/                       # Evaluation results per model and dataset
│   ├── summary.csv                   # Full results table
│   ├── summary_best.csv             # Best result per model/dataset
│   ├── summary.md                    # Markdown summary
│   ├── isolation_forest/<dataset>/
│   ├── sgd/<dataset>/
│   ├── mlp/<dataset>/
│   └── ensemble/<dataset>/
│
├── notebooks/                        # Jupyter notebooks for data & results analysis
│   ├── cross_dataset_comparison.ipynb
│   ├── netflow/
│   ├── kdd/
│   └── cores_iot/
│
├── src/
│   ├── datasets/
│   │   └── registry.py              # Central dataset configuration
│   ├── evaluation/
│   │   └── compare_results.py       # Cross-dataset comparison script
│   ├── preprocessing/               # Dataset-specific preprocessing scripts
│   │   ├── data_preprocessing_netflow.py
│   │   ├── data_preprocessing_kdd.py
│   │   └── data_preprocessing_cores_iot.py
│   ├── train/
│   │   ├── base_trainer.py          # Base class for online trainers
│   │   ├── iso_forest_train.py      # Isolation Forest (Stage 1)
│   │   ├── sgd_train.py             # SGD Classifier (Stage 2)
│   │   ├── mlp_train.py             # MLP Classifier (Stage 2)
│   │   └── ensemble_train.py        # Dynamic Ensemble (Stage 2)
│   └── run_experiments.py           # Unified experiment runner
│
├── requirements.txt
└── README.md
```

## Setup & Installation

### Prerequisites

- Python 3.8+

### Installation

```bash
git clone <repository-url>
cd intrusion-detection-AI
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

## Usage

All experiments are managed through `src/run_experiments.py`.

### Run Full Pipeline

```bash
# Preprocess + train + compare (all datasets, all models)
python src/run_experiments.py

# Single dataset
python src/run_experiments.py --dataset netflow

# Single dataset and model
python src/run_experiments.py --dataset kdd --model mlp
```

### Run Individual Phases

```bash
# Preprocessing only
python src/run_experiments.py --preprocess-only
python src/run_experiments.py --preprocess-only --dataset cores_iot

# Training only (assumes data is already preprocessed)
python src/run_experiments.py --train-only
python src/run_experiments.py --train-only --dataset netflow --model sgd

# Compare existing results only
python src/run_experiments.py --compare-only
```

### Training Options

```bash
# Custom Isolation Forest contamination
python src/run_experiments.py --dataset kdd --model all --contamination 0.3

# Skip Stage 1 training (use existing model)
python src/run_experiments.py --train-only --dataset netflow --model mlp --skip-stage1

# Specify a Stage 1 model file
python src/run_experiments.py --train-only --dataset netflow --model ensemble \
                              --stage1-model models/netflow/iso_forest_0.45.pkl
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--dataset` | `netflow`, `kdd`, `cores_iot`, or `all` (default: `all`) |
| `--model` | `sgd`, `mlp`, `ensemble`, or `all` (default: `all`) |
| `--preprocess-only` | Only preprocess, skip training |
| `--train-only` | Only train, skip preprocessing |
| `--compare-only` | Only compare existing results |
| `--no-compare` | Skip comparison at the end |
| `--contamination` | Isolation Forest contamination (default: 0.45) |
| `--skip-stage1` | Use existing Isolation Forest model |
| `--stage1-model` | Path to a specific Stage 1 model |

### Output

- **Processed data**: `datasets/processed/<dataset>/`
- **Trained models**: `models/<dataset>/`
- **Evaluation CSVs**: `evaluation/<model>/<dataset>/`
- **Summary**: `evaluation/summary.csv`, `evaluation/summary_best.csv`, `evaluation/summary.md`

## Results (Post Feature Cleaning)

### Best Model per Dataset

| Dataset | Best Model | Accuracy | Balanced Accuracy | F1 Macro | Stage 2 Call Rate | Time (s) |
|---------|------------|----------|-------------------|----------|-------------------|----------|
| **NetFlow** | Ensemble | **97.52%** | **97.66%** | **95.96%** | 54.64% | 130.1 |
| **CORES-IoT** | Ensemble | **95.64%** | 95.31% | **95.58%** | 74.65% | 55.5 |
| **KDD** | Ensemble | 95.01% | **96.13%** | 92.62% | 85.04% | 372.8 |

### Full Results Breakdown

| Model | Dataset | Accuracy | Balanced Acc. | F1 Anomaly | Recall Anomaly | Precision Anomaly | Time (s) |
|-------|---------|----------|---------------|------------|----------------|-------------------|----------|
| Ensemble | NetFlow | 97.52% | 97.66% | 93.45% | 97.89% | 89.40% | 130.1 |
| Ensemble | CORES-IoT | 95.64% | 95.31% | 96.08% | 99.98% | 92.47% | 55.5 |
| Ensemble | KDD | 95.01% | 96.13% | 96.82% | 94.29% | 99.48% | 372.8 |
| MLP | NetFlow | 95.05% | 93.10% | 86.81% | 90.04% | 83.81% | 14.9 |
| MLP | KDD | 94.84% | 95.42% | 96.72% | 94.47% | 99.08% | 27.8 |
| MLP | CORES-IoT | 89.23% | 89.38% | 89.65% | 87.25% | 92.19% | 6.6 |
| SGD | NetFlow | 91.08% | 83.08% | 74.10% | 70.55% | 78.03% | 11.1 |
| SGD | KDD | 91.55% | 90.13% | 94.63% | 92.46% | 96.91% | 16.8 |
| SGD | CORES-IoT | 90.45% | 90.35% | 91.13% | 91.74% | 90.53% | 5.1 |
| Iso. Forest | KDD | 90.89% | — | 94.50% | 97.13% | 92.01% | 3.9 |
| Iso. Forest | CORES-IoT | 78.83% | — | 83.48% | 100.00% | 71.64% | 3.7 |
| Iso. Forest | NetFlow | 63.05% | — | 49.20% | 98.92% | 32.74% | 9.2 |
