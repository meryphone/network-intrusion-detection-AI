# Network Intrusion Detection AI

This project implements a **Hybrid Two-Stage Network Intrusion Detection System (IDS)** using Machine Learning. It combines a lightweight static anomaly detector with an adaptive, dynamically updated classifier to detect network attacks efficiently and robustly.

## Project Goal
The objective is to detect network intrusions (attacks) in real-time network traffic data. Traditional static ML models struggle with evolving attack patterns, while fully dynamic models are computationally expensive. This project proposes a **Hybrid approach**:
1.  **Stage 1 (Static Screening)**: A lightweight, unsupervised **Isolation Forest** model filters all traffic. It is trained only on benign traffic to learn "normality". It flags deviations as "suspicious".
2.  **Stage 2 (Dynamic Detection)**: A supervised classifier (Online Learning) processes *only* the traffic flagged as suspicious by Stage 1. It learns incrementally from new data, adapting to new attack types over time.

## Architecture & Methods

The system supports multiple strategies for Stage 2, allowing comparison between different levels of complexity and adaptability:

1.  **Dynamic Single (Linear)**: `SGDClassifier`. Fast, linear updates.
2.  **Dynamic Single (Neural Net)**: `MLPClassifier`. Captures non-linear patterns (similar to baseline paper).
3.  **Dynamic Ensemble**: A pool of classifiers (SEA-inspired). Trains a new model for each batch and prunes the worst performer. Offers high stability.

### Pipeline Flow
1.  **Incoming Traffic** -> **Stage 1 (Isolation Forest)**
2.  **IF Stage 1 says Normal** -> Classification: **Benign** (Stop)
3.  **IF Stage 1 says Suspicious** -> **Stage 2 (Adaptive Classifier)**
    *   Stage 2 predicts **Attack** or **Benign**.
    *   Stage 2 is **updated** (partially fitted or ensemble updated) with the true label of this sample to adapt to the current threat landscape.

## Setup & Installation

### Prerequisites
*   Python 3.8+
*   Virtual Environment (recommended)

### Installation
1.  Clone the repository.
2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/Mac
    # venv\Scripts\activate   # On Windows
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The project is divided into three main scripts located in `src/`.

### 1. Data Preprocessing
Prepares the raw NetFlow datasets for training and testing. It handles cleaning, encoding, and scaling.
```bash
python src/data_preprocessing_netflow.py
```
*   **Input**: `datasets/raw/train_net.csv`, `datasets/raw/test_net.csv`
*   **Output**: 
    *   `datasets/processed/X_train_processed.csv` (Benign-only for Stage 1)
    *   `datasets/processed/X_train_full_processed.csv` (Full training set for Stage 2 warmup)
    *   `datasets/processed/Y_test_processed.csv` (Test set)

### 2. Stage 1 Training (Static)
Trains the Isolation Forest model on benign data and evaluates it across different contamination levels. It saves the best model (contamination=0.45 or 0.5) for use in the pipeline.
```bash
python src/isoForestTrain.py
```
*   **Output**: `models/iso_forest_0.45.pkl`, evaluation reports in `evaluation/eval_isoForest/`.

### 3. Hybrid Pipeline (Stage 1 + Stage 2)
Runs the full 2-stage pipeline. It warms up Stage 2 using the training data filtered by Stage 1, then simulates an online stream using the test data. It runs three experiments: **SGD**, **MLP**, and **Dynamic Ensemble**.
```bash
python src/hybrid_pipeline.py
```
*   **Output**: Console logs of progress and final classification report. Results saved to `evaluation/hybrid_results/`.

## Experiments & Results

We evaluated three variations of the Stage 2 classifier:
1.  **SGDClassifier**: A lightweight linear model.
2.  **MLPClassifier**: A Neural Network (Multi-layer Perceptron) capable of learning non-linear patterns.
3.  **Dynamic Ensemble**: A pool of 5 MLP classifiers that updates by adding new models and removing the worst performing one.

### Comparative Results (Test Set)

| Metric | SGD (Linear) | MLP (Neural Net) | Dynamic Ensemble | Best Performer |
| :--- | :--- | :--- | :--- | :--- |
| **Balanced Accuracy (BAC)** | 85.71% | 90.04% | **95.94%** | **Ensemble** |
| **Accuracy** | 91.37% | 92.99% | **94.90%** | **Ensemble** |
| **Recall (Attacks)** | 76.84% | 85.41% | **97.57%** | **Ensemble** |
| **F1-Score (Attacks)** | 0.7631 | 0.8150 | **0.8738** | **Ensemble** |
| **Processing Time** | **3.66s** | 6.29s | 38.58s | SGD (Fastest) |

### Analysis
*   **Performance**: The **Dynamic Ensemble** significantly outperformed both single models, achieving a remarkable **95.94% Balanced Accuracy** and catching **97.6%** of attacks. This confirms the baseline paper's finding that ensemble methods provide superior stability and accuracy in dynamic environments.
*   **Recall**: The Ensemble's ability to retain diverse models allowed it to maintain extremely high recall (97.6%), minimizing missed attacks compared to the single MLP (85.4%).
*   **Efficiency**: 
    *   **SGD** is the fastest but least accurate.
    *   **MLP** offers a good balance (2x slower than SGD, but +5% accuracy).
    *   **Ensemble** is the most expensive (~6x slower than MLP) but delivers state-of-the-art accuracy. However, thanks to the Stage 1 filter, even this "expensive" model only runs on ~55% of the traffic, making it feasible for real-world deployment.

## Future Work
*   **Drift Handling**: Implement explicit drift detection to reset or re-weight Stage 2 models if performance drops.
*   **Feature Engineering**: Improve input features for better separation of classes.
