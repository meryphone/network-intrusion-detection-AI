# Project Report Information

This document summarizes the key concepts, results, and analysis of the **Hybrid Two-Stage Network Intrusion Detection System (IDS)**. It is designed to assist in writing the final project report in LaTeX.

---

## 1. Core Concept: The Hybrid Architecture

The project's main innovation is the **Hybrid Two-Stage approach**.

*   **Problem**:
    *   **Static Models** (trained once) fail to detect new, evolving attacks.
    *   **Dynamic Models** (continuously updating) are accurate but computationally expensive to run on *every* packet.
*   **Solution**:
    *   **Stage 1 (The Filter)**: A lightweight, unsupervised **Isolation Forest**. It learns what "Normal" traffic looks like. Anything that deviates is flagged as "Suspicious". It is *fast* and filters out clear benign traffic.
    *   **Stage 2 (The Expert)**: A heavy, supervised **Dynamic Classifier**. It only looks at the "Suspicious" traffic. It learns from mistakes and updates itself to recognize new attack patterns.

**Benefit**: We get the accuracy of the heavy model but save computational resources by only running it on ~55% of the traffic (the "suspicious" subset).

---

## 2. Methodology: Stage 2 Strategies

We compared three strategies for the "Expert" (Stage 2) to see which performed best.

### A. SGDClassifier (Dynamic Single - Linear)
*   **What is it?**: A simple linear model (Logistic Regression) updated via Stochastic Gradient Descent.
*   **Pros**: Extremely fast.
*   **Cons**: Too simple. Cannot learn complex, non-linear attack patterns.
*   **Result**: Lowest accuracy (85% BAC).

### B. MLPClassifier (Dynamic Single - Neural Net)
*   **What is it?**: A Neural Network (Multi-Layer Perceptron) that updates incrementally (`partial_fit`).
*   **Pros**: Can learn non-linear patterns. Much better recall than SGD.
*   **Cons**: Slower than SGD.
*   **Result**: Good accuracy (90% BAC). Matches the baseline paper's single-model approach.

### C. Dynamic Ensemble (The "Committee" Approach)
*   **What is it?**: Based on the **Streaming Ensemble Algorithm (SEA)** from the baseline paper.
*   **How it works**:
    1.  Maintains a pool of 5 classifiers (Experts).
    2.  When a new batch of data arrives, it trains a **new expert** on just that batch.
    3.  It adds the new expert to the pool.
    4.  If the pool is full, it **fires (prunes)** the expert with the worst accuracy on the current data.
    5.  **Prediction**: The 5 experts vote. Majority wins.
*   **Pros**: High stability. If one expert is wrong, others overrule it. Adapts fast to new attacks (via new experts) while remembering old ones (via old experts).
*   **Result**: **Best Performance (96% BAC)**.

---

## 3. Key Results & Analysis

### Performance Comparison Table

| Metric | SGD (Linear) | MLP (Neural Net) | **Dynamic Ensemble** |
| :--- | :--- | :--- | :--- |
| **Balanced Accuracy** | 85.71% | 90.04% | **95.94%** |
| **Attack Recall** | 76.84% | 85.41% | **97.57%** |
| **Attack F1-Score** | 0.763 | 0.815 | **0.874** |
| **Processing Time** | 3.66s | 6.29s | 38.58s |

### Interpretation for the Report

1.  **Recall is King**:
    *   The most critical metric is **Attack Recall** (Detection Rate).
    *   The **Ensemble caught 97.6% of attacks**, whereas the single Neural Net missed ~15% of them. This massive jump validates the use of Ensembles for security.

2.  **The "Hybrid Benefit" (Efficiency)**:
    *   The Ensemble is computationally heavy (~6x slower than MLP).
    *   **However**, our Stage 1 filter successfully marked **45.25%** of traffic as "Benign" with high confidence.
    *   This means the expensive Ensemble only had to process **54.75%** of the data.
    *   **Conclusion**: The Hybrid architecture makes state-of-the-art Ensemble models feasible for real-time detection by drastically cutting their workload.

3.  **Confusion Matrix Analysis (Ensemble)**:
    *   **Attacks Caught**: 160,906
    *   **Attacks Missed**: 4,008 (Only ~2.4% missed)
    *   **False Alarms**: 42,504 (Benign traffic flagged as attack)
    *   *Note*: In cybersecurity, a high False Positive rate is often tolerated if it guarantees catching nearly all real attacks (High Recall).

---

## 4. Suggested Future Work

1.  **Reduce False Positives**: The Stage 1 filter is very aggressive (high recall, low precision). Tuning it to let more benign traffic pass (without missing attacks) would further optimize efficiency.
2.  **Drift Detection**: Currently, we update the model on every batch. Implementing explicit "Drift Detection" (only updating when performance drops) could save more resources.
3.  **Deep Learning Ensembles**: Replacing the small MLPs in the ensemble with more complex architectures (e.g., CNNs or Transformers) could potentially push accuracy to 99%+.


