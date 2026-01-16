"""
SGD Classifier Training Script with Dynamic Ensemble approach.
Implements Stage 2 of the hybrid intrusion detection system using online learning.
"""

import time
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    classification_report, 
    balanced_accuracy_score, 
    accuracy_score,
    confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
LOSS = 'log_loss'
RANDOM_STATE = 42
WARM_START = True

# Pipeline config
BATCH_SIZE_OPTIONS = [25000] # After experiments we leave the best options from this batch sizes [10000, 15000, 20000, 25000, 30000]
CLASSES = np.array([0, 1])

# =============================================================================
# PATHS
# =============================================================================
TRAIN_DATA_PATH = Path("../datasets/processed/X_train_full_processed.csv")
TEST_DATA_PATH = Path("../datasets/processed/Y_test_processed.csv")
STAGE1_MODEL_PATH = Path("models/iso_forest_0.45.pkl")
STAGE1_FALLBACK_PATH = Path("models/iso_forest_0.5.pkl")
MODELS_DIR = Path("models")
EVAL_DIR = Path("evaluation") / "sgd"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SGD Classifier")
    parser.add_argument(
        "--train", 
        type=str, 
        required=True,
        help="Path to training dataset CSV (with ANOMALY column)"
    )
    parser.add_argument(
        "--test", 
        type=str, 
        required=True,
        help="Path to test dataset CSV (with ANOMALY column)"
    )
    parser.add_argument(
        "--stage1-model",
        type=str,
        required=True,
        help="Path to Stage 1 (Isolation Forest) model"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for evaluation results"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory to save models"
    )
    return parser.parse_args()


def load_stage1_model(model_path=None):
    """Load the trained Isolation Forest model."""
    path = Path(model_path) if model_path else STAGE1_MODEL_PATH
    
    if not path.exists():
        print(f"Model {path} not found, trying {STAGE1_FALLBACK_PATH}")
        path = STAGE1_FALLBACK_PATH
    
    if not path.exists():
        raise FileNotFoundError("Stage 1 model not found. Train Isolation Forest first.")
    
    print(f"Loading Stage 1 model from {path}...")
    return joblib.load(path)


def create_sgd_model():
    """Create SGD Classifier with configured hyperparameters."""
    return SGDClassifier(
        loss=LOSS,
        random_state=RANDOM_STATE,
        warm_start=WARM_START
    )


def main(train_path, test_path, stage1_model_path, batch_size=None, output_dir=None, models_dir=None):
    """Main training function."""
    
    if batch_size is None:
        # Test all batch sizes
        eval_dir = Path(output_dir) if output_dir else EVAL_DIR
        eval_dir.mkdir(parents=True, exist_ok=True)

        all_results = []
        total_training_time = 0.0

        for bs in BATCH_SIZE_OPTIONS:
            print(f"\nTesting with batch_size: {bs}")
            try:
                result = _train_single(
                    train_path=train_path,
                    test_path=test_path,
                    stage1_model_path=stage1_model_path,
                    batch_size=bs,
                    output_dir=output_dir,
                    models_dir=models_dir
                )
                total_training_time += result["total_time_sec"]
                all_results.append(result)
            except Exception as e:
                print(f"Error with batch_size {bs}: {e}")
                continue

        # Save all results to DataFrame
        if all_results:
            # Add cumulative training time to all rows
            for result in all_results:
                result["total_training_time_sec"] = round(total_training_time, 2)

            results_df = pd.DataFrame(all_results)
            output_file = eval_dir / "sgd_evaluation.csv"
            results_df.to_csv(output_file, index=False)
            print(f"\nSaved all evaluation results to {output_file}")
            print(f"Total training time (all batch sizes): {total_training_time:.2f}s")

        return all_results
    else:
        # Single batch size
        return _train_single(
            train_path=train_path,
            test_path=test_path,
            stage1_model_path=stage1_model_path,
            batch_size=batch_size,
            output_dir=output_dir,
            models_dir=models_dir
        )


def _train_single(train_path, test_path, stage1_model_path, batch_size, output_dir=None, models_dir=None):
    
    train_data_path = Path(train_path)
    test_data_path = Path(test_path)
    eval_dir = Path(output_dir) if output_dir else EVAL_DIR
    model_dir = Path(models_dir) if models_dir else MODELS_DIR
    
    print("=" * 60)
    print("SGD CLASSIFIER TRAINING (DYNAMIC ONLINE LEARNING)")
    print("=" * 60)
    print(f"Train data: {train_data_path}")
    print(f"Test data: {test_data_path}")
    
    # Create directories
    eval_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading datasets...")
    df_train = pd.read_csv(train_data_path)
    df_test = pd.read_csv(test_data_path)
    
    X_train = df_train.drop(columns=["ANOMALY"])
    y_train = df_train["ANOMALY"]
    X_test = df_test.drop(columns=["ANOMALY"])
    y_test = df_test["ANOMALY"]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Load Stage 1 model
    stage1_model = load_stage1_model(stage1_model_path)
    
    # =========================================================================
    # PHASE 1: WARMUP
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: WARMUP")
    print("=" * 60)
    
    start_warmup = time.time()
    
    # Filter training data through Stage 1
    s1_pred_raw = stage1_model.predict(X_train)
    s1_pred = (s1_pred_raw == -1).astype(int)
    
    suspicious_idx = np.where(s1_pred == 1)[0]
    print(f"Stage 1 flagged {len(suspicious_idx)} / {len(X_train)} samples "
          f"({len(suspicious_idx)/len(X_train):.2%})")
    
    X_warmup = X_train.iloc[suspicious_idx]
    y_warmup = y_train.iloc[suspicious_idx]
    
    # Initialize and warmup model
    model = create_sgd_model()
    
    if len(X_warmup) > 0:
        model.partial_fit(X_warmup, y_warmup, classes=CLASSES)
        print("Warmup complete.")
    else:
        print("Warning: No suspicious samples for warmup.")
    
    warmup_time = time.time() - start_warmup
    print(f"Warmup time: {warmup_time:.2f}s")
    
    # =========================================================================
    # PHASE 2: ONLINE TESTING WITH DYNAMIC UPDATES
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: ONLINE TESTING (DYNAMIC UPDATES)")
    print("=" * 60)
    
    n_batches = int(np.ceil(len(X_test) / batch_size))
    
    y_true_all = []
    y_pred_all = []
    total_processed = 0
    stage2_calls = 0
    
    start_test = time.time()
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X_test))
        
        X_batch = X_test.iloc[start_idx:end_idx]
        y_batch = y_test.iloc[start_idx:end_idx]
        
        # Stage 1 screening
        s1_batch_raw = stage1_model.predict(X_batch)
        s1_batch = (s1_batch_raw == -1).astype(int)
        
        y_batch_pred = np.zeros(len(y_batch), dtype=int)
        
        # Process suspicious samples
        suspicious_mask = (s1_batch == 1)
        suspicious_X = X_batch[suspicious_mask]
        
        if len(suspicious_X) > 0:
            stage2_calls += len(suspicious_X)
            
            # Predict
            s2_pred = model.predict(suspicious_X)
            y_batch_pred[suspicious_mask] = s2_pred
            
            # Dynamic update (online learning)
            suspicious_y_true = y_batch[suspicious_mask]
            model.partial_fit(suspicious_X, suspicious_y_true)
        
        y_true_all.extend(y_batch)
        y_pred_all.extend(y_batch_pred)
        total_processed += len(X_batch)
        
        if (i + 1) % 10 == 0:
            print(f"Processed batch {i+1}/{n_batches}...")
    
    test_time = time.time() - start_test
    print(f"\nProcessing complete. Time: {test_time:.2f}s")
    
    # =========================================================================
    # EVALUATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    
    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])
    report_dict = classification_report(
        y_true_all, y_pred_all,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        digits=4,
        output_dict=True,
        zero_division=0
    )
    bac = balanced_accuracy_score(y_true_all, y_pred_all)
    
    print(f"Balanced Accuracy: {bac:.4f}")
    print(f"Accuracy: {report_dict['accuracy']:.4f}")
    print(f"F1 Macro: {report_dict['macro avg']['f1-score']:.4f}")
    print(f"Stage 2 Call Rate: {stage2_calls/total_processed:.2%}")
    
    # Save results to DataFrame
    results = {
        "model_name": "sgd_classifier",
        "loss": LOSS,
        "random_state": RANDOM_STATE,
        "batch_size": batch_size,
        "warmup_time_sec": round(warmup_time, 2),
        "test_time_sec": round(test_time, 2),
        "total_time_sec": round(warmup_time + test_time, 2),
        "total_processed": total_processed,
        "stage2_calls": stage2_calls,
        "stage2_call_rate": round(stage2_calls / total_processed, 4),
        "accuracy": round(report_dict["accuracy"], 4),
        "balanced_accuracy": round(bac, 4),
        "precision_normal": round(report_dict["Normal"]["precision"], 4),
        "recall_normal": round(report_dict["Normal"]["recall"], 4),
        "f1_normal": round(report_dict["Normal"]["f1-score"], 4),
        "support_normal": int(report_dict["Normal"]["support"]),
        "precision_anomaly": round(report_dict["Anomaly"]["precision"], 4),
        "recall_anomaly": round(report_dict["Anomaly"]["recall"], 4),
        "f1_anomaly": round(report_dict["Anomaly"]["f1-score"], 4),
        "support_anomaly": int(report_dict["Anomaly"]["support"]),
        "precision_macro": round(report_dict["macro avg"]["precision"], 4),
        "recall_macro": round(report_dict["macro avg"]["recall"], 4),
        "f1_macro": round(report_dict["macro avg"]["f1-score"], 4),
        "confusion_tn": int(cm[0, 0]),
        "confusion_fp": int(cm[0, 1]),
        "confusion_fn": int(cm[1, 0]),
        "confusion_tp": int(cm[1, 1]),
    }
    
    results_df = pd.DataFrame([results])
    output_file = eval_dir / "sgd_evaluation.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved evaluation results to {output_file}")
    
    # Save model
    model_file = model_dir / "sgd_classifier.pkl"
    joblib.dump(model, model_file)
    print(f"Saved model to {model_file}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    args = parse_args()
    main(
        train_path=args.train,
        test_path=args.test,
        stage1_model_path=args.stage1_model,
        batch_size=None,  # Test all batch sizes
        output_dir=args.output_dir,
        models_dir=args.models_dir
    )
