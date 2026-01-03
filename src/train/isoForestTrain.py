"""
Isolation Forest Training Script.
Trains Isolation Forest models with multiple contamination values for Stage 1.
"""

import time
from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
N_ESTIMATORS = 100
RANDOM_STATE = 42
N_JOBS = -1
CONTAMINATION_VALUES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.45, 0.5]
SAVE_MODEL_CONTAMINATION = [0.45]

# =============================================================================
# PATHS
# =============================================================================
TRAIN_DATA_PATH = "datasets/processed/X_train_processed.csv"
TEST_DATA_PATH = "datasets/processed/Y_test_processed.csv"
MODELS_DIR = Path("models")
EVAL_DIR = Path("evaluation") / "isolation_forest"


def main():
    """Main training function."""
    
    print("=" * 60)
    print("ISOLATION FOREST TRAINING")
    print("=" * 60)
    
    # Load data
    print("\nLoading training and test data...")
    try:
        X_train = pd.read_csv(TRAIN_DATA_PATH)
        test_df = pd.read_csv(TEST_DATA_PATH)
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
    
    print(f"Training rows: {len(X_train)}, Test rows: {len(test_df)}")
    
    X_test = test_df.drop(columns=["ANOMALY"])
    y_test = test_df["ANOMALY"]
    
    # Create output directories
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for c in CONTAMINATION_VALUES:
        print(f"\n{'='*60}")
        print(f"Training Isolation Forest with contamination = {c}")
        print('='*60)
        
        start_time = time.time()
        
        # Build and train model
        iso = IsolationForest(
            n_estimators=N_ESTIMATORS,
            contamination=c,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS
        )
        
        try:
            iso.fit(X_train)
        except Exception as e:
            print(f"Error training model: {e}")
            raise
        
        print("Model trained.")
        
        # Save model if in save list
        if c in SAVE_MODEL_CONTAMINATION:
            model_path = MODELS_DIR / f"iso_forest_{c}.pkl"
            try:
                joblib.dump(iso, model_path)
                print(f"Saved model to {model_path}")
            except Exception as e:
                print(f"Error saving model: {e}")
        
        # Predictions
        print("Making predictions on test...")
        try:
            y_pred_raw = iso.predict(X_test)
        except Exception as e:
            print(f"Error during prediction: {e}")
            raise
        
        # Convert from {1, -1} to {0, 1}
        y_pred = (y_pred_raw == -1).astype(int)
        
        # Evaluation
        elapsed_sec = time.time() - start_time
        
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        report_dict = classification_report(
            y_test, y_pred,
            labels=[0, 1],
            target_names=["Normal", "Anomaly"],
            digits=4,
            output_dict=True,
            zero_division=0
        )
        
        print(f"Accuracy: {report_dict['accuracy']:.4f}")
        print(f"F1 Macro: {report_dict['macro avg']['f1-score']:.4f}")
        print(f"Elapsed time: {elapsed_sec:.2f}s")
        
        # Store results
        results.append({
            "contamination": c,
            "n_estimators": N_ESTIMATORS,
            "random_state": RANDOM_STATE,
            "elapsed_sec": round(elapsed_sec, 2),
            "accuracy": round(report_dict["accuracy"], 4),
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
        })
    
    # Save all results to a single DataFrame
    results_df = pd.DataFrame(results)
    output_path = EVAL_DIR / "isolation_forest_evaluation.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n{'='*60}")
    print(f"Saved evaluation results to {output_path}")
    
    # Print summary
    best_idx = results_df["f1_macro"].idxmax()
    best_row = results_df.loc[best_idx]
    print(f"\nBest model by F1 Macro:")
    print(f"  Contamination: {best_row['contamination']}")
    print(f"  F1 Macro: {best_row['f1_macro']}")
    print(f"  Accuracy: {best_row['accuracy']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
