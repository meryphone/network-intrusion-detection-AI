"""Isolation Forest Training Script.
Trains Isolation Forest models with multiple contamination values for Stage 1.

Usage:
    python iso_forest_train.py --train path/to/train_benign.csv --test path/to/test.csv
"""

import time
import argparse
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
# OUTPUT PATHS
# =============================================================================
MODELS_DIR = Path("models")
EVAL_DIR = Path("evaluation") / "isolation_forest"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Isolation Forest model")
    parser.add_argument(
        "--train", 
        type=str, 
        required=True,
        help="Path to BENIGN-ONLY training dataset CSV"
    )
    parser.add_argument(
        "--test", 
        type=str, 
        required=True,
        help="Path to test dataset CSV (with ANOMALY column)"
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
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (used for output directory routing)"
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=None,
        help="Single contamination value to train (default: test all CONTAMINATION_VALUES)"
    )
    return parser.parse_args()


def main(train_path, test_path, output_dir=None, models_dir=None, contamination=None, dataset=None):
    """Main training function.
    
    Args:
        train_path: Path to benign-only training CSV
        test_path: Path to test CSV with ANOMALY column
        output_dir: Directory for evaluation results
        models_dir: Directory to save models
        contamination: Single contamination value, or None for all CONTAMINATION_VALUES
        dataset: Dataset name for output routing
    """
    
    train_data_path = Path(train_path)
    test_data_path = Path(test_path)
    
    # Build eval_dir with optional dataset subdirectory
    if output_dir:
        eval_dir = Path(output_dir)
    elif dataset:
        eval_dir = EVAL_DIR / dataset
    else:
        eval_dir = EVAL_DIR
    
    # Build model_dir with optional dataset subdirectory  
    if models_dir:
        model_dir = Path(models_dir)
    elif dataset:
        model_dir = MODELS_DIR / dataset
    else:
        model_dir = MODELS_DIR
    
    # Determine which contamination values to test
    contamination_values = [contamination] if contamination else CONTAMINATION_VALUES
    save_contamination = [contamination] if contamination else SAVE_MODEL_CONTAMINATION
    
    print("=" * 60)
    print("ISOLATION FOREST TRAINING")
    print("=" * 60)
    print(f"Train data (benign): {train_data_path}")
    print(f"Test data: {test_data_path}")
    
    # Load data
    print("\nLoading training and test data...")
    try:
        X_train = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
    
    print(f"Training rows: {len(X_train)}, Test rows: {len(test_df)}")
    
    X_test = test_df.drop(columns=["ANOMALY"])
    y_test = test_df["ANOMALY"]
    
    # Create output directories
    eval_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for c in contamination_values:
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
        if c in save_contamination:
            model_file = model_dir / f"iso_forest_{c}.pkl"
            try:
                joblib.dump(iso, model_file)
                print(f"Saved model to {model_file}")
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
    output_file = eval_dir / "isolation_forest_evaluation.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n{'='*60}")
    print(f"Saved evaluation results to {output_file}")
    
    # Print summary
    best_idx = results_df["f1_macro"].idxmax()
    best_row = results_df.loc[best_idx]
    print(f"\nBest model by F1 Macro:")
    print(f"  Contamination: {best_row['contamination']}")
    print(f"  F1 Macro: {best_row['f1_macro']}")
    print(f"  Accuracy: {best_row['accuracy']}")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    main(
        train_path=args.train,
        test_path=args.test,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        contamination=args.contamination,
        dataset=args.dataset
    )
