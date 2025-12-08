import time
from pathlib import Path
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report

print("Loading training and test data...")
try:
    X_train = pd.read_csv("datasets/processed/X_train_processed.csv")
    test_df = pd.read_csv("datasets/processed/Y_test_processed.csv")
except Exception as e:
    print(f"Error loading data: {e}")
    raise
print(f"Training rows: {len(X_train)}, test rows: {len(test_df)}")

X_test = test_df.drop(columns=["ANOMALY"])
y_test = test_df["ANOMALY"]

eval_dir = Path("evaluation") / "eval_isoForest"
eval_dir.mkdir(parents=True, exist_ok=True)

results = []
contamination = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

for c in contamination:
    print(f"---------Training Isolation Forest with contamination = {c}---------")
    start_time = time.time()

    iso = IsolationForest(
        n_estimators=100,
        contamination=c,
        random_state=42
    )


    try:
        iso.fit(X_train)
    except Exception as e:
        print(f"Error training model: {e}")
        raise

    print("Model trained.")

    print("Making predictions on test...")

    try:
        y_pred_if = iso.predict(X_test)
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise

    # Convertir de {1, -1} a {0, 1}
    y_pred = (y_pred_if == -1).astype(int)

    print("Evaluating results...")
    try:
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        cm_df = pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])
        report_dict = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["0", "1"],
            digits=4,
            output_dict=True,
            zero_division=0
        )
        report_str = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["0", "1"],
            digits=4,
            zero_division=0
        )
        iso_params = iso.get_params()
        elapsed_sec = time.time() - start_time
        print(cm)
        print(report_str)
        print(f"Elapsed time (s): {elapsed_sec:.2f}")

        eval_path = eval_dir / f"iso_forest_eval_cont_{c}.txt"
        with eval_path.open("w", encoding="utf-8") as f:
            f.write("Isolation Forest Evaluation\n")
            f.write(f"Contamination: {c}\n")
            f.write(f"Elapsed time (s): {elapsed_sec:.2f}\n")
            f.write("\nModel Parameters:\n")
            for k, v in iso_params.items():
                f.write(f"{k}: {v}\n")
            f.write("\nConfusion Matrix:\n")
            f.write(f"{cm}\n")
            f.write("\nClassification Report:\n")
            f.write(report_str)
        print(f"Saved evaluation results to {eval_path}")

        cm_path = eval_dir / f"iso_forest_confusion_cont_{c}.csv"
        cm_df.to_csv(cm_path, index=True)
        print(f"Saved confusion matrix to {cm_path}")

        results.append({
            "contamination": c,
            "elapsed_sec": elapsed_sec,
            "accuracy": report_dict.get("accuracy", 0.0),
            "precision_0": report_dict.get("0", {}).get("precision", 0.0),
            "recall_0": report_dict.get("0", {}).get("recall", 0.0),
            "f1_0": report_dict.get("0", {}).get("f1-score", 0.0),
            "precision_1": report_dict.get("1", {}).get("precision", 0.0),
            "recall_1": report_dict.get("1", {}).get("recall", 0.0),
            "f1_1": report_dict.get("1", {}).get("f1-score", 0.0),
            "precision_macro": report_dict.get("macro avg", {}).get("precision", 0.0),
            "recall_macro": report_dict.get("macro avg", {}).get("recall", 0.0),
            "f1_macro": report_dict.get("macro avg", {}).get("f1-score", 0.0),
        })
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise

try:
    results_df = pd.DataFrame(results)
    summary_path = eval_dir / "iso_forest_eval_summary.csv"
    results_df.to_csv(summary_path, index=False)
    print(f"Saved evaluation summary to {summary_path}")
except Exception as e:
    print(f"Error saving evaluation summary: {e}")
    raise
