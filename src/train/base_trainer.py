"""
Base Online Trainer - Common logic for Stage 2 training scripts.

Encapsulates the shared pipeline: data loading, Stage 1 filtering,
warmup, online testing with dynamic updates, evaluation, and model saving.

Subclasses only need to define model creation, warmup, prediction, and update logic.
"""

import time
import argparse
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    confusion_matrix,
)

import warnings
warnings.filterwarnings('ignore')

CLASSES = np.array([0, 1])
MODELS_DIR = Path("models")


def parse_stage2_args(description: str) -> argparse.Namespace:
    """Shared argument parser for all Stage 2 scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--train", type=str, required=True,
                        help="Path to training dataset CSV (with ANOMALY column)")
    parser.add_argument("--test", type=str, required=True,
                        help="Path to test dataset CSV (with ANOMALY column)")
    parser.add_argument("--stage1-model", type=str, required=True,
                        help="Path to Stage 1 (Isolation Forest) model")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for evaluation results")
    parser.add_argument("--models-dir", type=str, default=None,
                        help="Directory to save models")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Single batch size to test (default: test all configured sizes)")
    return parser.parse_args()


def load_stage1_model(model_path: str):
    """Load the trained Isolation Forest model."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Stage 1 model not found: {path}")
    print(f"Loading Stage 1 model from {path}...")
    return joblib.load(path)


class BaseOnlineTrainer(ABC):
    """
    Base class for Stage 2 online learning trainers.

    Subclasses must implement:
        - model_name: str property
        - eval_filename: str property
        - model_filename: str property
        - batch_size_options: list property
        - create_model(): create the ML model
        - warmup_model(model, X, y): initial training on suspicious samples
        - predict(model, X): make predictions
        - update_model(model, X, y): online update with new data
        - get_model_hyperparams(): dict of hyperparams for results CSV
    """

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def eval_filename(self) -> str: ...

    @property
    @abstractmethod
    def model_filename(self) -> str: ...

    @property
    @abstractmethod
    def batch_size_options(self) -> list: ...

    @abstractmethod
    def create_model(self): ...

    @abstractmethod
    def warmup_model(self, model, X: pd.DataFrame, y: pd.Series): ...

    @abstractmethod
    def predict(self, model, X: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def update_model(self, model, X: pd.DataFrame, y: pd.Series): ...

    @abstractmethod
    def get_model_hyperparams(self) -> dict: ...

    def get_eval_dir(self) -> Path:
        return Path("evaluation") / self.model_name.split("_classifier")[0].split("dynamic_")[0] or self.model_name

    def main(self, train_path, test_path, stage1_model_path,
             batch_size=None, output_dir=None, models_dir=None):
        """Main training function. Tests all batch sizes or a single one."""
        if batch_size is not None:
            return self._train_single(
                train_path, test_path, stage1_model_path,
                batch_size, output_dir, models_dir
            )

        eval_dir = Path(output_dir) if output_dir else self.get_eval_dir()
        eval_dir.mkdir(parents=True, exist_ok=True)

        all_results = []
        total_training_time = 0.0

        for bs in self.batch_size_options:
            print(f"\nTesting with batch_size: {bs}")
            try:
                result = self._train_single(
                    train_path, test_path, stage1_model_path,
                    bs, output_dir, models_dir
                )
                total_training_time += result["total_time_sec"]
                all_results.append(result)
            except Exception as e:
                print(f"Error with batch_size {bs}: {e}")
                continue

        if all_results:
            for result in all_results:
                result["total_training_time_sec"] = round(total_training_time, 2)

            results_df = pd.DataFrame(all_results)
            output_file = eval_dir / self.eval_filename
            results_df.to_csv(output_file, index=False)
            print(f"\nSaved all evaluation results to {output_file}")
            print(f"Total training time (all batch sizes): {total_training_time:.2f}s")

        return all_results

    def _train_single(self, train_path, test_path, stage1_model_path,
                       batch_size, output_dir=None, models_dir=None):
        """Train and evaluate with a single batch size."""
        eval_dir = Path(output_dir) if output_dir else self.get_eval_dir()
        model_dir = Path(models_dir) if models_dir else MODELS_DIR

        print("=" * 60)
        print(f"{self.model_name.upper()} TRAINING (ONLINE LEARNING)")
        print("=" * 60)
        print(f"Train data: {train_path}")
        print(f"Test data: {test_path}")

        eval_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        print("\nLoading datasets...")
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        X_train = df_train.drop(columns=["ANOMALY"])
        y_train = df_train["ANOMALY"]
        X_test = df_test.drop(columns=["ANOMALY"])
        y_test = df_test["ANOMALY"]

        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")

        # Load Stage 1 model
        stage1_model = load_stage1_model(stage1_model_path)

        # =================================================================
        # PHASE 1: WARMUP
        # =================================================================
        print("\n" + "=" * 60)
        print("PHASE 1: WARMUP")
        print("=" * 60)

        start_warmup = time.time()

        s1_pred = (stage1_model.predict(X_train) == -1).astype(int)
        suspicious_idx = np.where(s1_pred == 1)[0]
        print(f"Stage 1 flagged {len(suspicious_idx)} / {len(X_train)} samples "
              f"({len(suspicious_idx)/len(X_train):.2%})")

        X_warmup = X_train.iloc[suspicious_idx]
        y_warmup = y_train.iloc[suspicious_idx]

        model = self.create_model()

        if len(X_warmup) > 0:
            self.warmup_model(model, X_warmup, y_warmup)
            print("Warmup complete.")
        else:
            print("Warning: No suspicious samples for warmup.")

        warmup_time = time.time() - start_warmup
        print(f"Warmup time: {warmup_time:.2f}s")

        # =================================================================
        # PHASE 2: ONLINE TESTING WITH DYNAMIC UPDATES
        # =================================================================
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
            s1_batch = (stage1_model.predict(X_batch) == -1).astype(int)
            y_batch_pred = np.zeros(len(y_batch), dtype=int)

            suspicious_mask = (s1_batch == 1)
            suspicious_X = X_batch[suspicious_mask]

            if len(suspicious_X) > 0:
                stage2_calls += len(suspicious_X)
                s2_pred = self.predict(model, suspicious_X)
                y_batch_pred[suspicious_mask] = s2_pred

                suspicious_y_true = y_batch[suspicious_mask]
                self.update_model(model, suspicious_X, suspicious_y_true)

            y_true_all.extend(y_batch)
            y_pred_all.extend(y_batch_pred)
            total_processed += len(X_batch)

            if (i + 1) % 10 == 0:
                print(f"Processed batch {i+1}/{n_batches}...")

        test_time = time.time() - start_test
        print(f"\nProcessing complete. Time: {test_time:.2f}s")

        # =================================================================
        # EVALUATION
        # =================================================================
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
            digits=4, output_dict=True, zero_division=0
        )
        bac = balanced_accuracy_score(y_true_all, y_pred_all)

        print(f"Balanced Accuracy: {bac:.4f}")
        print(f"Accuracy: {report_dict['accuracy']:.4f}")
        print(f"F1 Macro: {report_dict['macro avg']['f1-score']:.4f}")
        print(f"Stage 2 Call Rate: {stage2_calls/total_processed:.2%}")

        # Build results dict
        results = {
            "model_name": self.model_name,
            **self.get_model_hyperparams(),
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
        output_file = eval_dir / self.eval_filename
        results_df.to_csv(output_file, index=False)
        print(f"\nSaved evaluation results to {output_file}")

        # Save model
        model_file = model_dir / self.model_filename
        joblib.dump(model, model_file)
        print(f"Saved model to {model_file}")

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)

        return results
