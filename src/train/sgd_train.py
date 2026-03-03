"""
SGD Classifier Training Script.
Implements Stage 2 of the hybrid IDS using online learning with SGDClassifier.

Usage:
    python sgd_train.py --train path/to/train.csv --test path/to/test.csv \
                        --stage1-model path/to/iso_forest.pkl
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

from train.base_trainer import BaseOnlineTrainer, parse_stage2_args, CLASSES

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
LOSS = 'log_loss'
RANDOM_STATE = 42
WARM_START = True

# Best batch size determined from experiments [10000, 15000, 20000, 25000, 30000]
BATCH_SIZE_OPTIONS = [25000]


class SGDTrainer(BaseOnlineTrainer):

    model_name = "sgd_classifier"
    eval_filename = "sgd_evaluation.csv"
    model_filename = "sgd_classifier.pkl"
    batch_size_options = BATCH_SIZE_OPTIONS

    def create_model(self):
        return SGDClassifier(
            loss=LOSS,
            random_state=RANDOM_STATE,
            warm_start=WARM_START,
        )

    def warmup_model(self, model, X: pd.DataFrame, y: pd.Series):
        model.partial_fit(X, y, classes=CLASSES)

    def predict(self, model, X: pd.DataFrame) -> np.ndarray:
        return model.predict(X)

    def update_model(self, model, X: pd.DataFrame, y: pd.Series):
        model.partial_fit(X, y)

    def get_model_hyperparams(self) -> dict:
        return {
            "loss": LOSS,
            "random_state": RANDOM_STATE,
        }

    def get_eval_dir(self):
        return Path("evaluation") / "sgd"


trainer = SGDTrainer()
main = trainer.main


if __name__ == "__main__":
    args = parse_stage2_args("Train SGD Classifier")
    main(
        train_path=args.train,
        test_path=args.test,
        stage1_model_path=args.stage1_model,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
    )
