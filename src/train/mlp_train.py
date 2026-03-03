"""
MLP Classifier Training Script.
Implements Stage 2 of the hybrid IDS using online learning with MLPClassifier.

Usage:
    python mlp_train.py --train path/to/train.csv --test path/to/test.csv \
                        --stage1-model path/to/iso_forest.pkl
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from train.base_trainer import BaseOnlineTrainer, parse_stage2_args, CLASSES

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
HIDDEN_LAYER_SIZES = (100,)
ACTIVATION = 'relu'
SOLVER = 'adam'
RANDOM_STATE = 42
WARM_START = True
MAX_ITER = 1

# Batch sizes to evaluate [10000, 15000, 20000, 25000, 30000]
BATCH_SIZE_OPTIONS = [10000, 15000, 20000, 25000, 30000]


class MLPTrainer(BaseOnlineTrainer):

    model_name = "mlp_classifier"
    eval_filename = "mlp_evaluation.csv"
    model_filename = "mlp_classifier.pkl"
    batch_size_options = BATCH_SIZE_OPTIONS

    def create_model(self):
        return MLPClassifier(
            hidden_layer_sizes=HIDDEN_LAYER_SIZES,
            activation=ACTIVATION,
            solver=SOLVER,
            random_state=RANDOM_STATE,
            warm_start=WARM_START,
            max_iter=MAX_ITER,
        )

    def warmup_model(self, model, X: pd.DataFrame, y: pd.Series):
        model.partial_fit(X, y, classes=CLASSES)

    def predict(self, model, X: pd.DataFrame) -> np.ndarray:
        return model.predict(X)

    def update_model(self, model, X: pd.DataFrame, y: pd.Series):
        model.partial_fit(X, y)

    def get_model_hyperparams(self) -> dict:
        return {
            "hidden_layer_sizes": str(HIDDEN_LAYER_SIZES),
            "activation": ACTIVATION,
            "solver": SOLVER,
            "random_state": RANDOM_STATE,
            "max_iter": MAX_ITER,
        }

    def get_eval_dir(self):
        return Path("evaluation") / "mlp"


trainer = MLPTrainer()
main = trainer.main


if __name__ == "__main__":
    args = parse_stage2_args("Train MLP Classifier")
    main(
        train_path=args.train,
        test_path=args.test,
        stage1_model_path=args.stage1_model,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
    )
