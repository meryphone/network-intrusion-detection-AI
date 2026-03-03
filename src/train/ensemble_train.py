"""
Dynamic Ensemble Training Script with SEA (Streaming Ensemble Algorithm).
Implements Stage 2 with a pool of MLP classifiers and pruning strategy.

Usage:
    python ensemble_train.py --train path/to/train.csv --test path/to/test.csv \
                             --stage1-model path/to/iso_forest.pkl
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from train.base_trainer import BaseOnlineTrainer, parse_stage2_args

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
POOL_SIZE = 5

# Base estimator config (MLP)
BASE_HIDDEN_LAYERS = (50,)
BASE_ACTIVATION = 'relu'
BASE_SOLVER = 'adam'
BASE_RANDOM_STATE = 42
BASE_MAX_ITER = 50

# Best batch size determined from experiments [10000, 15000, 20000, 25000, 30000]
BATCH_SIZE_OPTIONS = [25000]


class DynamicEnsemble:
    """
    Dynamic Ensemble using Streaming Ensemble Algorithm (SEA).
    Maintains a pool of classifiers and prunes the worst performer.
    """

    def __init__(self, pool_size: int = POOL_SIZE):
        self.pool_size = pool_size
        self.pool = []

    def _create_base_estimator(self):
        return MLPClassifier(
            hidden_layer_sizes=BASE_HIDDEN_LAYERS,
            activation=BASE_ACTIVATION,
            solver=BASE_SOLVER,
            random_state=BASE_RANDOM_STATE,
            max_iter=BASE_MAX_ITER,
        )

    def predict(self, X):
        if not self.pool:
            return np.zeros(len(X), dtype=int)
        preds = np.array([model.predict(X) for model in self.pool])
        return (np.mean(preds, axis=0) >= 0.5).astype(int)

    def update(self, X_batch, y_batch):
        if len(np.unique(y_batch)) < 2:
            return

        new_model = self._create_base_estimator()
        try:
            new_model.fit(X_batch, y_batch)
        except Exception as e:
            print(f"Ensemble update failed: {e}")
            return

        if len(self.pool) < self.pool_size:
            self.pool.append(new_model)
        else:
            accuracies = [
                accuracy_score(y_batch, m.predict(X_batch))
                for m in self.pool
            ]
            worst_idx = np.argmin(accuracies)
            self.pool[worst_idx] = new_model


class EnsembleTrainer(BaseOnlineTrainer):

    model_name = "dynamic_ensemble"
    eval_filename = "ensemble_evaluation.csv"
    model_filename = "dynamic_ensemble.pkl"
    batch_size_options = BATCH_SIZE_OPTIONS

    def create_model(self):
        return DynamicEnsemble(pool_size=POOL_SIZE)

    def warmup_model(self, model: DynamicEnsemble, X: pd.DataFrame, y: pd.Series):
        chunk_size = max(1, len(X) // POOL_SIZE)
        for i in range(POOL_SIZE):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(X))
            if start_idx >= len(X):
                break
            model.update(X.iloc[start_idx:end_idx], y.iloc[start_idx:end_idx])
        print(f"Ensemble initialized with {len(model.pool)} models.")

    def predict(self, model: DynamicEnsemble, X: pd.DataFrame) -> np.ndarray:
        return model.predict(X)

    def update_model(self, model: DynamicEnsemble, X: pd.DataFrame, y: pd.Series):
        model.update(X, y)

    def get_model_hyperparams(self) -> dict:
        return {
            "pool_size": POOL_SIZE,
            "base_hidden_layers": str(BASE_HIDDEN_LAYERS),
            "base_activation": BASE_ACTIVATION,
            "base_solver": BASE_SOLVER,
            "base_max_iter": BASE_MAX_ITER,
        }

    def get_eval_dir(self):
        return Path("evaluation") / "ensemble"


trainer = EnsembleTrainer()
main = trainer.main


if __name__ == "__main__":
    args = parse_stage2_args("Train Dynamic Ensemble")
    main(
        train_path=args.train,
        test_path=args.test,
        stage1_model_path=args.stage1_model,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
    )
