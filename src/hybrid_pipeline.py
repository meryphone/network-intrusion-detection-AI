import pandas as pd
import joblib
import time
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score, accuracy_score
from pathlib import Path

def load_data():
    print("Loading datasets...")
    try:
        # Load Stage 2 Warmup Data (Training set with attacks)
        df_train = pd.read_csv("datasets/processed/X_train_full_processed.csv")
        
        # Load Test Data
        df_test = pd.read_csv("datasets/processed/Y_test_processed.csv")
        
        return df_train, df_test
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def load_stage1_model(model_path="models/iso_forest_0.45.pkl"):
    # Fallback to 0.5 if 0.45 doesn't exist (e.g. if user skipped training step)
    path = Path(model_path)
    if not path.exists():
        print(f"Model {model_path} not found, trying models/iso_forest_0.5.pkl")
        model_path = "models/iso_forest_0.5.pkl"

    print(f"Loading Stage 1 model from {model_path}...")
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading Stage 1 model: {e}")
        raise

class DynamicEnsemble:
    def __init__(self, pool_size=5, base_estimator_func=None):
        self.pool_size = pool_size
        self.pool = [] # List of (model, accuracy) or just models. Paper implies pruning worst.
        # We need a function to create new base estimators
        if base_estimator_func is None:
            # Default to a small MLP as per paper or similar
            self.base_estimator_func = lambda: MLPClassifier(
                hidden_layer_sizes=(50,), 
                activation='relu', 
                solver='adam', 
                random_state=42, 
                max_iter=50 # Small iterations for speed per batch
            )
        else:
            self.base_estimator_func = base_estimator_func
            
    def predict(self, X):
        if not self.pool:
            # If no models yet, predict default (Normal=0) or random. 
            # Safest is 0 (Benign) or 1 (Attack) depending on risk. 
            # Let's assume 0 if empty.
            return np.zeros(X.shape[0], dtype=int)
        
        # Collect predictions from all models
        preds = np.array([model.predict(X) for model in self.pool])
        
        # Majority vote
        avg_pred = np.mean(preds, axis=0)
        final_pred = (avg_pred >= 0.5).astype(int)
        return final_pred

    def update(self, X_batch, y_batch):
        # 1. Train a new classifier on this batch
        new_model = self.base_estimator_func()
        try:
            # Check if batch has both classes, if not, partial_fit might complain or fit needs handling
            # MLP fit handles single class if we assume it's valid, but ideally we need both.
            # If only one class, we might skip training or handle it.
            if len(np.unique(y_batch)) < 2:
                # Can't train a binary classifier effectively with 1 class usually
                # But let's try fitting; sklearn might error or warn.
                # For simplicity, if < 2 classes, skip adding this model or handle gracefully.
                # Actually, standard fit works but might predict constant.
                pass
            
            new_model.fit(X_batch, y_batch)
            
        except Exception as e:
            print(f"Ensemble update failed: {e}")
            return

        # 2. Add to pool
        if len(self.pool) < self.pool_size:
            self.pool.append(new_model)
        else:
            # 3. Prune: Remove worst performing model
            # To know "worst", we need to evaluate them on the CURRENT batch 
            # (or previous batch, but paper usually implies evaluating on new data).
            # "The dynamic ensemble approach utilizes the Streaming Ensemble Algorithm (SEA)... 
            # prunes procedure that removes the worst-performing classifier from the pool."
            # Typically SEA evaluates existing pool on the NEW batch before training the new one?
            # Or evaluates on the batch used for update? Evaluating on train data is biased.
            # Standard SEA: Evaluate all pool members on the NEW chunk X_batch (before training on it? or after?)
            # Actually, standard SEA evaluates on the new chunk X_batch to determine weights/pruning.
            # Let's evaluate accuracy of current pool on X_batch.
            
            accuracies = []
            for m in self.pool:
                pred = m.predict(X_batch)
                acc = accuracy_score(y_batch, pred)
                accuracies.append(acc)
            
            # Also evaluate the NEW model? Usually the new model is always added, and we remove the worst of the (pool + new).
            # Let's evaluate the new model on X_batch (it will be high since it trained on it - wait, that's overfitting bias).
            # SEA usually keeps the new model. So we remove the worst of the OLD pool?
            # Or we assume the new model is "fresh" and good.
            # Let's stick to: Remove the model with lowest accuracy on the current batch.
            
            worst_idx = np.argmin(accuracies)
            # Replace the worst with the new model
            self.pool[worst_idx] = new_model

def create_stage2_model(model_type='sgd'):
    if model_type == 'sgd':
        # SGDClassifier (Linear)
        return SGDClassifier(loss='log_loss', random_state=42, warm_start=True)
    elif model_type == 'mlp':
        # MLPClassifier (Neural Network) - Matching paper specs where possible
        # Hidden layer 100, ReLU, Adam.
        return MLPClassifier(
            hidden_layer_sizes=(100,), 
            activation='relu', 
            solver='adam', 
            random_state=42, 
            warm_start=True,
            max_iter=1 # Important for partial_fit to simulate online learning one pass
        )
    elif model_type == 'ensemble':
        return DynamicEnsemble(pool_size=5)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def run_pipeline_for_model(model_type, df_train, df_test, stage1_model):
    print(f"\n\n################################################")
    print(f"RUNNING PIPELINE WITH MODEL: {model_type.upper()}")
    print(f"################################################")

    # Prepare Data
    X_train_full = df_train.drop(columns=["ANOMALY"])
    y_train_full = df_train["ANOMALY"]
    
    X_test = df_test.drop(columns=["ANOMALY"])
    y_test = df_test["ANOMALY"]

    # ==========================================
    # PHASE 1: WARMUP STAGE 2
    # ==========================================
    print("\n=== Phase 1: Warming up Stage 2 Model ===")
    
    start_warmup = time.time()
    
    # Stage 1 Prediction (Unsupervised)
    s1_pred_train_raw = stage1_model.predict(X_train_full)
    s1_pred_train = (s1_pred_train_raw == -1).astype(int)
    
    # Filter: Keep only what Stage 1 thinks is Anomaly
    suspicious_indices = np.where(s1_pred_train == 1)[0]
    print(f"Stage 1 flagged {len(suspicious_indices)} out of {len(X_train_full)} training samples as suspicious ({len(suspicious_indices)/len(X_train_full):.2%}).")
    
    X_warmup = X_train_full.iloc[suspicious_indices]
    y_warmup = y_train_full.iloc[suspicious_indices]
    
    stage2_model = create_stage2_model(model_type)
    classes = np.array([0, 1])
    
    if len(X_warmup) > 0:
        if model_type == 'ensemble':
            # For ensemble, we treat warmup as the first "batch" to initialize the pool
            # We can split warmup into chunks to fill the pool
            print("Initializing Ensemble with warmup data...")
            # Split warmup into pool_size chunks
            chunk_size = int(len(X_warmup) / 5) # Assuming pool_size=5
            if chunk_size > 0:
                for i in range(5):
                    start = i * chunk_size
                    end = (i + 1) * chunk_size
                    stage2_model.update(X_warmup.iloc[start:end], y_warmup.iloc[start:end])
            else:
                 stage2_model.update(X_warmup, y_warmup)
        else:
            stage2_model.partial_fit(X_warmup, y_warmup, classes=classes)
        print("Stage 2 Warmup Complete.")
    else:
        print("Warning: No suspicious data found in training set. Stage 2 not warmed up.")

    print(f"Warmup time: {time.time() - start_warmup:.2f}s")

    # ==========================================
    # PHASE 2: ONLINE TESTING (SIMULATION)
    # ==========================================
    print("\n=== Phase 2: Online Testing (Simulation) ===")
    
    batch_size = 10000
    n_batches = int(np.ceil(len(X_test) / batch_size))
    
    y_true_all = []
    y_pred_all = []
    
    total_processed = 0
    stage2_calls = 0
    
    start_test_time = time.time()
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X_test))
        
        X_batch = X_test.iloc[start_idx:end_idx]
        y_batch = y_test.iloc[start_idx:end_idx]
        
        # 1. Stage 1 Screening
        s1_batch_raw = stage1_model.predict(X_batch)
        s1_batch = (s1_batch_raw == -1).astype(int)
        
        y_batch_pred = np.zeros(len(y_batch), dtype=int)
        
        # Identify suspicious indices
        suspicious_mask = (s1_batch == 1)
        suspicious_X = X_batch[suspicious_mask]
        
        if len(suspicious_X) > 0:
            stage2_calls += len(suspicious_X)
            
            # 2. Stage 2 Prediction
            s2_pred = stage2_model.predict(suspicious_X)
            y_batch_pred[suspicious_mask] = s2_pred
            
            # 3. Dynamic Update
            suspicious_y_true = y_batch[suspicious_mask]
            
            if model_type == 'ensemble':
                stage2_model.update(suspicious_X, suspicious_y_true)
            else:
                stage2_model.partial_fit(suspicious_X, suspicious_y_true)
            
        y_true_all.extend(y_batch)
        y_pred_all.extend(y_batch_pred)
        
        total_processed += len(X_batch)
        
        if (i + 1) % 10 == 0:
            print(f"Processed batch {i+1}/{n_batches}...")

    elapsed_test = time.time() - start_test_time
    print(f"\nProcessing Complete. Time: {elapsed_test:.2f}s")
    
    # ==========================================
    # EVALUATION
    # ==========================================
    print("\n=== Final Evaluation ===")
    
    report_dict = classification_report(y_true_all, y_pred_all, digits=4, output_dict=True)
    report_str = classification_report(y_true_all, y_pred_all, digits=4)
    bac = balanced_accuracy_score(y_true_all, y_pred_all)
    
    print(f"Balanced Accuracy: {bac:.4f}")
    print(report_str)
    
    # Save results
    results_dir = Path("evaluation/hybrid_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"hybrid_report_{model_type}.txt"
    with open(results_dir / filename, "w") as f:
        f.write(f"Hybrid 2-Stage Detection System Results ({model_type.upper()})\n")
        f.write("=======================================\n")
        f.write(f"Stage 1 Model: {stage1_model}\n")
        f.write(f"Stage 2 Model Type: {model_type.upper()}\n")
        f.write(f"Traffic Processed: {total_processed}\n")
        f.write(f"Stage 2 Triggered: {stage2_calls} ({stage2_calls/total_processed:.2%})\n")
        f.write(f"Total Time: {elapsed_test:.2f}s\n")
        f.write(f"Balanced Accuracy: {bac:.4f}\n\n")
        f.write(report_str)
        
    print(f"Results saved to {results_dir / filename}")

def run_hybrid_pipeline():
    # 1. Load Resources Once
    df_train, df_test = load_data()
    stage1_model = load_stage1_model()

    # 2. Run SGD
    run_pipeline_for_model('sgd', df_train, df_test, stage1_model)
    
    # 3. Run MLP
    run_pipeline_for_model('mlp', df_train, df_test, stage1_model)

    # 4. Run Ensemble
    run_pipeline_for_model('ensemble', df_train, df_test, stage1_model)

if __name__ == "__main__":
    run_hybrid_pipeline()
