from adaptive_trainer import AdaptiveTrainer
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 1. Data Setup ---
X, y = make_classification(n_samples=1600, n_features=20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# --- 2. Training Loop ---
trainer = AdaptiveTrainer(alpha=0.3)

for episode in range(50):
    plan = trainer.plan({"dataset_size": len(X_train), "episode_num": episode})

    # --- DEBUG: Uncomment this if you want to see exactly what keys ARE in the plan ---
    # print(f"Plan keys: {plan.keys()}")

    # --- FIX: Safely get fidelity, defaulting to 1.0 if missing ---
    fidelity = plan.get('fidelity', 1.0)
    
    # Calculate subset size based on the safe fidelity value
    subset_size = int(len(X_train) * fidelity)
    subset_size = max(subset_size, 50) # Safety floor
    
    X_sub = X_train[:subset_size]
    y_sub = y_train[:subset_size]

    # Handle batch size (default to 32 if trainer doesn't provide it)
    planned_batch = plan.get('batch_size', 32)
    real_batch_size = min(planned_batch, subset_size)

    model = MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=plan.get('max_iter', 10), # Default if missing
        batch_size=real_batch_size,
        random_state=episode
    )
    
    model.fit(X_sub, y_sub)
    accuracy = model.score(X_val, y_val)

    # --- Update Cost Calculation ---
    # We use the 'fidelity' variable we created earlier, which is safe
    cost = plan.get('max_iter', 10) * (fidelity ** 2)
    
    trainer.observe(metric=accuracy, cost=cost)
    
    print(f"Ep {episode}: Acc={accuracy:.4f} | Fid={fidelity:.2f}")

print("Done.")