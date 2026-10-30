
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "workloads.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_FILE)

# Input features and target
features = [
    "instructions",
    "memory_mb",
    "io_operations",
    "priority"
]

X = df[features]
y = df["burst_time"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define ML models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
}

results = []
best_model = None
best_name = None
best_mae = float("inf")


for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    # Select model with lowest MAE
    if mae < best_mae:
        best_mae = mae
        best_model = model
        best_name = name

# Save trained model
model_path = MODEL_DIR / "best_model.pkl"
joblib.dump(best_model, model_path)

# Save evaluation results
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

print("\n" + "=" * 40)
print(f"Best model: {best_name}")
print(f"Saved model: {model_path}")

print("\nModel comparison:")
print(results_df.to_string(index=False))

# Compare actual and predicted values for five test samples
sample = X_test.iloc[:5]
sample_predictions = best_model.predict(sample)

print("\nSample predictions:")

for actual, predicted in zip(y_test.iloc[:5], sample_predictions):
    print(f"Actual: {actual:.2f} | Predicted: {predicted:.2f}")
