import os
import datetime as dt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


# =========================
# INPUT / OUTPUT PATHS
# =========================
INPUT_CSV   = "data/modeling_dataset.csv"
OUT_METRICS = "data/model_regression_metrics.csv"
OUT_LOG     = "logs_model_regression.txt"
MODEL_DIR   = "models"


# =========================
# MAIN
# =========================
def main():
    # Load dataset
    df = pd.read_csv(INPUT_CSV)

    # Define target and features
    # NOTE: Adjust target column if needed
    target_col = "ts_points" if "ts_points" in df.columns else "week"
    X = df.drop(columns=["game_id", "team", "opponent", target_col], errors="ignore")
    y = df[target_col]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.01)
    }

    metrics = []

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Train/evaluate each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)

        metrics.append({
            "model": name,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        })

        # Save model artifact
        joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))

    # Write metrics CSV
    pd.DataFrame(metrics).to_csv(OUT_METRICS, index=False)

    # Write log file
    with open(OUT_LOG, "w", encoding="utf-8") as f:
        f.write(f"snapshot_utc={dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
        f.write(f"input_rows={len(df)}\n")
        f.write(f"features={len(X.columns)}\n")
        for m in metrics:
            f.write(f"{m['model']}: rmse={m['rmse']:.4f}, mae={m['mae']:.4f}, r2={m['r2']:.4f}\n")

    print("Regression models complete.")


if __name__ == "__main__":
    main()
