# src/model_trainer.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

FEATURES = ["points_diff", "neutral_venue", "h2h_home_win_rate", "h2h_match_count"]
TARGET = "outcome"

def train_and_save_model(data_path, model_output_path, scaler_output_path):
    df = pd.read_csv(data_path)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    print(classification_report(y_test, model.predict(X_test_scaled),
                                 target_names=["away_win", "draw", "home_win"]))
    print("Coefficients:", model.coef_)

    joblib.dump(model, model_output_path)
    joblib.dump(scaler, scaler_output_path)
    print(f"Model saved to {model_output_path}")
    print(f"Scaler saved to {scaler_output_path}")


if __name__ == "__main__":
    train_and_save_model(
        "data/processed/training_data.csv",
        "models/classifier.pkl",
        "models/scaler.pkl",
    )