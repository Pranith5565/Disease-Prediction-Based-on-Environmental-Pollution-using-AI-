import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

def train_disease_prediction_model(data_path="environmental_health_data.csv"):
    try:
        df = pd.read_csv(data_path)
        print(f"Data successfully loaded. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}. Please run generate_dataset.py first.")
        return

    # Drop the continuous "Risk_Score" as it's a direct proxy of the label, keeping it would artificially inflate performance
    X = df.drop(columns=["Disease_Risk", "Risk_Score"])
    y = df["Disease_Risk"]

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Init Model
    # A robust algorithm resistant to overfitting and does extremely well with tabular data
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    # Evaluate Model
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Feature Importance
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False)
    
    print("\n--- Feature Importance ---")
    print(importance_df)

    # Save Model and Scaler
    print("\nSaving the model and scaler to disk...")
    joblib.dump(model, "disease_prediction_model.pkl")
    joblib.dump(scaler, "disease_prediction_scaler.pkl")
    print("Done! Model saved as 'disease_prediction_model.pkl'")

if __name__ == "__main__":
    train_disease_prediction_model()
