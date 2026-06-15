import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def train_model():
    # Load dataset
    df = pd.read_csv('data/crowd_dataset.csv')

    # Separate features and label
    X = df.drop('label', axis=1)
    y = df['label']

    # Split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create XGBoost model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )

    # Train the model
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)

    # Print results
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred,
          target_names=['Safe', 'Warning', 'Critical']))

    # Save the model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/risk_model.pkl')
    print("\nModel saved to models/risk_model.pkl")

if __name__ == "__main__":
    train_model()