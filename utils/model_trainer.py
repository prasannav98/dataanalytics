import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import os

def load_model(model_path='models/trained_model.pkl'):
    """
    Load a pre-trained ML model from disk
    """
    return joblib.load(model_path)

def train_model_from_user_data(df, target_column):
    """
    Train model from user-uploaded data based on selected target column.
    Returns trained model.
    """
    X = df.drop(columns=[target_column]).select_dtypes(include='number')
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Save model (optional)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/trained_model.pkl')

    # Print accuracy to terminal (optional feedback)
    y_pred = model.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, y_pred))

    return model
