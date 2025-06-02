import streamlit as st
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

def load_model(model_path='models/trained_model.pkl'):
    """
    Load a pre-trained ML model from disk
    """
    return joblib.load(model_path)

def load_model(model_path='models/trained_model.pkl'):
    """Load a pre-trained model"""
    return joblib.load(model_path)

def train_categorical_model_from_user_data(df, target_column, model_choice):
    X = df.drop(columns=[target_column]).select_dtypes(include='number')
    y = df[target_column]

    le = None

    # Encode labels if needed
    if y.dtype == "object" or pd.api.types.is_categorical_dtype(y):
        le = LabelEncoder()
        y = le.fit_transform(y)
    elif not pd.api.types.is_integer_dtype(y):
        st.error("The selected target must be categorical or convertible to categorical.")
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_choice == "Random Forest":
        model = RandomForestClassifier()
    elif model_choice == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    elif model_choice == "Gradient Boosting":
        model = GradientBoostingClassifier()
    elif model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    else:
        st.error("Invalid model selection.")
        return None, None

    model.fit(X_train, y_train)

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/trained_model.pkl')

    y_pred = model.predict(X_test)
    st.subheader("Classification Accuracy")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Always return model and encoder (encoder can be None if not used)
    return model, le

def train_numerical_model_from_user_data(df, target_column, model_choice):
    """Train a regression model with user-selected algorithm"""
    X = df.drop(columns=[target_column]).select_dtypes(include='number')
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model selection
    if model_choice == "Random Forest":
        model = RandomForestRegressor()
    elif model_choice == "XGBoost":
        model = XGBRegressor()
    elif model_choice == "Gradient Boosting":
        model = GradientBoostingRegressor()
    elif model_choice == "Linear Regression":
        model = LinearRegression()
    else:
        st.error("Invalid model selection.")
        return None

    # Train and evaluate
    model.fit(X_train, y_train)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/trained_model.pkl')

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Regression Metrics")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R² Score: {r2:.2f}")

    return model