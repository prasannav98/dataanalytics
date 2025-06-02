import streamlit as st
import pandas as pd
from utils.model_trainer import load_model, train_categorical_model_from_user_data, train_numerical_model_from_user_data

def categorical_writer(df_clean, target_column):
    model_type = st.selectbox("Choose classification model", ["Random Forest", "XGBoost", "Gradient Boosting", "Logistic Regression"])
    
    if st.button("Train Classification Model"):
        result = train_categorical_model_from_user_data(df_clean, target_column, model_type)
        
        if result:
            model, le = result
            X_test = df_clean.drop(columns=[target_column]).select_dtypes(include='number')
            y_pred = model.predict(X_test)
            if le:
                y_pred = le.inverse_transform(y_pred)
            
            predictions_df = pd.DataFrame({
                "Predicted": y_pred,
                "Actual": df_clean[target_column].reset_index(drop=True)
            })
            st.subheader("Predictions vs Actual")
            st.write(predictions_df)


def numerical_writer(df_clean, target_column):
    model_type = st.selectbox("Choose regression model", ["Random Forest", "XGBoost", "Gradient Boosting", "Linear Regression"])

    if st.button("Train Regression Model"):
        model = train_numerical_model_from_user_data(df_clean, target_column, model_type)
        if model:
            X_test = df_clean.drop(columns=[target_column]).select_dtypes(include='number')
            predictions = model.predict(X_test)
            predictions_df = pd.DataFrame({
                "Predicted": predictions.squeeze().tolist(),
                "Actual": df_clean[target_column].reset_index(drop=True)
            })
            st.subheader("Predictions vs Actual")
            st.write(predictions_df)