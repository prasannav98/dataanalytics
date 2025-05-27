import streamlit as st
import pandas as pd
from utils.data_handler import load_and_clean_data
from utils.eda import generate_summary
from utils.model_trainer import load_model, train_model_from_user_data
from utils.predictor import make_prediction

st.set_page_config(page_title="Data Analytics & Prediction App", layout="centered")
st.title("Data Analytics and Prediction Website")

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Step 2: Display raw data
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # Step 3: Clean and validate data
    df_clean = load_and_clean_data(df)

    # Step 4: EDA Summary
    st.subheader("Data Summary")
    st.text(generate_summary(df_clean))

    # Step 5: Let user select the target column
    st.subheader("Select Target Column")
    target_column = st.selectbox("Choose the column to predict (target)", options=df_clean.columns)

    if target_column:
        model = train_model_from_user_data(df_clean, target_column)
        X_test = df_clean.drop(columns=[target_column])
        predictions = make_prediction(X_test, model)
        st.subheader("Predictions")
        st.write(predictions)
else:
    st.info("Please upload a CSV file to proceed.")