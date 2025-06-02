import streamlit as st
import pandas as pd
from utils.data_handler import load_and_clean_data
from utils.eda import generate_summary
from writer.model_writer import categorical_writer
from writer.model_writer import numerical_writer
from writer.plot_writer import show_plot_interface

def writer(uploaded_file):
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

        show_plot_interface(df_clean)

        st.subheader("Type of target variable")
        type_of_target = st.selectbox("Choose if it's a numerical or categorical target variable", options = ["Categorical", "Numerical"])

        # Step 5: Let user select the target column
        st.subheader("Select Target Column")
        target_column = st.selectbox("Choose the column to predict (target)", options=df_clean.columns)

        if target_column:
            if (type_of_target == "Categorical"):
                categorical_writer(df_clean, target_column)
            else:
                numerical_writer(df_clean, target_column)
    else:
        st.info("Please upload a CSV file to proceed.")
    
