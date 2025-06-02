import streamlit as st
import pandas as pd
from writer.basic import writer

st.set_page_config(page_title="Data Analytics & Prediction App", layout="centered")
st.title("Data Analytics and Prediction Website")

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

writer(uploaded_file)