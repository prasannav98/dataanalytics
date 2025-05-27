import pandas as pd

def load_and_clean_data(df):
    df = df.dropna()  # Drop rows with missing values
    df.columns = [col.strip() for col in df.columns]  # Trim column names
    return df
