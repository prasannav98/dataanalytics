import pandas as pd

def load_and_clean_data(df):
    """
    Cleans the input DataFrame by:
    - Dropping missing values
    - Trimming column names
    - Removing ID-like and irrelevant columns
    """
    # Drop rows with missing values
    df = df.dropna()

    # Strip whitespace from column names
    df.columns = [col.strip() for col in df.columns]

    # Drop ID-like columns (case-insensitive)
    id_like_cols = [col for col in df.columns if 'id' in col.lower()]
    
    # Drop known irrelevant columns (you can extend this list)
    irrelevant_keywords = ['timestamp', 'date', 'name', 'address']
    irrelevant_cols = [col for col in df.columns if any(key in col.lower() for key in irrelevant_keywords)]

    # Combine all columns to drop
    drop_cols = list(set(id_like_cols + irrelevant_cols))
    df = df.drop(columns=drop_cols, errors='ignore')

    return df
