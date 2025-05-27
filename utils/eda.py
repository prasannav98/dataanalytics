def generate_summary(df):
    """
    Generates basic stats: rows, columns, dtypes, missing values
    """
    summary = f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n"
    summary += f"Data Types:\n{df.dtypes}\n"
    summary += f"Missing Values:\n{df.isnull().sum()}"
    return summary