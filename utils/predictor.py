import pandas as pd

def make_prediction(df, model):
    """
    Use the loaded model to make predictions
    """
    X = df.select_dtypes(include='number')  # Use only numeric columns
    predictions = model.predict(X)
    return pd.DataFrame({"Prediction": predictions})