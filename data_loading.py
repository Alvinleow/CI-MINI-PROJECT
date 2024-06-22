import pandas as pd

def load_data(file_path):
    """
    Load the Parkinson's dataset.

    Parameters:
    file_path (str): The path to the dataset file.

    Returns:
    DataFrame: The loaded dataset.
    """
    return pd.read_csv(file_path)
