import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):

    """
    Preprocess the dataset by standardizing the features and applying SMOTE for class balancing.

    Parameters:
    df (DataFrame): The raw dataset.

    Returns:
    DataFrame: The processed features.
    Series: The target variable after resampling.
    """

    features = df.drop(columns=['name', 'status'])
    status = df['status']

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return scaled_features, status
