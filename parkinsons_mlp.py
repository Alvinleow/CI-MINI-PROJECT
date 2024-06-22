import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Load the dataset
parkinsons_data = pd.read_csv('parkinsons.data')

# Display dataset
print(parkinsons_data.head())

## ====================================================================================================================================
##                                                          Standardize
## ====================================================================================================================================

# Separate the features and the target variable
features = parkinsons_data.drop(columns=['name', 'status'])
status = parkinsons_data['status']

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the features and transform them
scaled_features = scaler.fit_transform(features)

# Create a new DataFrame with the scaled features
scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)

# Add the 'status' column back to the scaled DataFrame
scaled_features_df['status'] = status.values

# Display Standardized Data
print(scaled_features_df.head())

## ====================================================================================================================================
##                                                              Split
## ====================================================================================================================================

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_features_df.drop(columns=['status']), scaled_features_df['status'], 
    test_size=0.2, random_state=42
)

# Print the shapes of the training and testing sets
print(f'Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}')

## ====================================================================================================================================
##                                                          Training
## ====================================================================================================================================

# Initialize the MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, alpha=0.0001,
                    solver='adam', random_state=42, learning_rate_init=0.001)

# Train the model
mlp.fit(X_train, y_train)