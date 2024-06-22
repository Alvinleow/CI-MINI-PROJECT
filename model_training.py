from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

def train_model(X_train, y_train):
    """
    Train an MLPClassifier on the given training data.

    Parameters:
    X_train (DataFrame): The training features.
    y_train (Series): The training labels.

    Returns:
    MLPClassifier: The trained model.
    """
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, alpha=0.0001,
                        solver='adam', random_state=42, learning_rate_init=0.0001)
    
    # Train the model
    mlp.fit(X_train, y_train)
    
    return mlp
