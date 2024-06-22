from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

def grid_search_model(X_train, y_train, random_seed=42):
    """
    Perform Grid Search to find the best hyperparameters for MLPClassifier.

    Parameters:
    X_train (DataFrame): The training features.
    y_train (Series): The training labels.

    Returns:
    MLPClassifier: The best model found by Grid Search.
    """
    mlp = MLPClassifier(random_state=random_seed)
    parameter_space = {
        'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs','sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive'],
        'max_iter': [3500, 4000]
    }
    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=5)
    clf.fit(X_train, y_train)
    
    print('Best parameters found:\n', clf.best_params_)
    return clf.best_estimator_
