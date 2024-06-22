from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neural_network import MLPClassifier

def cross_validate_model(X, y, random_seed=42):
    """
    Perform cross-validation on the MLPClassifier.

    Parameters:
    X (DataFrame): The features.
    y (Series): The target variable.

    Returns:
    float: The mean accuracy score from cross-validation.
    float: The standard deviation of accuracy scores from cross-validation.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    
    model = MLPClassifier(random_state=random_seed, max_iter=3500)

    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

    return scores.mean(), scores.std()
