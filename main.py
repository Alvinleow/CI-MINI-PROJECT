from sklearn.model_selection import train_test_split, StratifiedKFold
import data_loading
import data_preprocessing
import model_training
import model_evaluation
import model_validation
import model_tuning
import joblib
from imblearn.over_sampling import SMOTE

def main():
    
    """
    Main function to load the dataset, preprocess the data, train the MLP model, and evaluate its performance.
    """
    # Set a random seed for reproducibility
    random_seed = 42
    
    # Load the dataset
    file_path = 'parkinsons.data'
    df = data_loading.load_data(file_path)

    # Preprocess the data
    X, y = data_preprocessing.preprocess_data(df)

    # Split the data into 70% training and 30% remaining
    X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, test_size=0.3, random_state=random_seed)

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=random_seed)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # Split the remaining data into 50% validation and 50% test, which is 15% each of the original data
    X_valid, X_test, y_valid, y_test = train_test_split(X_remaining, y_remaining, test_size=0.5, random_state=random_seed)

    print(f'Training set shape: {X_train_smote.shape}, Validation set shape: {X_valid.shape}, Test set shape: {X_test.shape}')

    # Perform Grid Search to find the best model
    best_model = model_tuning.grid_search_model(X_train_smote, y_train_smote, random_seed)

    # Evaluate the best model on the validation set 
    print("Evaluation on validation set:")
    model_evaluation.evaluate_model(best_model, X_valid, y_valid)

    # Save the best model
    joblib.dump(best_model, 'parkinsons_best_model.pkl')

    # Load the best model
    best_model = joblib.load('parkinsons_best_model.pkl')

    # Final evaluation on the test set
    print("\nFinal evaluation on test set:")
    model_evaluation.evaluate_model(best_model, X_test, y_test)

    # Perform cross-validation
    mean_cv_score, std_cv_score = model_validation.cross_validate_model(X, y, random_seed)
    print(f'\nMean cross-validation accuracy: {mean_cv_score:.4f} ± {std_cv_score:.4f}')

     # Plot learning curve
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    model_evaluation.plot_learning_curve(best_model, X_train_smote, y_train_smote, cv=skf)

if __name__ == '__main__':
    main()
