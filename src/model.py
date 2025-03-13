import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV

def create_models():
    """
    Create a dictionary of models to try
    
    Returns:
    dict: Dictionary of model names and instances
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    return models

def create_param_grids():
    """
    Create parameter grids for hyperparameter tuning
    
    Returns:
    dict: Dictionary of parameter grids for each model
    """
    param_grids = {
        'Logistic Regression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga'],
            'penalty': ['l1', 'l2']
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        },
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'linear']
        }
    }
    
    return param_grids

def train_and_evaluate_models(X_train, X_val, y_train, y_val, output_dir='models'):
    """
    Train and evaluate multiple models
    
    Parameters:
    X_train (pd.DataFrame): Training features
    X_val (pd.DataFrame): Validation features
    y_train (pd.Series): Training labels
    y_val (pd.Series): Validation labels
    output_dir (str): Directory to save model results
    
    Returns:
    dict: Dictionary of model results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create preprocessing pipeline
    preprocessing = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Apply preprocessing
    X_train_processed = preprocessing.fit_transform(X_train)
    X_val_processed = preprocessing.transform(X_val)
    
    # Save preprocessing pipeline
    joblib.dump(preprocessing, os.path.join(output_dir, 'preprocessing_pipeline.pkl'))
    
    # Get models
    models = create_models()
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_processed, y_train)
        
        # Evaluate on validation set
        y_pred = model.predict(X_val_processed)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"{name} Validation Accuracy: {accuracy:.4f}")
        print(classification_report(y_val, y_pred))
        
        # Save confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title(f'{name} Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks([0, 1], ['Real', 'Fake'])
        plt.yticks([0, 1], ['Real', 'Fake'])
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), 
                         horizontalalignment='center', 
                         color='white' if cm[i, j] > cm.max() / 2 else 'black')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name.replace(" ", "_")}_confusion_matrix.png'))
        plt.close()
        
        # ROC curve
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_val_processed)[:, 1]
            fpr, tpr, _ = roc_curve(y_val, y_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{name} ROC Curve')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(output_dir, f'{name.replace(" ", "_")}_roc_curve.png'))
            plt.close()
        
        # Save model
        joblib.dump(model, os.path.join(output_dir, f'{name.replace(" ", "_")}_model.pkl'))
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'report': classification_report(y_val, y_pred, output_dict=True)
        }
    
    # Select the best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    print(f"\nBest model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.4f}")
    
    return results, best_model_name, preprocessing

def tune_hyperparameters(X_train, y_train, best_model_name, preprocessing, output_dir='models'):
    """
    Tune hyperparameters for the best model
    
    Parameters:
    X_train (pd.DataFrame): Training features
    y_train (pd.Series): Training labels
    best_model_name (str): Name of the best model
    preprocessing (Pipeline): Preprocessing pipeline
    output_dir (str): Directory to save model results
    
    Returns:
    object: Best model after hyperparameter tuning
    """
    print(f"\nTuning hyperparameters for {best_model_name}...")
    
    # Get models and parameter grids
    models = create_models()
    param_grids = create_param_grids()
    
    # Apply preprocessing
    X_train_processed = preprocessing.transform(X_train)
    
    # Create grid search
    grid_search = GridSearchCV(
        models[best_model_name],
        param_grids[best_model_name],
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X_train_processed, y_train)
    
    # Get best parameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Save best model
    joblib.dump(best_model, os.path.join(output_dir, f'{best_model_name.replace(" ", "_")}_tuned_model.pkl'))
    
    return best_model

def evaluate_final_model(best_model, preprocessing, X_test, y_test, output_dir='models'):
    """
    Evaluate the final model on the test set
    
    Parameters:
    best_model (object): Best model after hyperparameter tuning
    preprocessing (Pipeline): Preprocessing pipeline
    X_test (pd.DataFrame): Test features
    y_test (pd.Series): Test labels
    output_dir (str): Directory to save model results
    
    Returns:
    dict: Dictionary of test results
    """
    print("\nEvaluating final model on test set...")
    
    # Apply preprocessing
    X_test_processed = preprocessing.transform(X_test)
    
    # Make predictions
    y_pred = best_model.predict(X_test_processed)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(report)
    
    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Final Model Confusion Matrix (Test Set)')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks([0, 1], ['Real', 'Fake'])
    plt.yticks([0, 1], ['Real', 'Fake'])
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), 
                     horizontalalignment='center', 
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_model_confusion_matrix.png'))
    plt.close()
    
    return {
        'accuracy': accuracy,
        'report': report
    }