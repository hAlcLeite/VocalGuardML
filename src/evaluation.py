import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

def load_model_and_pipeline(model_path, pipeline_path):
    """
    Load the trained model and preprocessing pipeline
    
    Parameters:
    model_path (str): Path to the saved model
    pipeline_path (str): Path to the saved preprocessing pipeline
    
    Returns:
    tuple: (model, preprocessing_pipeline)
    """
    model = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)
    
    print(f"Loaded model from '{model_path}'")
    print(f"Loaded preprocessing pipeline from '{pipeline_path}'")
    
    return model, pipeline

def evaluate_model(model, pipeline, features_path, output_dir='results'):
    """
    Evaluate the model on new data
    
    Parameters:
    model (object): Trained model
    pipeline (object): Preprocessing pipeline
    features_path (str): Path to the CSV file containing features
    output_dir (str): Directory to save evaluation results
    
    Returns:
    dict: Dictionary of evaluation results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load features
    if not os.path.exists(features_path):
        raise ValueError(f"Features file does not exist: {features_path}")
    
    features_df = pd.read_csv(features_path)
    print(f"Loaded features from '{features_path}'")
    print(f"Shape: {features_df.shape}")
    
    # Check if label column exists
    if 'label' not in features_df.columns:
        print("Warning: 'label' column not found. Performing prediction only.")
        has_labels = False
    else:
        has_labels = True
    
    # Prepare features
    X = features_df.drop(['label', 'file_name'], axis=1, errors='ignore')
    if has_labels:
        y = features_df['label']
    
    # Apply preprocessing
    X_processed = pipeline.transform(X)
    
    # Make predictions
    y_pred = model.predict(X_processed)
    
    # Save predictions
    results_df = pd.DataFrame({
        'file_name': features_df['file_name'] if 'file_name' in features_df.columns else range(len(y_pred)),
        'predicted_label': y_pred
    })
    
    if has_labels:
        results_df['true_label'] = y
        results_df['correct'] = results_df['true_label'] == results_df['predicted_label']
    
    results_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    
    # Calculate metrics if labels are available
    if has_labels:
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(report)
        
        # Save confusion matrix
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title('Model Evaluation Confusion Matrix')
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
        plt.savefig(os.path.join(output_dir, 'evaluation_confusion_matrix.png'))
        plt.close()
        
        # ROC curve if model supports probability predictions
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_processed)[:, 1]
            fpr, tpr, _ = roc_curve(y, y_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(output_dir, 'evaluation_roc_curve.png'))
            plt.close()
        
        return {
            'accuracy': accuracy,
            'report': report,
            'predictions': results_df
        }
    else:
        return {
            'predictions': results_df
        }

def analyze_errors(results_df, features_df, output_dir='results'):
    """
    Analyze prediction errors
    
    Parameters:
    results_df (pd.DataFrame): DataFrame with prediction results
    features_df (pd.DataFrame): DataFrame with features
    output_dir (str): Directory to save analysis results
    
    Returns:
    None
    """
    if 'correct' not in results_df.columns:
        print("Error analysis requires true labels. Skipping.")
        return
    
    # Merge results with features
    analysis_df = pd.merge(
        results_df, 
        features_df, 
        on='file_name', 
        how='inner'
    )
    
    # Separate correct and incorrect predictions
    correct_df = analysis_df[analysis_df['correct']]
    incorrect_df = analysis_df[~analysis_df['correct']]
    
    print(f"Total samples: {len(analysis_df)}")
    print(f"Correct predictions: {len(correct_df)} ({len(correct_df)/len(analysis_df)*100:.2f}%)")
    print(f"Incorrect predictions: {len(incorrect_df)} ({len(incorrect_df)/len(analysis_df)*100:.2f}%)")
    
    # Analyze feature distributions for correct vs incorrect predictions
    feature_cols = features_df.drop(['label', 'file_name'], axis=1, errors='ignore').columns
    
    for feature in feature_cols[:10]:  # Analyze top 10 features
        plt.figure(figsize=(10, 6))
        plt.hist(correct_df[feature], alpha=0.5, label='Correct Predictions')
        plt.hist(incorrect_df[feature], alpha=0.5, label='Incorrect Predictions')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.title(f'Distribution of {feature} for Correct vs Incorrect Predictions')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'error_analysis_{feature}.png'))
        plt.close()
    
    # Save error analysis summary
    error_summary = pd.DataFrame({
        'feature': feature_cols,
        'correct_mean': [correct_df[col].mean() for col in feature_cols],
        'incorrect_mean': [incorrect_df[col].mean() for col in feature_cols],
        'mean_difference': [abs(correct_df[col].mean() - incorrect_df[col].mean()) for col in feature_cols]
    })
    
    error_summary = error_summary.sort_values('mean_difference', ascending=False)
    error_summary.to_csv(os.path.join(output_dir, 'error_analysis_summary.csv'), index=False)
    
    print(f"Error analysis saved to '{output_dir}'")

if __name__ == "__main__":
    # Define paths
    model_path = 'models/Random_Forest_tuned_model.pkl'  # Update with your best model
    pipeline_path = 'models/preprocessing_pipeline.pkl'
    features_path = 'data/processed/audio_features.csv'  # Update with your test data
    output_dir = 'results'
    
    # Load model and pipeline
    model, pipeline = load_model_and_pipeline(model_path, pipeline_path)
    
    # Evaluate model
    results = evaluate_model(model, pipeline, features_path, output_dir)
    
    # Analyze errors
    if 'predictions' in results and 'true_label' in results['predictions'].columns:
        features_df = pd.read_csv(features_path)
        analyze_errors(results['predictions'], features_df, output_dir)
