import os
import argparse
from feature_extraction import process_dataset
from preprocessing import preprocess_data
from model import train_and_evaluate_models, tune_hyperparameters, evaluate_final_model

def main(extract_features=True, preprocess=True, train=True):
    # Define paths
    base_dir = os.path.dirname(os.getcwd())
    fake_dir = os.path.join(base_dir, 'data', 'AUDIO', 'FAKE')
    real_dir = os.path.join(base_dir, 'data', 'AUDIO', 'REAL')
    features_path = os.path.join(base_dir, 'data', 'processed', 'audio_features.csv')
    models_dir = os.path.join(base_dir, 'models')
    
    # Step 1: Extract features
    if extract_features:
        print("Step 1: Extracting features...")
        features_df = process_dataset(fake_dir, real_dir, features_path)
        print("Feature extraction complete.\n")
    
    # Step 2: Preprocess data
    if preprocess:
        print("Step 2: Preprocessing data...")
        X_train, X_val, X_test, y_train, y_val, y_test, selected_features = preprocess_data(
            features_path, os.path.join(base_dir, 'data', 'processed')
        )
        print("Preprocessing complete.\n")
    else:
        # Load preprocessed data
        import joblib
        X_train, X_val, X_test, y_train, y_val, y_test = joblib.load(
            os.path.join(base_dir, 'data', 'processed', 'preprocessed_data.pkl')
        )
    
    # Step 3: Train and evaluate models
    if train:
        print("Step 3: Training and evaluating models...")
        results, best_model_name, preprocessing = train_and_evaluate_models(
            X_train, X_val, y_train, y_val, models_dir
        )
        
        # Step 4: Tune hyperparameters
        print("Step 4: Tuning hyperparameters...")
        best_model = tune_hyperparameters(
            X_train, y_train, best_model_name, preprocessing, models_dir
        )
        
        # Step 5: Evaluate final model
        print("Step 5: Evaluating final model...")
        test_results = evaluate_final_model(
            best_model, preprocessing, X_test, y_test, models_dir
        )
        
        print(f"\nFinal model test accuracy: {test_results['accuracy']:.4f}")
        print("Training and evaluation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run audio deepfake detection pipeline')
    parser.add_argument('--skip-extraction', action='store_true', help='Skip feature extraction')
    parser.add_argument('--skip-preprocessing', action='store_true', help='Skip preprocessing')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training')
    
    args = parser.parse_args()
    
    main(
        extract_features=not args.skip_extraction,
        preprocess=not args.skip_preprocessing,
        train=not args.skip_training
    )