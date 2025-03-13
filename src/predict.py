import os
import joblib
import argparse
from feature_extraction import extract_audio_features
import pandas as pd

def predict_audio(audio_path, model_path, preprocessing_path, selected_features_path):
    """
    Predict whether an audio file is real or fake
    
    Parameters:
    audio_path (str): Path to the audio file
    model_path (str): Path to the trained model
    preprocessing_path (str): Path to the preprocessing pipeline
    selected_features_path (str): Path to the selected features file
    
    Returns:
    dict: Prediction results
    """
    # Extract features
    features = extract_audio_features(audio_path)
    if not features:
        return {"error": f"Failed to extract features from {audio_path}"}
    
    # Convert to DataFrame
    features_df = pd.DataFrame([features])
    
    # Load selected features
    with open(selected_features_path, 'r') as f:
        selected_features = [line.strip() for line in f.readlines()]
    
    # Select only the features used during training
    # Handle missing features
    for feature in selected_features:
        if feature not in features_df.columns:
            features_df[feature] = 0  # Default value for missing features
    
    features_df = features_df[selected_features]
    
    # Load preprocessing pipeline and model
    preprocessing = joblib.load(preprocessing_path)
    model = joblib.load(model_path)
    
    # Preprocess features
    features_processed = preprocessing.transform(features_df)
    
    # Make prediction
    prediction = model.predict(features_processed)[0]
    probability = model.predict_proba(features_processed)[0][1]
    
    return {
        'file': os.path.basename(audio_path),
        'prediction': 'Fake' if prediction == 1 else 'Real',
        'confidence': probability if prediction == 1 else 1 - probability
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict if an audio file is real or fake')
    parser.add_argument('audio_path', help='Path to the audio file')
    parser.add_argument('--model', default='models/Random_Forest_tuned_model.pkl', 
                        help='Path to the trained model')
    parser.add_argument('--preprocessing', default='models/preprocessing_pipeline.pkl',
                        help='Path to the preprocessing pipeline')
    parser.add_argument('--features', default='data/processed/selected_features.txt',
                        help='Path to the selected features file')
    
    args = parser.parse_args()
    
    # Adjust paths if running from src directory
    base_dir = os.path.dirname(os.getcwd())
    model_path = os.path.join(base_dir, args.model)
    preprocessing_path = os.path.join(base_dir, args.preprocessing)
    features_path = os.path.join(base_dir, args.features)
    
    result = predict_audio(args.audio_path, model_path, preprocessing_path, features_path)
    print(result)