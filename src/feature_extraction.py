import os
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm

def extract_audio_features(audio_path):
    """
    Extract various audio features from a WAV file
    
    Parameters:
    audio_path (str): Path to the audio file
    
    Returns:
    dict: Dictionary containing extracted features
    """
    try:
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Extract features
        features = {}
        
        # Basic properties
        features['duration'] = librosa.get_duration(y=y, sr=sr)
        features['sample_rate'] = sr
        
        # Extract each feature with try-except blocks
        try:
            features['zero_crossing_rate_mean'] = np.mean(librosa.feature.zero_crossing_rate(y))
            features['zero_crossing_rate_std'] = np.std(librosa.feature.zero_crossing_rate(y))
        except Exception as e:
            print(f"Error extracting zero crossing rate: {str(e)}")
        
        try:
            rms = librosa.feature.rms(y=y)
            features['rms_energy_mean'] = np.mean(rms)
            features['rms_energy_std'] = np.std(rms)
        except Exception as e:
            print(f"Error extracting RMS energy: {str(e)}")
        
        # Spectral features
        features['spectral_centroid_mean'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        features['spectral_centroid_std'] = np.std(librosa.feature.spectral_centroid(y=y, sr=sr))
        features['spectral_bandwidth_mean'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        features['spectral_bandwidth_std'] = np.std(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        features['spectral_rolloff_mean'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        features['spectral_rolloff_std'] = np.std(librosa.feature.spectral_rolloff(y=y, sr=sr))
        
        # MFCCs (Mel-Frequency Cepstral Coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i, mfcc in enumerate(mfccs):
            features[f'mfcc_{i}_mean'] = np.mean(mfcc)
            features[f'mfcc_{i}_std'] = np.std(mfcc)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        for i, chroma_val in enumerate(chroma):
            features[f'chroma_{i}_mean'] = np.mean(chroma_val)
            features[f'chroma_{i}_std'] = np.std(chroma_val)
        
        # Tempo and beat features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features['spectral_contrast_mean'] = np.mean(contrast)
        features['spectral_contrast_std'] = np.std(contrast)
        
        # Spectral flatness
        flatness = librosa.feature.spectral_flatness(y=y)
        features['spectral_flatness_mean'] = np.mean(flatness)
        features['spectral_flatness_std'] = np.std(flatness)
        
        return features
    
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None

def process_dataset(fake_dir, real_dir, output_path='data/processed/audio_features.csv', visualize=True):
    """
    Process all audio files in the dataset and extract features
    
    Parameters:
    fake_dir (str): Directory containing fake audio files
    real_dir (str): Directory containing real audio files
    output_path (str): Path to save the extracted features
    visualize (bool): Whether to create visualizations
    
    Returns:
    pd.DataFrame: DataFrame containing extracted features
    """
    # Check if directories exist
    if not os.path.exists(fake_dir):
        raise ValueError(f"Fake directory does not exist: {fake_dir}")
    if not os.path.exists(real_dir):
        raise ValueError(f"Real directory does not exist: {real_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get list of audio files
    fake_files = [f for f in os.listdir(fake_dir) if f.endswith('.wav')]
    real_files = [f for f in os.listdir(real_dir) if f.endswith('.wav')]
    
    print(f"Found {len(fake_files)} fake audio files and {len(real_files)} real audio files")
    
    # Create a list of all audio files with their labels
    audio_files = []
    labels = []
    
    # Add fake audio files
    for file in fake_files:
        audio_files.append(os.path.join(fake_dir, file))
        labels.append(1)  # 1 for fake
    
    # Add real audio files
    for file in real_files:
        audio_files.append(os.path.join(real_dir, file))
        labels.append(0)  # 0 for real
    
    # Extract features from all audio files (with progress bar)
    features_list = []
    for i, audio_path in tqdm(enumerate(audio_files), total=len(audio_files), desc="Extracting features"):
        features = extract_audio_features(audio_path)
        if features:
            features['file_name'] = os.path.basename(audio_path)
            features['label'] = labels[i]  # 0 for real, 1 for fake
            features_list.append(features)
    
    # Create a DataFrame from the features
    features_df = pd.DataFrame(features_list)
    print(f"Created features for {len(features_df)} audio files")
    print(f"Number of features extracted: {features_df.shape[1]}")
    
    # Save the features to a CSV file
    features_df.to_csv(output_path, index=False)
    print(f"Features saved to '{output_path}'")
    
    # Check for missing values
    missing_values = features_df.isnull().sum()
    missing_features = missing_values[missing_values > 0]
    if len(missing_features) > 0:
        print("\nMissing values in features:")
        print(missing_features)
    
    # Visualize feature distributions if requested
    if visualize:
        visualize_features(features_df)
    
    return features_df

def visualize_features(features_df, output_dir=None):
    """
    Create visualizations of feature distributions
    
    Parameters:
    features_df (pd.DataFrame): DataFrame containing features
    output_dir (str): Directory to save visualizations
    """
    # Set default output directory if not provided
    if output_dir is None:
        base_dir = os.path.dirname(os.getcwd())  # Changed from dirname(dirname(getcwd()))
        output_dir = os.path.join(base_dir, 'data', 'visualizations')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Select key features to visualize
    key_features = [
        'spectral_centroid_mean', 
        'spectral_bandwidth_mean', 
        'zero_crossing_rate_mean',
        'rms_energy_mean',
        'mfcc_0_mean',
        'spectral_flatness_mean',
        'tempo'
    ]
    
    # Create a figure with subplots
    plt.figure(figsize=(15, 12))
    
    for i, feature in enumerate(key_features):
        if feature in features_df.columns:
            plt.subplot(3, 3, i+1)
            
            # Plot histograms for real and fake audio
            plt.hist(features_df[features_df['label'] == 0][feature], alpha=0.5, bins=30, label='Real')
            plt.hist(features_df[features_df['label'] == 1][feature], alpha=0.5, bins=30, label='Fake')
            
            plt.title(feature)
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_distributions.png'))
    plt.close()
    
    print(f"Feature distributions saved to '{os.path.join(output_dir, 'feature_distributions.png')}'")
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    corr = features_df.select_dtypes(include=['float64', 'int64']).corr()
    plt.imshow(corr, cmap='coolwarm')
    plt.colorbar()
    plt.title('Feature Correlation Matrix')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()
    
    print(f"Correlation matrix saved to '{os.path.join(output_dir, 'correlation_matrix.png')}'")

if __name__ == "__main__":
    # Debug: Print current working directory and check directories
    print(f"Current working directory: {os.getcwd()}")
    
    # Define paths - using correct path calculation
    # Go up just one level to the NewModel directory
    base_dir = os.path.dirname(os.getcwd())  # Changed from dirname(dirname(getcwd()))
    
    fake_dir = os.path.join(base_dir, 'data', 'AUDIO', 'FAKE')
    real_dir = os.path.join(base_dir, 'data', 'AUDIO', 'REAL')
    output_path = os.path.join(base_dir, 'data', 'processed', 'audio_features.csv')
    
    print(f"Base directory: {base_dir}")
    print(f"Fake directory path: {fake_dir}")
    print(f"Real directory path: {real_dir}")
    
    # Check if directories exist and list their contents
    print(f"Fake directory exists: {os.path.exists(fake_dir)}")
    if os.path.exists(fake_dir):
        print(f"Contents of {fake_dir}:")
        print(os.listdir(fake_dir))
    
    print(f"Real directory exists: {os.path.exists(real_dir)}")
    if os.path.exists(real_dir):
        print(f"Contents of {real_dir}:")
        print(os.listdir(real_dir))
    
    # Process the dataset
    try:
        features_df = process_dataset(fake_dir, real_dir, output_path)
        
        # Only do analysis if we have data
        if len(features_df) > 0:
            print("\nTop features with largest difference between real and fake audio:")
            real_means = features_df[features_df['label'] == 0].mean(numeric_only=True)
            fake_means = features_df[features_df['label'] == 1].mean(numeric_only=True)
            mean_diff = abs(real_means - fake_means).sort_values(ascending=False)
            print(mean_diff.head(10))
            
            print("\nTop features with highest correlation to the label:")
            correlations = features_df.corr()['label'].sort_values(ascending=False)
            print(correlations.head(10))
        else:
            print("\nNo data was processed. Cannot perform analysis.")
    except Exception as e:
        print(f"Error during processing: {str(e)}")