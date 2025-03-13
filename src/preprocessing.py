import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import joblib
import librosa
import librosa.display
import os


def load_features(features_path):
    """
    Load features from CSV file
    
    Parameters:
    features_path (str): Path to the CSV file containing features
    
    Returns:
    pd.DataFrame: DataFrame containing features
    """
    if not os.path.exists(features_path):
        raise ValueError(f"Features file does not exist: {features_path}")
    
    features_df = pd.read_csv(features_path)
    print(f"Loaded features from '{features_path}'")
    print(f"Shape: {features_df.shape}")
    
    return features_df

def handle_missing_values(df):
    """
    Handle missing values in the DataFrame
    
    Parameters:
    df (pd.DataFrame): DataFrame containing features
    
    Returns:
    pd.DataFrame: DataFrame with missing values handled
    """
    # Check for missing values
    missing_values = df.isnull().sum()
    missing_features = missing_values[missing_values > 0]
    
    if len(missing_features) > 0:
        print(f"Found {len(missing_features)} features with missing values")
        print(missing_features)
        
        # For now, we'll use median imputation
        # This will be replaced by the pipeline later
        imputer = SimpleImputer(strategy='median')
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        print("Missing values handled with median imputation")
    else:
        print("No missing values found")
    
    return df

def handle_outliers(df, columns=None):
    """
    Handle outliers in the DataFrame using IQR method
    
    Parameters:
    df (pd.DataFrame): DataFrame containing features
    columns (list): List of columns to check for outliers. If None, all numeric columns are used.
    
    Returns:
    pd.DataFrame: DataFrame with outliers handled
    """
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Exclude the label column if it's in the list
    if 'label' in columns:
        columns = columns.drop('label')
    
    print(f"Checking {len(columns)} features for outliers")
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if outliers > 0:
            print(f"Found {outliers} outliers in {col}")
            
            # Cap outliers
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    
    print("Outliers handled with capping")
    return df

def select_features(X, y, n_features=None, threshold='mean'):
    """
    Select important features using Random Forest feature importance
    
    Parameters:
    X (pd.DataFrame): Feature matrix
    y (pd.Series): Target variable
    n_features (int): Number of features to select. If None, threshold is used.
    threshold (str or float): Threshold for feature selection
    
    Returns:
    list: List of selected feature names
    """
    print("Selecting important features using Random Forest")
    
    # Use Random Forest to select important features
    selector = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=42),
        threshold=threshold,
        max_features=n_features
    )
    selector.fit(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    
    print(f"Selected {len(selected_features)} features out of {X.shape[1]}")
    print("Top 10 selected features:")
    print(selected_features[:10])
    
    return selected_features

def split_data(df, selected_features, test_size=0.2, val_size=0.25, random_state=42):
    """
    Split data into train, validation, and test sets
    
    Parameters:
    df (pd.DataFrame): DataFrame containing features and label
    selected_features (list): List of selected feature names
    test_size (float): Proportion of data to use for testing
    val_size (float): Proportion of training data to use for validation
    random_state (int): Random seed for reproducibility
    
    Returns:
    tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Ensure all selected features are in the DataFrame
    for feature in selected_features:
        if feature not in df.columns:
            raise ValueError(f"Selected feature '{feature}' not found in DataFrame")
    
    # Split features and target
    X = df[selected_features]
    y = df['label']
    
    # First split: training + validation vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: training vs validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state, stratify=y_train_val
    )
    
    print(f"Data split into train ({X_train.shape[0]} samples), "
          f"validation ({X_val.shape[0]} samples), and test ({X_test.shape[0]} samples) sets")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def extract_audio_features(file_path, sr=22050, n_mfcc=13):
    """
    Extract key audio features from a .wav file. This will mainly be used to gather information from the FE

    Parameters:
    file_path (str): Path to the audio file
    sr (int): Sample rate for loading the audio
    n_mfcc (int): Number of MFCCs to extract

    Returns:
    dict: Extracted features
    """
    try:
        y, sr = librosa.load(file_path, sr=sr)
        
        # Extract features
        features = {
            'chroma_stft': np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
            'rmse': np.mean(librosa.feature.rms(y=y)),
            'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            'rolloff': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(y)),
        }
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        for i in range(n_mfcc):
            features[f'mfcc_{i+1}'] = np.mean(mfccs[i])

        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def preprocess_data(features_path, output_dir='data/processed', n_features=None):
    """
    Preprocess the data: load, clean, select features, and split
    
    Parameters:
    features_path (str): Path to the CSV file containing features
    output_dir (str): Directory to save preprocessed data
    n_features (int): Number of features to select. If None, threshold is used.
    
    Returns:
    tuple: (X_train, X_val, X_test, y_train, y_val, y_test, selected_features)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load features
    df = load_features(features_path)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Handle outliers
    df = handle_outliers(df)
    
    # Select features
    X = df.drop(['label', 'file_name'], axis=1, errors='ignore')
    y = df['label']
    selected_features = select_features(X, y, n_features)
    
    # Save selected features
    with open(os.path.join(output_dir, 'selected_features.txt'), 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, selected_features)
    
    # Save preprocessed data
    joblib.dump((X_train, X_val, X_test, y_train, y_val, y_test), 
                os.path.join(output_dir, 'preprocessed_data.pkl'))
    
    print(f"Preprocessed data saved to '{os.path.join(output_dir, 'preprocessed_data.pkl')}'")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, selected_features


def process_audio_files(audio_dir, output_csv='data/processed/audio_features.csv'):
    """
    Process all .wav files in a directory and save extracted features. This will be used to gather information from the FE

    Parameters:
    audio_dir (str): Directory containing .wav files
    output_csv (str): Path to save extracted features as a CSV file

    Returns:
    pd.DataFrame: DataFrame containing extracted features
    """
    feature_list = []
    
    for file in os.listdir(audio_dir):
        if file.endswith(".wav"):
            file_path = os.path.join(audio_dir, file)
            features = extract_audio_features(file_path)
            if features:
                features['file_name'] = file  # Store file name for reference
                feature_list.append(features)
    
    # Convert to DataFrame
    df = pd.DataFrame(feature_list)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"Extracted features saved to {output_csv}")
    return df


if __name__ == "__main__":
    # Define paths with correct relative paths
    base_dir = os.path.dirname(os.getcwd())  # Go up one level to the project root
    audio_dir = os.path.join(base_dir, 'data', 'raw', 'audio')  # Raw .wav files
    features_path = os.path.join(base_dir, 'data', 'processed', 'audio_features.csv')
    output_dir = os.path.join(base_dir, 'data', 'processed')
    
    print(f"Looking for features file at: {features_path}")
    print(f"Output directory: {output_dir}")
    
    print(f"Extracting features from {audio_dir}...")
    process_audio_files(audio_dir, features_path)  # Convert .wav to CSV
    
    # Preprocess the data
    X_train, X_val, X_test, y_train, y_val, y_test, selected_features = preprocess_data(
        features_path, output_dir
    )