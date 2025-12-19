import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def load_data(filepath):
    """Load dataset dari file CSV"""
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Data loaded successfully")
        print(f"  Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None

def handle_zero_values(df):
    """Replace nilai 0 yang tidak wajar dengan NaN"""
    df_clean = df.copy()
    zero_not_allowed = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for col in zero_not_allowed:
        df_clean[col] = df_clean[col].replace(0, np.nan)
    
    print(f"✓ Zero values handled")
    return df_clean

def impute_missing_values(df):
    """Impute missing values dengan median"""
    df_imputed = df.copy()
    columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for col in columns_to_impute:
        median_val = df_imputed[col].median()
        df_imputed[col] = df_imputed[col].fillna(median_val)
    
    print(f"✓ Missing values imputed")
    return df_imputed

def remove_duplicates(df):
    """Hapus data duplikat"""
    before = len(df)
    df_clean = df.drop_duplicates()
    after = len(df_clean)
    
    print(f"✓ Duplicates removed: {before - after} rows")
    return df_clean

def remove_outliers_iqr(df, columns):
    """Remove outliers menggunakan IQR method"""
    df_clean = df.copy()
    initial_shape = df_clean.shape[0]
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    removed = initial_shape - df_clean.shape[0]
    print(f"✓ Outliers removed: {removed} rows")
    return df_clean

def scale_features(df, target_col='Outcome'):
    """Standardize features"""
    df_scaled = df.copy()
    X = df_scaled.drop(target_col, axis=1)
    y = df_scaled[target_col]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    df_final = X_scaled_df.copy()
    df_final[target_col] = y.values
    
    print(f"✓ Features scaled")
    return df_final, scaler

def preprocess_pipeline(input_path, output_path):
    """
    Complete preprocessing pipeline
    
    Args:
        input_path: Path ke raw dataset
        output_path: Path untuk menyimpan preprocessed dataset
    
    Returns:
        df_processed: DataFrame yang sudah dipreprocess
    """
    print("=" * 60)
    print("STARTING PREPROCESSING PIPELINE - DIABETES DATASET")
    print("=" * 60)
    
    # 1. Load data
    df = load_data(input_path)
    if df is None:
        return None
    
    # 2. Handle zero values
    df = handle_zero_values(df)
    
    # 3. Impute missing values
    df = impute_missing_values(df)
    
    # 4. Remove duplicates
    df = remove_duplicates(df)
    
    # 5. Remove outliers
    numeric_features = df.columns[:-1].tolist()
    df = remove_outliers_iqr(df, numeric_features)
    
    # 6. Scale features
    df_processed, scaler = scale_features(df)
    
    # 7. Save processed data
    # Create directory if not exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    print(f"✓ Processed data saved to: {output_path}")
    
    print("=" * 60)
    print("PREPROCESSING COMPLETED")
    print(f"Final shape: {df_processed.shape}")
    print("=" * 60)
    
    return df_processed

if __name__ == "__main__":
    # Define paths
    input_path = "../diabetes-datasets.csv"
    output_path = "diabetes_preprocessing/preprocessed_diabetes.csv"
    
    # Run preprocessing
    df_final = preprocess_pipeline(input_path, output_path)
    
    if df_final is not None:
        print("\n✅ Preprocessing successful!")
        print(f"Data ready for training with shape: {df_final.shape}")
        
        # Also split and save train/test sets
        X = df_final.drop('Outcome', axis=1)
        y = df_final['Outcome']
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Save train-test split
        X_train.to_csv('diabetes_preprocessing/X_train.csv', index=False)
        X_test.to_csv('diabetes_preprocessing/X_test.csv', index=False)
        y_train.to_csv('diabetes_preprocessing/y_train.csv', index=False)
        y_test.to_csv('diabetes_preprocessing/y_test.csv', index=False)
        print("✓ Train-test split saved")
        print(f"  Training set: {X_train.shape[0]} samples")
        print(f"  Test set: {X_test.shape[0]} samples")