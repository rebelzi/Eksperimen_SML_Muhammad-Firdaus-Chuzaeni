import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path for import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from automate_MuhammadFirdausChuzaeni import preprocess_pipeline

def test_preprocessing_consistency():
    """
    Test bahwa preprocessing manual dan automate menghasilkan output yang sama
    """
    print("=" * 70)
    print("UNIT TEST: Preprocessing Consistency")
    print("=" * 70)
    
    # Load hasil manual (dari notebook)
    df_manual = pd.read_csv('diabetes_preprocessing/preprocessed_diabetes.csv')
    print(f"\n✓ Loaded manual preprocessing result: {df_manual.shape}")
    
    # Run automate (simpan ke file berbeda untuk perbandingan)
    df_automate = preprocess_pipeline(
        '../diabetes-datasets.csv',
        'diabetes_preprocessing/preprocessed_diabetes_automate.csv'
    )
    print(f"✓ Run automate preprocessing: {df_automate.shape}")
    
    # Test 1: Shape
    assert df_manual.shape == df_automate.shape, \
        f"Shape mismatch: {df_manual.shape} != {df_automate.shape}"
    print("\n✅ TEST 1 PASSED: Shape identik")
    
    # Test 2: Columns
    assert df_manual.columns.tolist() == df_automate.columns.tolist(), \
        "Columns mismatch"
    print("✅ TEST 2 PASSED: Columns identik")
    
    # Test 3: Data types
    assert (df_manual.dtypes == df_automate.dtypes).all(), \
        "Data types mismatch"
    print("✅ TEST 3 PASSED: Data types identik")
    
    # Test 4: Values (dengan toleransi)
    assert np.allclose(df_manual.values, df_automate.values, rtol=1e-10, atol=1e-10), \
        "Values mismatch"
    print("✅ TEST 4 PASSED: Values identik (dengan toleransi)")
    
    # Test 5: Summary statistics
    for col in df_manual.columns:
        manual_mean = df_manual[col].mean()
        automate_mean = df_automate[col].mean()
        assert np.isclose(manual_mean, automate_mean, rtol=1e-10), \
            f"Mean mismatch for {col}: {manual_mean} != {automate_mean}"
    print("✅ TEST 5 PASSED: Summary statistics identik")
    
    # Test 6: Check outliers removal consistency
    manual_count = len(df_manual)
    automate_count = len(df_automate)
    assert manual_count == automate_count, \
        f"Row count mismatch: {manual_count} != {automate_count}"
    print("✅ TEST 6 PASSED: Outlier removal konsisten")
    
    print("\n" + "=" * 70)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 70)
    print("\n📊 Summary:")
    print(f"  • Shape: {df_manual.shape}")
    print(f"  • Columns: {len(df_manual.columns)}")
    print(f"  • Manual preprocessing: diabetes_preprocessing/preprocessed_diabetes.csv")
    print(f"  • Automate preprocessing: diabetes_preprocessing/preprocessed_diabetes_automate.csv")
    print("\n✅ Kedua metode preprocessing menghasilkan output yang IDENTIK")

if __name__ == "__main__":
    test_preprocessing_consistency()