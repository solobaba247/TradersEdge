# diagnostic.py - Run this script to debug your model loading issues

import os
import sys

def diagnose_model_files():
    """Comprehensive diagnostic for model file issues."""
    
    print("=" * 60)
    print("ML MODEL FILES DIAGNOSTIC")
    print("=" * 60)
    
    # 1. Basic Python/System Info
    print(f"\n1. SYSTEM INFO:")
    print(f"   Python version: {sys.version}")
    print(f"   Current working directory: {os.getcwd()}")
    print(f"   Script location: {__file__}")
    print(f"   Script directory: {os.path.dirname(__file__)}")
    
    # 2. Check current directory structure
    print(f"\n2. CURRENT DIRECTORY STRUCTURE:")
    current_dir = os.getcwd()
    try:
        for item in os.listdir(current_dir):
            item_path = os.path.join(current_dir, item)
            if os.path.isdir(item_path):
                print(f"   📁 {item}/")
                # If it's ml_models, show contents
                if item == 'ml_models':
                    try:
                        for subitem in os.listdir(item_path):
                            subitem_path = os.path.join(item_path, subitem)
                            size = os.path.getsize(subitem_path) if os.path.isfile(subitem_path) else 0
                            print(f"      📄 {subitem} ({size} bytes)")
                    except Exception as e:
                        print(f"      ❌ Error reading ml_models: {e}")
            else:
                size = os.path.getsize(item_path)
                print(f"   📄 {item} ({size} bytes)")
    except Exception as e:
        print(f"   ❌ Error listing current directory: {e}")
    
    # 3. Check specific model files
    print(f"\n3. MODEL FILES CHECK:")
    model_files = [
        'ml_models/model.joblib',
        'ml_models/scaler.joblib', 
        'ml_models/feature_columns.csv'
    ]
    
    for file_path in model_files:
        full_path = os.path.abspath(file_path)
        exists = os.path.exists(file_path)
        print(f"   {file_path}:")
        print(f"      Exists: {exists}")
        print(f"      Full path: {full_path}")
        
        if exists:
            try:
                size = os.path.getsize(file_path)
                print(f"      Size: {size} bytes")
                
                # Additional checks for each file type
                if file_path.endswith('.csv'):
                    try:
                        import pandas as pd
                        df = pd.read_csv(file_path)
                        print(f"      CSV rows: {len(df)}")
                        print(f"      CSV columns: {list(df.columns)}")
                    except Exception as e:
                        print(f"      CSV read error: {e}")
                        
                elif file_path.endswith('.joblib'):
                    try:
                        import joblib
                        obj = joblib.load(file_path)
                        print(f"      Joblib type: {type(obj)}")
                        if hasattr(obj, 'n_features_in_'):
                            print(f"      Model features: {obj.n_features_in_}")
                    except Exception as e:
                        print(f"      Joblib load error: {e}")
            except Exception as e:
                print(f"      Size check error: {e}")
        print()
    
    # 4. Check environment variables
    print(f"4. ENVIRONMENT VARIABLES:")
    relevant_env_vars = ['PORT', 'PYTHON_VERSION', 'WEB_CONCURRENCY', 'RENDER']
    for var in relevant_env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"   {var}: {value}")
    
    # 5. Check app directory structure (if app exists)
    print(f"\n5. APP DIRECTORY STRUCTURE:")
    app_dir = os.path.join(os.getcwd(), 'app')
    if os.path.exists(app_dir):
        try:
            for item in os.listdir(app_dir):
                print(f"   📄 app/{item}")
        except Exception as e:
            print(f"   ❌ Error listing app directory: {e}")
    else:
        print(f"   ❌ app/ directory not found")
    
    # 6. Try importing required modules
    print(f"\n6. DEPENDENCY CHECK:")
    required_modules = ['flask', 'pandas', 'joblib', 'numpy', 'sklearn']
    for module in required_modules:
        try:
            __import__(module)
            print(f"   ✅ {module}: OK")
        except ImportError as e:
            print(f"   ❌ {module}: FAILED - {e}")
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    diagnose_model_files()
