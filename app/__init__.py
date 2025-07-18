# Updated app/__init__.py with enhanced debugging

import os
from flask import Flask
from flask_cors import CORS
import joblib
import pandas as pd

def create_app():
    app = Flask(__name__)

    # --- Enable CORS for all routes, allowing requests from any origin ---
    CORS(app)

    # --- Configuration ---
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(APP_DIR)
    ML_MODELS_FOLDER = os.path.join(PROJECT_ROOT, 'ml_models/')

    # --- ENHANCED DEBUGGING ---
    print(f"\n=== DEBUGGING MODEL LOADING ===")
    print(f"APP_DIR: {APP_DIR}")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"ML_MODELS_FOLDER: {ML_MODELS_FOLDER}")
    
    # Check if ml_models directory exists
    print(f"ML_MODELS_FOLDER exists: {os.path.exists(ML_MODELS_FOLDER)}")
    
    # List contents of project root
    print(f"Contents of PROJECT_ROOT ({PROJECT_ROOT}):")
    try:
        for item in os.listdir(PROJECT_ROOT):
            item_path = os.path.join(PROJECT_ROOT, item)
            if os.path.isdir(item_path):
                print(f"  📁 {item}/")
                # If it's ml_models, list its contents
                if item == 'ml_models':
                    try:
                        for subitem in os.listdir(item_path):
                            print(f"    📄 {subitem}")
                    except Exception as e:
                        print(f"    ❌ Error listing ml_models contents: {e}")
            else:
                print(f"  📄 {item}")
    except Exception as e:
        print(f"❌ Error listing PROJECT_ROOT: {e}")
    
    # Alternative paths to check
    alternative_paths = [
        os.path.join(APP_DIR, 'ml_models'),
        os.path.join(os.getcwd(), 'ml_models'),
        './ml_models',
        '/opt/render/project/src/ml_models',
        os.path.join(os.path.dirname(__file__), '..', 'ml_models')
    ]
    
    print(f"\n=== CHECKING ALTERNATIVE PATHS ===")
    for alt_path in alternative_paths:
        abs_path = os.path.abspath(alt_path)
        exists = os.path.exists(abs_path)
        print(f"Path: {alt_path}")
        print(f"  Absolute: {abs_path}")
        print(f"  Exists: {exists}")
        if exists:
            try:
                contents = os.listdir(abs_path)
                print(f"  Contents: {contents}")
            except Exception as e:
                print(f"  Error listing: {e}")
        print()

    app.config['ASSET_CLASSES'] = {
        "Forex": [
            "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X", "NZDUSD=X",
            "EURGBP=X", "EURJPY=X", "EURAUD=X", "EURCAD=X", "EURCHF=X",
            "GBPJPY=X", "AUDJPY=X", "CADJPY=X", "CHFJPY=X", "NZDJPY=X",
            "GBPAUD=X", "GBPCAD=X", "GBPCHF=X",
            "USDZAR=X", "USDMXN=X", "USDTRY=X", "USDSGD=X"
        ],
        "Crypto": [
            "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "BNB-USD", "ADA-USD", "AVAX-USD",
            "DOGE-USD", "DOT-USD", "MATIC-USD", "SHIB-USD", "TRX-USD", "LTC-USD", "LINK-USD", "BCH-USD"
        ],
        "Stocks": [
            "AAPL", "GOOGL", "MSFT", "AMZN", "NVDA", "META", "TSLA", "AMD", "INTC", "CRM",
            "JPM", "BAC", "WFC", "GS", "MS", "V", "MA",
            "WMT", "COST", "HD", "NKE", "MCD", "KO",
            "JNJ", "PFE", "MRNA", "UNH",
            "XOM", "CVX", "BA"
        ],
        "Indices": [
            "^GSPC", "^DJI", "^IXIC", "^RUT",
            "^FTSE", "^GDAXI", "^FCHI",
            "^N225", "^HSI",
            "^VIX", "^TNX"
        ]
    }

    app.config['TIMEFRAMES'] = {
        "1 Hour": "1h",
        "4 Hours": "4h",
        "1 Day": "1d",
        "1 Week": "1wk"
    }

    # --- Load Model Artifacts with Enhanced Error Handling ---
    print("\n=== ATTEMPTING MODEL LOADING ===")
    
    # Try multiple possible locations
    possible_locations = [
        ML_MODELS_FOLDER,
        os.path.join(APP_DIR, 'ml_models'),
        os.path.join(os.getcwd(), 'ml_models'),
        './ml_models',
        '/opt/render/project/src/ml_models'
    ]
    
    model_loaded = False
    for location in possible_locations:
        print(f"\n--- Trying location: {location} ---")
        
        model_path = os.path.join(location, 'model.joblib')
        scaler_path = os.path.join(location, 'scaler.joblib')
        features_path = os.path.join(location, 'feature_columns.csv')
        
        print(f"  model.joblib: {model_path} -> {os.path.exists(model_path)}")
        print(f"  scaler.joblib: {scaler_path} -> {os.path.exists(scaler_path)}")
        print(f"  feature_columns.csv: {features_path} -> {os.path.exists(features_path)}")
        
        if all([os.path.exists(model_path), os.path.exists(scaler_path), os.path.exists(features_path)]):
            print(f"  ✅ All files found at: {location}")
            try:
                app.model = joblib.load(model_path)
                print("  ✅ SUCCESS: Model loaded.")

                app.scaler = joblib.load(scaler_path)
                print("  ✅ SUCCESS: Scaler loaded.")

                app.feature_columns = pd.read_csv(features_path)['feature_name'].tolist()
                print(f"  ✅ SUCCESS: Feature columns ({len(app.feature_columns)}) loaded.")

                app.config['MODELS_LOADED'] = True
                model_loaded = True
                break
                
            except Exception as e:
                print(f"  ❌ Error loading models from {location}: {e}")
                continue
        else:
            missing_files = []
            if not os.path.exists(model_path):
                missing_files.append('model.joblib')
            if not os.path.exists(scaler_path):
                missing_files.append('scaler.joblib')
            if not os.path.exists(features_path):
                missing_files.append('feature_columns.csv')
            print(f"  ❌ Missing files: {missing_files}")
    
    if not model_loaded:
        print("\n🔥🔥🔥 CRITICAL ERROR: Could not load models from any location!")
        app.model = None
        app.scaler = None
        app.feature_columns = None
        app.config['MODELS_LOADED'] = False
        
        # Final diagnostic
        print("\n=== FINAL DIAGNOSTIC ===")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Python __file__: {__file__}")
        print(f"App directory: {APP_DIR}")
        print(f"Project root: {PROJECT_ROOT}")
        
        # List all directories in project root
        try:
            print(f"All items in project root:")
            for item in os.listdir(PROJECT_ROOT):
                full_path = os.path.join(PROJECT_ROOT, item)
                if os.path.isdir(full_path):
                    print(f"  📁 {item}/")
                else:
                    print(f"  📄 {item}")
        except Exception as e:
            print(f"Error listing project root: {e}")
    
    # Register routes
    with app.app_context():
        from . import routes

    return app
