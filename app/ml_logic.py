# app/ml_logic.py - COMPLETE RECURSION FIX

import pandas as pd
import numpy as np
import requests
import pandas_ta as ta
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Financial Modeling Prep API Configuration
FMP_API_KEY = "3V5meXmuiupLM1fyL4vs6GeDB7RFA0LM"
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

def get_fmp_interval_mapping(interval):
    """Maps our internal intervals to FMP API intervals."""
    interval_map = {
        '1m': '1min',
        '5m': '5min', 
        '15m': '15min',
        '30m': '30min',
        '1h': '1hour',
        '4h': '4hour',
        '1d': '1day',
        '1wk': '1week'
    }
    return interval_map.get(interval, '1hour')

def get_fmp_period_days(period):
    """Converts period string to number of days for FMP API."""
    period_map = {
        '1d': 1,
        '5d': 5,
        '1mo': 30,
        '3mo': 90,
        '6mo': 180,
        '1y': 365,
        '2y': 730,
        '5y': 1825,
        '10y': 3650,
        '90d': 90,
        'max': 365
    }
    return period_map.get(period, 90)

def fetch_yfinance_data(symbol, period='90d', interval='1h'):
    """
    MAIN DATA FETCHING FUNCTION - DIRECT FMP API CALL
    This is the ONLY function that should be called by routes.
    NO MORE RECURSION - Everything happens here.
    """
    print(f"=== DIRECT FMP FETCH for {symbol} ===")
    
    # Determine symbol format based on asset type
    api_symbol = symbol
    
    if symbol.endswith('=X'):
        # Forex: Convert EURUSD=X -> EUR/USD
        base_quote = symbol.replace('=X', '')
        if len(base_quote) == 6:
            api_symbol = f"{base_quote[:3]}/{base_quote[3:]}"
        print(f"   Converted forex symbol: {symbol} -> {api_symbol}")
    elif '-USD' in symbol:
        # Crypto: Convert BTC-USD -> BTCUSD
        api_symbol = symbol.replace('-USD', 'USD')
        print(f"   Converted crypto symbol: {symbol} -> {api_symbol}")
    elif symbol.startswith('^'):
        # Index: Use as-is
        print(f"   Using index symbol: {symbol}")
    else:
        # Stock: Use as-is
        print(f"   Using stock symbol: {symbol}")
    
    # Get FMP API parameters
    fmp_interval = get_fmp_interval_mapping(interval)
    days = get_fmp_period_days(period)
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    from_date = start_date.strftime('%Y-%m-%d')
    to_date = end_date.strftime('%Y-%m-%d')
    
    # Build FMP API URL
    url = f"{FMP_BASE_URL}/historical-chart/{fmp_interval}/{api_symbol}"
    params = {
        'from': from_date,
        'to': to_date,
        'apikey': FMP_API_KEY
    }
    
    print(f"   URL: {url}")
    print(f"   Params: {params}")
    
    try:
        # Make the API request
        print(f"   Making HTTP request...")
        response = requests.get(url, params=params, timeout=30)
        
        print(f"   Response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"   ❌ HTTP Error {response.status_code}: {response.text}")
            return None
        
        # Parse JSON response
        data = response.json()
        print(f"   Response type: {type(data)}")
        
        if not data:
            print(f"   ⚠️ Empty response from FMP API")
            return None
        
        if not isinstance(data, list):
            print(f"   ⚠️ Unexpected response format. Expected list, got {type(data)}")
            print(f"   Response content: {str(data)[:200]}...")
            return None
        
        print(f"   📊 Got {len(data)} data points from FMP")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        print(f"   DataFrame shape: {df.shape}")
        print(f"   DataFrame columns: {list(df.columns)}")
        
        # Validate required columns
        if 'date' not in df.columns:
            print(f"   ❌ Missing 'date' column")
            return None
        
        required_price_cols = ['open', 'high', 'low', 'close']
        missing_price_cols = [col for col in required_price_cols if col not in df.columns]
        if missing_price_cols:
            print(f"   ❌ Missing price columns: {missing_price_cols}")
            return None
        
        # Process dates
        print(f"   Processing dates...")
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df = df.set_index('date')
        df = df.sort_index()
        
        # Standardize column names
        column_mapping = {
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        df = df.rename(columns=column_mapping)
        
        # Handle missing volume (common for forex/indices)
        if 'Volume' not in df.columns:
            print(f"   📊 Adding dummy volume data")
            df['Volume'] = 1000000
        
        # Convert to numeric
        print(f"   Converting to numeric...")
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean data
        print(f"   Cleaning data...")
        initial_rows = len(df)
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        final_rows = len(df)
        
        if final_rows < initial_rows:
            print(f"   🧹 Removed {initial_rows - final_rows} rows with NaN values")
        
        if df.empty:
            print(f"   ❌ No valid data after cleaning")
            return None
        
        # Final result
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        result_df = df[required_cols]
        
        print(f"   ✅ SUCCESS! Returning {len(result_df)} rows")
        print(f"   📈 Price range: {result_df['Close'].min():.4f} - {result_df['Close'].max():.4f}")
        print(f"   📅 Date range: {result_df.index[0]} to {result_df.index[-1]}")
        
        return result_df
        
    except requests.exceptions.Timeout:
        print(f"   ❌ Request timeout after 30 seconds")
        return None
    except requests.exceptions.RequestException as e:
        print(f"   ❌ HTTP request failed: {e}")
        return None
    except ValueError as e:
        print(f"   ❌ JSON parsing failed: {e}")
        return None
    except Exception as e:
        print(f"   ❌ Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_features_for_prediction(data, feature_columns_list):
    """Creates all necessary features for the model from raw price data."""
    if data is None or data.empty:
        print("   ❌ create_features_for_prediction: No input data")
        return pd.DataFrame()

    print(f"   📊 Creating features from {len(data)} rows of data")
    df = data.copy()

    try:
        # Calculate technical indicators
        print(f"   📈 Calculating RSI...")
        df.ta.rsi(length=14, append=True)
        
        print(f"   📈 Calculating EMA...")
        df.ta.ema(length=200, append=True)
        
        print(f"   📈 Calculating ATR...")
        df.ta.atr(length=14, append=True)
        
        # Create all required features
        print(f"   🔧 Creating custom features...")
        df['channel_slope'] = 0.0
        df['channel_width_atr'] = 1.0
        df['bars_outside_zone'] = 0
        df['breakout_distance_norm'] = 0.0
        df['breakout_candle_body_ratio'] = 0.5
        df['rsi_14'] = df.get('RSI_14', 50.0)
        
        # Price vs EMA200
        ema_200 = df.get('EMA_200', df['Close'])
        df['price_vs_ema200'] = df['Close'] / ema_200
        
        # Volume ratio
        volume_ma = df['Volume'].rolling(20, min_periods=1).mean()
        df['volume_ratio'] = df['Volume'] / volume_ma
        
        # Time features
        df['hour_of_day'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # Trading features
        df['risk_reward_ratio'] = 2.0
        df['stop_loss_in_atrs'] = 1.5
        df['entry_pos_in_channel_norm'] = 0.5
        
        # Historical features
        for i in range(24):
            df[f'hist_close_channel_dev_t_minus_{i}'] = 0.0
        
        # Interaction features
        df['volume_rsi_interaction'] = df['volume_ratio'] * df['rsi_14']
        df['breakout_strength'] = 0.0
        df['channel_efficiency'] = 0.0
        
        # Boolean features
        df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
        df['price_above_ema'] = (df['Close'] > ema_200).astype(int)
        df['high_risk_trade'] = 0
        df['trade_type_encoded'] = 0
        
        # Ensure all required columns exist
        for col in feature_columns_list:
            if col not in df.columns:
                df[col] = 0.0
        
        # Handle NaN values
        print(f"   🧹 Handling NaN values...")
        df = df.ffill().bfill().fillna(0)
        
        # Select required columns
        required_cols = feature_columns_list + ['Close']
        available_cols = [col for col in required_cols if col in df.columns]
        result_df = df[available_cols]
        
        print(f"   ✅ Features created successfully: {len(result_df)} rows, {len(result_df.columns)} columns")
        return result_df
        
    except Exception as e:
        print(f"   ❌ Error creating features: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def get_model_prediction(data, model, scaler, feature_columns):
    """Generates a prediction for a single asset."""
    if data is None or data.empty:
        return {"error": "Cannot generate prediction, input data is missing."}
    
    print(f"   🤖 Generating model prediction...")
    
    try:
        features_df = create_features_for_prediction(data, feature_columns)
        if features_df.empty:
            return {"error": "Could not create features for prediction."}

        latest_features = features_df.iloc[-1].copy()
        last_price = latest_features['Close']
        
        print(f"   💰 Latest price: {last_price}")

        # Create features for BUY and SELL scenarios
        buy_features = latest_features.copy()
        buy_features['trade_type_encoded'] = 0
        sell_features = latest_features.copy()
        sell_features['trade_type_encoded'] = 1

        # Prepare data for model
        buy_df = pd.DataFrame([buy_features])[feature_columns]
        sell_df = pd.DataFrame([sell_features])[feature_columns]

        # Scale features
        buy_scaled = scaler.transform(buy_df)
        sell_scaled = scaler.transform(sell_df)
        
        # Get predictions
        buy_prob = model.predict_proba(buy_scaled)[0][1]
        sell_prob = model.predict_proba(sell_scaled)[0][1]
        
        print(f"   📊 Buy probability: {buy_prob:.3f}, Sell probability: {sell_prob:.3f}")

        # Determine signal
        confidence_threshold = 0.55
        signal_type = "HOLD"
        confidence = max(buy_prob, sell_prob)

        if buy_prob > sell_prob and buy_prob > confidence_threshold:
            signal_type = "BUY"
            confidence = buy_prob
        elif sell_prob > buy_prob and sell_prob > confidence_threshold:
            signal_type = "SELL"
            confidence = sell_prob
        else:
            confidence = 0.5

        print(f"   🎯 Final signal: {signal_type} with {confidence:.1%} confidence")

        return {
            "signal": signal_type,
            "confidence": confidence,
            "latest_price": last_price,
            "buy_prob": buy_prob,
            "sell_prob": sell_prob,
            "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        print(f"   ❌ Prediction failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Prediction failed: {str(e)}"}

# BACKWARD COMPATIBILITY ALIAS - NO RECURSION
def fetch_data_via_proxies(symbol, period='90d', interval='1h'):
    """Alias for fetch_yfinance_data for backward compatibility."""
    return fetch_yfinance_data(symbol, period, interval)
