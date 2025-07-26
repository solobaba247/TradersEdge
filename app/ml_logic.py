# app/ml_logic.py - FIXED VERSION (Recursion Issue Resolved)

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
        '90d': 90,  # Default period used in your app
        'max': 365  # Limit to 1 year for free tier
    }
    return period_map.get(period, 90)

def fetch_fmp_data_unified(symbol, period='90d', interval='1h'):
    """
    UNIFIED FMP data fetching function to prevent recursion.
    This replaces all the individual fetch functions.
    """
    print(f"--- Fetching data for {symbol} from FMP ---")
    
    # Determine symbol format based on asset type
    api_symbol = symbol
    
    if symbol.endswith('=X'):
        # Forex: Convert EURUSD=X -> EUR/USD
        base_quote = symbol.replace('=X', '')
        if len(base_quote) == 6:
            api_symbol = f"{base_quote[:3]}/{base_quote[3:]}"
    elif '-USD' in symbol:
        # Crypto: Convert BTC-USD -> BTCUSD
        api_symbol = symbol.replace('-USD', 'USD')
    # Stocks and indices use symbol as-is
    
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
    
    try:
        print(f"   Requesting: {url}")
        print(f"   Params: {params}")
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if not data or not isinstance(data, list):
            print(f"   ⚠️ FMP returned no data for {api_symbol}")
            return None
        
        print(f"   📊 FMP returned {len(data)} data points")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Check if we have the date column
        if 'date' not in df.columns:
            print(f"   ❌ No 'date' column found. Available columns: {list(df.columns)}")
            return None
        
        # Parse dates and set as index
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df = df.set_index('date')
        df = df.sort_index()  # Ensure chronological order
        
        # Standardize column names
        column_mapping = {
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Ensure we have required OHLC columns
        required_ohlc = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_ohlc if col not in df.columns]
        
        if missing_cols:
            print(f"   ❌ Missing OHLC columns: {missing_cols}")
            return None
        
        # Handle Volume column (might not exist for forex/indices)
        if 'Volume' not in df.columns:
            print(f"   📊 No volume data, using dummy volume")
            df['Volume'] = 1000000  # Dummy volume
        
        # Convert to numeric and clean data
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with NaN values in OHLC
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        if df.empty:
            print(f"   ⚠️ No valid data after cleaning for {api_symbol}")
            return None
        
        # Final validation
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        result_df = df[required_cols]
        
        print(f"   ✅ Success! Got {len(result_df)} clean rows for {symbol}")
        print(f"   📈 Price range: {result_df['Close'].min():.4f} - {result_df['Close'].max():.4f}")
        
        return result_df
        
    except requests.exceptions.RequestException as e:
        print(f"   ❌ FMP API request failed for {symbol}: {e}")
        return None
    except Exception as e:
        print(f"   ❌ Error processing FMP data for {symbol}: {e}")
        return None

# CRITICAL FIX: Replace the recursive function with direct call
def fetch_yfinance_data(symbol, period='90d', interval='1h'):
    """
    Main data fetching function - FIXED to prevent recursion.
    This is the function called by your routes.
    """
    return fetch_fmp_data_unified(symbol, period, interval)

def create_features_for_prediction(data, feature_columns_list):
    """Creates all necessary features for the model from raw price data."""
    df = data.copy()
    if df.empty:
        return pd.DataFrame()

    try:
        # --- Standard Technical Indicators ---
        df.ta.rsi(length=14, append=True)
        df.ta.ema(length=200, append=True)
        df.ta.atr(length=14, append=True)
        
        # --- Feature Creation (Matching feature_columns.csv) ---
        df['channel_slope'] = 0.0
        df['channel_width_atr'] = 1.0
        df['bars_outside_zone'] = 0
        df['breakout_distance_norm'] = 0.0
        df['breakout_candle_body_ratio'] = 0.5
        df['rsi_14'] = df.get('RSI_14', 50.0)
        ema_200 = df.get('EMA_200', df['Close'])
        df['price_vs_ema200'] = df['Close'] / ema_200
        volume_ma = df['Volume'].rolling(20, min_periods=1).mean()
        df['volume_ratio'] = df['Volume'] / volume_ma
        df['hour_of_day'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['risk_reward_ratio'] = 2.0
        df['stop_loss_in_atrs'] = 1.5
        df['entry_pos_in_channel_norm'] = 0.5
        for i in range(24):
            df[f'hist_close_channel_dev_t_minus_{i}'] = 0.0
        df['volume_rsi_interaction'] = df['volume_ratio'] * df['rsi_14']
        df['breakout_strength'] = 0.0
        df['channel_efficiency'] = 0.0
        df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
        df['price_above_ema'] = (df['Close'] > ema_200).astype(int)
        df['high_risk_trade'] = 0
        df['trade_type_encoded'] = 0
        
        for col in feature_columns_list:
            if col not in df.columns:
                df[col] = 0.0
        
        # Handle NaN values properly for newer pandas versions
        df = df.ffill().bfill().fillna(0)
        
        required_cols = feature_columns_list + ['Close']
        available_cols = [col for col in required_cols if col in df.columns]
        
        return df[available_cols]
        
    except Exception as e:
        print(f"Error creating features: {e}")
        return pd.DataFrame()

def get_model_prediction(data, model, scaler, feature_columns):
    """Generates a prediction for a single asset."""
    if data is None or data.empty:
        return {"error": "Cannot generate prediction, input data is missing."}
    
    try:
        features_df = create_features_for_prediction(data, feature_columns)
        if features_df.empty:
            return {"error": "Could not create features for prediction."}

        latest_features = features_df.iloc[-1].copy()
        last_price = latest_features['Close']

        buy_features = latest_features.copy()
        buy_features['trade_type_encoded'] = 0
        sell_features = latest_features.copy()
        sell_features['trade_type_encoded'] = 1

        buy_df = pd.DataFrame([buy_features])[feature_columns]
        sell_df = pd.DataFrame([sell_features])[feature_columns]

        buy_scaled = scaler.transform(buy_df)
        sell_scaled = scaler.transform(sell_df)
        
        buy_prob = model.predict_proba(buy_scaled)[0][1]
        sell_prob = model.predict_proba(sell_scaled)[0][1]

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

        return {
            "signal": signal_type,
            "confidence": confidence,
            "latest_price": last_price,
            "buy_prob": buy_prob,
            "sell_prob": sell_prob,
            "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

def fetch_data_via_proxies(symbol, period='90d', interval='1h'):
    """Alias for fetch_yfinance_data for backward compatibility."""
    return fetch_yfinance_data(symbol, period, interval)
