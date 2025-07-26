# app/ml_logic.py

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

def fetch_fmp_stock_data(symbol, period='90d', interval='1h'):
    """Fetch stock data from Financial Modeling Prep API."""
    print(f"--- Fetching stock data for {symbol} from FMP ---")
    
    fmp_interval = get_fmp_interval_mapping(interval)
    days = get_fmp_period_days(period)
    
    # Calculate from and to dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Format dates for API
    from_date = start_date.strftime('%Y-%m-%d')
    to_date = end_date.strftime('%Y-%m-%d')
    
    # Build URL for historical chart data
    url = f"{FMP_BASE_URL}/historical-chart/{fmp_interval}/{symbol}"
    params = {
        'from': from_date,
        'to': to_date,
        'apikey': FMP_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if not data or not isinstance(data, list):
            print(f"   ⚠️ FMP returned no data for {symbol}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # FMP returns data with 'date' column in format '2024-01-01 09:30:00'
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], utc=True)
            df = df.set_index('date')
            df = df.sort_index()  # Ensure chronological order
        else:
            print(f"   ⚠️ No date column found in FMP response for {symbol}")
            return None
        
        # Standardize column names to match your existing code
        column_mapping = {
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"   ⚠️ Missing columns {missing_cols} for {symbol}")
            return None
        
        # Convert to numeric and clean data
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        
        # Remove rows with NaN values
        df = df.dropna()
        
        if df.empty:
            print(f"   ⚠️ No valid data after cleaning for {symbol}")
            return None
        
        print(f"   ✅ Success with FMP for {symbol}! Got {len(df)} rows")
        return df[required_cols]
        
    except requests.exceptions.RequestException as e:
        print(f"   ❌ FMP API request failed for {symbol}: {e}")
        return None
    except Exception as e:
        print(f"   ❌ Error processing FMP data for {symbol}: {e}")
        return None

def fetch_fmp_forex_data(symbol, period='90d', interval='1h'):
    """Fetch forex data from Financial Modeling Prep API."""
    print(f"--- Fetching forex data for {symbol} from FMP ---")
    
    # Convert symbol format (EURUSD=X -> EUR/USD)
    if symbol.endswith('=X'):
        # Remove =X and format as XXX/YYY
        base_quote = symbol.replace('=X', '')
        if len(base_quote) == 6:
            forex_pair = f"{base_quote[:3]}/{base_quote[3:]}"
        else:
            forex_pair = base_quote
    else:
        forex_pair = symbol
    
    fmp_interval = get_fmp_interval_mapping(interval)
    days = get_fmp_period_days(period)
    
    # Calculate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    from_date = start_date.strftime('%Y-%m-%d')
    to_date = end_date.strftime('%Y-%m-%d')
    
    # FMP Forex endpoint
    url = f"{FMP_BASE_URL}/historical-chart/{fmp_interval}/{forex_pair}"
    params = {
        'from': from_date,
        'to': to_date,
        'apikey': FMP_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if not data or not isinstance(data, list):
            print(f"   ⚠️ FMP returned no forex data for {forex_pair}")
            return None
        
        df = pd.DataFrame(data)
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], utc=True)
            df = df.set_index('date')
            df = df.sort_index()
        else:
            print(f"   ⚠️ No date column in forex response for {forex_pair}")
            return None
        
        # Standardize columns
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low', 
            'close': 'Close',
            'volume': 'Volume'
        }
        
        df = df.rename(columns=column_mapping)
        
        # For forex, volume might not be available, so we'll create dummy volume
        if 'Volume' not in df.columns:
            df['Volume'] = 1000000  # Dummy volume for forex
        
        # Ensure required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols[:4]:  # OHLC
            if col not in df.columns:
                print(f"   ⚠️ Missing {col} column for {forex_pair}")
                return None
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        df = df.dropna()
        
        if df.empty:
            print(f"   ⚠️ No valid forex data after cleaning for {forex_pair}")
            return None
        
        print(f"   ✅ Success with FMP forex for {forex_pair}! Got {len(df)} rows")
        return df[required_cols]
        
    except Exception as e:
        print(f"   ❌ Error fetching FMP forex data for {forex_pair}: {e}")
        return None

def fetch_fmp_crypto_data(symbol, period='90d', interval='1h'):
    """Fetch cryptocurrency data from Financial Modeling Prep API."""
    print(f"--- Fetching crypto data for {symbol} from FMP ---")
    
    # Convert symbol format (BTC-USD -> BTCUSD)
    if '-USD' in symbol:
        crypto_symbol = symbol.replace('-USD', 'USD')
    else:
        crypto_symbol = symbol
    
    fmp_interval = get_fmp_interval_mapping(interval)
    days = get_fmp_period_days(period)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    from_date = start_date.strftime('%Y-%m-%d')
    to_date = end_date.strftime('%Y-%m-%d')
    
    # FMP Crypto endpoint
    url = f"{FMP_BASE_URL}/historical-chart/{fmp_interval}/{crypto_symbol}"
    params = {
        'from': from_date,
        'to': to_date,
        'apikey': FMP_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if not data or not isinstance(data, list):
            print(f"   ⚠️ FMP returned no crypto data for {crypto_symbol}")
            return None
        
        df = pd.DataFrame(data)
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], utc=True)
            df = df.set_index('date')
            df = df.sort_index()
        else:
            print(f"   ⚠️ No date column in crypto response for {crypto_symbol}")
            return None
        
        # Standardize columns
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close', 
            'volume': 'Volume'
        }
        
        df = df.rename(columns=column_mapping)
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                if col == 'Volume':
                    df[col] = 1000000  # Dummy volume if not available
                else:
                    print(f"   ⚠️ Missing {col} column for {crypto_symbol}")
                    return None
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        if df.empty:
            print(f"   ⚠️ No valid crypto data after cleaning for {crypto_symbol}")
            return None
        
        print(f"   ✅ Success with FMP crypto for {crypto_symbol}! Got {len(df)} rows")
        return df[required_cols]
        
    except Exception as e:
        print(f"   ❌ Error fetching FMP crypto data for {crypto_symbol}: {e}")
        return None

def fetch_fmp_index_data(symbol, period='90d', interval='1h'):
    """Fetch index data from Financial Modeling Prep API."""
    print(f"--- Fetching index data for {symbol} from FMP ---")
    
    # Convert symbol format (^GSPC -> ^GSPC or map to FMP equivalent)
    index_symbol_map = {
        '^GSPC': '^GSPC',  # S&P 500
        '^DJI': '^DJI',    # Dow Jones
        '^IXIC': '^IXIC',  # NASDAQ
        '^RUT': '^RUT',    # Russell 2000
        '^VIX': '^VIX',    # VIX
        '^TNX': '^TNX',    # 10-Year Treasury
        '^FTSE': '^FTSE',  # FTSE 100
        '^GDAXI': '^GDAXI', # DAX
        '^FCHI': '^FCHI',  # CAC 40
        '^N225': '^N225',  # Nikkei 225
        '^HSI': '^HSI'     # Hang Seng
    }
    
    fmp_symbol = index_symbol_map.get(symbol, symbol)
    
    fmp_interval = get_fmp_interval_mapping(interval)
    days = get_fmp_period_days(period)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    from_date = start_date.strftime('%Y-%m-%d')
    to_date = end_date.strftime('%Y-%m-%d')
    
    url = f"{FMP_BASE_URL}/historical-chart/{fmp_interval}/{fmp_symbol}"
    params = {
        'from': from_date,
        'to': to_date,
        'apikey': FMP_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if not data or not isinstance(data, list):
            print(f"   ⚠️ FMP returned no index data for {fmp_symbol}")
            return None
        
        df = pd.DataFrame(data)
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], utc=True)
            df = df.set_index('date')
            df = df.sort_index()
        else:
            print(f"   ⚠️ No date column in index response for {fmp_symbol}")
            return None
        
        # Standardize columns
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        
        df = df.rename(columns=column_mapping)
        
        # For indices, volume might not be meaningful, create dummy volume
        if 'Volume' not in df.columns:
            df['Volume'] = 1000000
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols[:4]:  # OHLC
            if col not in df.columns:
                print(f"   ⚠️ Missing {col} column for {fmp_symbol}")
                return None
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        df = df.dropna()
        
        if df.empty:
            print(f"   ⚠️ No valid index data after cleaning for {fmp_symbol}")
            return None
        
        print(f"   ✅ Success with FMP index for {fmp_symbol}! Got {len(df)} rows")
        return df[required_cols]
        
    except Exception as e:
        print(f"   ❌ Error fetching FMP index data for {fmp_symbol}: {e}")
        return None

def fetch_yfinance_data(symbol, period='90d', interval='1h'):
    """Main function to fetch data from Financial Modeling Prep API based on symbol type."""
    print(f"--- Starting FMP fetch for {symbol} ---")
    
    try:
        # Determine asset type and route to appropriate function
        if "=X" in symbol:
            # Forex pair
            return fetch_fmp_forex_data(symbol, period, interval)
        elif "-USD" in symbol:
            # Cryptocurrency
            return fetch_fmp_crypto_data(symbol, period, interval)
        elif symbol.startswith('^'):
            # Index
            return fetch_fmp_index_data(symbol, period, interval)
        else:
            # Stock
            return fetch_fmp_stock_data(symbol, period, interval)
            
    except Exception as e:
        print(f"   ❌ Error in main fetch function for {symbol}: {e}")
        return None

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
        
        # Use newer fillna syntax
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
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
