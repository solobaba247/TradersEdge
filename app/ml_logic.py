# app/ml_logic.py - COMPLETE FMP INTEGRATION

import pandas as pd
import numpy as np
import requests
import warnings
import time
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Financial Modeling Prep API Configuration
FMP_API_KEY = "3V5meXmuiupLM1fyL4vs6GeDB7RFA0LM"
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

class FMPRateLimiter:
    """Rate limiter for FMP API to respect free tier limits"""
    def __init__(self, requests_per_day=200, min_interval=0.6):
        self.requests_per_day = requests_per_day  # Leave buffer from 250 limit
        self.min_interval = min_interval  # 600ms between requests
        self.requests_made = 0
        self.last_reset = datetime.now().date()
        self.last_request_time = 0
    
    def can_make_request(self):
        """Check if we can make another request today"""
        today = datetime.now().date()
        if today > self.last_reset:
            self.requests_made = 0
            self.last_reset = today
        
        return self.requests_made < self.requests_per_day
    
    def wait_if_needed(self):
        """Ensure minimum time between requests and track usage"""
        if not self.can_make_request():
            raise Exception("Daily API limit reached. Try again tomorrow.")
        
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.requests_made += 1
        
        print(f"🌐 API Request #{self.requests_made}/{self.requests_per_day} today")

# Global rate limiter instance
fmp_limiter = FMPRateLimiter()

def convert_to_fmp_symbol(symbol):
    """Convert various symbol formats to FMP API format."""
    print(f"🔄 Converting symbol: {symbol}")
    
    # Forex pairs - FMP expects EURUSD format, not EUR/USD
    if symbol.endswith('=X'):
        base_quote = symbol.replace('=X', '')
        if len(base_quote) == 6:
            converted = base_quote  # EURUSD=X → EURUSD
            print(f"   Forex: {symbol} → {converted}")
            return converted
    
    # Crypto pairs - FMP expects BTCUSD format
    elif '-USD' in symbol:
        converted = symbol.replace('-USD', 'USD')  # BTC-USD → BTCUSD
        print(f"   Crypto: {symbol} → {converted}")
        return converted
    
    # Stock indices - FMP has specific formats
    elif symbol.startswith('^'):
        index_map = {
            '^GSPC': 'SPX',    # S&P 500
            '^DJI': 'DJI',     # Dow Jones
            '^IXIC': 'IXIC',   # NASDAQ
            '^RUT': 'RUT',     # Russell 2000
            '^VIX': 'VIX',     # Volatility Index
            '^TNX': 'TNX'      # 10-Year Treasury
        }
        converted = index_map.get(symbol, symbol[1:])
        print(f"   Index: {symbol} → {converted}")
        return converted
    
    # Regular stocks - no change needed
    print(f"   Stock: {symbol} → {symbol} (no change)")
    return symbol

def calculate_rsi_manual(prices, window=14):
    """Calculate RSI manually without pandas-ta."""
    try:
        prices = pd.Series(prices)
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        
        # Avoid division by zero
        loss = loss.replace(0, 0.0001)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    except Exception as e:
        print(f"   ⚠️ RSI calculation error: {e}")
        return pd.Series([50] * len(prices))

def calculate_ema_manual(prices, window=200):
    """Calculate EMA manually without pandas-ta."""
    try:
        prices = pd.Series(prices)
        ema = prices.ewm(span=window, min_periods=1).mean()
        return ema
    except Exception as e:
        print(f"   ⚠️ EMA calculation error: {e}")
        return pd.Series(prices)

def calculate_atr_manual(high, low, close, window=14):
    """Calculate ATR manually without pandas-ta."""
    try:
        high = pd.Series(high)
        low = pd.Series(low) 
        close = pd.Series(close)
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window, min_periods=1).mean()
        return atr.fillna(1.0)
    except Exception as e:
        print(f"   ⚠️ ATR calculation error: {e}")
        return pd.Series([1.0] * len(high))

def fetch_fmp_data(symbol, period='90d', interval='1h'):
    """
    Fetch data from Financial Modeling Prep API - COMPLETE IMPLEMENTATION
    """
    print(f"🔍 === FMP FETCH for {symbol} ===")
    print(f"🔍 Request: period={period}, interval={interval}")
    
    try:
        # Rate limiting
        fmp_limiter.wait_if_needed()
        
        # Convert symbol format
        api_symbol = convert_to_fmp_symbol(symbol)
        
        # Map intervals to FMP format
        interval_map = {
            '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min', 
            '1h': '1hour', '4h': '4hour', '1d': '1day', '1wk': '1week'
        }
        fmp_interval = interval_map.get(interval, '1hour')
        print(f"🔍 Interval mapping: {interval} → {fmp_interval}")
        
        # Calculate date range with buffer for more data
        period_days_map = {
            '1d': 7, '5d': 14, '1mo': 60, '3mo': 120, 
            '6mo': 240, '1y': 365, '2y': 730, '90d': 180
        }
        days = period_days_map.get(period, 180)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')
        
        print(f"🔍 Date range: {from_date} to {to_date} ({days} days)")
        
        # Build API request
        url = f"{FMP_BASE_URL}/historical-chart/{fmp_interval}/{api_symbol}"
        params = {
            'from': from_date, 
            'to': to_date, 
            'apikey': FMP_API_KEY
        }
        
        print(f"🔍 API URL: {url}")
        print(f"🔍 Params: from={from_date}, to={to_date}")
        
        # Make HTTP request
        print(f"🌐 Making FMP API request...")
        response = requests.get(url, params=params, timeout=30)
        print(f"🔍 Response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")
            return None
        
        # Parse JSON response
        try:
            data = response.json()
        except ValueError as e:
            print(f"❌ JSON parsing failed: {e}")
            print(f"❌ Response preview: {response.text[:500]}...")
            return None
        
        print(f"🔍 JSON parsed successfully")
        
        # Validate response
        if isinstance(data, dict):
            if 'error' in data or 'Error Message' in data:
                print(f"❌ API Error: {data}")
                return None
        
        if not data or not isinstance(data, list):
            print(f"❌ Invalid data format: expected list, got {type(data)}")
            return None
        
        print(f"🔍 Raw data points from FMP: {len(data)}")
        
        # Process data points
        records = []
        for i, item in enumerate(data):
            try:
                if not isinstance(item, dict):
                    continue
                
                # Validate required fields
                required_fields = ['date', 'open', 'high', 'low', 'close']
                if not all(field in item for field in required_fields):
                    continue
                
                # Convert to standardized format
                record = {
                    'date': item.get('date'),
                    'Open': float(item.get('open', 0)),
                    'High': float(item.get('high', 0)), 
                    'Low': float(item.get('low', 0)),
                    'Close': float(item.get('close', 0)),
                    'Volume': float(item.get('volume', 1000000))
                }
                
                # Validate prices are positive
                if any(val <= 0 for val in [record['Open'], record['High'], record['Low'], record['Close']]):
                    continue
                
                records.append(record)
                
            except (ValueError, TypeError) as e:
                continue
        
        print(f"🔍 Valid records after processing: {len(records)}")
        
        if not records:
            print(f"❌ No valid records after processing")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        # Process dates with UTC timezone
        df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
        df = df.dropna(subset=['date'])
        
        if df.empty:
            print(f"❌ No data after date processing")
            return None
        
        # Set index and sort
        df = df.set_index('date')
        df = df.sort_index()
        
        # Final cleaning
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        if df.empty:
            print(f"❌ No data after final cleaning")
            return None
        
        result_df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        print(f"✅ SUCCESS! Final result: {len(result_df)} rows")
        print(f"📈 Price range: {result_df['Close'].min():.4f} - {result_df['Close'].max():.4f}")
        print(f"📊 Latest price: {result_df['Close'].iloc[-1]:.4f}")
        print(f"📅 Date range: {result_df.index[0]} to {result_df.index[-1]}")
        
        return result_df
        
    except requests.exceptions.Timeout:
        print(f"❌ Request timeout after 30 seconds")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ HTTP request failed: {type(e).__name__}: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")
        return None

def create_features_for_prediction(data, feature_columns_list):
    """Create features WITHOUT pandas-ta to avoid recursion."""
    if data is None or data.empty:
        print("   ❌ No input data for features")
        return pd.DataFrame()

    print(f"   🔧 Creating features from {len(data)} rows of data")
    
    try:
        df = data.copy()
        
        # Manual technical indicators
        print(f"   📈 Calculating RSI manually...")
        df['RSI_14'] = calculate_rsi_manual(df['Close'], 14)
        df['rsi_14'] = df['RSI_14']
        
        print(f"   📈 Calculating EMA manually...")
        df['EMA_200'] = calculate_ema_manual(df['Close'], 200)
        df['price_vs_ema200'] = df['Close'] / df['EMA_200']
        
        print(f"   📈 Calculating ATR manually...")
        df['ATR_14'] = calculate_atr_manual(df['High'], df['Low'], df['Close'], 14)
        
        # Simple volume ratio
        print(f"   📊 Calculating volume features...")
        volume_ma = df['Volume'].rolling(20, min_periods=1).mean()
        df['volume_ratio'] = df['Volume'] / volume_ma
        
        # Time features
        print(f"   🕐 Adding time features...")
        df['hour_of_day'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # Static features (from your original model)
        print(f"   🎯 Adding static features...")
        df['channel_slope'] = 0.0
        df['channel_width_atr'] = 1.0
        df['bars_outside_zone'] = 0
        df['breakout_distance_norm'] = 0.0
        df['breakout_candle_body_ratio'] = 0.5
        df['risk_reward_ratio'] = 2.0
        df['stop_loss_in_atrs'] = 1.5
        df['entry_pos_in_channel_norm'] = 0.5
        df['breakout_strength'] = 0.0
        df['channel_efficiency'] = 0.0
        df['high_risk_trade'] = 0
        df['trade_type_encoded'] = 0
        
        # Historical features (dummy values)
        for i in range(24):
            df[f'hist_close_channel_dev_t_minus_{i}'] = 0.0
        
        # Interaction features
        df['volume_rsi_interaction'] = df['volume_ratio'] * df['rsi_14']
        
        # Boolean features
        df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
        df['price_above_ema'] = (df['Close'] > df['EMA_200']).astype(int)
        
        # Ensure all required columns exist
        for col in feature_columns_list:
            if col not in df.columns:
                df[col] = 0.0
        
        # Fill NaN values using forward fill, then backward fill, then zero
        print(f"   🧹 Cleaning NaN values...")
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Select required columns
        required_cols = feature_columns_list + ['Close']
        available_cols = [col for col in required_cols if col in df.columns]
        result_df = df[available_cols]
        
        print(f"   ✅ Features created: {len(result_df)} rows, {len(result_df.columns)} cols")
        return result_df
        
    except Exception as e:
        print(f"   ❌ Feature creation error: {type(e).__name__}: {e}")
        return pd.DataFrame()

def get_model_prediction(data, model, scaler, feature_columns):
    """Generate prediction with minimal processing."""
    if data is None or data.empty:
        return {"error": "No input data for prediction"}
    
    print(f"   🤖 Generating prediction from {len(data)} data points...")
    
    try:
        features_df = create_features_for_prediction(data, feature_columns)
        if features_df.empty:
            return {"error": "Could not create features"}

        latest_features = features_df.iloc[-1].copy()
        last_price = latest_features['Close']

        # BUY scenario
        buy_features = latest_features.copy()
        buy_features['trade_type_encoded'] = 0
        buy_df = pd.DataFrame([buy_features])[feature_columns]
        buy_scaled = scaler.transform(buy_df)
        buy_prob = model.predict_proba(buy_scaled)[0][1]

        # SELL scenario  
        sell_features = latest_features.copy()
        sell_features['trade_type_encoded'] = 1
        sell_df = pd.DataFrame([sell_features])[feature_columns]
        sell_scaled = scaler.transform(sell_df)
        sell_prob = model.predict_proba(sell_scaled)[0][1]

        # Determine signal
        confidence_threshold = 0.55
        if buy_prob > sell_prob and buy_prob > confidence_threshold:
            signal_type = "BUY"
            confidence = buy_prob
        elif sell_prob > buy_prob and sell_prob > confidence_threshold:
            signal_type = "SELL" 
            confidence = sell_prob
        else:
            signal_type = "HOLD"
            confidence = 0.5

        print(f"   🎯 Signal: {signal_type} ({confidence:.1%})")

        return {
            "signal": signal_type,
            "confidence": confidence,
            "latest_price": last_price,
            "buy_prob": buy_prob,
            "sell_prob": sell_prob,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        print(f"   ❌ Prediction error: {type(e).__name__}: {e}")
        return {"error": f"Prediction failed: {str(e)}"}

# Backward compatibility aliases
def fetch_data_via_proxies(symbol, period='90d', interval='1h'):
    """Backward compatibility - redirects to FMP function"""
    return fetch_fmp_data(symbol, period, interval)

def fetch_yfinance_data(symbol, period='90d', interval='1h'):
    """Backward compatibility - redirects to FMP function"""
    return fetch_fmp_data(symbol, period, interval)
