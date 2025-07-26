# app/ml_logic.py - DIAGNOSTIC VERSION TO DEBUG FMP DATA ISSUES

import pandas as pd
import numpy as np
import requests
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Financial Modeling Prep API Configuration
FMP_API_KEY = "3V5meXmuiupLM1fyL4vs6GeDB7RFA0LM"
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

def calculate_rsi_manual(prices, window=14):
    """Calculate RSI manually without pandas-ta."""
    try:
        prices = pd.Series(prices)
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    except:
        return pd.Series([50] * len(prices))

def calculate_ema_manual(prices, window=200):
    """Calculate EMA manually without pandas-ta."""
    try:
        prices = pd.Series(prices)
        return prices.ewm(span=window).mean()
    except:
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
        atr = tr.rolling(window=window).mean()
        return atr.fillna(1.0)
    except:
        return pd.Series([1.0] * len(high))

def fetch_yfinance_data(symbol, period='90d', interval='1h'):
    """
    DIAGNOSTIC VERSION - Enhanced logging to debug FMP data issues
    """
    print(f"🔍 === DIAGNOSTIC FMP FETCH for {symbol} ===")
    print(f"🔍 Original request: period={period}, interval={interval}")
    
    # Convert symbol format
    api_symbol = symbol
    if symbol.endswith('=X'):
        base_quote = symbol.replace('=X', '')
        if len(base_quote) == 6:
            api_symbol = f"{base_quote[:3]}/{base_quote[3:]}"
    elif '-USD' in symbol:
        api_symbol = symbol.replace('-USD', 'USD')
    
    print(f"🔍 Symbol conversion: {symbol} -> {api_symbol}")
    
    # Map intervals with more options
    interval_map = {
        '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min', 
        '1h': '1hour', '4h': '4hour', '1d': '1day', '1wk': '1week'
    }
    fmp_interval = interval_map.get(interval, '1hour')
    print(f"🔍 Interval mapping: {interval} -> {fmp_interval}")
    
    # Try different period strategies for more data
    if period == '90d':
        # For 90 days, try getting more data by extending the period
        days = 180  # Get 6 months of data instead
        print(f"🔍 Extended period: 90d -> 180 days for more data points")
    else:
        period_map = {'1d': 7, '5d': 14, '1mo': 60, '3mo': 120, '6mo': 240, 
                      '1y': 365, '2y': 730}
        days = period_map.get(period, 180)
        print(f"🔍 Period mapping: {period} -> {days} days")
    
    # Date range calculation
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    from_date = start_date.strftime('%Y-%m-%d')
    to_date = end_date.strftime('%Y-%m-%d')
    
    print(f"🔍 Date range: {from_date} to {to_date} ({days} days)")
    
    # Build API request
    url = f"{FMP_BASE_URL}/historical-chart/{fmp_interval}/{api_symbol}"
    params = {'from': from_date, 'to': to_date, 'apikey': FMP_API_KEY}
    
    print(f"🔍 Full URL: {url}")
    print(f"🔍 Params: {params}")
    
    try:
        print(f"🔍 Making HTTP request...")
        response = requests.get(url, params=params, timeout=30)
        print(f"🔍 Response status: {response.status_code}")
        print(f"🔍 Response headers: {dict(response.headers)}")
        
        if response.status_code != 200:
            print(f"❌ HTTP Error {response.status_code}")
            print(f"❌ Response text: {response.text}")
            return None
        
        # Check response size
        response_text = response.text
        print(f"🔍 Response size: {len(response_text)} characters")
        
        try:
            data = response.json()
        except ValueError as e:
            print(f"❌ JSON parsing failed: {e}")
            print(f"❌ Response preview: {response_text[:500]}...")
            return None
        
        print(f"🔍 JSON parsed successfully")
        print(f"🔍 Response type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"🔍 Response is dict with keys: {list(data.keys())}")
            # Check if it's an error response
            if 'error' in data or 'Error Message' in data:
                print(f"❌ API Error: {data}")
                return None
        
        if not data:
            print(f"❌ Empty response from FMP API")
            return None
        
        if not isinstance(data, list):
            print(f"❌ Expected list, got {type(data)}")
            print(f"❌ Data content: {str(data)[:200]}...")
            return None
        
        print(f"🔍 Raw data points from FMP: {len(data)}")
        
        # Show first few data points for debugging
        if len(data) > 0:
            print(f"🔍 First data point: {data[0]}")
            if len(data) > 1:
                print(f"🔍 Last data point: {data[-1]}")
        
        # Process data with detailed logging
        print(f"🔍 Processing data points...")
        records = []
        for i, item in enumerate(data):
            try:
                if not isinstance(item, dict):
                    print(f"⚠️ Item {i} is not dict: {type(item)}")
                    continue
                
                required_fields = ['date', 'open', 'high', 'low', 'close']
                missing_fields = [field for field in required_fields if field not in item]
                if missing_fields:
                    print(f"⚠️ Item {i} missing fields: {missing_fields}")
                    continue
                
                record = {
                    'date': item.get('date'),
                    'Open': float(item.get('open', 0)),
                    'High': float(item.get('high', 0)), 
                    'Low': float(item.get('low', 0)),
                    'Close': float(item.get('close', 0)),
                    'Volume': float(item.get('volume', 1000000))
                }
                
                # Validate numeric values
                if any(val <= 0 for val in [record['Open'], record['High'], record['Low'], record['Close']]):
                    print(f"⚠️ Item {i} has invalid prices: {record}")
                    continue
                
                records.append(record)
                
            except (ValueError, TypeError) as e:
                print(f"⚠️ Error processing item {i}: {e}")
                continue
        
        print(f"🔍 Valid records after processing: {len(records)}")
        
        if not records:
            print(f"❌ No valid records after processing")
            return None
        
        # Create DataFrame
        print(f"🔍 Creating DataFrame...")
        df = pd.DataFrame(records)
        print(f"🔍 DataFrame shape before date processing: {df.shape}")
        
        # Process dates
        print(f"🔍 Processing dates...")
        df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
        
        # Check for NaT (Not a Time) values
        nat_count = df['date'].isna().sum()
        if nat_count > 0:
            print(f"⚠️ Found {nat_count} invalid dates, removing them")
            df = df.dropna(subset=['date'])
        
        if df.empty:
            print(f"❌ No data after date processing")
            return None
        
        df = df.set_index('date')
        df = df.sort_index()
        
        print(f"🔍 DataFrame shape after date processing: {df.shape}")
        print(f"🔍 Date range in data: {df.index[0]} to {df.index[-1]}")
        
        # Final cleaning
        print(f"🔍 Final data cleaning...")
        initial_count = len(df)
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        final_count = len(df)
        
        if final_count < initial_count:
            print(f"🔍 Removed {initial_count - final_count} rows with NaN prices")
        
        if df.empty:
            print(f"❌ No data after final cleaning")
            return None
        
        result_df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        print(f"✅ SUCCESS! Final result: {len(result_df)} rows")
        print(f"📈 Price range: {result_df['Close'].min():.4f} - {result_df['Close'].max():.4f}")
        print(f"📊 Latest price: {result_df['Close'].iloc[-1]:.4f}")
        print(f"📅 Final date range: {result_df.index[0]} to {result_df.index[-1]}")
        
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
        
        # Fill NaN values using simple methods to avoid recursion
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

# Backward compatibility
def fetch_data_via_proxies(symbol, period='90d', interval='1h'):
    return fetch_yfinance_data(symbol, period, interval)
