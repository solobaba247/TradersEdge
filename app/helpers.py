# app/helpers.py - UPDATED FOR FMP INTEGRATION

import pandas as pd
import pandas_ta as ta
from flask import jsonify
from .ml_logic import fetch_fmp_data

def calculate_stop_loss_value(symbol, entry_price, sl_price):
    """Calculate stop loss value with currency formatting."""
    price_diff = abs(entry_price - sl_price)
    currency_map = {'USD': '$', 'JPY': '¥', 'GBP': '£', 'EUR': '€', 'CHF': 'Fr.'}
    
    try:
        if "=X" in symbol:
            # Forex pairs
            value = price_diff * 1000  # Standard lot size
            quote_currency = symbol[3:6]
            currency_symbol = currency_map.get(quote_currency, quote_currency + ' ')
            return f"({currency_symbol}{value:,.2f})"
        elif "-USD" in symbol:
            # Crypto pairs
            value = price_diff * 0.01  # Small position size for crypto
            return f"(~${value:,.2f})"
        else:
            # Stocks and indices
            value = price_diff * 1  # 1 share/unit
            return f"(~${value:,.2f})"
    except Exception as e:
        print(f"Error calculating stop loss value: {e}")
        return ""

def get_latest_price(symbol):
    """Get the latest price for a symbol using FMP API."""
    if not symbol:
        return jsonify({"error": "Symbol parameter is required."}), 400
    
    try:
        print(f"🔍 Fetching latest price for {symbol}")
        
        # Fetch recent data (1 day with 1 minute intervals for most recent price)
        data = fetch_fmp_data(symbol, period='1d', interval='1m')
        
        if data is None or data.empty:
            print(f"❌ No data returned for {symbol}")
            return jsonify({"error": f"Could not fetch latest price for {symbol}."}), 500
        
        latest_price = data['Close'].iloc[-1]
        latest_timestamp = data.index[-1].strftime('%Y-%m-%d %H:%M:%S UTC')
        
        print(f"✅ Latest price for {symbol}: ${latest_price:.5f}")
        
        return jsonify({
            "symbol": symbol,
            "price": latest_price,
            "timestamp": latest_timestamp,
            "data_source": "Financial Modeling Prep API"
        })
        
    except Exception as e:
        print(f"❌ Error fetching latest price for {symbol}: {e}")
        return jsonify({"error": f"Failed to fetch latest price for {symbol}: {str(e)}"}), 500

def calculate_rsi_simple(prices, window=14):
    """Simple RSI calculation without pandas_ta."""
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
    except:
        return pd.Series([50] * len(prices))

def calc_simple_macd(prices, fast=12, slow=26, signal=9):
    """Simple MACD calculation without pandas_ta."""
    try:
        prices = pd.Series(prices)
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line, signal_line
    except:
        return pd.Series([0] * len(prices)), pd.Series([0] * len(prices))

def calc_bollinger_bands(prices, window=20, std_dev=2):
    """Simple Bollinger Bands calculation without pandas_ta."""
    try:
        prices = pd.Series(prices)
        sma = prices.rolling(window=window, min_periods=1).mean()
        std = prices.rolling(window=window, min_periods=1).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    except:
        return pd.Series(prices), pd.Series(prices), pd.Series(prices)

def get_technical_indicators(symbol, timeframe):
    """Calculate and return technical indicators for a symbol using FMP data."""
    if not symbol:
        return jsonify({"error": "Symbol parameter is required."}), 400
    
    try:
        print(f"🔍 Calculating technical indicators for {symbol} on {timeframe}")
        
        # Fetch sufficient data for technical indicators
        data = fetch_fmp_data(symbol, period='90d', interval=timeframe)
        
        if data is None or len(data) < 20:
            print(f"❌ Insufficient data for {symbol}: {len(data) if data is not None else 0} points")
            return jsonify({
                "error": f"Could not fetch sufficient historical data for {symbol}. Need at least 20 data points."
            }), 500

        print(f"📊 Calculating indicators from {len(data)} data points")
        
        # Calculate technical indicators using simple methods
        results = {}
        
        # RSI Analysis
        try:
            rsi_values = calculate_rsi_simple(data['Close'], 14)
            rsi_val = rsi_values.iloc[-1]
            
            summary = f"{rsi_val:.2f}"
            if rsi_val > 70:
                summary += " (Overbought)"
            elif rsi_val < 30:
                summary += " (Oversold)"
            else:
                summary += " (Neutral)"
            results['RSI (14)'] = summary
            print(f"✅ RSI calculated: {rsi_val:.2f}")
        except Exception as e:
            print(f"⚠️ RSI calculation failed: {e}")
            results['RSI (14)'] = "Calculation failed"

        # MACD Analysis
        try:
            macd_line, macd_signal = calc_simple_macd(data['Close'], 12, 26, 9)
            macd_val = macd_line.iloc[-1]
            signal_val = macd_signal.iloc[-1]
            
            summary = f"MACD: {macd_val:.5f}, Signal: {signal_val:.5f}"
            if macd_val > signal_val:
                summary += " (Bullish)"
            else:
                summary += " (Bearish)"
            results['MACD (12, 26, 9)'] = summary
            print(f"✅ MACD calculated: {macd_val:.5f}")
        except Exception as e:
            print(f"⚠️ MACD calculation failed: {e}")
            results['MACD (12, 26, 9)'] = "Calculation failed"

        # Bollinger Bands Analysis
        try:
            bb_upper, bb_middle, bb_lower = calc_bollinger_bands(data['Close'], 20, 2)
            upper_val = bb_upper.iloc[-1]
            middle_val = bb_middle.iloc[-1]
            lower_val = bb_lower.iloc[-1]
            current_close = data['Close'].iloc[-1]
            
            summary = f"Upper: {upper_val:.4f}, Middle: {middle_val:.4f}, Lower: {lower_val:.4f}"
            if current_close > upper_val:
                summary += " (Price Above Upper Band - Strong Uptrend)"
            elif current_close < lower_val:
                summary += " (Price Below Lower Band - Strong Downtrend)"
            elif current_close > middle_val:
                summary += " (Price Above Middle - Bullish Bias)"
            else:
                summary += " (Price Below Middle - Bearish Bias)"
            results['Bollinger Bands (20, 2)'] = summary
            print(f"✅ Bollinger Bands calculated")
        except Exception as e:
            print(f"⚠️ Bollinger Bands calculation failed: {e}")
            results['Bollinger Bands (20, 2)'] = "Calculation failed"

        # Current Price and additional info
        current_close = data['Close'].iloc[-1]
        results['Latest Close'] = f"{current_close:.5f}"
        
        # Calculate price change
        if len(data) > 1:
            prev_close = data['Close'].iloc[-2]
            price_change = current_close - prev_close
            price_change_pct = (price_change / prev_close) * 100
            results['Price Change'] = f"{price_change:+.5f} ({price_change_pct:+.2f}%)"
        
        # Volume info
        current_volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].tail(20).mean()
        volume_ratio = current_volume / avg_volume
        results['Volume Ratio'] = f"{volume_ratio:.2f}x (vs 20-day avg)"
        
        # Data source and timestamp
        results['Data Source'] = "Financial Modeling Prep API"
        results['Last Updated'] = data.index[-1].strftime('%Y-%m-%d %H:%M:%S UTC')
        results['Data Points Used'] = len(data)
        
        print(f"✅ Technical indicators calculated successfully for {symbol}")
        return jsonify(results)
        
    except Exception as e:
        print(f"❌ Error calculating technical indicators for {symbol}: {e}")
        return jsonify({"error": f"Failed to calculate technical indicators for {symbol}: {str(e)}"}), 500
