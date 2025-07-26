# app/helpers.py - Updated for Financial Modeling Prep API

import pandas as pd
import pandas_ta as ta
from flask import jsonify
from .ml_logic import fetch_data_via_proxies

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
        # Fetch recent data (1 day with 1 minute intervals for most recent price)
        data = fetch_data_via_proxies(symbol, period='1d', interval='1m')
        
        if data is None or data.empty:
            return jsonify({"error": f"Could not fetch latest price for {symbol}."}), 500
        
        latest_price = data['Close'].iloc[-1]
        latest_timestamp = data.index[-1].strftime('%Y-%m-%d %H:%M:%S UTC')
        
        return jsonify({
            "symbol": symbol,
            "price": latest_price,
            "timestamp": latest_timestamp
        })
        
    except Exception as e:
        print(f"Error fetching latest price for {symbol}: {e}")
        return jsonify({"error": f"Failed to fetch latest price for {symbol}: {str(e)}"}), 500

def get_technical_indicators(symbol, timeframe):
    """Calculate and return technical indicators for a symbol."""
    if not symbol:
        return jsonify({"error": "Symbol parameter is required."}), 400
    
    try:
        # Fetch sufficient data for technical indicators
        data = fetch_data_via_proxies(symbol, period='90d', interval=timeframe)
        
        if data is None or len(data) < 20:
            return jsonify({"error": f"Could not fetch sufficient historical data for {symbol}. Need at least 20 data points."}), 500

        # Calculate technical indicators using pandas_ta
        data.ta.rsi(length=14, append=True)
        data.ta.macd(fast=12, slow=26, signal=9, append=True)
        data.ta.bbands(length=20, std=2, append=True)
        
        # Get the latest values
        latest = data.iloc[-1]
        results = {}

        # RSI Analysis
        rsi_val = latest.get('RSI_14')
        if pd.notna(rsi_val):
            summary = f"{rsi_val:.2f}"
            if rsi_val > 70:
                summary += " (Overbought)"
            elif rsi_val < 30:
                summary += " (Oversold)"
            else:
                summary += " (Neutral)"
            results['RSI (14)'] = summary

        # MACD Analysis
        macd_line = latest.get('MACD_12_26_9')
        macd_signal = latest.get('MACDs_12_26_9')
        
        if pd.notna(macd_line) and pd.notna(macd_signal):
            summary = f"MACD: {macd_line:.5f}, Signal: {macd_signal:.5f}"
            if macd_line > macd_signal:
                summary += " (Bullish)"
            else:
                summary += " (Bearish)"
            results['MACD (12, 26, 9)'] = summary

        # Bollinger Bands Analysis
        bb_upper = latest.get('BBU_20_2.0')
        bb_middle = latest.get('BBM_20_2.0') 
        bb_lower = latest.get('BBL_20_2.0')
        current_close = latest.get('Close')
        
        if all(pd.notna(val) for val in [bb_upper, bb_middle, bb_lower, current_close]):
            summary = f"Upper: {bb_upper:.4f}, Middle: {bb_middle:.4f}, Lower: {bb_lower:.4f}"
            if current_close > bb_upper:
                summary += " (Price Above Upper Band - Strong Uptrend)"
            elif current_close < bb_lower:
                summary += " (Price Below Lower Band - Strong Downtrend)"
            elif current_close > bb_middle:
                summary += " (Price Above Middle - Bullish Bias)"
            else:
                summary += " (Price Below Middle - Bearish Bias)"
            results['Bollinger Bands (20, 2)'] = summary

        # Current Price
        results['Latest Close'] = f"{current_close:.5f}"
        
        # Additional market information
        results['Data Source'] = "Financial Modeling Prep API"
        results['Last Updated'] = data.index[-1].strftime('%Y-%m-%d %H:%M:%S UTC')
        
        return jsonify(results)
        
    except Exception as e:
        print(f"Error calculating technical indicators for {symbol}: {e}")
        return jsonify({"error": f"Failed to calculate technical indicators for {symbol}: {str(e)}"}), 500
