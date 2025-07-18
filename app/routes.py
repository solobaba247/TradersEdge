# app/routes.py

from flask import current_app, request, jsonify
import pandas as pd
import concurrent.futures
import time
from .ml_logic import get_model_prediction, fetch_yfinance_data
from .helpers import calculate_stop_loss_value, get_latest_price, get_technical_indicators

@current_app.route('/')
def api_root():
    """Root endpoint for the API, providing a welcome message and status."""
    return jsonify({
        "message": "Welcome to the ML Trading Signal API",
        "status": "ok",
        "documentation": "Please see the frontend client for usage."
    })

@current_app.route('/api/health')
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "healthy", "timestamp": time.time()}), 200

@current_app.route('/api/check_model_status')
def check_model_status():
    """Health check endpoint for model loading status."""
    models_loaded = current_app.config.get('MODELS_LOADED', False)
    
    if models_loaded and current_app.model is not None and current_app.scaler is not None:
        return jsonify({
            "status": "ok", 
            "models_loaded": True,
            "message": "Models are loaded and ready."
        }), 200
    else:
        return jsonify({
            "status": "error", 
            "models_loaded": False,
            "message": "Models failed to load or are not available."
        }), 503

@current_app.route('/api/generate_signal')
def generate_signal_route():
    """Generate trading signal for a single asset."""
    symbol = request.args.get('symbol')
    timeframe = request.args.get('timeframe', '1h')
    
    if not symbol:
        return jsonify({"error": "Symbol parameter is required."}), 400
    
    if not current_app.config.get('MODELS_LOADED', False):
        return jsonify({"error": "Models are not loaded. Please wait for initialization."}), 503
    
    try:
        data = fetch_yfinance_data(symbol, period='90d', interval=timeframe)
        if data is None or len(data) < 50:
            return jsonify({"error": f"Insufficient data for {symbol}. Need at least 50 data points."}), 400
        
        prediction = get_model_prediction(
            data, 
            current_app.model, 
            current_app.scaler, 
            current_app.feature_columns
        )
        
        if "error" in prediction:
            return jsonify(prediction), 500
        
        latest_price = prediction['latest_price']
        signal = prediction['signal']
        confidence = prediction['confidence']
        
        if signal == "BUY":
            entry_price, stop_loss, exit_price = latest_price, latest_price * 0.99, latest_price * 1.02
        elif signal == "SELL":
            entry_price, stop_loss, exit_price = latest_price, latest_price * 1.01, latest_price * 0.98
        else: # HOLD
            entry_price, stop_loss, exit_price = latest_price, latest_price, latest_price
        
        response = {
            "symbol": symbol, "signal": signal, "confidence": f"{confidence:.2%}",
            "entry_price": f"{entry_price:.5f}", "exit_price": f"{exit_price:.5f}",
            "stop_loss": f"{stop_loss:.5f}",
            "stop_loss_value": calculate_stop_loss_value(symbol, entry_price, stop_loss),
            "timestamp": prediction['timestamp']
        }
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": f"Failed to generate signal: {str(e)}"}), 500

def get_prediction_for_symbol_sync(symbol, timeframe, model, scaler, feature_columns):
    """Synchronous version of prediction function for concurrent execution."""
    try:
        # Shorter timeout for individual symbol processing
        start_time = time.time()
        
        data = fetch_yfinance_data(symbol, period='90d', interval=timeframe)
        if data is None or len(data) < 50: 
            return None
        
        # Check if we're running out of time
        if time.time() - start_time > 25:  # 25 second timeout per symbol
            print(f"⏰ Symbol {symbol} taking too long, skipping...")
            return None
        
        prediction = get_model_prediction(data, model, scaler, feature_columns)
        if "error" in prediction or prediction['signal'] == "HOLD": 
            return None
        
        signal, confidence, latest_price = prediction['signal'], prediction['confidence'], prediction['latest_price']
        
        if signal == "BUY":
            entry_price, stop_loss, exit_price = latest_price, latest_price * 0.99, latest_price * 1.02
        else: # SELL
            entry_price, stop_loss, exit_price = latest_price, latest_price * 1.01, latest_price * 0.98
        
        return {
            "symbol": symbol, "signal": signal, "confidence": f"{confidence:.2%}",
            "entry_price": f"{entry_price:.5f}", "exit_price": f"{exit_price:.5f}",
            "stop_loss": f"{stop_loss:.5f}",
            "stop_loss_value": calculate_stop_loss_value(symbol, entry_price, stop_loss),
            "timestamp": prediction['timestamp']
        }
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        return None

@current_app.route('/api/scan_market', methods=['POST'])
def scan_market_route():
    """Scan multiple assets concurrently for trading signals."""
    try:
        data = request.get_json()
        asset_type = data.get('asset_type')
        timeframe = data.get('timeframe', '1h')
        
        asset_classes = current_app.config.get('ASSET_CLASSES', {})
        if not asset_type or asset_type not in asset_classes:
            return jsonify({"error": "Invalid asset type"}), 400
        
        if not current_app.config.get('MODELS_LOADED', False):
            return jsonify({"error": "Models are not loaded. Please wait for initialization."}), 503
        
        symbols_to_scan = asset_classes[asset_type]
        
        # Limit symbols to prevent timeouts on free tier
        max_symbols = 15  # Reduced from full list
        if len(symbols_to_scan) > max_symbols:
            symbols_to_scan = symbols_to_scan[:max_symbols]
            print(f"⚠️ Limiting scan to first {max_symbols} symbols to prevent timeout")
        
        results = []
        # Use only 2 workers to reduce resource usage
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_to_symbol = {
                executor.submit(get_prediction_for_symbol_sync, symbol, timeframe, current_app.model, current_app.scaler, current_app.feature_columns): symbol for symbol in symbols_to_scan
            }
            
            completed_count = 0
            for future in concurrent.futures.as_completed(future_to_symbol, timeout=90):  # 90 second total timeout
                symbol_name = future_to_symbol[future]
                try:
                    result = future.result(timeout=30)  # 30 second timeout per symbol
                    if result is not None: 
                        results.append(result)
                    completed_count += 1
                    
                    # Progress logging
                    if completed_count % 5 == 0:
                        print(f"📊 Processed {completed_count}/{len(symbols_to_scan)} symbols")
                        
                except concurrent.futures.TimeoutError:
                    print(f"⏰ Timeout processing {symbol_name} after 30 seconds. Skipping.")
                except Exception as e:
                    print(f"Error processing {symbol_name}: {e}")
                    continue
        
        print(f"✅ Market scan completed: {len(results)} signals found from {len(symbols_to_scan)} symbols")
        return jsonify(results)
        
    except Exception as e:
        print(f"🔥 Market scan failed: {e}")
        return jsonify({"error": f"Failed to scan market: {str(e)}"}), 500

@current_app.route('/api/latest_price')
def latest_price_route():
    return get_latest_price(request.args.get('symbol'))

@current_app.route('/api/technical_indicators')
def technical_indicators_route():
    symbol, timeframe = request.args.get('symbol'), request.args.get('timeframe', '1h')
    return get_technical_indicators(symbol, timeframe)
