#!/usr/bin/env python3
# test_fmp_integration.py - Test Financial Modeling Prep API Integration

import sys
import os
import pandas as pd
from datetime import datetime

# Add the current directory to Python path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_fmp_integration():
    """Test the Financial Modeling Prep API integration."""
    
    print("=" * 70)
    print("FINANCIAL MODELING PREP API INTEGRATION TEST")
    print("=" * 70)
    
    try:
        # Import our updated ml_logic module
        from app.ml_logic import (
            fetch_yfinance_data,
            fetch_fmp_stock_data,
            fetch_fmp_forex_data,
            fetch_fmp_crypto_data,
            fetch_fmp_index_data
        )
        print("✅ Successfully imported FMP modules")
        
    except ImportError as e:
        print(f"❌ Failed to import modules: {e}")
        return False
    
    # Test symbols for each asset class
    test_symbols = {
        "Stock": "AAPL",
        "Forex": "EURUSD=X", 
        "Crypto": "BTC-USD",
        "Index": "^GSPC"
    }
    
    print(f"\n🧪 Testing {len(test_symbols)} asset types...")
    
    results = {}
    
    for asset_type, symbol in test_symbols.items():
        print(f"\n--- Testing {asset_type}: {symbol} ---")
        
        try:
            # Test with different timeframes
            for timeframe in ['1h', '1d']:
                print(f"  Testing {timeframe} timeframe...")
                
                data = fetch_yfinance_data(symbol, period='5d', interval=timeframe)
                
                if data is not None and not data.empty:
                    print(f"    ✅ {timeframe}: Got {len(data)} rows")
                    print(f"    📊 Columns: {list(data.columns)}")
                    print(f"    📈 Price range: ${data['Close'].min():.4f} - ${data['Close'].max():.4f}")
                    print(f"    🕐 Date range: {data.index[0]} to {data.index[-1]}")
                    
                    # Store result
                    if asset_type not in results:
                        results[asset_type] = {}
                    results[asset_type][timeframe] = {
                        'success': True,
                        'rows': len(data),
                        'latest_price': data['Close'].iloc[-1]
                    }
                else:
                    print(f"    ❌ {timeframe}: No data returned")
                    if asset_type not in results:
                        results[asset_type] = {}
                    results[asset_type][timeframe] = {
                        'success': False,
                        'error': 'No data returned'
                    }
                    
        except Exception as e:
            print(f"    ❌ Error testing {symbol}: {e}")
            results[asset_type] = {'error': str(e)}
    
    # Summary Report
    print("\n" + "=" * 70)
    print("TEST SUMMARY REPORT")
    print("=" * 70)
    
    total_tests = 0
    successful_tests = 0
    
    for asset_type, result in results.items():
        print(f"\n📊 {asset_type.upper()}:")
        
        if 'error' in result:
            print(f"  ❌ Failed: {result['error']}")
            total_tests += 1
        else:
            for timeframe, test_result in result.items():
                total_tests += 1
                if test_result['success']:
                    successful_tests += 1
                    print(f"  ✅ {timeframe}: {test_result['rows']} rows, Latest: ${test_result['latest_price']:.4f}")
                else:
                    print(f"  ❌ {timeframe}: {test_result.get('error', 'Unknown error')}")
    
    success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"\n🎯 OVERALL RESULTS:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Successful: {successful_tests}")
    print(f"   Failed: {total_tests - successful_tests}")
    print(f"   Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 75:
        print(f"\n🎉 FMP Integration Status: GOOD ({success_rate:.1f}% success rate)")
        return True
    else:
        print(f"\n⚠️ FMP Integration Status: NEEDS ATTENTION ({success_rate:.1f}% success rate)")
        return False

def test_technical_indicators():
    """Test technical indicators calculation with FMP data."""
    
    print(f"\n" + "=" * 50)
    print("TECHNICAL INDICATORS TEST")
    print("=" * 50)
    
    try:
        from app.ml_logic import fetch_yfinance_data, create_features_for_prediction
        
        # Test with AAPL stock data
        print("Testing technical indicators with AAPL...")
        
        data = fetch_yfinance_data("AAPL", period='90d', interval='1h')
        
        if data is None or data.empty:
            print("❌ No data available for technical indicators test")
            return False
        
        print(f"✅ Got {len(data)} rows of AAPL data")
        
        # Create dummy feature columns list
        feature_columns = [
            'channel_slope', 'channel_width_atr', 'bars_outside_zone', 
            'rsi_14', 'price_vs_ema200', 'volume_ratio', 'hour_of_day',
            'day_of_week', 'risk_reward_ratio', 'trade_type_encoded'
        ]
        
        features_df = create_features_for_prediction(data, feature_columns)
        
        if features_df.empty:
            print("❌ Failed to create features")
            return False
        
        print(f"✅ Created {len(features_df)} feature rows with {len(features_df.columns)} columns")
        
        # Check specific indicators
        if 'rsi_14' in features_df.columns:
            latest_rsi = features_df['rsi_14'].iloc[-1]
            print(f"📈 Latest RSI: {latest_rsi:.2f}")
        
        if 'price_vs_ema200' in features_df.columns:
            price_ema_ratio = features_df['price_vs_ema200'].iloc[-1]
            print(f"📈 Price vs EMA200: {price_ema_ratio:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Technical indicators test failed: {e}")
        return False

def check_api_limits():
    """Check FMP API rate limits and provide recommendations."""
    
    print(f"\n" + "=" * 50)
    print("API LIMITS & RECOMMENDATIONS")
    print("=" * 50)
    
    print("📊 Financial Modeling Prep API (Free Tier):")
    print("   • 250 requests per day")
    print("   • Historical data up to 5 years")
    print("   • Real-time and intraday data available")
    print("   • Multiple asset classes supported")
    
    print("\n💡 RECOMMENDATIONS:")
    print("   1. Cache responses when possible to reduce API calls")
    print("   2. Limit market scans to 10-15 symbols per request")
    print("   3. Use longer timeframes (1h, 4h, 1d) to get more history per call")
    print("   4. Consider upgrading to paid plan for production use")
    
    print("\n⚠️ RATE LIMITING STRATEGY:")
    print("   • Current implementation: No rate limiting")
    print("   • Recommended: Add 1-2 second delays between requests")
    print("   • Implement exponential backoff on API errors")

if __name__ == "__main__":
    print(f"🚀 Starting FMP Integration Tests at {datetime.now()}")
    
    # Run all tests
    api_test_passed = test_fmp_integration()
    indicators_test_passed = test_technical_indicators()
    
    # Show recommendations
    check_api_limits()
    
    # Final status
    print(f"\n" + "=" * 70)
    print("FINAL TEST STATUS")
    print("=" * 70)
    
    if api_test_passed and indicators_test_passed:
        print("🎉 ALL TESTS PASSED - FMP Integration is ready!")
        print("✅ You can now deploy your app with Financial Modeling Prep API")
    else:
        print("⚠️ SOME TESTS FAILED - Review the errors above")
        print("🔧 Fix the issues before deploying to production")
    
    print(f"\n📅 Test completed at {datetime.now()}")
    print("=" * 70)
