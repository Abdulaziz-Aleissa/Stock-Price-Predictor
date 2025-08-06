#!/usr/bin/env python3
"""
Test script to verify the functionality of key features
"""
import requests
import sys
import time
import os
sys.path.append('/home/runner/work/Stock-Price-Predictor/Stock-Price-Predictor')

def test_home_page_ticker_input():
    """Test home page ticker input functionality"""
    print("🔍 Testing home page ticker input...")
    
    try:
        # Test basic GET to home page
        response = requests.get('http://127.0.0.1:5000')
        if response.status_code != 200:
            print(f"❌ Home page not accessible: {response.status_code}")
            return False
        print("✅ Home page loads correctly")
        
        # Test POST to predict endpoint with valid ticker
        test_data = {'ticker': 'AAPL'}
        response = requests.post('http://127.0.0.1:5000/predict', data=test_data)
        
        if response.status_code == 200:
            print("✅ Predict route works with valid ticker")
            return True
        else:
            print(f"❌ Predict route failed: {response.status_code}")
            if response.text:
                print(f"Response: {response.text[:200]}...")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to app - is it running on port 5000?")
        return False
    except Exception as e:
        print(f"❌ Error testing home page: {str(e)}")
        return False

def test_watchlist_functionality():
    """Test watchlist add/remove functionality"""
    print("\n🔍 Testing watchlist functionality...")
    
    try:
        # First we need to check if login works (watchlist requires auth)
        session = requests.Session()
        
        # Try to access dashboard without login (should redirect or fail)
        response = session.get('http://127.0.0.1:5000/dashboard', allow_redirects=False)
        if response.status_code == 302:  # Redirect to login
            print("✅ Dashboard correctly requires authentication")
        elif response.status_code == 200:
            print("❌ Dashboard accessible without login - authentication may be broken")
            return False
        else:
            print(f"❌ Unexpected dashboard response: {response.status_code}")
            return False
        return True
        
    except Exception as e:
        print(f"❌ Error testing watchlist: {str(e)}")
        return False

def test_compare_functionality():
    """Test stock comparison functionality"""
    print("\n🔍 Testing compare stocks functionality...")
    
    try:
        session = requests.Session()
        test_data = {
            'symbol1': 'AAPL',
            'symbol2': 'GOOGL', 
            'timeframe': '1mo'
        }
        
        response = session.post('http://127.0.0.1:5000/compare_stocks', data=test_data, allow_redirects=False)
        
        if response.status_code == 302:  # Redirect to login
            print("✅ Compare route correctly requires authentication")
            return True
        elif response.status_code == 200:
            print("❌ Compare route accessible without login")
            return False
        else:
            print(f"❌ Compare route unexpected response: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing compare: {str(e)}")
        return False

def test_paper_trading():
    """Test paper trading functionality"""
    print("\n🔍 Testing paper trading functionality...")
    
    try:
        session = requests.Session()
        test_data = {
            'symbol': 'AAPL',
            'shares': '10',
            'price': '150.00'
        }
        
        response = session.post('http://127.0.0.1:5000/paper_buy', data=test_data, allow_redirects=False)
        
        if response.status_code == 302:  # Redirect to login
            print("✅ Paper trading correctly requires authentication")
            return True
        elif response.status_code == 200:
            print("❌ Paper trading accessible without login")
            return False
        else:
            print(f"❌ Paper trading unexpected response: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing paper trading: {str(e)}")
        return False

def test_advanced_analytics():
    """Test advanced analytics functionality"""
    print("\n🔍 Testing advanced analytics functionality...")
    
    try:
        session = requests.Session()
        test_data = {
            'symbol': 'AAPL',
            'portfolio_value': '10000',
            'holding_period': '1'
        }
        
        response = session.post('http://127.0.0.1:5000/value_at_risk', data=test_data, allow_redirects=False)
        
        if response.status_code == 302:  # Redirect to login
            print("✅ Advanced analytics correctly requires authentication")
            return True
        elif response.status_code == 200:
            print("❌ Advanced analytics accessible without login")
            return False
        else:
            print(f"❌ Advanced analytics unexpected response: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing advanced analytics: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting functionality tests...\n")
    
    tests = [
        test_home_page_ticker_input,
        test_watchlist_functionality, 
        test_compare_functionality,
        test_paper_trading,
        test_advanced_analytics
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            time.sleep(1)  # Brief pause between tests
        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed - issues need to be fixed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)