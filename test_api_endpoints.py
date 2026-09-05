#!/usr/bin/env python
"""
TEST API ENDPOINTS - CloudBurst Prediction System
Tests all database-integrated API endpoints
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://127.0.0.1:5000"
API_KEY = "devkey"  # Default API key

def print_header(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def test_health_check():
    """Test /api/health endpoint"""
    print_header("1. HEALTH CHECK")
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_recent_predictions():
    """Test /api/recent_predictions endpoint"""
    print_header("2. GET RECENT PREDICTIONS")
    try:
        response = requests.get(
            f"{BASE_URL}/api/recent_predictions",
            params={"hours": 24, "limit": 10},
            timeout=5
        )
        print(f"✓ Status: {response.status_code}")
        data = response.json()
        print(f"✓ Recent predictions: {data['count']} records")
        if data['predictions']:
            print(f"✓ Latest prediction: {data['predictions'][0]}")
        return response.status_code == 200
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_high_risk_alerts():
    """Test /api/high_risk_alerts endpoint"""
    print_header("3. GET HIGH-RISK ALERTS")
    try:
        response = requests.get(
            f"{BASE_URL}/api/high_risk_alerts",
            params={"days": 7},
            timeout=5
        )
        print(f"✓ Status: {response.status_code}")
        data = response.json()
        print(f"✓ High-risk alerts: {data['count']} total")
        print(f"✓ Unacknowledged: {data['unacknowledged']}")
        if data['alerts']:
            print(f"✓ Latest alert: {data['alerts'][0]}")
        return response.status_code == 200
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_database_summary():
    """Test /api/database_summary endpoint"""
    print_header("4. DATABASE SUMMARY")
    try:
        response = requests.get(f"{BASE_URL}/api/database_summary", timeout=5)
        print(f"✓ Status: {response.status_code}")
        data = response.json()
        print(f"✓ Total predictions: {data.get('total_predictions', 0)}")
        print(f"✓ High-risk alerts: {data.get('unacknowledged_alerts', 0)}")
        print(f"✓ Risk distribution: {data.get('risk_distribution', {})}")
        return response.status_code == 200
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_make_prediction():
    """Test /api/predict endpoint with sample data"""
    print_header("5. MAKE PREDICTION (with database storage)")
    try:
        # Create sample features (26 features)
        features = [[0.5] * 26] * 6  # 6 timesteps x 26 features
        
        payload = {
            "features": features,
            "latitude": 35.6,
            "longitude": 73.6
        }
        
        response = requests.post(
            f"{BASE_URL}/api/predict",
            json=payload,
            timeout=10
        )
        print(f"✓ Status: {response.status_code}")
        data = response.json()
        print(f"✓ Cloudburst probability: {data.get('cloudburst_probability')}%")
        print(f"✓ Classification: {data.get('cloudburst')}")
        if 'database_stored' in data:
            print(f"✓ Database result: {data['database_stored']}")
        return response.status_code == 200
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_acknowledge_alert():
    """Test /api/acknowledge_alert endpoint"""
    print_header("6. ACKNOWLEDGE ALERT")
    try:
        # First get an alert ID from high-risk alerts
        response = requests.get(
            f"{BASE_URL}/api/high_risk_alerts",
            params={"days": 7},
            timeout=5
        )
        
        if response.status_code == 200:
            alerts = response.json().get('alerts', [])
            if alerts:
                alert_id = alerts[0].get('id')
                print(f"✓ Found alert ID: {alert_id}")
                
                # Acknowledge the alert
                ack_response = requests.post(
                    f"{BASE_URL}/api/acknowledge_alert/{alert_id}",
                    json={"acknowledged_by": "test_user"},
                    headers={"X-API-Key": API_KEY},
                    timeout=5
                )
                print(f"✓ Status: {ack_response.status_code}")
                print(f"✓ Response: {json.dumps(ack_response.json(), indent=2)}")
                return ack_response.status_code == 200
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def run_all_tests():
    """Run all API tests"""
    print("\n" + "█"*70)
    print("  CLOUDBURST PREDICTION SYSTEM - API TEST SUITE")
    print("█"*70)
    print(f"Server: {BASE_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Health Check", test_health_check),
        ("Recent Predictions", test_recent_predictions),
        ("High-Risk Alerts", test_high_risk_alerts),
        ("Database Summary", test_database_summary),
        ("Make Prediction", test_make_prediction),
        ("Acknowledge Alert", test_acknowledge_alert),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            time.sleep(0.5)  # Small delay between tests
        except Exception as e:
            print(f"\n✗ Test '{test_name}' failed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print_header("TEST SUMMARY")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is production-ready.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check server logs.")
    
    print("\n" + "█"*70 + "\n")

if __name__ == '__main__':
    print("\n⏳ Waiting for server to start (checking connection)...")
    max_retries = 10
    for i in range(max_retries):
        try:
            requests.get(f"{BASE_URL}/api/health", timeout=2)
            print("✓ Server is running!\n")
            break
        except Exception:
            if i < max_retries - 1:
                print(f"  Attempt {i+1}/{max_retries}: Server not ready, retrying...")
                time.sleep(1)
            else:
                print(f"\n✗ Could not connect to server after {max_retries} attempts")
                print(f"  Make sure the Flask app is running: python app.py")
                exit(1)
    
    run_all_tests()
