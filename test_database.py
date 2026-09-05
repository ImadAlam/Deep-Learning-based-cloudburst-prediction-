"""
Test Script for CloudBurst Database
Demonstrates database functionality with sample data
"""

from database import PredictionDatabase
from datetime import datetime, timedelta
import random

def test_database():
    """Test database with sample data"""
    
    print("\n" + "="*70)
    print("CLOUDBURST DATABASE TEST")
    print("="*70)
    
    # Initialize database
    db = PredictionDatabase()
    
    # Test data
    locations = [
        (35.6, 73.6),  # Babusar Top
        (35.61, 73.61),
        (35.59, 73.59),
    ]
    
    print("\n1️⃣  Adding test predictions...")
    
    # Add sample predictions
    for i in range(20):
        # Random probability
        probability = random.uniform(0.1, 0.95)
        
        # Random time in last 7 days
        hours_ago = random.randint(0, 168)
        prediction_time = datetime.now() - timedelta(hours=hours_ago)
        
        # Random location
        lat, lon = random.choice(locations)
        lat += random.uniform(-0.02, 0.02)
        lon += random.uniform(-0.02, 0.02)
        
        # Sample features
        features = {
            'temperature_500hpa': random.uniform(240, 270),
            'humidity_700hpa': random.uniform(30, 90),
            'wind_speed_850hpa': random.uniform(0, 20),
            'cape': random.uniform(500, 3000),
            'column_water': random.uniform(20, 60)
        }
        
        prediction_id = db.store_prediction(
            probability=probability,
            prediction_time=prediction_time,
            latitude=lat,
            longitude=lon,
            features=features
        )
        
        if probability >= 0.7:
            print(f"   ✓ Prediction {prediction_id}: HIGH RISK (Prob: {probability:.2%})")
        elif probability >= 0.5:
            print(f"   ✓ Prediction {prediction_id}: MEDIUM RISK (Prob: {probability:.2%})")
        else:
            print(f"   ✓ Prediction {prediction_id}: LOW RISK (Prob: {probability:.2%})")
    
    print("\n2️⃣  Updating daily statistics...")
    db.update_daily_statistics()
    print("   ✓ Statistics updated")
    
    print("\n3️⃣  Database Summary:")
    summary = db.get_database_summary()
    print(f"   Total Predictions: {summary['total_predictions']}")
    print(f"   High-Risk Alerts: {summary['total_alerts']}")
    print(f"   Risk Distribution:")
    for risk, count in summary['risk_distribution'].items():
        percentage = (count / summary['total_predictions'] * 100)
        print(f"      {risk}: {count} ({percentage:.1f}%)")
    
    print("\n4️⃣  Recent Predictions:")
    predictions = db.get_recent_predictions(hours=168, limit=5)
    for p in predictions:
        print(f"   - ID: {p['id']:3d} | Time: {p['timestamp']} | "
              f"Prob: {p['probability']:.3f} | Risk: {p['risk_level']}")
    
    print("\n5️⃣  High-Risk Alerts:")
    alerts = db.get_high_risk_predictions(days=7)
    print(f"   Total: {len(alerts)}")
    for a in alerts[:5]:
        print(f"   - Alert ID: {a['id']} | Prob: {a['probability']:.3f} | {a['alert_message'][:50]}")
    
    print("\n✓ Database test completed successfully!")
    print("="*70)

if __name__ == '__main__':
    test_database()
