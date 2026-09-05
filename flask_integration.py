"""
Flask Integration Module for CloudBurst Prediction Database
Adds database storage functionality to the prediction API
"""

from database import PredictionDatabase
from datetime import datetime
from flask import jsonify
import traceback

# Initialize database
db = PredictionDatabase()

def store_prediction_and_check_alert(probability, latitude=None, longitude=None, 
                                     features=None, model_version='BiLSTM-v1.0'):
    """
    Store prediction in database and create alert if high-risk
    
    Returns: dict with storage status and alert info
    """
    try:
        prediction_time = datetime.now()
        prediction_id = db.store_prediction(
            probability=probability,
            prediction_time=prediction_time,
            latitude=latitude,
            longitude=longitude,
            features=features,
            model_version=model_version
        )
        
        # Determine risk level
        if probability >= 0.7:
            risk_level = 'HIGH'
            is_alert = True
        elif probability >= 0.5:
            risk_level = 'MEDIUM'
            is_alert = False
        else:
            risk_level = 'LOW'
            is_alert = False
        
        return {
            'stored': True,
            'prediction_id': prediction_id,
            'risk_level': risk_level,
            'is_high_risk_alert': is_alert,
            'message': f'Prediction stored (ID: {prediction_id}, Risk: {risk_level})'
        }
    
    except Exception as e:
        print(f"✗ Error storing prediction: {e}")
        traceback.print_exc()
        return {
            'stored': False,
            'error': str(e),
            'message': 'Failed to store prediction'
        }

def get_api_recent_predictions(hours=24, limit=100):
    """Get recent predictions for API response"""
    try:
        predictions = db.get_recent_predictions(hours=hours, limit=limit)
        return [dict(p) for p in predictions]
    except Exception as e:
        print(f"✗ Error retrieving predictions: {e}")
        return []

def get_api_high_risk_alerts(days=7):
    """Get high-risk alerts for API response"""
    try:
        alerts = db.get_high_risk_predictions(days=days)
        return [dict(a) for a in alerts]
    except Exception as e:
        print(f"✗ Error retrieving alerts: {e}")
        return []

def get_api_database_summary():
    """Get database summary for dashboard"""
    try:
        summary = db.get_database_summary()
        return summary
    except Exception as e:
        print(f"✗ Error getting summary: {e}")
        return {}

# Flask route examples to add to app.py

FLASK_ROUTES_EXAMPLE = '''
# Add these routes to your Flask app (app.py)

@app.route('/api/store_prediction', methods=['POST'])
def store_prediction_route():
    """Store a prediction in the database"""
    from flask_integration import store_prediction_and_check_alert
    
    data = request.get_json()
    probability = data.get('probability')
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    features = data.get('features')
    
    if probability is None:
        return jsonify({'error': 'Probability required'}), 400
    
    result = store_prediction_and_check_alert(
        probability=probability,
        latitude=latitude,
        longitude=longitude,
        features=features
    )
    
    return jsonify(result)

@app.route('/api/recent_predictions', methods=['GET'])
def recent_predictions_route():
    """Get recent predictions"""
    from flask_integration import get_api_recent_predictions
    
    hours = request.args.get('hours', 24, type=int)
    limit = request.args.get('limit', 100, type=int)
    
    predictions = get_api_recent_predictions(hours=hours, limit=limit)
    
    return jsonify({
        'count': len(predictions),
        'predictions': predictions
    })

@app.route('/api/high_risk_alerts', methods=['GET'])
def high_risk_alerts_route():
    """Get high-risk alerts"""
    from flask_integration import get_api_high_risk_alerts
    
    days = request.args.get('days', 7, type=int)
    alerts = get_api_high_risk_alerts(days=days)
    
    return jsonify({
        'count': len(alerts),
        'alerts': alerts
    })

@app.route('/api/database_summary', methods=['GET'])
def database_summary_route():
    """Get database summary"""
    from flask_integration import get_api_database_summary
    
    summary = get_api_database_summary()
    
    return jsonify(summary)

@app.route('/api/acknowledge_alert/<int:alert_id>', methods=['POST'])
def acknowledge_alert_route(alert_id):
    """Acknowledge a high-risk alert"""
    from database import PredictionDatabase
    
    db = PredictionDatabase()
    acknowledged_by = request.json.get('acknowledged_by') if request.json else None
    
    try:
        db.acknowledge_alert(alert_id, acknowledged_by=acknowledged_by)
        return jsonify({
            'status': 'success',
            'message': f'Alert {alert_id} acknowledged'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
'''
