"""
SQLite Database Setup for CloudBurst Prediction System
Stores prediction records with focus on high-risk alerts
"""

import sqlite3
import os
from datetime import datetime
import json

# Database file path
DB_PATH = os.path.join(os.path.dirname(__file__), 'cloudburst_predictions.db')

class PredictionDatabase:
    """SQLite database handler for cloudburst predictions"""
    
    def __init__(self, db_path=DB_PATH):
        """Initialize database connection"""
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        """Initialize database schema"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                prediction_time DATETIME NOT NULL,
                probability REAL NOT NULL,
                is_cloudburst INTEGER DEFAULT 0,
                risk_level TEXT CHECK(risk_level IN ('LOW', 'MEDIUM', 'HIGH')) DEFAULT 'LOW',
                features_json TEXT,
                latitude REAL,
                longitude REAL,
                model_version TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # High-risk alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS high_risk_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                alert_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                probability REAL NOT NULL,
                risk_level TEXT,
                alert_message TEXT,
                location_lat REAL,
                location_lon REAL,
                acknowledged INTEGER DEFAULT 0,
                acknowledged_by TEXT,
                acknowledged_at DATETIME,
                FOREIGN KEY (prediction_id) REFERENCES predictions(id)
            )
        ''')
        
        # Statistics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE UNIQUE,
                total_predictions INTEGER DEFAULT 0,
                high_risk_count INTEGER DEFAULT 0,
                medium_risk_count INTEGER DEFAULT 0,
                low_risk_count INTEGER DEFAULT 0,
                true_positives INTEGER DEFAULT 0,
                false_positives INTEGER DEFAULT 0,
                accuracy REAL DEFAULT 0.0,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_risk ON predictions(risk_level)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_time ON high_risk_alerts(alert_time)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_probability ON high_risk_alerts(probability)')
        
        conn.commit()
        conn.close()
        
        print(f"✓ Database initialized: {self.db_path}")
    
    def store_prediction(self, probability, prediction_time, latitude=None, 
                        longitude=None, features=None, model_version='BiLSTM-v1.0'):
        """
        Store a prediction record
        
        Args:
            probability: Float (0-1), prediction probability
            prediction_time: datetime object when prediction was made
            latitude: Optional location latitude
            longitude: Optional location longitude
            features: Optional dict of input features
            model_version: Model version string
        
        Returns:
            prediction_id: ID of stored prediction
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Determine risk level
        if probability >= 0.7:
            risk_level = 'HIGH'
        elif probability >= 0.5:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        is_cloudburst = 1 if probability >= 0.5 else 0
        features_json = json.dumps(features) if features else None
        
        cursor.execute('''
            INSERT INTO predictions 
            (probability, is_cloudburst, risk_level, prediction_time, 
             features_json, latitude, longitude, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (probability, is_cloudburst, risk_level, prediction_time, 
              features_json, latitude, longitude, model_version))
        
        prediction_id = cursor.lastrowid
        conn.commit()
        
        # Create alert if high-risk
        if risk_level == 'HIGH':
            self.create_high_risk_alert(prediction_id, probability, 
                                       risk_level, latitude, longitude)
        
        conn.close()
        return prediction_id
    
    def create_high_risk_alert(self, prediction_id, probability, risk_level, 
                               latitude=None, longitude=None):
        """Create a high-risk alert record"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        alert_message = (
            f"HIGH RISK CLOUDBURST ALERT - Probability: {probability:.2%} "
            f"Location: ({latitude:.2f}°N, {longitude:.2f}°E)" 
            if latitude and longitude 
            else f"HIGH RISK CLOUDBURST ALERT - Probability: {probability:.2%}"
        )
        
        cursor.execute('''
            INSERT INTO high_risk_alerts 
            (prediction_id, probability, risk_level, alert_message, 
             location_lat, location_lon)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (prediction_id, probability, risk_level, alert_message, 
              latitude, longitude))
        
        conn.commit()
        conn.close()
        
        print(f"🚨 {alert_message}")
    
    def get_recent_predictions(self, hours=24, limit=100):
        """Get recent predictions"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions 
            WHERE timestamp >= datetime('now', '-' || ? || ' hours')
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (hours, limit))
        
        results = cursor.fetchall()
        conn.close()
        return results
    
    def get_high_risk_predictions(self, days=7):
        """Get all high-risk predictions from last N days"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM high_risk_alerts
            WHERE alert_time >= datetime('now', '-' || ? || ' days')
            ORDER BY alert_time DESC
        ''', (days,))
        
        results = cursor.fetchall()
        conn.close()
        return results
    
    def get_predictions_by_risk_level(self, risk_level='HIGH', limit=50):
        """Get predictions by risk level"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions 
            WHERE risk_level = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (risk_level, limit))
        
        results = cursor.fetchall()
        conn.close()
        return results
    
    def acknowledge_alert(self, alert_id, acknowledged_by=None):
        """Mark an alert as acknowledged"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE high_risk_alerts 
            SET acknowledged = 1, acknowledged_by = ?, acknowledged_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (acknowledged_by, alert_id))
        
        conn.commit()
        conn.close()
    
    def get_statistics(self, date=None):
        """Get statistics for a specific date"""
        if date is None:
            date = datetime.now().date()
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM statistics WHERE date = ?
        ''', (date,))
        
        result = cursor.fetchone()
        conn.close()
        return result
    
    def update_daily_statistics(self, date=None):
        """Calculate and update daily statistics"""
        if date is None:
            date = datetime.now().date()
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Count predictions by risk level
        cursor.execute('''
            SELECT risk_level, COUNT(*) as count 
            FROM predictions 
            WHERE DATE(timestamp) = ?
            GROUP BY risk_level
        ''', (date,))
        
        risk_counts = dict(cursor.fetchall())
        high_count = risk_counts.get('HIGH', 0)
        medium_count = risk_counts.get('MEDIUM', 0)
        low_count = risk_counts.get('LOW', 0)
        total = high_count + medium_count + low_count
        
        # Get total predictions for the day
        cursor.execute('''
            SELECT COUNT(*) as total FROM predictions WHERE DATE(timestamp) = ?
        ''', (date,))
        
        total_predictions = cursor.fetchone()[0]
        
        # Insert or update statistics
        cursor.execute('''
            INSERT INTO statistics 
            (date, total_predictions, high_risk_count, medium_risk_count, 
             low_risk_count, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(date) DO UPDATE SET 
                total_predictions = ?,
                high_risk_count = ?,
                medium_risk_count = ?,
                low_risk_count = ?,
                updated_at = CURRENT_TIMESTAMP
        ''', (date, total_predictions, high_count, medium_count, low_count,
              total_predictions, high_count, medium_count, low_count))
        
        conn.commit()
        conn.close()
    
    def get_database_summary(self):
        """Get overall database summary"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM predictions')
        total_predictions = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM high_risk_alerts')
        total_alerts = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT risk_level, COUNT(*) as count 
            FROM predictions GROUP BY risk_level
        ''')
        risk_distribution = dict(cursor.fetchall())
        
        cursor.execute('''
            SELECT COUNT(*) FROM high_risk_alerts WHERE acknowledged = 0
        ''')
        unacknowledged_alerts = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_predictions': total_predictions,
            'total_alerts': total_alerts,
            'risk_distribution': risk_distribution,
            'unacknowledged_alerts': unacknowledged_alerts
        }

# Initialize database when module is imported
if __name__ == '__main__':
    db = PredictionDatabase()
    print("\n✓ CloudBurst Prediction Database initialized!")
    print(f"Database location: {DB_PATH}")
