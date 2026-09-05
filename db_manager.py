"""
Database Management Tool for CloudBurst Predictions
Query, analyze, and manage the SQLite database
"""

from database import PredictionDatabase
from datetime import datetime, timedelta
import json
from tabulate import tabulate

class DatabaseManager:
    """Management and analytics for prediction database"""
    
    def __init__(self):
        self.db = PredictionDatabase()
    
    def show_summary(self):
        """Display database summary"""
        print("\n" + "="*70)
        print("CLOUDBURST PREDICTION DATABASE SUMMARY")
        print("="*70)
        
        summary = self.db.get_database_summary()
        
        print(f"\n📊 Overall Statistics:")
        print(f"   Total Predictions: {summary['total_predictions']}")
        print(f"   Total High-Risk Alerts: {summary['total_alerts']}")
        print(f"   Unacknowledged Alerts: {summary['unacknowledged_alerts']}")
        
        print(f"\n📈 Risk Distribution:")
        for risk, count in summary['risk_distribution'].items():
            percentage = (count / summary['total_predictions'] * 100) if summary['total_predictions'] > 0 else 0
            print(f"   {risk}: {count} ({percentage:.1f}%)")
    
    def show_recent_predictions(self, hours=24, limit=10):
        """Display recent predictions"""
        print("\n" + "="*70)
        print(f"RECENT PREDICTIONS (Last {hours} hours)")
        print("="*70)
        
        predictions = self.db.get_recent_predictions(hours=hours, limit=limit)
        
        if not predictions:
            print("No predictions found.")
            return
        
        data = []
        for p in predictions:
            data.append([
                p['id'],
                p['timestamp'][:19],
                f"{p['probability']:.3f}",
                p['risk_level'],
                p['is_cloudburst'],
                f"({p['latitude']:.2f}, {p['longitude']:.2f})" if p['latitude'] and p['longitude'] else "N/A"
            ])
        
        print(tabulate(data, headers=['ID', 'Time', 'Probability', 'Risk', 'CB', 'Location'], 
                      tablefmt='grid'))
    
    def show_high_risk_alerts(self, days=7):
        """Display high-risk alerts"""
        print("\n" + "="*70)
        print(f"HIGH-RISK ALERTS (Last {days} days)")
        print("="*70)
        
        alerts = self.db.get_high_risk_predictions(days=days)
        
        if not alerts:
            print("No high-risk alerts found.")
            return
        
        data = []
        for a in alerts:
            ack_status = "✓ Yes" if a['acknowledged'] else "✗ No"
            data.append([
                a['id'],
                a['alert_time'][:19],
                f"{a['probability']:.3f}",
                a['risk_level'],
                ack_status,
                a['alert_message'][:40] + "..." if len(a['alert_message']) > 40 else a['alert_message']
            ])
        
        print(tabulate(data, headers=['Alert ID', 'Time', 'Probability', 'Risk', 'Acknowledged', 'Message'], 
                      tablefmt='grid'))
    
    def show_risk_distribution(self):
        """Show predictions by risk level"""
        print("\n" + "="*70)
        print("PREDICTIONS BY RISK LEVEL")
        print("="*70)
        
        for risk_level in ['HIGH', 'MEDIUM', 'LOW']:
            predictions = self.db.get_predictions_by_risk_level(risk_level, limit=1000)
            print(f"\n{risk_level} Risk Predictions: {len(predictions)}")
            
            if predictions:
                # Show top 5 most recent
                print(f"  Most recent 5:")
                for p in predictions[:5]:
                    print(f"    - {p['timestamp'][:19]} | Prob: {p['probability']:.3f} | ID: {p['id']}")
    
    def export_predictions(self, filename='predictions_export.json', hours=24):
        """Export predictions to JSON file"""
        print(f"\n📤 Exporting predictions from last {hours} hours...")
        
        predictions = self.db.get_recent_predictions(hours=hours, limit=10000)
        
        export_data = {
            'export_time': datetime.now().isoformat(),
            'count': len(predictions),
            'predictions': [dict(p) for p in predictions]
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"✓ Exported {len(predictions)} predictions to {filename}")
    
    def show_statistics(self, days=7):
        """Show statistics for last N days"""
        print("\n" + "="*70)
        print(f"DAILY STATISTICS (Last {days} days)")
        print("="*70)
        
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM statistics 
            WHERE date >= DATE('now', '-' || ? || ' days')
            ORDER BY date DESC
        ''', (days,))
        
        stats = cursor.fetchall()
        conn.close()
        
        if not stats:
            print("No statistics available yet.")
            return
        
        data = []
        for s in stats:
            data.append([
                s['date'],
                s['total_predictions'],
                s['high_risk_count'],
                s['medium_risk_count'],
                s['low_risk_count']
            ])
        
        print(tabulate(data, headers=['Date', 'Total', 'High Risk', 'Medium Risk', 'Low Risk'], 
                      tablefmt='grid'))
    
    def cleanup_old_records(self, days=90):
        """Delete predictions older than N days"""
        print(f"\n🗑️  Cleaning up records older than {days} days...")
        
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) FROM predictions 
            WHERE timestamp < datetime('now', '-' || ? || ' days')
        ''', (days,))
        
        count = cursor.fetchone()[0]
        
        if count > 0:
            cursor.execute('''
                DELETE FROM predictions 
                WHERE timestamp < datetime('now', '-' || ? || ' days')
            ''', (days,))
            conn.commit()
            print(f"✓ Deleted {count} old prediction records")
        else:
            print(f"✓ No records older than {days} days found")
        
        conn.close()
    
    def interactive_menu(self):
        """Interactive menu for database management"""
        while True:
            print("\n" + "="*70)
            print("CLOUDBURST DATABASE MANAGEMENT MENU")
            print("="*70)
            print("\n1. Show Summary")
            print("2. Show Recent Predictions")
            print("3. Show High-Risk Alerts")
            print("4. Show Risk Distribution")
            print("5. Show Daily Statistics")
            print("6. Export Predictions")
            print("7. Cleanup Old Records")
            print("8. Exit")
            
            choice = input("\nSelect option (1-8): ").strip()
            
            if choice == '1':
                self.show_summary()
            elif choice == '2':
                hours = input("Hours to look back (default 24): ").strip()
                self.show_recent_predictions(hours=int(hours) if hours else 24)
            elif choice == '3':
                days = input("Days to look back (default 7): ").strip()
                self.show_high_risk_alerts(days=int(days) if days else 7)
            elif choice == '4':
                self.show_risk_distribution()
            elif choice == '5':
                days = input("Days to show (default 7): ").strip()
                self.show_statistics(days=int(days) if days else 7)
            elif choice == '6':
                filename = input("Output filename (default: predictions_export.json): ").strip()
                self.export_predictions(filename if filename else 'predictions_export.json')
            elif choice == '7':
                days = input("Delete records older than N days (default 90): ").strip()
                confirm = input(f"This will delete records older than {days if days else 90} days. Continue? (y/N): ")
                if confirm.lower() == 'y':
                    self.cleanup_old_records(int(days) if days else 90)
            elif choice == '8':
                print("\nGoodbye!")
                break
            else:
                print("\n✗ Invalid option. Please try again.")

def main():
    """Main function"""
    import sys
    
    manager = DatabaseManager()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'summary':
            manager.show_summary()
        elif command == 'recent':
            hours = int(sys.argv[2]) if len(sys.argv) > 2 else 24
            manager.show_recent_predictions(hours=hours)
        elif command == 'alerts':
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
            manager.show_high_risk_alerts(days=days)
        elif command == 'stats':
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
            manager.show_statistics(days=days)
        elif command == 'export':
            filename = sys.argv[2] if len(sys.argv) > 2 else 'predictions_export.json'
            manager.export_predictions(filename)
        elif command == 'cleanup':
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 90
            manager.cleanup_old_records(days)
        else:
            print("Unknown command")
    else:
        # Interactive mode
        manager.interactive_menu()

if __name__ == '__main__':
    print("="*70)
    print("CloudBurst Prediction Database Manager")
    print("="*70)
    main()
