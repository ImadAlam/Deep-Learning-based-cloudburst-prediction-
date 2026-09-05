#!/usr/bin/env python
"""
PRODUCTION STARTUP SCRIPT - CloudBurst Prediction System
Starts the Flask app with database integration and displays status
"""

import sys
import os
import subprocess
import time
import platform

def print_banner():
    banner = """
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║         CLOUDBURST PREDICTION SYSTEM - PRODUCTION STARTUP                  ║
║                                                                            ║
║              BiLSTM Model + SQLite Database Integration                    ║
║              Real-time Prediction with Alert Management                    ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def check_dependencies():
    """Verify all required dependencies are installed"""
    print("📦 Checking dependencies...")
    
    required = {
        'flask': 'Flask web framework',
        'tensorflow': 'TensorFlow/Keras (ML model)',
        'numpy': 'NumPy (array processing)',
        'joblib': 'Joblib (model serialization)',
    }
    
    missing = []
    for package, description in required.items():
        try:
            __import__(package)
            print(f"   ✓ {package:15} ({description})")
        except ImportError:
            print(f"   ✗ {package:15} ({description}) - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("    Run: pip install -r requirements.txt")
        return False
    
    print("✓ All dependencies available\n")
    return True

def check_files():
    """Verify all required project files exist"""
    print("📁 Checking project files...")
    
    required_files = {
        'app.py': 'Flask application',
        'database.py': 'Database module',
        'flask_integration.py': 'Flask integration functions',
        'cloudburst_final_bilstm_only.keras': 'ML model file',
        'scaler_final.pkl': 'Feature scaler',
        'feature_cols.pkl': 'Feature column names',
        'templates/index.html': 'Web dashboard',
    }
    
    missing = []
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            size_str = f"{size/1024/1024:.1f}MB" if size > 1024*1024 else f"{size/1024:.1f}KB"
            print(f"   ✓ {file_path:40} ({size_str})")
        else:
            print(f"   ✗ {file_path:40} - MISSING")
            missing.append(file_path)
    
    if missing:
        print(f"\n⚠️  Missing files: {', '.join(missing)}")
        return False
    
    print("✓ All project files present\n")
    return True

def check_database():
    """Verify database is initialized"""
    print("🗄️  Checking database...")
    
    db_file = 'cloudburst_predictions.db'
    if os.path.exists(db_file):
        size = os.path.getsize(db_file)
        size_str = f"{size/1024:.1f}KB"
        print(f"   ✓ Database exists: {db_file} ({size_str})")
        print("   ✓ Ready for predictions")
    else:
        print(f"   ⚠️  Database file not found: {db_file}")
        print("   → Database will be auto-created on first run")
    
    print()
    return True

def start_flask_app():
    """Start the Flask application"""
    print("🚀 Starting Flask application...\n")
    
    try:
        # Use subprocess to run app.py
        cmd = [sys.executable, 'app.py']
        
        print("╔" + "="*78 + "╗")
        print("║ Flask Application Starting... Press CTRL+C to stop" + " "*29 + "║")
        print("╚" + "="*78 + "╝\n")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Read and display output line by line
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
        except KeyboardInterrupt:
            print("\n\n⏹️  Shutting down...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            print("✓ Application stopped\n")
    
    except Exception as e:
        print(f"\n✗ Error starting application: {e}")
        return False

def main():
    """Main startup routine"""
    os.system('cls' if platform.system() == 'Windows' else 'clear')
    
    print_banner()
    
    # Run checks
    if not check_dependencies():
        sys.exit(1)
    
    if not check_files():
        sys.exit(1)
    
    check_database()
    
    # Display system info
    print("📊 System Information:")
    print(f"   • Python: {sys.version.split()[0]}")
    print(f"   • OS: {platform.system()} {platform.release()}")
    print(f"   • Working Dir: {os.getcwd()}\n")
    
    # Display startup info
    print("🌐 Web Server URLs:")
    print("   • Local:     http://127.0.0.1:5000")
    print("   • Dashboard: http://127.0.0.1:5000/")
    print()
    
    print("📡 API Endpoints:")
    print("   • POST   /api/predict              - Make prediction (stored)")
    print("   • GET    /api/recent_predictions   - Query predictions")
    print("   • GET    /api/high_risk_alerts     - Get alerts")
    print("   • GET    /api/database_summary     - Statistics")
    print()
    
    print("🧪 Testing the API:")
    print("   • In another terminal run: python test_api_endpoints.py")
    print()
    
    print("📖 Documentation:")
    print("   • See DATABASE_GUIDE.md for integration details")
    print("   • See DATABASE_SETUP_COMPLETE.txt for quick reference")
    print()
    
    # Start app
    start_flask_app()

if __name__ == '__main__':
    main()
