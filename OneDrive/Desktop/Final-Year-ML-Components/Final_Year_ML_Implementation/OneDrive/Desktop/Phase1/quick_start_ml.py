"""
Quick Start - ML Security System
Database-driven approach without blockchain/IoT devices
"""

from database.ml_security_manager import MLSecurityManager
from database.data_generator import initialize_database_with_dummy_data, IoTDatabase
from datetime import datetime


def main():
    print("=" * 60)
    print("  🤖 IoT ML Security System - Quick Start")
    print("  Database-Driven Approach")
    print("=" * 60 + "\n")
    
    # Step 1: Initialize database
    print("1️⃣  Checking database...")
    db = IoTDatabase()
    stats = db.get_statistics()
    
    if stats['total_access_logs'] == 0:
        print("   Database empty. Initializing with dummy data...")
        db.close()
        initialize_database_with_dummy_data()
        db = IoTDatabase()
        stats = db.get_statistics()
    
    print(f"   ✓ Database ready with {stats['total_access_logs']} access logs")
    print(f"   ✓ Users: {stats['total_users']}, Devices: {stats['total_devices']}\n")
    db.close()
    
    # Step 2: Initialize ML Security Manager
    print("2️⃣  Initializing ML Security Manager...")
    manager = MLSecurityManager()
    
    # Step 3: Test access anomaly detection
    print("3️⃣  Testing access anomaly detection...")
    test_access = {
        'timestamp': datetime.now().isoformat(),
        'device_id': 'smart_lock',
        'user_id': 'Adish',
        'action': 'unlock',
        'ip_address': '192.168.1.100',
        'location': 'home',
        'access_count': 3,
        'time_since_last': 3600,
        'duration': 45,
        'success': True
    }
    
    result = manager.analyze_access_log(test_access)
    print(f"   ✓ Access analyzed")
    print(f"   Anomaly detected: {result['is_anomaly']}")
    print(f"   Confidence: {result['confidence']:.2f}\n")
    
    # Step 4: Test power profiling
    print("4️⃣  Testing power consumption profiling...")
    test_power = {
        'device_id': 'smart_light',
        'power_watts': 12.5,
        'voltage': 120.2,
        'current_amps': 0.104,
        'power_factor': 0.95,
        'avg_power': 12.0,
        'power_variance': 1.5,
        'peak_power': 15.0,
        'device_state': 'on',
        'cpu_usage': 15,
        'network_activity': 50,
        'temperature': 26.2
    }
    
    result = manager.analyze_power_consumption(test_power)
    print(f"   ✓ Power analyzed")
    print(f"   Anomaly detected: {result['is_anomaly']}")
    if result['is_anomaly']:
        print(f"   Anomaly type: {result['anomaly_type']}\n")
    else:
        print()
    
    # Step 5: Test behavior prediction
    print("5️⃣  Testing behavior prediction...")
    test_context = {
        'hour': 19,
        'day_of_week': 2,
        'user_id': 'Adish',
        'device_id': 'smart_tv',
        'previous_state': 'off',
        'time_since_last': 3600,
        'interactions_today': 5,
        'typical_usage_hour': 19,
        'is_home': True,
        'ambient_light': 25,
        'temperature': 22.5
    }
    
    result = manager.predict_device_behavior(test_context)
    print(f"   ✓ Behavior predicted")
    print(f"   Predicted state: {result['predicted_state']}\n")
    
    # Step 6: System status
    print("6️⃣  System status:")
    status = manager.get_system_status()
    
    print(f"   Database:")
    print(f"     Access logs: {status['database']['access_logs']}")
    print(f"     Power logs: {status['database']['power_logs']}")
    print(f"     Behavior logs: {status['database']['behavior_logs']}")
    print(f"     Alerts: {status['database']['alerts']}")
    
    print(f"\n   ML Models:")
    print(f"     Anomaly Detector: {status['ml_models']['anomaly_detector']}")
    print(f"     Power Profiler: {status['ml_models']['power_profiler']}")
    print(f"     Behavior System: {status['ml_models']['behavior_system']}")
    
    print(f"\n   Anomalies Detected:")
    print(f"     Access anomalies: {status['anomalies']['access_anomalies']}")
    print(f"     Power anomalies: {status['anomalies']['power_anomalies']}\n")
    
    # Cleanup
    manager.close()
    
    print("=" * 60)
    print("  ✅ Quick Start Complete!")
    print("=" * 60 + "\n")
    
    print("🎯 What you just did:")
    print("   • Initialized database with dummy IoT data")
    print("   • Loaded ML models (Isolation Forest, LSTM, Autoencoder, Random Forest)")
    print("   • Analyzed access patterns for anomalies")
    print("   • Profiled power consumption")
    print("   • Predicted device behavior")
    print("   • Monitored system status\n")
    
    print("📚 Next steps:")
    print("   • Train ML models: python ml_models/model_trainer.py")
    print("   • Run full demo: python demo_ml.py")
    print("   • Scan database: python database/ml_security_manager.py")
    print("   • View data: python -c \"from database.data_generator import IoTDatabase; db=IoTDatabase(); print(db.get_recent_access_logs())\"\n")
    
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
