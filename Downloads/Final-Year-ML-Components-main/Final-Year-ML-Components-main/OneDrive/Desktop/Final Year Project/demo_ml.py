"""
ML-Focused Demo
Demonstrates all ML functionalities using database with dummy data
"""

from database.ml_security_manager import MLSecurityManager
from database.data_generator import initialize_database_with_dummy_data, IoTDatabase
from datetime import datetime
import pandas as pd


def print_header(title):
    """Print section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def demo_anomaly_detection(manager):
    """Demonstrate anomaly detection"""
    print_header("1. ACCESS ANOMALY DETECTION")
    
    print("🔍 Testing behavioral anomaly detection...\n")
    
    # Test 1: Normal access
    print("  Test 1: Normal access (7 PM, home)")
    normal_access = {
        'timestamp': '2024-11-13T19:00:00',
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
    
    result = manager.analyze_access_log(normal_access)
    print(f"  Anomaly: {'✓ YES' if result['is_anomaly'] else '✗ NO'}")
    print(f"  Confidence: {result['confidence']:.2f}\n")
    
    # Test 2: Suspicious access
    print("  Test 2: Suspicious access (3:42 AM, unknown location)")
    suspicious_access = {
        'timestamp': '2024-11-13T03:42:00',
        'device_id': 'smart_lock',
        'user_id': 'Guest001',
        'action': 'unlock',
        'ip_address': '203.45.67.89',
        'location': 'unknown_city',
        'access_count': 75,
        'time_since_last': 120,
        'duration': 300,
        'success': True
    }
    
    result = manager.analyze_access_log(suspicious_access)
    print(f"  Anomaly: {'✓ YES' if result['is_anomaly'] else '✗ NO'}")
    print(f"  Confidence: {result['confidence']:.2f}\n")


def demo_power_profiling(manager):
    """Demonstrate power consumption profiling"""
    print_header("2. POWER CONSUMPTION PROFILING")
    
    print("⚡ Testing power consumption anomaly detection...\n")
    
    # Test 1: Normal power
    print("  Test 1: Normal power consumption")
    normal_power = {
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
    
    result = manager.analyze_power_consumption(normal_power)
    print(f"  Anomaly: {'✓ YES' if result['is_anomaly'] else '✗ NO'}\n")
    
    # Test 2: Crypto mining attack
    print("  Test 2: Crypto mining attack (high power + CPU)")
    crypto_mining = {
        'device_id': 'smart_light',
        'power_watts': 150.0,
        'voltage': 120.1,
        'current_amps': 1.25,
        'power_factor': 0.95,
        'avg_power': 148.0,
        'power_variance': 5.0,
        'peak_power': 160.0,
        'device_state': 'on',
        'cpu_usage': 95,
        'network_activity': 180,
        'temperature': 45.8
    }
    
    result = manager.analyze_power_consumption(crypto_mining)
    print(f"  Anomaly: {'✓ YES' if result['is_anomaly'] else '✗ NO'}")
    if result['is_anomaly']:
        print(f"  Type: {result['anomaly_type']}")
        print(f"\n  🚨 ALERT: Possible {result['anomaly_type']} detected!\n")
    
    # Test 3: Botnet activity
    print("  Test 3: Botnet activity (high network traffic)")
    botnet = {
        'device_id': 'security_camera',
        'power_watts': 18.0,
        'voltage': 120.0,
        'current_amps': 0.15,
        'power_factor': 0.95,
        'avg_power': 17.5,
        'power_variance': 2.0,
        'peak_power': 20.0,
        'device_state': 'active',
        'cpu_usage': 45,
        'network_activity': 1200,
        'temperature': 35.0
    }
    
    result = manager.analyze_power_consumption(botnet)
    print(f"  Anomaly: {'✓ YES' if result['is_anomaly'] else '✗ NO'}")
    if result['is_anomaly']:
        print(f"  Type: {result['anomaly_type']}\n")


def demo_behavior_prediction(manager):
    """Demonstrate contextual behavior prediction"""
    print_header("3. CONTEXTUAL BEHAVIOR PREDICTION")
    
    print("🧠 Testing device behavior prediction...\n")
    
    # Test 1: Expected behavior
    print("  Test 1: Expected behavior (TV on at 7 PM)")
    context = {
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
    
    result = manager.predict_device_behavior(context)
    print(f"  Predicted: {result['predicted_state']}")
    print(f"  Confidence: {result.get('confidence', 0.0):.2f}")
    
    actual_state = 'on'
    anomaly_result = manager.detect_behavioral_anomaly(context, actual_state)
    print(f"  Actual: {actual_state}")
    print(f"  Anomaly: {'✓ YES' if anomaly_result['is_anomaly'] else '✗ NO'}\n")
    
    # Test 2: Unexpected behavior
    print("  Test 2: Unexpected behavior (lights on at 3 AM, user away)")
    context = {
        'hour': 3,
        'day_of_week': 2,
        'user_id': 'Adish',
        'device_id': 'smart_light',
        'previous_state': 'off',
        'time_since_last': 28800,
        'interactions_today': 0,
        'typical_usage_hour': 19,
        'is_home': False,
        'ambient_light': 0,
        'temperature': 20.0
    }
    
    result = manager.predict_device_behavior(context)
    print(f"  Predicted: {result['predicted_state']}")
    
    actual_state = 'on'
    anomaly_result = manager.detect_behavioral_anomaly(context, actual_state)
    print(f"  Actual: {actual_state}")
    print(f"  Anomaly: {'✓ YES' if anomaly_result['is_anomaly'] else '✗ NO'}\n")


def demo_database_scan(manager):
    """Demonstrate scanning database for anomalies"""
    print_header("4. DATABASE ANOMALY SCAN")
    
    # Scan access logs
    manager.scan_recent_access_logs(limit=50)
    
    # Scan power logs
    manager.scan_recent_power_logs(limit=50)


def demo_alerts(manager):
    """Demonstrate security alerts"""
    print_header("5. SECURITY ALERTS")
    
    print("🚨 Recent security alerts:\n")
    
    alerts = manager.get_alerts(limit=10)
    
    if len(alerts) == 0:
        print("  No alerts found.\n")
    else:
        for idx, alert in alerts.iterrows():
            print(f"  Alert {idx + 1}:")
            print(f"    Type: {alert['alert_type']}")
            print(f"    Severity: {alert['severity']}")
            print(f"    Time: {alert['timestamp']}")
            if alert['device_id']:
                print(f"    Device: {alert['device_id']}")
            if alert['description']:
                print(f"    Description: {alert['description']}")
            print()


def demo_system_status(manager):
    """Demonstrate system status"""
    print_header("6. SYSTEM STATUS")
    
    print("📊 Overall System Status:\n")
    
    status = manager.get_system_status()
    
    for category, data in status.items():
        print(f"  {category.upper()}:")
        for key, value in data.items():
            print(f"    {key}: {value}")
        print()


def main():
    """Main demo function"""
    print("\n" + "=" * 70)
    print("  🤖 IoT ML Security System - Comprehensive Demo")
    print("  Database-Driven Approach with Dummy Data")
    print("=" * 70)
    
    # Check if database exists, if not initialize it
    db = IoTDatabase()
    stats = db.get_statistics()
    db.close()
    
    if stats['total_access_logs'] == 0:
        print("\n📦 Database is empty. Initializing with dummy data...")
        initialize_database_with_dummy_data()
    else:
        print(f"\n✅ Database already initialized with {stats['total_access_logs']} access logs")
    
    # Initialize security manager
    manager = MLSecurityManager()
    
    # Run demos
    demo_anomaly_detection(manager)
    demo_power_profiling(manager)
    demo_behavior_prediction(manager)
    demo_database_scan(manager)
    demo_alerts(manager)
    demo_system_status(manager)
    
    # Cleanup
    manager.close()
    
    print_header("DEMO COMPLETE")
    
    print("✅ Successfully demonstrated:")
    print("   • ML-based anomaly detection (Isolation Forest + LSTM)")
    print("   • Power consumption profiling (Autoencoder)")
    print("   • Contextual behavior prediction (Random Forest)")
    print("   • Database-driven architecture")
    print("   • Security alert system")
    print("   • System monitoring and statistics\n")
    
    print("🔒 Your IoT ecosystem is secured with ML!\n")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
