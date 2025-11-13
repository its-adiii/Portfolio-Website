"""
ML-focused Security Manager
Works with database instead of blockchain/IoT devices
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.data_generator import IoTDatabase
from ml_models.anomaly_detection import EnsembleAnomalyDetector
from ml_models.power_profiling import PowerProfiler
from ml_models.behavior_prediction import ContextualBehaviorSystem


class MLSecurityManager:
    """Security manager focused on ML functionality with database backend"""
    
    def __init__(self, db_path='database/iot_data.db'):
        print("🔐 Initializing ML Security Manager...")
        
        # Initialize database
        self.db = IoTDatabase(db_path)
        
        # Initialize ML models
        self.anomaly_detector = None
        self.power_profiler = None
        self.behavior_system = None
        
        # Load ML models
        self.load_ml_models()
        
        print("✅ ML Security Manager initialized\n")
    
    def load_ml_models(self):
        """Load all ML models"""
        try:
            print("  📦 Loading ML models...")
            
            # Load anomaly detector
            self.anomaly_detector = EnsembleAnomalyDetector()
            self.anomaly_detector.load('models/anomaly_detection')
            print("    ✓ Anomaly detector loaded")
            
            # Load power profiler
            self.power_profiler = PowerProfiler()
            self.power_profiler.load_profiles('models/power_profiles')
            print("    ✓ Power profiler loaded")
            
            # Load behavior system
            self.behavior_system = ContextualBehaviorSystem()
            self.behavior_system.load('models/behavior_prediction')
            print("    ✓ Behavior system loaded")
            
        except Exception as e:
            print(f"  ⚠️  Warning: Could not load ML models: {e}")
            print("  ℹ️  Run 'python ml_models/model_trainer.py' to train models first")
    
    def analyze_access_log(self, access_data):
        """Analyze access log for anomalies"""
        if not self.anomaly_detector:
            return {'is_anomaly': False, 'confidence': 0.0, 'reason': 'Model not loaded'}
        
        # Detect anomaly (pass the dict directly, not numpy array)
        result = self.anomaly_detector.predict(access_data)
        
        # Store result in database
        confidence = result.get('combined_confidence', 0.0)
        if result['is_anomaly']:
            self.db.insert_alert(
                alert_type='access_anomaly',
                severity='MEDIUM' if confidence > 0.7 else 'LOW',
                device_id=access_data.get('device_id'),
                user_id=access_data.get('user_id'),
                description=f"Anomalous access detected (confidence: {confidence:.2f})",
                data=access_data
            )
        
        # Return simplified result
        return {
            'is_anomaly': result['is_anomaly'],
            'confidence': confidence
        }
    
    def analyze_power_consumption(self, power_data):
        """Analyze power consumption for anomalies"""
        if not self.power_profiler:
            return {'is_anomaly': False, 'anomaly_type': 'normal', 'reason': 'Model not loaded'}
        
        device_id = power_data.get('device_id')
        
        # Analyze power
        result = self.power_profiler.check_power_consumption(device_id, power_data)
        
        # Store result in database
        if result['is_anomaly']:
            severity = 'CRITICAL' if result['anomaly_type'] == 'crypto_mining' else 'MEDIUM'
            self.db.insert_alert(
                alert_type='power_anomaly',
                severity=severity,
                device_id=device_id,
                description=f"Power anomaly detected: {result['anomaly_type']}",
                data=power_data
            )
        
        return result
    
    def predict_device_behavior(self, context_data):
        """Predict expected device behavior"""
        if not self.behavior_system:
            return {'predicted_state': 'unknown', 'confidence': 0.0, 'reason': 'Model not loaded'}
        
        # Predict behavior using the predictor
        result = self.behavior_system.predictor.predict(context_data)
        
        return result
    
    def detect_behavioral_anomaly(self, context_data, actual_state):
        """Detect if actual behavior differs from expected"""
        if not self.behavior_system:
            return {'is_anomaly': False, 'reason': 'Model not loaded'}
        
        # Detect anomaly
        result = self.behavior_system.check_behavior(context_data, actual_state)
        
        # Store result in database
        if result['is_anomaly']:
            self.db.insert_alert(
                alert_type='behavior_anomaly',
                severity='LOW',
                device_id=context_data.get('device_id'),
                user_id=context_data.get('user_id'),
                description=f"Unexpected behavior: expected {result['expected_state']}, got {actual_state}",
                data=context_data
            )
        
        return result
    
    def _prepare_access_features(self, access_data):
        """Prepare access log features for ML model"""
        # Extract timestamp features
        timestamp = pd.to_datetime(access_data.get('timestamp', datetime.now().isoformat()))
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Encode IP address
        ip = access_data.get('ip_address', '192.168.1.100')
        ip_parts = ip.split('.')
        ip_encoded = int(ip_parts[0]) * 1000 + int(ip_parts[1]) * 100 + int(ip_parts[2]) * 10 + int(ip_parts[3])
        
        # Encode location
        location_map = {'home': 0, 'remote': 1, 'unknown_city': 2}
        location_encoded = location_map.get(access_data.get('location', 'home'), 0)
        
        # Create feature vector (9 features)
        features = np.array([
            hour,
            day_of_week,
            ip_encoded,
            location_encoded,
            access_data.get('access_count', 1),
            access_data.get('time_since_last', 3600),
            access_data.get('duration', 60),
            1 if access_data.get('success', True) else 0,
            1 if access_data.get('location') == 'home' else 0
        ]).reshape(1, -1)
        
        return features
    
    def scan_recent_access_logs(self, limit=100):
        """Scan recent access logs for anomalies"""
        print(f"\n🔍 Scanning {limit} recent access logs...")
        
        logs = self.db.get_recent_access_logs(limit)
        anomalies_found = 0
        
        for idx, log in logs.iterrows():
            result = self.analyze_access_log(log.to_dict())
            if result['is_anomaly']:
                anomalies_found += 1
                print(f"  ⚠️  Anomaly detected: {log['device_id']} at {log['timestamp']}")
                print(f"      Confidence: {result['confidence']:.2f}")
        
        print(f"\n  Total anomalies found: {anomalies_found}/{len(logs)}")
        return anomalies_found
    
    def scan_recent_power_logs(self, limit=100):
        """Scan recent power logs for anomalies"""
        print(f"\n⚡ Scanning {limit} recent power logs...")
        
        logs = self.db.get_recent_power_logs(limit=limit)
        anomalies_found = 0
        
        for idx, log in logs.iterrows():
            result = self.analyze_power_consumption(log.to_dict())
            if result['is_anomaly']:
                anomalies_found += 1
                print(f"  ⚠️  Power anomaly: {log['device_id']} - {result['anomaly_type']}")
        
        print(f"\n  Total power anomalies found: {anomalies_found}/{len(logs)}")
        return anomalies_found
    
    def get_system_status(self):
        """Get overall system status"""
        stats = self.db.get_statistics()
        
        status = {
            'database': {
                'access_logs': stats['total_access_logs'],
                'power_logs': stats['total_power_logs'],
                'behavior_logs': stats['total_behavior_logs'],
                'alerts': stats['total_alerts']
            },
            'entities': {
                'users': stats['total_users'],
                'devices': stats['total_devices']
            },
            'ml_models': {
                'anomaly_detector': 'Loaded' if self.anomaly_detector else 'Not loaded',
                'power_profiler': 'Loaded' if self.power_profiler else 'Not loaded',
                'behavior_system': 'Loaded' if self.behavior_system else 'Not loaded'
            },
            'anomalies': {
                'access_anomalies': stats['anomalies_detected'],
                'power_anomalies': stats['power_anomalies']
            }
        }
        
        return status
    
    def get_alerts(self, severity=None, limit=50):
        """Get security alerts"""
        return self.db.get_alerts(severity=severity, limit=limit)
    
    def close(self):
        """Close database connection"""
        self.db.close()


if __name__ == '__main__':
    # Test the manager
    manager = MLSecurityManager()
    
    # Test access log analysis
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
    
    print("\n🧪 Testing access log analysis...")
    result = manager.analyze_access_log(test_access)
    print(f"  Result: {result}")
    
    # Test power analysis
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
    
    print("\n🧪 Testing power analysis...")
    result = manager.analyze_power_consumption(test_power)
    print(f"  Result: {result}")
    
    # Get system status
    print("\n📊 System Status:")
    status = manager.get_system_status()
    for category, data in status.items():
        print(f"\n  {category.upper()}:")
        for key, value in data.items():
            print(f"    {key}: {value}")
    
    manager.close()
