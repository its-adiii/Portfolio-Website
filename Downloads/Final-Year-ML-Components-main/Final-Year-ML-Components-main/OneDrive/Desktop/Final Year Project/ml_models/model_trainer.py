"""
ML Model Training Utilities
Generates synthetic training data and trains all models
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_models.anomaly_detection import EnsembleAnomalyDetector
from ml_models.power_profiling import PowerProfiler
from ml_models.behavior_prediction import ContextualBehaviorSystem


class SyntheticDataGenerator:
    """Generate synthetic IoT data for training"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        self.devices = ['smart_lock', 'smart_light', 'security_camera', 
                       'smart_tv', 'thermostat', 'smart_speaker']
        self.users = ['Adish', 'Guest001', 'Guest002', 'Family001']
        self.actions = ['unlock', 'lock', 'view', 'control', 'power_on', 'power_off']
    
    def generate_access_logs(self, num_samples: int = 1000) -> list:
        """Generate synthetic access logs for anomaly detection"""
        logs = []
        start_date = datetime.now() - timedelta(days=30)
        
        for i in range(num_samples):
            # Normal pattern: most access during 7AM-10PM
            hour = random.choices(
                range(24),
                weights=[1, 1, 1, 1, 1, 2, 5, 8, 8, 8, 6, 6, 
                        6, 6, 6, 7, 8, 9, 9, 8, 7, 5, 3, 2]
            )[0]
            
            timestamp = start_date + timedelta(
                days=random.randint(0, 29),
                hours=hour,
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59)
            )
            
            # Calculate time since last access
            if i > 0:
                last_timestamp = datetime.fromisoformat(logs[-1]['timestamp'])
                time_since_last = (timestamp - last_timestamp).total_seconds()
            else:
                time_since_last = 0
            
            log = {
                'timestamp': timestamp.isoformat(),
                'device_id': random.choice(self.devices),
                'user_id': random.choice(self.users),
                'action': random.choice(self.actions),
                'ip_address': f"192.168.1.{random.randint(1, 254)}",
                'location': random.choice(['home', 'home', 'home', 'remote']),
                'access_count': random.randint(1, 10),
                'time_since_last': time_since_last,
                'duration': random.randint(1, 300),
                'success': random.random() > 0.05  # 95% success rate
            }
            
            logs.append(log)
        
        # Add some anomalies (10%)
        num_anomalies = int(num_samples * 0.1)
        for _ in range(num_anomalies):
            idx = random.randint(0, num_samples - 1)
            
            # Create anomalous access
            anomaly_type = random.choice(['unusual_time', 'unusual_location', 'high_frequency'])
            
            if anomaly_type == 'unusual_time':
                # Access at 3 AM
                timestamp = start_date + timedelta(
                    days=random.randint(0, 29),
                    hours=3,
                    minutes=random.randint(0, 59)
                )
                logs[idx]['timestamp'] = timestamp.isoformat()
            
            elif anomaly_type == 'unusual_location':
                logs[idx]['location'] = 'unknown_city'
                logs[idx]['ip_address'] = f"203.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"
            
            elif anomaly_type == 'high_frequency':
                logs[idx]['access_count'] = random.randint(50, 100)
        
        return logs
    
    def generate_power_logs(self, device_id: str, num_samples: int = 1000) -> list:
        """Generate synthetic power consumption logs"""
        logs = []
        start_date = datetime.now() - timedelta(days=30)
        
        # Device-specific power profiles
        power_profiles = {
            'smart_lock': {'base': 5, 'active': 15, 'variance': 2},
            'smart_light': {'base': 10, 'active': 60, 'variance': 5},
            'security_camera': {'base': 8, 'active': 12, 'variance': 1},
            'smart_tv': {'base': 2, 'active': 150, 'variance': 20},
            'thermostat': {'base': 3, 'active': 8, 'variance': 1},
            'smart_speaker': {'base': 3, 'active': 10, 'variance': 2}
        }
        
        profile = power_profiles.get(device_id, {'base': 5, 'active': 20, 'variance': 3})
        
        for i in range(num_samples):
            timestamp = start_date + timedelta(
                days=random.randint(0, 29),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            # Device is more likely to be active during day
            hour = timestamp.hour
            is_active = random.random() < (0.7 if 7 <= hour <= 22 else 0.2)
            
            base_power = profile['active'] if is_active else profile['base']
            power_watts = base_power + np.random.normal(0, profile['variance'])
            power_watts = max(0, power_watts)
            
            voltage = 120 + np.random.normal(0, 2)
            current_amps = power_watts / voltage
            
            log = {
                'timestamp': timestamp.isoformat(),
                'device_id': device_id,
                'power_watts': power_watts,
                'voltage': voltage,
                'current_amps': current_amps,
                'power_factor': 0.95 + np.random.normal(0, 0.02),
                'avg_power': power_watts * (0.9 + np.random.random() * 0.2),
                'power_variance': np.random.uniform(0, 5),
                'peak_power': power_watts * (1.1 + np.random.random() * 0.3),
                'device_state': 'on' if is_active else 'off',
                'cpu_usage': random.randint(10, 40) if is_active else random.randint(0, 10),
                'network_activity': random.randint(50, 200) if is_active else random.randint(0, 20),
                'temperature': 25 + np.random.normal(0, 5)
            }
            
            logs.append(log)
        
        # Add power anomalies (5%)
        num_anomalies = int(num_samples * 0.05)
        for _ in range(num_anomalies):
            idx = random.randint(0, num_samples - 1)
            
            anomaly_type = random.choice(['crypto_mining', 'botnet', 'hardware_issue'])
            
            if anomaly_type == 'crypto_mining':
                logs[idx]['power_watts'] *= 2.5
                logs[idx]['cpu_usage'] = random.randint(85, 100)
            
            elif anomaly_type == 'botnet':
                logs[idx]['network_activity'] = random.randint(800, 1500)
                logs[idx]['power_watts'] *= 1.5
            
            elif anomaly_type == 'hardware_issue':
                logs[idx]['voltage'] = random.choice([90, 140])
                logs[idx]['power_variance'] = random.uniform(20, 50)
        
        return logs
    
    def generate_behavior_logs(self, num_samples: int = 1000) -> list:
        """Generate synthetic behavior logs for pattern learning"""
        logs = []
        start_date = datetime.now() - timedelta(days=30)
        
        # User-specific patterns
        user_patterns = {
            'Adish': {'peak_hours': [7, 8, 19, 20, 21], 'devices': ['smart_tv', 'smart_light']},
            'Guest001': {'peak_hours': [10, 11, 12, 13, 14], 'devices': ['smart_lock']},
            'Guest002': {'peak_hours': [15, 16, 17], 'devices': ['smart_lock', 'smart_light']},
            'Family001': {'peak_hours': [18, 19, 20], 'devices': ['smart_tv', 'thermostat']}
        }
        
        for i in range(num_samples):
            user_id = random.choice(self.users)
            pattern = user_patterns.get(user_id, {'peak_hours': [12], 'devices': ['smart_lock']})
            
            # More likely during peak hours
            hour = random.choices(
                range(24),
                weights=[2 if h in pattern['peak_hours'] else 1 for h in range(24)]
            )[0]
            
            timestamp = start_date + timedelta(
                days=random.randint(0, 29),
                hours=hour,
                minutes=random.randint(0, 59)
            )
            
            device_id = random.choice(pattern['devices'])
            
            # Calculate interactions today
            interactions_today = sum(1 for log in logs 
                                   if log['user_id'] == user_id and 
                                   datetime.fromisoformat(log['timestamp']).date() == timestamp.date())
            
            log = {
                'timestamp': timestamp.isoformat(),
                'user_id': user_id,
                'device_id': device_id,
                'device_state': random.choice(['on', 'off', 'active']),
                'previous_state': random.choice(['on', 'off']),
                'time_since_last': random.randint(0, 7200),
                'interactions_today': interactions_today,
                'typical_usage_hour': random.choice(pattern['peak_hours']),
                'is_home': random.random() > 0.2,
                'ambient_light': random.randint(20, 80) if 6 <= hour <= 20 else random.randint(0, 30),
                'temperature': 20 + np.random.normal(0, 3)
            }
            
            logs.append(log)
        
        return logs


def train_all_models(output_dir: str = 'models'):
    """Train all ML models with synthetic data"""
    print("=" * 60)
    print("IoT Security ML Model Training")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize data generator
    print("\n📊 Generating synthetic training data...")
    generator = SyntheticDataGenerator()
    
    # Generate data
    access_logs = generator.generate_access_logs(num_samples=2000)
    behavior_logs = generator.generate_behavior_logs(num_samples=2000)
    
    print(f"✓ Generated {len(access_logs)} access logs")
    print(f"✓ Generated {len(behavior_logs)} behavior logs")
    
    # Train Anomaly Detection Models
    print("\n" + "=" * 60)
    print("1. Training Anomaly Detection Models")
    print("=" * 60)
    
    anomaly_detector = EnsembleAnomalyDetector()
    anomaly_detector.train(access_logs)
    anomaly_detector.save(os.path.join(output_dir, 'anomaly_detection'))
    
    # Train Power Profiling Models
    print("\n" + "=" * 60)
    print("2. Training Power Profiling Models")
    print("=" * 60)
    
    power_profiler = PowerProfiler()
    
    for device in generator.devices:
        print(f"\n  Training profile for {device}...")
        power_logs = generator.generate_power_logs(device, num_samples=1500)
        power_profiler.create_profile(device, power_logs)
    
    power_profiler.save_profiles(os.path.join(output_dir, 'power_profiles'))
    
    # Train Behavior Prediction Models
    print("\n" + "=" * 60)
    print("3. Training Behavior Prediction Models")
    print("=" * 60)
    
    behavior_system = ContextualBehaviorSystem()
    behavior_system.train(behavior_logs)
    behavior_system.save(os.path.join(output_dir, 'behavior_prediction'))
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"\nModels saved to: {output_dir}/")
    print("\nTrained models:")
    print("  ✓ Isolation Forest (Anomaly Detection)")
    print("  ✓ LSTM (Temporal Anomaly Detection)")
    print("  ✓ Autoencoder (Power Profiling)")
    print("  ✓ Random Forest (Behavior Prediction)")
    print("  ✓ User Pattern Analyzer")
    print(f"\nTotal devices profiled: {len(generator.devices)}")
    print(f"Total users analyzed: {len(generator.users)}")
    
    return {
        'anomaly_detector': anomaly_detector,
        'power_profiler': power_profiler,
        'behavior_system': behavior_system
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train IoT Security ML Models')
    parser.add_argument('--output', '-o', default='models', 
                       help='Output directory for trained models')
    
    args = parser.parse_args()
    
    train_all_models(output_dir=args.output)
