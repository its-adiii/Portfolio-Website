"""
Database Module with Dummy Data Generation
Replaces IoT devices and blockchain with SQLite database
"""

import sqlite3
import json
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path


class DummyDataGenerator:
    """Generate realistic dummy data matching original features"""
    
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
        self.devices = ['smart_lock', 'smart_light', 'security_camera', 
                       'smart_tv', 'thermostat', 'smart_speaker']
        self.users = ['Adish', 'Guest001', 'Guest002', 'Family001']
        self.locations = ['home', 'remote', 'unknown_city']
        # Use same generic actions as original
        self.actions = ['unlock', 'lock', 'view', 'control', 'power_on', 'power_off']
        
        # Device power profiles
        self.power_profiles = {
            'smart_lock': {'base': 5, 'active': 15, 'variance': 2},
            'smart_light': {'base': 10, 'active': 60, 'variance': 5},
            'security_camera': {'base': 8, 'active': 12, 'variance': 1},
            'smart_tv': {'base': 2, 'active': 150, 'variance': 20},
            'thermostat': {'base': 3, 'active': 8, 'variance': 1},
            'smart_speaker': {'base': 3, 'active': 10, 'variance': 2}
        }
    
    def generate_access_logs(self, n_samples=2000):
        """Generate access log data (9 features) - EXACT match with original"""
        logs = []
        start_date = datetime.now() - timedelta(days=30)
        
        for i in range(n_samples):
            # Normal pattern: most access during 7AM-10PM (same weights as original)
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
            
            # Calculate time since last access (same as original)
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
        
        # Add some anomalies (10%) - same as original
        num_anomalies = int(n_samples * 0.1)
        for _ in range(num_anomalies):
            idx = random.randint(0, n_samples - 1)
            
            # Create anomalous access (same types as original)
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
        
        return pd.DataFrame(logs)
    
    def generate_power_logs(self, device_id, n_samples=1500):
        """Generate power consumption data (12 features) - EXACT match with original"""
        logs = []
        start_date = datetime.now() - timedelta(days=30)
        
        # Device-specific power profiles (same as original)
        profile = self.power_profiles.get(device_id, {'base': 5, 'active': 20, 'variance': 3})
        
        for i in range(n_samples):
            timestamp = start_date + timedelta(
                days=random.randint(0, 29),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            # Device is more likely to be active during day (same as original)
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
        
        # Add power anomalies (5%) - same as original
        num_anomalies = int(n_samples * 0.05)
        for _ in range(num_anomalies):
            idx = random.randint(0, n_samples - 1)
            
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
        
        return pd.DataFrame(logs)
    
    def generate_behavior_logs(self, n_samples=2000):
        """Generate behavior data (10 features) - EXACT match with original"""
        logs = []
        start_date = datetime.now() - timedelta(days=30)
        
        # User-specific patterns (same as original)
        user_patterns = {
            'Adish': {'peak_hours': [7, 8, 19, 20, 21], 'devices': ['smart_tv', 'smart_light']},
            'Guest001': {'peak_hours': [10, 11, 12, 13, 14], 'devices': ['smart_lock']},
            'Guest002': {'peak_hours': [15, 16, 17], 'devices': ['smart_lock', 'smart_light']},
            'Family001': {'peak_hours': [18, 19, 20], 'devices': ['smart_tv', 'thermostat']}
        }
        
        for i in range(n_samples):
            user_id = random.choice(self.users)
            pattern = user_patterns.get(user_id, {'peak_hours': [12], 'devices': ['smart_lock']})
            
            # More likely during peak hours (same as original)
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
            
            # Calculate interactions today (same as original)
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
        
        return pd.DataFrame(logs)


class IoTDatabase:
    """SQLite database to store all IoT data"""
    
    def __init__(self, db_path='database/iot_data.db'):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.create_tables()
    
    def create_tables(self):
        """Create database tables"""
        cursor = self.conn.cursor()
        
        # Access logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS access_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                device_id TEXT,
                user_id TEXT,
                action TEXT,
                ip_address TEXT,
                location TEXT,
                access_count INTEGER,
                time_since_last INTEGER,
                duration INTEGER,
                success BOOLEAN,
                anomaly_detected BOOLEAN DEFAULT 0,
                anomaly_confidence REAL DEFAULT 0.0
            )
        ''')
        
        # Power logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS power_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                device_id TEXT,
                power_watts REAL,
                voltage REAL,
                current_amps REAL,
                power_factor REAL,
                avg_power REAL,
                power_variance REAL,
                peak_power REAL,
                device_state TEXT,
                cpu_usage INTEGER,
                network_activity INTEGER,
                temperature REAL,
                anomaly_detected BOOLEAN DEFAULT 0,
                anomaly_type TEXT
            )
        ''')
        
        # Behavior logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS behavior_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user_id TEXT,
                device_id TEXT,
                device_state TEXT,
                previous_state TEXT,
                time_since_last INTEGER,
                interactions_today INTEGER,
                typical_usage_hour INTEGER,
                is_home BOOLEAN,
                ambient_light INTEGER,
                temperature REAL,
                anomaly_detected BOOLEAN DEFAULT 0
            )
        ''')
        
        # Alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                alert_type TEXT,
                severity TEXT,
                device_id TEXT,
                user_id TEXT,
                description TEXT,
                data TEXT
            )
        ''')
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE,
                name TEXT,
                created_at TEXT
            )
        ''')
        
        # Devices table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS devices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT UNIQUE,
                device_type TEXT,
                manufacturer TEXT,
                firmware_version TEXT,
                created_at TEXT
            )
        ''')
        
        self.conn.commit()
    
    def insert_access_logs(self, df):
        """Insert access logs from DataFrame"""
        df.to_sql('access_logs', self.conn, if_exists='append', index=False)
    
    def insert_power_logs(self, df):
        """Insert power logs from DataFrame"""
        df.to_sql('power_logs', self.conn, if_exists='append', index=False)
    
    def insert_behavior_logs(self, df):
        """Insert behavior logs from DataFrame"""
        df.to_sql('behavior_logs', self.conn, if_exists='append', index=False)
    
    def insert_alert(self, alert_type, severity, device_id=None, user_id=None, description='', data=None):
        """Insert security alert"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO alerts (timestamp, alert_type, severity, device_id, user_id, description, data)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (datetime.now().isoformat(), alert_type, severity, device_id, user_id, 
              description, json.dumps(data) if data else None))
        self.conn.commit()
    
    def register_user(self, user_id, name):
        """Register a user"""
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO users (user_id, name, created_at)
                VALUES (?, ?, ?)
            ''', (user_id, name, datetime.now().isoformat()))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # User already exists
    
    def register_device(self, device_id, device_type, manufacturer='Unknown', firmware_version='1.0.0'):
        """Register a device"""
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO devices (device_id, device_type, manufacturer, firmware_version, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (device_id, device_type, manufacturer, firmware_version, datetime.now().isoformat()))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # Device already exists
    
    def get_recent_access_logs(self, limit=100):
        """Get recent access logs"""
        return pd.read_sql_query(f'SELECT * FROM access_logs ORDER BY timestamp DESC LIMIT {limit}', self.conn)
    
    def get_recent_power_logs(self, device_id=None, limit=100):
        """Get recent power logs"""
        if device_id:
            return pd.read_sql_query(
                f'SELECT * FROM power_logs WHERE device_id=? ORDER BY timestamp DESC LIMIT {limit}',
                self.conn, params=(device_id,))
        return pd.read_sql_query(f'SELECT * FROM power_logs ORDER BY timestamp DESC LIMIT {limit}', self.conn)
    
    def get_recent_behavior_logs(self, limit=100):
        """Get recent behavior logs"""
        return pd.read_sql_query(f'SELECT * FROM behavior_logs ORDER BY timestamp DESC LIMIT {limit}', self.conn)
    
    def get_alerts(self, severity=None, limit=50):
        """Get security alerts"""
        if severity:
            return pd.read_sql_query(
                f'SELECT * FROM alerts WHERE severity=? ORDER BY timestamp DESC LIMIT {limit}',
                self.conn, params=(severity,))
        return pd.read_sql_query(f'SELECT * FROM alerts ORDER BY timestamp DESC LIMIT {limit}', self.conn)
    
    def get_statistics(self):
        """Get database statistics"""
        cursor = self.conn.cursor()
        
        stats = {}
        stats['total_access_logs'] = cursor.execute('SELECT COUNT(*) FROM access_logs').fetchone()[0]
        stats['total_power_logs'] = cursor.execute('SELECT COUNT(*) FROM power_logs').fetchone()[0]
        stats['total_behavior_logs'] = cursor.execute('SELECT COUNT(*) FROM behavior_logs').fetchone()[0]
        stats['total_alerts'] = cursor.execute('SELECT COUNT(*) FROM alerts').fetchone()[0]
        stats['total_users'] = cursor.execute('SELECT COUNT(*) FROM users').fetchone()[0]
        stats['total_devices'] = cursor.execute('SELECT COUNT(*) FROM devices').fetchone()[0]
        
        stats['anomalies_detected'] = cursor.execute(
            'SELECT COUNT(*) FROM access_logs WHERE anomaly_detected=1').fetchone()[0]
        stats['power_anomalies'] = cursor.execute(
            'SELECT COUNT(*) FROM power_logs WHERE anomaly_detected=1').fetchone()[0]
        
        return stats
    
    def close(self):
        """Close database connection"""
        self.conn.close()


def initialize_database_with_dummy_data():
    """Initialize database with dummy data"""
    print("🗄️  Initializing database with dummy data...")
    
    # Create database
    db = IoTDatabase()
    
    # Generate dummy data
    generator = DummyDataGenerator(seed=42)
    
    print("  📊 Generating access logs...")
    access_logs = generator.generate_access_logs(2000)
    db.insert_access_logs(access_logs)
    
    print("  ⚡ Generating power logs...")
    for device in generator.devices:
        power_logs = generator.generate_power_logs(device, 1500)
        db.insert_power_logs(power_logs)
    
    print("  🧠 Generating behavior logs...")
    behavior_logs = generator.generate_behavior_logs(2000)
    db.insert_behavior_logs(behavior_logs)
    
    print("  👥 Registering users...")
    for user in generator.users:
        db.register_user(user, user)
    
    print("  📱 Registering devices...")
    for device in generator.devices:
        db.register_device(device, device, 'TestManufacturer', '1.0.0')
    
    stats = db.get_statistics()
    print(f"\n✅ Database initialized successfully!")
    print(f"  Access logs: {stats['total_access_logs']}")
    print(f"  Power logs: {stats['total_power_logs']}")
    print(f"  Behavior logs: {stats['total_behavior_logs']}")
    print(f"  Users: {stats['total_users']}")
    print(f"  Devices: {stats['total_devices']}")
    
    db.close()
    return True


if __name__ == '__main__':
    initialize_database_with_dummy_data()
