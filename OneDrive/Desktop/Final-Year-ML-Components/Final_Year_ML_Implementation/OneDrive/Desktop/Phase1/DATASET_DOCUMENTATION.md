# 📊 Complete Dataset Documentation

## Table of Contents
1. [Dataset Overview](#dataset-overview)
2. [Storage Architecture](#storage-architecture)
3. [Data Access Methods](#data-access-methods)
4. [Feature Specifications](#feature-specifications)
5. [Data Generation Process](#data-generation-process)
6. [Usage Examples](#usage-examples)

---

## 1. Dataset Overview

### What is the Dataset?

The dataset consists of **synthetic IoT security data** that simulates real-world smart home device behavior, including:
- **Access Logs**: User interactions with IoT devices
- **Power Consumption Logs**: Device power usage patterns
- **Behavior Logs**: Contextual device usage patterns

### Dataset Size

| Data Type | Records | Features | Size |
|-----------|---------|----------|------|
| Access Logs | 4,000 | 10 core + 3 tracking | ~500 KB |
| Power Logs | 18,000 | 13 core + 3 tracking | ~2.5 MB |
| Behavior Logs | 4,000 | 11 core + 2 tracking | ~600 KB |
| **Total** | **26,000** | **34 unique** | **~10 MB** |

### Anomaly Distribution

- **Access Logs**: 90% normal, 10% anomalous
- **Power Logs**: 95% normal, 5% anomalous
- **Behavior Logs**: Realistic user patterns with natural variations

---

## 2. Storage Architecture

### Database Type: SQLite

**Location**: `database/iot_data.db`

**Why SQLite?**
- ✅ Lightweight (no server required)
- ✅ Fast queries (<10ms)
- ✅ ACID compliant
- ✅ Single file storage
- ✅ Built into Python
- ✅ Perfect for ML prototyping

### Database Schema

```
iot_data.db
│
├── access_logs          (4,000 records)
├── power_logs           (18,000 records)
├── behavior_logs        (4,000 records)
├── alerts               (25+ records, dynamic)
├── users                (4 records)
└── devices              (6 records)
```

---

## 3. Data Access Methods

### Method 1: Direct Database Access

```python
from database.data_generator import IoTDatabase

# Initialize database connection
db = IoTDatabase()

# Access methods
access_logs = db.get_recent_access_logs(limit=100)
power_logs = db.get_recent_power_logs(device_id='smart_lock', limit=50)
behavior_logs = db.get_recent_behavior_logs(limit=100)
alerts = db.get_alerts(severity='CRITICAL', limit=10)
stats = db.get_statistics()

# Close connection
db.close()
```

### Method 2: Through ML Security Manager

```python
from database.ml_security_manager import MLSecurityManager

# Initialize (automatically connects to database)
manager = MLSecurityManager()

# Access through manager methods
status = manager.get_system_status()
alerts = manager.get_alerts(severity='MEDIUM')

# Scan database for anomalies
manager.scan_recent_access_logs(limit=100)
manager.scan_recent_power_logs(limit=100)

manager.close()
```

### Method 3: Raw SQL Queries

```python
import sqlite3

conn = sqlite3.connect('database/iot_data.db')
cursor = conn.cursor()

# Custom queries
cursor.execute("""
    SELECT * FROM access_logs 
    WHERE anomaly_detected = 1 
    ORDER BY timestamp DESC 
    LIMIT 10
""")

results = cursor.fetchall()
conn.close()
```

### Method 4: Pandas DataFrame

```python
import pandas as pd
import sqlite3

conn = sqlite3.connect('database/iot_data.db')

# Load as DataFrame
df = pd.read_sql_query("SELECT * FROM access_logs", conn)
print(df.head())
print(df.describe())

conn.close()
```

---

## 4. Feature Specifications

### 4.1 Access Logs (10 Core Features + 3 Tracking)

**Table**: `access_logs`

#### Core Features (Used by ML Models)

| Feature | Type | Description | Example | Range/Values |
|---------|------|-------------|---------|--------------|
| `timestamp` | TEXT | ISO 8601 datetime | `2024-11-13T19:30:00` | Last 30 days |
| `device_id` | TEXT | Device identifier | `smart_lock` | 6 devices |
| `user_id` | TEXT | User identifier | `Adish` | 4 users |
| `action` | TEXT | Action performed | `unlock` | 6 actions |
| `ip_address` | TEXT | IP address | `192.168.1.100` | Local/External |
| `location` | TEXT | Access location | `home` | home/remote/unknown_city |
| `access_count` | INTEGER | Number of accesses | `3` | 1-100 |
| `time_since_last` | INTEGER | Seconds since last | `3600` | 0-86400 |
| `duration` | INTEGER | Duration in seconds | `45` | 1-300 |
| `success` | BOOLEAN | Success flag | `True` | True/False |

#### Tracking Features (Not used by ML)

| Feature | Type | Description |
|---------|------|-------------|
| `id` | INTEGER | Auto-increment primary key |
| `anomaly_detected` | BOOLEAN | ML prediction result |
| `anomaly_confidence` | REAL | ML confidence score |

#### Devices
- `smart_lock` - Door lock
- `smart_light` - Lighting system
- `security_camera` - Camera
- `smart_tv` - Television
- `thermostat` - Temperature control
- `smart_speaker` - Voice assistant

#### Users
- `Adish` - Primary user
- `Guest001` - Guest user 1
- `Guest002` - Guest user 2
- `Family001` - Family member

#### Actions
- `unlock`, `lock` - Lock operations
- `view` - Camera viewing
- `control` - General control
- `power_on`, `power_off` - Power operations

#### Access Patterns

**Normal Access**:
- Time: 7 AM - 10 PM (weighted distribution)
- Location: `home` (75% probability)
- IP: `192.168.1.x` (local network)
- Access count: 1-10
- Success rate: 95%

**Anomalous Access** (10% of data):
- **Unusual Time**: 3 AM access
- **Unusual Location**: `unknown_city` with external IP `203.x.x.x`
- **High Frequency**: 50-100 access count

---

### 4.2 Power Logs (13 Core Features + 3 Tracking)

**Table**: `power_logs`

#### Core Features (Used by ML Models)

| Feature | Type | Description | Example | Unit | Range |
|---------|------|-------------|---------|------|-------|
| `timestamp` | TEXT | ISO 8601 datetime | `2024-11-13T14:00:00` | - | Last 30 days |
| `device_id` | TEXT | Device identifier | `smart_light` | - | 6 devices |
| `power_watts` | REAL | Power consumption | `12.5` | Watts | 0-200 |
| `voltage` | REAL | Voltage level | `120.2` | Volts | 90-140 |
| `current_amps` | REAL | Current draw | `0.104` | Amperes | 0-2 |
| `power_factor` | REAL | Power factor | `0.95` | - | 0.9-1.0 |
| `avg_power` | REAL | Average power | `12.0` | Watts | 0-200 |
| `power_variance` | REAL | Power variance | `1.5` | Watts | 0-50 |
| `peak_power` | REAL | Peak power | `15.0` | Watts | 0-250 |
| `device_state` | TEXT | Device state | `on` | - | on/off |
| `cpu_usage` | INTEGER | CPU usage | `15` | % | 0-100 |
| `network_activity` | INTEGER | Network traffic | `50` | KB/s | 0-1500 |
| `temperature` | REAL | Device temperature | `26.2` | °C | 15-50 |

#### Tracking Features

| Feature | Type | Description |
|---------|------|-------------|
| `id` | INTEGER | Auto-increment primary key |
| `anomaly_detected` | BOOLEAN | ML prediction result |
| `anomaly_type` | TEXT | Type of anomaly detected |

#### Device Power Profiles

| Device | Base Power | Active Power | Variance | Description |
|--------|-----------|--------------|----------|-------------|
| `smart_lock` | 5W | 15W | 2W | Low power, spikes when locking |
| `smart_light` | 10W | 60W | 5W | Variable based on brightness |
| `security_camera` | 8W | 12W | 1W | Constant, slight increase when recording |
| `smart_tv` | 2W | 150W | 20W | High power when active |
| `thermostat` | 3W | 8W | 1W | Low power, heating/cooling cycles |
| `smart_speaker` | 3W | 10W | 2W | Low power, spikes with audio |

#### Power Patterns

**Normal Power**:
- Day (7 AM - 10 PM): 70% probability device is active
- Night (10 PM - 7 AM): 20% probability device is active
- Voltage: 120V ± 2V (normal range)
- Temperature: 25°C ± 5°C

**Anomalous Power** (5% of data):

1. **Crypto Mining**:
   - Power: 2.5× normal
   - CPU: 85-100%
   - Network: 100-200 KB/s
   - Temperature: 40-50°C

2. **Botnet Activity**:
   - Power: 1.5× normal
   - CPU: 40-60%
   - Network: 800-1500 KB/s (very high!)
   - Temperature: 30-40°C

3. **Hardware Issue**:
   - Power: 0.5-1.5× normal (erratic)
   - Voltage: 90V or 140V (abnormal)
   - Power variance: 20-50W (high fluctuation)

---

### 4.3 Behavior Logs (11 Core Features + 2 Tracking)

**Table**: `behavior_logs`

#### Core Features (Used by ML Models)

| Feature | Type | Description | Example | Range/Values |
|---------|------|-------------|---------|--------------|
| `timestamp` | TEXT | ISO 8601 datetime | `2024-11-13T19:30:00` | Last 30 days |
| `user_id` | TEXT | User identifier | `Adish` | 4 users |
| `device_id` | TEXT | Device identifier | `smart_tv` | 6 devices |
| `device_state` | TEXT | Current state | `on` | on/off/active |
| `previous_state` | TEXT | Previous state | `off` | on/off |
| `time_since_last` | INTEGER | Seconds since last | `3600` | 0-7200 |
| `interactions_today` | INTEGER | Daily interaction count | `5` | 0-50 |
| `typical_usage_hour` | INTEGER | User's typical hour | `19` | 0-23 |
| `is_home` | BOOLEAN | User at home | `True` | True/False |
| `ambient_light` | INTEGER | Light level | `25` | 0-100 lux |
| `temperature` | REAL | Room temperature | `22.5` | 15-30°C |

#### Tracking Features

| Feature | Type | Description |
|---------|------|-------------|
| `id` | INTEGER | Auto-increment primary key |
| `anomaly_detected` | BOOLEAN | ML prediction result |

#### User Behavior Patterns

| User | Peak Hours | Preferred Devices | Typical Behavior |
|------|-----------|-------------------|------------------|
| `Adish` | 7-8 AM, 7-9 PM | smart_tv, smart_light | Morning routine, evening entertainment |
| `Guest001` | 10 AM - 2 PM | smart_lock | Daytime access |
| `Guest002` | 3-5 PM | smart_lock, smart_light | Afternoon access |
| `Family001` | 6-8 PM | smart_tv, thermostat | Evening family time |

#### Contextual Features

**Time-based**:
- Peak hours: 2× probability of usage
- Off-peak hours: 1× probability
- Weighted selection based on user patterns

**Environmental**:
- **Ambient Light**:
  - Day (6 AM - 8 PM): 20-80 lux
  - Night (8 PM - 6 AM): 0-30 lux
- **Temperature**: 20°C ± 3°C (normal distribution)
- **Home Presence**: 80% probability user is home

---

## 5. Data Generation Process

### 5.1 Generation Algorithm

```python
class DummyDataGenerator:
    def __init__(self, seed=42):
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
```

### 5.2 Access Log Generation

**Step 1**: Generate base logs
```python
for i in range(n_samples):
    # Weighted hour distribution (realistic patterns)
    hour = random.choices(range(24), weights=[...])
    
    # Random day within last 30 days
    timestamp = start_date + timedelta(days=random.randint(0, 29), hours=hour)
    
    # Calculate time_since_last from actual previous timestamp
    if i > 0:
        time_since_last = (timestamp - last_timestamp).total_seconds()
```

**Step 2**: Inject anomalies (10%)
```python
num_anomalies = int(n_samples * 0.1)
for _ in range(num_anomalies):
    anomaly_type = random.choice(['unusual_time', 'unusual_location', 'high_frequency'])
    # Modify specific records to create anomalies
```

### 5.3 Power Log Generation

**Step 1**: Device-specific profiles
```python
power_profiles = {
    'smart_lock': {'base': 5, 'active': 15, 'variance': 2},
    # ... for each device
}
```

**Step 2**: Time-based activity
```python
hour = timestamp.hour
is_active = random.random() < (0.7 if 7 <= hour <= 22 else 0.2)
base_power = profile['active'] if is_active else profile['base']
power_watts = base_power + np.random.normal(0, profile['variance'])
```

**Step 3**: Inject anomalies (5%)
```python
if anomaly_type == 'crypto_mining':
    logs[idx]['power_watts'] *= 2.5
    logs[idx]['cpu_usage'] = random.randint(85, 100)
```

### 5.4 Behavior Log Generation

**Step 1**: User-specific patterns
```python
user_patterns = {
    'Adish': {'peak_hours': [7, 8, 19, 20, 21], 'devices': ['smart_tv', 'smart_light']},
    # ... for each user
}
```

**Step 2**: Weighted hour selection
```python
hour = random.choices(
    range(24),
    weights=[2 if h in pattern['peak_hours'] else 1 for h in range(24)]
)[0]
```

**Step 3**: Calculate contextual features
```python
interactions_today = sum(1 for log in logs 
                        if log['user_id'] == user_id and 
                        same_date(log['timestamp'], timestamp))
```

---

## 6. Usage Examples

### Example 1: Load and Explore Data

```python
from database.data_generator import IoTDatabase
import pandas as pd

db = IoTDatabase()

# Load access logs
access_df = db.get_recent_access_logs(limit=1000)

# Basic statistics
print("Dataset Shape:", access_df.shape)
print("\nColumn Types:")
print(access_df.dtypes)

print("\nBasic Statistics:")
print(access_df.describe())

print("\nDevice Distribution:")
print(access_df['device_id'].value_counts())

print("\nLocation Distribution:")
print(access_df['location'].value_counts())

db.close()
```

### Example 2: Filter and Analyze

```python
# Get anomalous access logs only
anomalous = pd.read_sql_query("""
    SELECT * FROM access_logs 
    WHERE anomaly_detected = 1
    ORDER BY anomaly_confidence DESC
    LIMIT 50
""", db.conn)

print(f"Found {len(anomalous)} anomalies")
print(anomalous[['timestamp', 'device_id', 'location', 'anomaly_confidence']])
```

### Example 3: Device-Specific Analysis

```python
# Get power logs for specific device
smart_lock_power = db.get_recent_power_logs(device_id='smart_lock', limit=500)

print(f"Smart Lock Power Stats:")
print(f"  Average: {smart_lock_power['power_watts'].mean():.2f}W")
print(f"  Max: {smart_lock_power['power_watts'].max():.2f}W")
print(f"  Min: {smart_lock_power['power_watts'].min():.2f}W")
print(f"  Std Dev: {smart_lock_power['power_watts'].std():.2f}W")
```

### Example 4: Time-Series Analysis

```python
import matplotlib.pyplot as plt

# Get power consumption over time
power_df = db.get_recent_power_logs(limit=1000)
power_df['timestamp'] = pd.to_datetime(power_df['timestamp'])
power_df = power_df.sort_values('timestamp')

# Plot for each device
for device in power_df['device_id'].unique():
    device_data = power_df[power_df['device_id'] == device]
    plt.plot(device_data['timestamp'], device_data['power_watts'], 
             label=device, alpha=0.7)

plt.xlabel('Time')
plt.ylabel('Power (Watts)')
plt.title('Power Consumption Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### Example 5: Export Data

```python
# Export to CSV
access_df = db.get_recent_access_logs(limit=10000)
access_df.to_csv('access_logs_export.csv', index=False)

# Export to JSON
access_df.to_json('access_logs_export.json', orient='records', indent=2)

# Export specific columns
access_df[['timestamp', 'device_id', 'user_id', 'action']].to_csv(
    'access_logs_minimal.csv', index=False
)
```

---

## 7. Data Quality & Validation

### Data Integrity Checks

```python
from database.data_generator import IoTDatabase

db = IoTDatabase()
stats = db.get_statistics()

print("Data Quality Report:")
print(f"✓ Total Access Logs: {stats['total_access_logs']}")
print(f"✓ Total Power Logs: {stats['total_power_logs']}")
print(f"✓ Total Behavior Logs: {stats['total_behavior_logs']}")
print(f"✓ Total Users: {stats['total_users']}")
print(f"✓ Total Devices: {stats['total_devices']}")
print(f"✓ Anomalies Detected: {stats['anomalies_detected']}")
print(f"✓ Power Anomalies: {stats['power_anomalies']}")

db.close()
```

### Expected Distributions

| Metric | Expected | Actual Check |
|--------|----------|--------------|
| Access anomaly rate | ~10% | `anomalies / total_access` |
| Power anomaly rate | ~5% | `power_anomalies / total_power` |
| Success rate | ~95% | Count `success=True` |
| Home location | ~75% | Count `location='home'` |
| Peak hour usage | Higher | Group by hour |

---

## 8. Database Schema Details

### Complete Table Structures

```sql
-- Access Logs Table
CREATE TABLE access_logs (
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
);

-- Power Logs Table
CREATE TABLE power_logs (
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
);

-- Behavior Logs Table
CREATE TABLE behavior_logs (
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
);

-- Alerts Table
CREATE TABLE alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    alert_type TEXT,
    severity TEXT,
    device_id TEXT,
    user_id TEXT,
    description TEXT,
    data TEXT
);

-- Users Table
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT UNIQUE,
    name TEXT,
    created_at TEXT
);

-- Devices Table
CREATE TABLE devices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT UNIQUE,
    device_type TEXT,
    manufacturer TEXT,
    firmware_version TEXT,
    created_at TEXT
);
```

---

## Summary

### Key Points

✅ **Storage**: SQLite database (`database/iot_data.db`)  
✅ **Size**: ~10 MB, 26,000 records  
✅ **Access**: Direct SQL, Pandas, or through ML Security Manager  
✅ **Features**: 34 unique features across 3 data types  
✅ **Quality**: Realistic distributions with proper anomaly injection  
✅ **Purpose**: ML training and security analysis  

### Feature Count Summary

- **Access Logs**: 10 core + 3 tracking = **13 total**
- **Power Logs**: 13 core + 3 tracking = **16 total**
- **Behavior Logs**: 11 core + 2 tracking = **13 total**
- **Total Unique Features**: **34 features**

All features are carefully designed to match real-world IoT security scenarios and are used by the ML models for anomaly detection, power profiling, and behavior prediction.
