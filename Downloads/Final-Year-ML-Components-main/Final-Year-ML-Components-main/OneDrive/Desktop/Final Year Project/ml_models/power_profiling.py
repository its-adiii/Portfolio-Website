"""
Power Consumption Profiling using Autoencoder
Detects anomalies in device power consumption patterns
Identifies malware, hijacked devices, or hardware issues
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import os


class PowerConsumptionAutoencoder:
    """
    Autoencoder for power consumption anomaly detection
    
    Detects:
    - Crypto mining malware (unusual power spikes)
    - Botnet activity (abnormal network-related power usage)
    - Hardware malfunctions (irregular power patterns)
    - Unauthorized device usage
    """
    
    def __init__(self, encoding_dim: int = 8):
        """
        Args:
            encoding_dim: Dimension of the encoded representation
        """
        self.encoding_dim = encoding_dim
        self.model = None
        self.encoder = None
        self.decoder = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.is_trained = False
        self.feature_names = []
    
    def extract_features(self, power_logs: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extract power consumption features
        
        Features:
        - Current power draw (watts)
        - Voltage
        - Current (amperes)
        - Power factor
        - Average power (last N readings)
        - Power variance
        - Peak power
        - Time of day
        - Device state
        """
        features = []
        
        for log in power_logs:
            timestamp = datetime.fromisoformat(log.get('timestamp', datetime.now().isoformat()))
            
            feature_vector = [
                log.get('power_watts', 0),
                log.get('voltage', 120),
                log.get('current_amps', 0),
                log.get('power_factor', 1.0),
                log.get('avg_power', 0),
                log.get('power_variance', 0),
                log.get('peak_power', 0),
                timestamp.hour / 24.0,  # Normalized hour
                int(log.get('device_state', 'off') == 'on'),
                log.get('cpu_usage', 0) / 100.0,  # Normalized CPU usage
                log.get('network_activity', 0) / 1000.0,  # Normalized KB/s
                log.get('temperature', 25) / 100.0,  # Normalized temperature
            ]
            
            features.append(feature_vector)
        
        self.feature_names = [
            'power_watts', 'voltage', 'current_amps', 'power_factor',
            'avg_power', 'power_variance', 'peak_power', 'hour_normalized',
            'device_state', 'cpu_usage', 'network_activity', 'temperature'
        ]
        
        return np.array(features)
    
    def _build_model(self, input_dim: int):
        """Build autoencoder model"""
        # Encoder
        input_layer = layers.Input(shape=(input_dim,))
        
        encoded = layers.Dense(64, activation='relu')(input_layer)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dropout(0.2)(encoded)
        
        encoded = layers.Dense(32, activation='relu')(encoded)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dropout(0.2)(encoded)
        
        encoded = layers.Dense(16, activation='relu')(encoded)
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='encoding')(encoded)
        
        # Decoder
        decoded = layers.Dense(16, activation='relu')(encoded)
        
        decoded = layers.Dense(32, activation='relu')(decoded)
        decoded = layers.BatchNormalization()(decoded)
        decoded = layers.Dropout(0.2)(decoded)
        
        decoded = layers.Dense(64, activation='relu')(decoded)
        decoded = layers.BatchNormalization()(decoded)
        decoded = layers.Dropout(0.2)(decoded)
        
        output_layer = layers.Dense(input_dim, activation='linear')(decoded)
        
        # Full autoencoder
        autoencoder = keras.Model(input_layer, output_layer)
        autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Separate encoder model
        encoder = keras.Model(input_layer, encoded)
        
        return autoencoder, encoder
    
    def train(self, power_logs: List[Dict[str, Any]], epochs: int = 100,
              batch_size: int = 32, validation_split: float = 0.2):
        """Train the autoencoder on normal power consumption patterns"""
        # Extract features
        features = self.extract_features(power_logs)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Build model
        self.model, self.encoder = self._build_model(input_dim=features_scaled.shape[1])
        
        # Train
        history = self.model.fit(
            features_scaled, features_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]
        )
        
        # Calculate reconstruction error threshold
        predictions = self.model.predict(features_scaled, verbose=0)
        mse = np.mean(np.power(features_scaled - predictions, 2), axis=1)
        self.threshold = np.percentile(mse, 95)  # 95th percentile
        
        self.is_trained = True
        
        print(f"✓ Power Autoencoder trained on {len(power_logs)} samples")
        print(f"  Final loss: {history.history['loss'][-1]:.4f}")
        print(f"  Anomaly threshold: {self.threshold:.4f}")
        
        return history
    
    def predict(self, power_log: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict if power consumption is anomalous
        
        Returns:
            Anomaly detection result with details
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Extract features
        features = self.extract_features([power_log])
        features_scaled = self.scaler.transform(features)
        
        # Get reconstruction
        reconstruction = self.model.predict(features_scaled, verbose=0)
        
        # Calculate reconstruction error
        mse = np.mean(np.power(features_scaled - reconstruction, 2))
        
        is_anomaly = mse > self.threshold
        
        # Get feature-wise errors to identify which features are anomalous
        feature_errors = np.abs(features_scaled[0] - reconstruction[0])
        top_anomalous_features = np.argsort(feature_errors)[-3:][::-1]
        
        anomalous_features = [
            {
                'feature': self.feature_names[idx],
                'error': float(feature_errors[idx])
            }
            for idx in top_anomalous_features
        ]
        
        # Determine anomaly type
        anomaly_type = self._classify_anomaly(power_log, anomalous_features)
        
        return {
            'is_anomaly': is_anomaly,
            'reconstruction_error': float(mse),
            'threshold': float(self.threshold),
            'confidence': float(min(mse / self.threshold, 2.0)),
            'anomalous_features': anomalous_features,
            'anomaly_type': anomaly_type,
            'model': 'PowerAutoencoder',
            'timestamp': datetime.now().isoformat()
        }
    
    def _classify_anomaly(self, power_log: Dict[str, Any], 
                         anomalous_features: List[Dict[str, Any]]) -> str:
        """Classify the type of power anomaly"""
        feature_names = [f['feature'] for f in anomalous_features]
        
        # Check for crypto mining (high power + high CPU)
        if ('power_watts' in feature_names and 
            'cpu_usage' in feature_names and
            power_log.get('cpu_usage', 0) > 80):
            return 'possible_crypto_mining'
        
        # Check for botnet (high network + unusual power)
        if ('network_activity' in feature_names and
            'power_watts' in feature_names and
            power_log.get('network_activity', 0) > 500):
            return 'possible_botnet_activity'
        
        # Check for hardware issue (voltage/current anomaly)
        if 'voltage' in feature_names or 'current_amps' in feature_names:
            return 'possible_hardware_issue'
        
        # Check for unauthorized usage (unexpected state)
        if 'device_state' in feature_names:
            return 'unauthorized_usage'
        
        # Check for overheating
        if 'temperature' in feature_names and power_log.get('temperature', 0) > 70:
            return 'overheating_detected'
        
        return 'unknown_anomaly'
    
    def get_encoding(self, power_log: Dict[str, Any]) -> np.ndarray:
        """Get the encoded representation of power consumption"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        features = self.extract_features([power_log])
        features_scaled = self.scaler.transform(features)
        encoding = self.encoder.predict(features_scaled, verbose=0)
        
        return encoding[0]
    
    def save(self, filepath: str):
        """Save model to disk"""
        # Save Keras model
        model_path = filepath.replace('.pkl', '_model.h5')
        encoder_path = filepath.replace('.pkl', '_encoder.h5')
        
        self.model.save(model_path)
        self.encoder.save(encoder_path)
        
        # Save other components
        model_data = {
            'scaler': self.scaler,
            'threshold': self.threshold,
            'encoding_dim': self.encoding_dim,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        
        print(f"✓ Power Autoencoder saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from disk"""
        # Load Keras models
        model_path = filepath.replace('.pkl', '_model.h5')
        encoder_path = filepath.replace('.pkl', '_encoder.h5')
        
        self.model = keras.models.load_model(model_path)
        self.encoder = keras.models.load_model(encoder_path)
        
        # Load other components
        model_data = joblib.load(filepath)
        self.scaler = model_data['scaler']
        self.threshold = model_data['threshold']
        self.encoding_dim = model_data['encoding_dim']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        
        print(f"✓ Power Autoencoder loaded from {filepath}")


class PowerProfiler:
    """
    High-level power profiling system
    Maintains device-specific power profiles
    """
    
    def __init__(self):
        self.device_profiles: Dict[str, PowerConsumptionAutoencoder] = {}
        self.device_baselines: Dict[str, Dict[str, float]] = {}
    
    def create_profile(self, device_id: str, power_logs: List[Dict[str, Any]],
                      encoding_dim: int = 8):
        """Create a power profile for a device"""
        autoencoder = PowerConsumptionAutoencoder(encoding_dim=encoding_dim)
        autoencoder.train(power_logs)
        
        self.device_profiles[device_id] = autoencoder
        
        # Calculate baseline metrics
        features = autoencoder.extract_features(power_logs)
        self.device_baselines[device_id] = {
            'avg_power': float(np.mean(features[:, 0])),
            'std_power': float(np.std(features[:, 0])),
            'max_power': float(np.max(features[:, 0])),
            'min_power': float(np.min(features[:, 0]))
        }
        
        print(f"✓ Power profile created for {device_id}")
        print(f"  Baseline: {self.device_baselines[device_id]}")
    
    def check_power_consumption(self, device_id: str, 
                               power_log: Dict[str, Any]) -> Dict[str, Any]:
        """Check if power consumption is anomalous for a device"""
        if device_id not in self.device_profiles:
            return {
                'error': f'No profile found for device {device_id}',
                'is_anomaly': False
            }
        
        autoencoder = self.device_profiles[device_id]
        result = autoencoder.predict(power_log)
        
        # Add baseline comparison
        baseline = self.device_baselines[device_id]
        current_power = power_log.get('power_watts', 0)
        
        result['baseline_comparison'] = {
            'current_power': current_power,
            'avg_power': baseline['avg_power'],
            'deviation_percent': abs(current_power - baseline['avg_power']) / baseline['avg_power'] * 100
        }
        
        return result
    
    def save_profiles(self, directory: str):
        """Save all device profiles"""
        os.makedirs(directory, exist_ok=True)
        
        for device_id, autoencoder in self.device_profiles.items():
            filepath = os.path.join(directory, f'{device_id}_power_profile.pkl')
            autoencoder.save(filepath)
        
        # Save baselines
        baselines_path = os.path.join(directory, 'baselines.pkl')
        joblib.dump(self.device_baselines, baselines_path)
        
        print(f"✓ All power profiles saved to {directory}")
    
    def load_profiles(self, directory: str):
        """Load all device profiles"""
        import glob
        
        profile_files = glob.glob(os.path.join(directory, '*_power_profile.pkl'))
        
        for filepath in profile_files:
            device_id = os.path.basename(filepath).replace('_power_profile.pkl', '')
            autoencoder = PowerConsumptionAutoencoder()
            autoencoder.load(filepath)
            self.device_profiles[device_id] = autoencoder
        
        # Load baselines
        baselines_path = os.path.join(directory, 'baselines.pkl')
        if os.path.exists(baselines_path):
            self.device_baselines = joblib.load(baselines_path)
        
        print(f"✓ Loaded {len(self.device_profiles)} power profiles from {directory}")
