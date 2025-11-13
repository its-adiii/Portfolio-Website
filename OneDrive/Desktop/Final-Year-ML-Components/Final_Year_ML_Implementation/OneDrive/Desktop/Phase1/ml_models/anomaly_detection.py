"""
Behavioral Anomaly Detection Models
Implements Isolation Forest and LSTM for detecting abnormal IoT device behavior
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import json
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import os


class IsolationForestDetector:
    """
    Isolation Forest for unsupervised anomaly detection
    Detects anomalies in device access patterns and behavior
    """
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100):
        """
        Args:
            contamination: Expected proportion of outliers (0.1 = 10%)
            n_estimators: Number of trees in the forest
        """
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
    
    def extract_features(self, access_logs: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extract features from access logs
        
        Features:
        - Hour of day
        - Day of week
        - Access frequency
        - Time since last access
        - IP address hash (categorical)
        - Location hash (categorical)
        - Action type (encoded)
        """
        features = []
        
        for i, log in enumerate(access_logs):
            timestamp = datetime.fromisoformat(log.get('timestamp', datetime.now().isoformat()))
            
            feature_vector = [
                timestamp.hour,  # Hour of day (0-23)
                timestamp.weekday(),  # Day of week (0-6)
                log.get('access_count', 1),  # Access frequency
                log.get('time_since_last', 0),  # Seconds since last access
                hash(log.get('ip_address', '')) % 1000,  # IP hash
                hash(log.get('location', '')) % 100,  # Location hash
                self._encode_action(log.get('action', 'unknown')),
                log.get('duration', 0),  # Action duration in seconds
                int(log.get('success', True)),  # Success flag
            ]
            
            features.append(feature_vector)
        
        self.feature_names = [
            'hour', 'day_of_week', 'access_count', 'time_since_last',
            'ip_hash', 'location_hash', 'action_encoded', 'duration', 'success'
        ]
        
        return np.array(features)
    
    def _encode_action(self, action: str) -> int:
        """Encode action type to integer"""
        action_map = {
            'unlock': 1, 'lock': 2, 'view': 3, 'control': 4,
            'power_on': 5, 'power_off': 6, 'unknown': 0
        }
        return action_map.get(action.lower(), 0)
    
    def train(self, access_logs: List[Dict[str, Any]]):
        """Train the Isolation Forest model"""
        features = self.extract_features(access_logs)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train model
        self.model.fit(features_scaled)
        self.is_trained = True
        
        print(f"✓ Isolation Forest trained on {len(access_logs)} samples")
    
    def predict(self, access_log: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict if access log is anomalous
        
        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        features = self.extract_features([access_log])
        features_scaled = self.scaler.transform(features)
        
        # Predict: -1 for anomaly, 1 for normal
        prediction = self.model.predict(features_scaled)[0]
        
        # Get anomaly score (lower = more anomalous)
        anomaly_score = self.model.score_samples(features_scaled)[0]
        
        is_anomaly = prediction == -1
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': float(anomaly_score),
            'confidence': abs(anomaly_score),
            'model': 'IsolationForest',
            'timestamp': datetime.now().isoformat()
        }
    
    def save(self, filepath: str):
        """Save model to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from disk"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        print(f"✓ Model loaded from {filepath}")


class LSTMDetector:
    """
    LSTM-based anomaly detection for time-series access patterns
    Learns temporal patterns and detects deviations
    """
    
    def __init__(self, sequence_length: int = 10, lstm_units: int = 64):
        """
        Args:
            sequence_length: Number of time steps to look back
            lstm_units: Number of LSTM units
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.is_trained = False
    
    def _build_model(self, input_shape: Tuple[int, int]):
        """Build LSTM autoencoder model"""
        # Encoder
        inputs = layers.Input(shape=input_shape)
        
        # LSTM layers
        x = layers.LSTM(self.lstm_units, return_sequences=True)(inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(self.lstm_units // 2, return_sequences=False)(x)
        x = layers.Dropout(0.2)(x)
        
        # Bottleneck
        encoded = layers.Dense(self.lstm_units // 4, activation='relu')(x)
        
        # Decoder
        x = layers.RepeatVector(input_shape[0])(encoded)
        x = layers.LSTM(self.lstm_units // 2, return_sequences=True)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(self.lstm_units, return_sequences=True)(x)
        x = layers.Dropout(0.2)(x)
        
        # Output
        outputs = layers.TimeDistributed(layers.Dense(input_shape[1]))(x)
        
        model = keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def extract_features(self, access_logs: List[Dict[str, Any]]) -> np.ndarray:
        """Extract time-series features from access logs"""
        features = []
        
        for log in access_logs:
            timestamp = datetime.fromisoformat(log.get('timestamp', datetime.now().isoformat()))
            
            feature_vector = [
                timestamp.hour / 24.0,  # Normalized hour
                timestamp.weekday() / 7.0,  # Normalized day
                log.get('access_count', 1) / 100.0,  # Normalized count
                min(log.get('time_since_last', 0) / 3600.0, 24.0) / 24.0,  # Normalized hours
                self._encode_action(log.get('action', 'unknown')) / 6.0,  # Normalized action
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _encode_action(self, action: str) -> int:
        """Encode action type to integer"""
        action_map = {
            'unlock': 1, 'lock': 2, 'view': 3, 'control': 4,
            'power_on': 5, 'power_off': 6, 'unknown': 0
        }
        return action_map.get(action.lower(), 0)
    
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length):
            seq = data[i:i + self.sequence_length]
            target = data[i:i + self.sequence_length]  # Autoencoder: reconstruct input
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def train(self, access_logs: List[Dict[str, Any]], epochs: int = 50, 
              batch_size: int = 32, validation_split: float = 0.2):
        """Train the LSTM model"""
        # Extract features
        features = self.extract_features(access_logs)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = self.create_sequences(features_scaled)
        
        if len(X) == 0:
            raise ValueError(f"Not enough data. Need at least {self.sequence_length + 1} samples.")
        
        # Build model
        self.model = self._build_model(input_shape=(X.shape[1], X.shape[2]))
        
        # Train
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            ]
        )
        
        # Calculate reconstruction error threshold
        predictions = self.model.predict(X, verbose=0)
        mse = np.mean(np.power(X - predictions, 2), axis=(1, 2))
        self.threshold = np.percentile(mse, 95)  # 95th percentile
        
        self.is_trained = True
        
        print(f"✓ LSTM trained on {len(X)} sequences")
        print(f"  Final loss: {history.history['loss'][-1]:.4f}")
        print(f"  Anomaly threshold: {self.threshold:.4f}")
    
    def predict(self, access_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Predict if sequence is anomalous
        
        Args:
            access_logs: List of recent access logs (at least sequence_length)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        if len(access_logs) < self.sequence_length:
            return {
                'is_anomaly': False,
                'error': f'Need at least {self.sequence_length} samples',
                'model': 'LSTM'
            }
        
        # Extract features from recent logs
        features = self.extract_features(access_logs[-self.sequence_length:])
        features_scaled = self.scaler.transform(features)
        
        # Reshape for prediction
        X = features_scaled.reshape(1, self.sequence_length, -1)
        
        # Predict
        prediction = self.model.predict(X, verbose=0)
        
        # Calculate reconstruction error
        mse = np.mean(np.power(X - prediction, 2))
        
        is_anomaly = mse > self.threshold
        
        return {
            'is_anomaly': is_anomaly,
            'reconstruction_error': float(mse),
            'threshold': float(self.threshold),
            'confidence': float(min(mse / self.threshold, 2.0)),
            'model': 'LSTM',
            'timestamp': datetime.now().isoformat()
        }
    
    def save(self, filepath: str):
        """Save model to disk"""
        # Save Keras model
        model_path = filepath.replace('.pkl', '_model.h5')
        self.model.save(model_path)
        
        # Save other components
        model_data = {
            'scaler': self.scaler,
            'threshold': self.threshold,
            'sequence_length': self.sequence_length,
            'lstm_units': self.lstm_units,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        
        print(f"✓ LSTM model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from disk"""
        # Load Keras model
        model_path = filepath.replace('.pkl', '_model.h5')
        self.model = keras.models.load_model(model_path)
        
        # Load other components
        model_data = joblib.load(filepath)
        self.scaler = model_data['scaler']
        self.threshold = model_data['threshold']
        self.sequence_length = model_data['sequence_length']
        self.lstm_units = model_data['lstm_units']
        self.is_trained = model_data['is_trained']
        
        print(f"✓ LSTM model loaded from {filepath}")


class EnsembleAnomalyDetector:
    """
    Ensemble detector combining Isolation Forest and LSTM
    Provides more robust anomaly detection
    """
    
    def __init__(self):
        self.isolation_forest = IsolationForestDetector()
        self.lstm = LSTMDetector()
        self.access_history: List[Dict[str, Any]] = []
    
    def train(self, access_logs: List[Dict[str, Any]]):
        """Train both models"""
        print("Training Isolation Forest...")
        self.isolation_forest.train(access_logs)
        
        print("\nTraining LSTM...")
        self.lstm.train(access_logs)
        
        print("\n✓ Ensemble training complete")
    
    def predict(self, access_log: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict using ensemble approach
        
        Returns combined prediction from both models
        """
        # Add to history
        self.access_history.append(access_log)
        
        # Keep only recent history
        if len(self.access_history) > 1000:
            self.access_history = self.access_history[-1000:]
        
        # Get predictions from both models
        if_result = self.isolation_forest.predict(access_log)
        
        lstm_result = {'is_anomaly': False, 'model': 'LSTM', 'note': 'Not enough history'}
        if len(self.access_history) >= self.lstm.sequence_length:
            lstm_result = self.lstm.predict(self.access_history)
        
        # Combine results (anomaly if either detects it)
        is_anomaly = if_result['is_anomaly'] or lstm_result.get('is_anomaly', False)
        
        # Calculate combined confidence
        if_conf = if_result.get('confidence', 0)
        lstm_conf = lstm_result.get('confidence', 0)
        combined_confidence = (if_conf + lstm_conf) / 2
        
        return {
            'is_anomaly': is_anomaly,
            'combined_confidence': combined_confidence,
            'isolation_forest': if_result,
            'lstm': lstm_result,
            'model': 'Ensemble',
            'timestamp': datetime.now().isoformat()
        }
    
    def save(self, directory: str):
        """Save both models"""
        os.makedirs(directory, exist_ok=True)
        self.isolation_forest.save(os.path.join(directory, 'isolation_forest.pkl'))
        self.lstm.save(os.path.join(directory, 'lstm.pkl'))
        print(f"✓ Ensemble saved to {directory}")
    
    def load(self, directory: str):
        """Load both models"""
        self.isolation_forest.load(os.path.join(directory, 'isolation_forest.pkl'))
        self.lstm.load(os.path.join(directory, 'lstm.pkl'))
        print(f"✓ Ensemble loaded from {directory}")
