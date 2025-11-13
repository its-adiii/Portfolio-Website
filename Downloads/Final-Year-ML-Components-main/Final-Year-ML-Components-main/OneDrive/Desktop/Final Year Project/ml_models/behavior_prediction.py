"""
Contextual Device Behavior Prediction
Learns user patterns and predicts expected device behavior
Detects anomalies when actual behavior deviates from predictions
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import json


class DeviceBehaviorPredictor:
    """
    Predicts expected device behavior based on context
    Uses Random Forest for classification
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.device_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_names = []
        self.device_states = []
    
    def extract_features(self, behavior_logs: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from behavior logs
        
        Features:
        - Hour of day
        - Day of week
        - Is weekend
        - User ID (encoded)
        - Previous device state
        - Time since last interaction
        - Number of interactions today
        - Typical usage pattern
        
        Target:
        - Expected device state/action
        """
        features = []
        targets = []
        
        for log in behavior_logs:
            timestamp = datetime.fromisoformat(log.get('timestamp', datetime.now().isoformat()))
            
            feature_vector = [
                timestamp.hour,
                timestamp.weekday(),
                int(timestamp.weekday() >= 5),  # Is weekend
                hash(log.get('user_id', 'unknown')) % 100,
                self._encode_state(log.get('previous_state', 'off')),
                log.get('time_since_last', 0) / 3600.0,  # Hours
                log.get('interactions_today', 0),
                log.get('typical_usage_hour', 0),
                int(log.get('is_home', True)),
                log.get('ambient_light', 50) / 100.0,  # Normalized
                log.get('temperature', 22) / 50.0,  # Normalized
            ]
            
            features.append(feature_vector)
            targets.append(log.get('device_state', 'off'))
        
        self.feature_names = [
            'hour', 'day_of_week', 'is_weekend', 'user_hash',
            'previous_state', 'time_since_last', 'interactions_today',
            'typical_usage_hour', 'is_home', 'ambient_light', 'temperature'
        ]
        
        return np.array(features), np.array(targets)
    
    def _encode_state(self, state: str) -> int:
        """Encode device state to integer"""
        state_map = {
            'off': 0, 'on': 1, 'standby': 2, 'active': 3,
            'locked': 4, 'unlocked': 5, 'unknown': -1
        }
        return state_map.get(state.lower(), -1)
    
    def train(self, behavior_logs: List[Dict[str, Any]]):
        """Train the behavior prediction model"""
        # Extract features and targets
        X, y = self.extract_features(behavior_logs)
        
        # Encode targets
        y_encoded = self.label_encoder.fit_transform(y)
        self.device_states = list(self.label_encoder.classes_)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y_encoded)
        self.is_trained = True
        
        # Calculate feature importances
        importances = self.model.feature_importances_
        feature_importance = sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )
        
        print(f"✓ Behavior Predictor trained on {len(behavior_logs)} samples")
        print(f"  Device states: {self.device_states}")
        print(f"  Top features: {feature_importance[:3]}")
    
    def predict(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict expected device behavior given context
        
        Args:
            context: Current context (time, user, environment, etc.)
        
        Returns:
            Prediction with probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Extract features
        features, _ = self.extract_features([context])
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        predicted_state = self.label_encoder.inverse_transform([prediction])[0]
        
        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_predictions = [
            {
                'state': self.label_encoder.inverse_transform([idx])[0],
                'probability': float(probabilities[idx])
            }
            for idx in top_indices
        ]
        
        return {
            'predicted_state': predicted_state,
            'confidence': float(probabilities[prediction]),
            'top_predictions': top_predictions,
            'model': 'BehaviorPredictor',
            'timestamp': datetime.now().isoformat()
        }
    
    def check_anomaly(self, context: Dict[str, Any], actual_state: str) -> Dict[str, Any]:
        """
        Check if actual behavior matches prediction
        
        Returns:
            Anomaly detection result
        """
        prediction = self.predict(context)
        
        predicted_state = prediction['predicted_state']
        confidence = prediction['confidence']
        
        is_anomaly = actual_state != predicted_state and confidence > 0.6
        
        return {
            'is_anomaly': is_anomaly,
            'predicted_state': predicted_state,
            'actual_state': actual_state,
            'confidence': confidence,
            'severity': 'high' if confidence > 0.8 else 'medium' if confidence > 0.6 else 'low',
            'model': 'BehaviorPredictor',
            'timestamp': datetime.now().isoformat()
        }
    
    def save(self, filepath: str):
        """Save model to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'device_encoder': self.device_encoder,
            'feature_names': self.feature_names,
            'device_states': self.device_states,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        print(f"✓ Behavior Predictor saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from disk"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.device_encoder = model_data['device_encoder']
        self.feature_names = model_data['feature_names']
        self.device_states = model_data['device_states']
        self.is_trained = model_data['is_trained']
        print(f"✓ Behavior Predictor loaded from {filepath}")


class UserPatternAnalyzer:
    """
    Analyzes user patterns and builds usage profiles
    """
    
    def __init__(self):
        self.user_patterns: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'hourly_usage': defaultdict(int),
            'daily_usage': defaultdict(int),
            'device_preferences': defaultdict(int),
            'typical_sequences': [],
            'total_interactions': 0
        })
    
    def analyze_logs(self, behavior_logs: List[Dict[str, Any]]):
        """Analyze behavior logs to extract patterns"""
        for log in behavior_logs:
            user_id = log.get('user_id', 'unknown')
            timestamp = datetime.fromisoformat(log.get('timestamp', datetime.now().isoformat()))
            device_id = log.get('device_id', 'unknown')
            
            pattern = self.user_patterns[user_id]
            
            # Update hourly usage
            pattern['hourly_usage'][timestamp.hour] += 1
            
            # Update daily usage
            pattern['daily_usage'][timestamp.weekday()] += 1
            
            # Update device preferences
            pattern['device_preferences'][device_id] += 1
            
            # Update total interactions
            pattern['total_interactions'] += 1
        
        print(f"✓ Analyzed patterns for {len(self.user_patterns)} users")
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get usage profile for a user"""
        if user_id not in self.user_patterns:
            return {'error': f'No pattern data for user {user_id}'}
        
        pattern = self.user_patterns[user_id]
        
        # Find peak usage hours
        hourly_usage = pattern['hourly_usage']
        peak_hours = sorted(hourly_usage.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Find preferred devices
        device_prefs = pattern['device_preferences']
        preferred_devices = sorted(device_prefs.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Find most active days
        daily_usage = pattern['daily_usage']
        active_days = sorted(daily_usage.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'user_id': user_id,
            'total_interactions': pattern['total_interactions'],
            'peak_hours': [{'hour': h, 'count': c} for h, c in peak_hours],
            'preferred_devices': [{'device': d, 'count': c} for d, c in preferred_devices],
            'active_days': [{'day': d, 'count': c} for d, c in active_days],
            'hourly_distribution': dict(hourly_usage),
            'daily_distribution': dict(daily_usage)
        }
    
    def is_typical_behavior(self, user_id: str, context: Dict[str, Any]) -> bool:
        """Check if behavior is typical for user"""
        if user_id not in self.user_patterns:
            return True  # No data, assume typical
        
        pattern = self.user_patterns[user_id]
        timestamp = datetime.fromisoformat(context.get('timestamp', datetime.now().isoformat()))
        
        # Check if hour is typical
        hour_usage = pattern['hourly_usage'].get(timestamp.hour, 0)
        avg_hourly_usage = sum(pattern['hourly_usage'].values()) / max(len(pattern['hourly_usage']), 1)
        
        # Typical if usage in this hour is above 50% of average
        return hour_usage >= avg_hourly_usage * 0.5
    
    def save(self, filepath: str):
        """Save patterns to disk"""
        # Convert defaultdicts to regular dicts for serialization
        patterns_serializable = {}
        for user_id, pattern in self.user_patterns.items():
            patterns_serializable[user_id] = {
                'hourly_usage': dict(pattern['hourly_usage']),
                'daily_usage': dict(pattern['daily_usage']),
                'device_preferences': dict(pattern['device_preferences']),
                'typical_sequences': pattern['typical_sequences'],
                'total_interactions': pattern['total_interactions']
            }
        
        joblib.dump(patterns_serializable, filepath)
        print(f"✓ User patterns saved to {filepath}")
    
    def load(self, filepath: str):
        """Load patterns from disk"""
        patterns_loaded = joblib.load(filepath)
        
        # Convert back to defaultdicts
        for user_id, pattern in patterns_loaded.items():
            self.user_patterns[user_id] = {
                'hourly_usage': defaultdict(int, pattern['hourly_usage']),
                'daily_usage': defaultdict(int, pattern['daily_usage']),
                'device_preferences': defaultdict(int, pattern['device_preferences']),
                'typical_sequences': pattern['typical_sequences'],
                'total_interactions': pattern['total_interactions']
            }
        
        print(f"✓ User patterns loaded from {filepath}")


class ContextualBehaviorSystem:
    """
    Complete contextual behavior analysis system
    Combines prediction and pattern analysis
    """
    
    def __init__(self):
        self.predictor = DeviceBehaviorPredictor()
        self.pattern_analyzer = UserPatternAnalyzer()
    
    def train(self, behavior_logs: List[Dict[str, Any]]):
        """Train both components"""
        print("Training Behavior Predictor...")
        self.predictor.train(behavior_logs)
        
        print("\nAnalyzing User Patterns...")
        self.pattern_analyzer.analyze_logs(behavior_logs)
        
        print("\n✓ Contextual Behavior System ready")
    
    def check_behavior(self, context: Dict[str, Any], actual_state: str) -> Dict[str, Any]:
        """
        Comprehensive behavior check
        
        Combines prediction-based and pattern-based anomaly detection
        """
        user_id = context.get('user_id', 'unknown')
        
        # Get prediction-based anomaly
        prediction_result = self.predictor.check_anomaly(context, actual_state)
        
        # Check if behavior is typical for user
        is_typical = self.pattern_analyzer.is_typical_behavior(user_id, context)
        
        # Combined decision
        is_anomaly = prediction_result['is_anomaly'] or not is_typical
        
        return {
            'is_anomaly': is_anomaly,
            'prediction_based': prediction_result,
            'pattern_based': {
                'is_typical': is_typical,
                'user_id': user_id
            },
            'severity': prediction_result['severity'],
            'model': 'ContextualBehaviorSystem',
            'timestamp': datetime.now().isoformat()
        }
    
    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive insights for a user"""
        return self.pattern_analyzer.get_user_profile(user_id)
    
    def save(self, directory: str):
        """Save both components"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        self.predictor.save(os.path.join(directory, 'behavior_predictor.pkl'))
        self.pattern_analyzer.save(os.path.join(directory, 'user_patterns.pkl'))
        
        print(f"✓ Contextual Behavior System saved to {directory}")
    
    def load(self, directory: str):
        """Load both components"""
        import os
        
        self.predictor.load(os.path.join(directory, 'behavior_predictor.pkl'))
        self.pattern_analyzer.load(os.path.join(directory, 'user_patterns.pkl'))
        
        print(f"✓ Contextual Behavior System loaded from {directory}")
