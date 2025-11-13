"""
Edge ML Inference Engine
Lightweight ML inference for IoT devices
Optimized for resource-constrained environments
"""

import numpy as np
from typing import Dict, Any, Optional
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class EdgeInferenceEngine:
    """
    Lightweight inference engine for edge devices
    Runs optimized ML models on IoT devices
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.models = {}
        self.inference_count = 0
        self.total_inference_time = 0
    
    def load_model(self, model_name: str, model_path: str):
        """Load a model for edge inference"""
        try:
            import joblib
            model = joblib.load(model_path)
            self.models[model_name] = model
            print(f"✓ Loaded {model_name} for edge inference")
            return True
        except Exception as e:
            print(f"✗ Failed to load {model_name}: {e}")
            return False
    
    def predict_anomaly(self, features: np.ndarray, model_name: str = 'anomaly') -> Dict[str, Any]:
        """
        Run anomaly detection inference
        
        Args:
            features: Feature vector
            model_name: Name of the model to use
        
        Returns:
            Prediction result
        """
        import time
        
        if model_name not in self.models:
            return {'error': f'Model {model_name} not loaded'}
        
        start_time = time.time()
        
        try:
            model = self.models[model_name]
            
            # Simple threshold-based detection for edge
            if hasattr(model, 'predict'):
                prediction = model.predict(features.reshape(1, -1))[0]
                is_anomaly = prediction == -1
            else:
                # Fallback to simple statistical method
                is_anomaly = self._simple_anomaly_detection(features)
            
            inference_time = time.time() - start_time
            self.inference_count += 1
            self.total_inference_time += inference_time
            
            return {
                'is_anomaly': bool(is_anomaly),
                'inference_time_ms': inference_time * 1000,
                'model': model_name,
                'edge_device': True
            }
        
        except Exception as e:
            return {'error': str(e)}
    
    def _simple_anomaly_detection(self, features: np.ndarray) -> bool:
        """
        Simple statistical anomaly detection for edge devices
        Uses z-score method
        """
        # Calculate z-scores
        mean = np.mean(features)
        std = np.std(features)
        
        if std == 0:
            return False
        
        z_scores = np.abs((features - mean) / std)
        
        # Anomaly if any feature has z-score > 3
        return np.any(z_scores > 3)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics"""
        avg_time = (self.total_inference_time / self.inference_count * 1000 
                   if self.inference_count > 0 else 0)
        
        return {
            'total_inferences': self.inference_count,
            'avg_inference_time_ms': avg_time,
            'loaded_models': list(self.models.keys())
        }


class TinyMLOptimizer:
    """
    Optimizer for deploying ML models to edge devices
    Reduces model size and complexity
    """
    
    @staticmethod
    def quantize_model(model, bits: int = 8):
        """
        Quantize model weights to reduce size
        
        Args:
            model: Model to quantize
            bits: Number of bits (8 or 16)
        """
        # Simplified quantization for demonstration
        print(f"Quantizing model to {bits}-bit precision...")
        
        # In production, use TensorFlow Lite or ONNX quantization
        return model
    
    @staticmethod
    def prune_model(model, sparsity: float = 0.5):
        """
        Prune model by removing low-weight connections
        
        Args:
            model: Model to prune
            sparsity: Fraction of weights to remove (0-1)
        """
        print(f"Pruning model with {sparsity*100}% sparsity...")
        
        # In production, use TensorFlow Model Optimization Toolkit
        return model
    
    @staticmethod
    def convert_to_tflite(model_path: str, output_path: str):
        """
        Convert model to TensorFlow Lite format
        
        Args:
            model_path: Path to original model
            output_path: Path for TFLite model
        """
        print(f"Converting model to TensorFlow Lite...")
        print(f"  Input: {model_path}")
        print(f"  Output: {output_path}")
        
        # In production, use TensorFlow Lite converter
        # converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
        # tflite_model = converter.convert()
        
        return True
    
    @staticmethod
    def estimate_model_size(model) -> Dict[str, Any]:
        """Estimate model size and complexity"""
        import sys
        
        size_bytes = sys.getsizeof(model)
        size_kb = size_bytes / 1024
        size_mb = size_kb / 1024
        
        return {
            'size_bytes': size_bytes,
            'size_kb': round(size_kb, 2),
            'size_mb': round(size_mb, 2),
            'suitable_for_edge': size_mb < 10  # Models under 10MB suitable for edge
        }


class EdgeSecurityMonitor:
    """
    Security monitoring system for edge devices
    Runs lightweight checks locally
    """
    
    def __init__(self):
        self.inference_engine = EdgeInferenceEngine()
        self.alert_threshold = 0.7
        self.alerts = []
    
    def check_power_anomaly(self, power_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check for power consumption anomalies
        Lightweight check suitable for edge devices
        """
        power_watts = power_data.get('power_watts', 0)
        expected_power = power_data.get('avg_power', power_watts)
        
        # Simple threshold-based check
        deviation = abs(power_watts - expected_power) / max(expected_power, 1)
        
        is_anomaly = deviation > 0.5  # 50% deviation threshold
        
        if is_anomaly:
            alert = {
                'type': 'power_anomaly',
                'device_id': power_data.get('device_id'),
                'deviation': deviation,
                'current_power': power_watts,
                'expected_power': expected_power,
                'timestamp': power_data.get('timestamp')
            }
            self.alerts.append(alert)
        
        return {
            'is_anomaly': is_anomaly,
            'deviation': deviation,
            'severity': 'high' if deviation > 1.0 else 'medium'
        }
    
    def check_access_pattern(self, access_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check for unusual access patterns
        Lightweight check for edge devices
        """
        from datetime import datetime
        
        timestamp = datetime.fromisoformat(access_data.get('timestamp', datetime.now().isoformat()))
        hour = timestamp.hour
        
        # Simple rule-based checks
        is_unusual_time = hour < 6 or hour > 23
        is_unusual_location = access_data.get('location', 'home') not in ['home', 'local']
        
        is_anomaly = is_unusual_time or is_unusual_location
        
        if is_anomaly:
            alert = {
                'type': 'access_anomaly',
                'device_id': access_data.get('device_id'),
                'unusual_time': is_unusual_time,
                'unusual_location': is_unusual_location,
                'timestamp': access_data.get('timestamp')
            }
            self.alerts.append(alert)
        
        return {
            'is_anomaly': is_anomaly,
            'unusual_time': is_unusual_time,
            'unusual_location': is_unusual_location,
            'severity': 'high' if is_unusual_time and is_unusual_location else 'medium'
        }
    
    def get_alerts(self, limit: int = 10) -> list:
        """Get recent alerts"""
        return self.alerts[-limit:]
    
    def clear_alerts(self):
        """Clear all alerts"""
        self.alerts = []


def deploy_to_edge(model_path: str, device_id: str) -> bool:
    """
    Deploy ML model to edge device
    
    Args:
        model_path: Path to trained model
        device_id: Target device identifier
    
    Returns:
        Success status
    """
    print(f"Deploying model to edge device: {device_id}")
    print(f"Model path: {model_path}")
    
    # Optimize model for edge
    optimizer = TinyMLOptimizer()
    
    # Estimate size
    try:
        import joblib
        model = joblib.load(model_path)
        size_info = optimizer.estimate_model_size(model)
        
        print(f"\nModel size: {size_info['size_mb']} MB")
        
        if not size_info['suitable_for_edge']:
            print("⚠ Warning: Model may be too large for edge device")
            print("  Consider quantization or pruning")
        
        # In production, transfer model to device via secure channel
        print(f"\n✓ Model ready for deployment to {device_id}")
        return True
    
    except Exception as e:
        print(f"✗ Deployment failed: {e}")
        return False


if __name__ == '__main__':
    # Test edge inference
    print("Testing Edge Inference Engine...")
    
    engine = EdgeInferenceEngine()
    monitor = EdgeSecurityMonitor()
    
    # Test power anomaly detection
    normal_power = {'device_id': 'test', 'power_watts': 10, 'avg_power': 10, 'timestamp': '2024-01-01T12:00:00'}
    anomalous_power = {'device_id': 'test', 'power_watts': 50, 'avg_power': 10, 'timestamp': '2024-01-01T12:00:00'}
    
    print("\nNormal power:", monitor.check_power_anomaly(normal_power))
    print("Anomalous power:", monitor.check_power_anomaly(anomalous_power))
    
    print("\n✓ Edge inference engine operational")
