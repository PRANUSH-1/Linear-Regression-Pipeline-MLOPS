"""
Monitoring Module for Linear Regression Model
Handles model performance tracking, drift detection, and alerting
"""

import numpy as np
import pandas as pd
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import joblib
from collections import deque
import threading
import requests
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """Record for storing prediction data"""
    timestamp: str
    features: Dict[str, float]
    prediction: float
    actual: Optional[float] = None
    model_version: str = "unknown"
    response_time: float = 0.0


@dataclass
class DriftMetrics:
    """Metrics for data drift detection"""
    feature_name: str
    reference_mean: float
    reference_std: float
    current_mean: float
    current_std: float
    drift_score: float
    is_drifted: bool
    timestamp: str


class ModelMonitor:
    """Monitor model performance and detect drift"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ModelMonitor
        
        Args:
            config: Monitoring configuration
        """
        self.config = config
        self.predictions = deque(maxlen=config.get('max_predictions', 10000))
        self.reference_data = None
        self.drift_threshold = config.get('drift_threshold', 0.1)
        self.performance_threshold = config.get('performance_threshold', 0.8)
        self.alert_webhook = config.get('alert_webhook', None)
        self.monitoring_active = False
        self.lock = threading.Lock()
        
        # Performance metrics
        self.performance_history = {
            'r2_scores': deque(maxlen=1000),
            'rmse_scores': deque(maxlen=1000),
            'response_times': deque(maxlen=1000),
            'prediction_counts': deque(maxlen=1000)
        }
    
    def set_reference_data(self, data: pd.DataFrame):
        """
        Set reference data for drift detection
        
        Args:
            data: Reference dataset
        """
        self.reference_data = data
        logger.info(f"Reference data set with shape: {data.shape}")
    
    def record_prediction(self, prediction_record: PredictionRecord):
        """
        Record a prediction for monitoring
        
        Args:
            prediction_record: Prediction record to store
        """
        with self.lock:
            self.predictions.append(prediction_record)
            
            # Update performance metrics
            if prediction_record.actual is not None:
                # Calculate R² for recent predictions
                recent_predictions = [p for p in self.predictions if p.actual is not None]
                if len(recent_predictions) >= 10:
                    actuals = [p.actual for p in recent_predictions[-100:]]
                    preds = [p.prediction for p in recent_predictions[-100:]]
                    
                    from sklearn.metrics import r2_score, mean_squared_error
                    r2 = r2_score(actuals, preds)
                    rmse = np.sqrt(mean_squared_error(actuals, preds))
                    
                    self.performance_history['r2_scores'].append(r2)
                    self.performance_history['rmse_scores'].append(rmse)
            
            self.performance_history['response_times'].append(prediction_record.response_time)
            self.performance_history['prediction_counts'].append(len(self.predictions))
    
    def detect_data_drift(self, window_size: int = 1000) -> List[DriftMetrics]:
        """
        Detect data drift in recent predictions
        
        Args:
            window_size: Number of recent predictions to analyze
            
        Returns:
            List of drift metrics for each feature
        """
        if self.reference_data is None:
            logger.warning("No reference data available for drift detection")
            return []
        
        if len(self.predictions) < window_size:
            logger.warning(f"Not enough predictions for drift detection. Need {window_size}, have {len(self.predictions)}")
            return []
        
        # Get recent predictions
        recent_predictions = list(self.predictions)[-window_size:]
        recent_features = pd.DataFrame([p.features for p in recent_predictions])
        
        drift_metrics = []
        
        for feature in recent_features.columns:
            if feature in self.reference_data.columns:
                ref_mean = self.reference_data[feature].mean()
                ref_std = self.reference_data[feature].std()
                curr_mean = recent_features[feature].mean()
                curr_std = recent_features[feature].std()
                
                # Calculate drift score (normalized difference in means)
                drift_score = abs(curr_mean - ref_mean) / (ref_std + 1e-8)
                
                is_drifted = drift_score > self.drift_threshold
                
                drift_metrics.append(DriftMetrics(
                    feature_name=feature,
                    reference_mean=ref_mean,
                    reference_std=ref_std,
                    current_mean=curr_mean,
                    current_std=curr_std,
                    drift_score=drift_score,
                    is_drifted=is_drifted,
                    timestamp=datetime.now().isoformat()
                ))
        
        return drift_metrics
    
    def check_performance_degradation(self) -> Dict[str, Any]:
        """
        Check for performance degradation
        
        Returns:
            Performance degradation metrics
        """
        if len(self.performance_history['r2_scores']) < 10:
            return {'status': 'insufficient_data'}
        
        recent_r2 = list(self.performance_history['r2_scores'])[-10:]
        avg_r2 = np.mean(recent_r2)
        
        is_degraded = avg_r2 < self.performance_threshold
        
        return {
            'status': 'degraded' if is_degraded else 'healthy',
            'average_r2': avg_r2,
            'threshold': self.performance_threshold,
            'recent_scores': recent_r2,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive monitoring report
        
        Returns:
            Monitoring report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_predictions': len(self.predictions),
            'model_performance': self.check_performance_degradation(),
            'data_drift': self.detect_data_drift(),
            'response_time_stats': {
                'mean': np.mean(self.performance_history['response_times']) if self.performance_history['response_times'] else 0,
                'std': np.std(self.performance_history['response_times']) if self.performance_history['response_times'] else 0,
                'p95': np.percentile(self.performance_history['response_times'], 95) if self.performance_history['response_times'] else 0
            }
        }
        
        return report
    
    def send_alert(self, alert_type: str, message: str, severity: str = "warning"):
        """
        Send alert notification
        
        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity
        """
        if not self.alert_webhook:
            logger.warning(f"Alert webhook not configured. Alert: {alert_type} - {message}")
            return
        
        alert_data = {
            'alert_type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
            'model_version': 'unknown'  # Could be extracted from model metadata
        }
        
        try:
            response = requests.post(self.alert_webhook, json=alert_data, timeout=10)
            if response.status_code == 200:
                logger.info(f"Alert sent successfully: {alert_type}")
            else:
                logger.error(f"Failed to send alert. Status: {response.status_code}")
        except Exception as e:
            logger.error(f"Error sending alert: {str(e)}")
    
    def plot_monitoring_dashboard(self, save_path: Optional[str] = None):
        """
        Create monitoring dashboard plots
        
        Args:
            save_path: Path to save dashboard
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Performance over time
        if self.performance_history['r2_scores']:
            r2_scores = list(self.performance_history['r2_scores'])
            axes[0, 0].plot(r2_scores, alpha=0.7)
            axes[0, 0].axhline(y=self.performance_threshold, color='red', linestyle='--', alpha=0.7)
            axes[0, 0].set_title('R² Score Over Time')
            axes[0, 0].set_ylabel('R² Score')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Response time distribution
        if self.performance_history['response_times']:
            response_times = list(self.performance_history['response_times'])
            axes[0, 1].hist(response_times, bins=30, alpha=0.7)
            axes[0, 1].set_title('Response Time Distribution')
            axes[0, 1].set_xlabel('Response Time (s)')
            axes[0, 1].set_ylabel('Frequency')
        
        # 3. Prediction volume
        if self.performance_history['prediction_counts']:
            prediction_counts = list(self.performance_history['prediction_counts'])
            axes[0, 2].plot(prediction_counts, alpha=0.7)
            axes[0, 2].set_title('Prediction Volume Over Time')
            axes[0, 2].set_ylabel('Total Predictions')
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Data drift visualization
        drift_metrics = self.detect_data_drift()
        if drift_metrics:
            feature_names = [d.feature_name for d in drift_metrics]
            drift_scores = [d.drift_score for d in drift_metrics]
            colors = ['red' if d.is_drifted else 'green' for d in drift_metrics]
            
            axes[1, 0].bar(feature_names, drift_scores, color=colors, alpha=0.7)
            axes[1, 0].axhline(y=self.drift_threshold, color='red', linestyle='--', alpha=0.7)
            axes[1, 0].set_title('Data Drift Detection')
            axes[1, 0].set_ylabel('Drift Score')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. RMSE over time
        if self.performance_history['rmse_scores']:
            rmse_scores = list(self.performance_history['rmse_scores'])
            axes[1, 1].plot(rmse_scores, alpha=0.7, color='orange')
            axes[1, 1].set_title('RMSE Over Time')
            axes[1, 1].set_ylabel('RMSE')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Feature distribution comparison (if reference data available)
        if self.reference_data is not None and len(self.predictions) > 0:
            recent_features = pd.DataFrame([p.features for p in list(self.predictions)[-100:]])
            if len(recent_features.columns) > 0:
                feature = recent_features.columns[0]  # Show first feature
                axes[1, 2].hist(self.reference_data[feature], alpha=0.5, label='Reference', bins=20)
                axes[1, 2].hist(recent_features[feature], alpha=0.5, label='Current', bins=20)
                axes[1, 2].set_title(f'Feature Distribution: {feature}')
                axes[1, 2].set_xlabel(feature)
                axes[1, 2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Monitoring dashboard saved to {save_path}")
        
        plt.show()
    
    def save_monitoring_data(self, filepath: str):
        """
        Save monitoring data to file
        
        Args:
            filepath: Path to save monitoring data
        """
        monitoring_data = {
            'predictions': [asdict(p) for p in self.predictions],
            'performance_history': {
                'r2_scores': list(self.performance_history['r2_scores']),
                'rmse_scores': list(self.performance_history['rmse_scores']),
                'response_times': list(self.performance_history['response_times']),
                'prediction_counts': list(self.performance_history['prediction_counts'])
            },
            'reference_data_stats': self.reference_data.describe().to_dict() if self.reference_data is not None else None,
            'config': self.config
        }
        
        joblib.dump(monitoring_data, filepath)
        logger.info(f"Monitoring data saved to {filepath}")
    
    def load_monitoring_data(self, filepath: str):
        """
        Load monitoring data from file
        
        Args:
            filepath: Path to load monitoring data from
        """
        monitoring_data = joblib.load(filepath)
        
        # Restore predictions
        self.predictions = deque([PredictionRecord(**p) for p in monitoring_data['predictions']], 
                               maxlen=self.config.get('max_predictions', 10000))
        
        # Restore performance history
        for key, values in monitoring_data['performance_history'].items():
            self.performance_history[key] = deque(values, maxlen=1000)
        
        logger.info(f"Monitoring data loaded from {filepath}")


class PerformanceTracker:
    """Track model performance metrics"""
    
    def __init__(self):
        """Initialize PerformanceTracker"""
        self.metrics = {
            'predictions_made': 0,
            'predictions_with_feedback': 0,
            'total_response_time': 0.0,
            'errors': 0,
            'start_time': datetime.now()
        }
    
    def record_prediction(self, response_time: float, has_feedback: bool = False, error: bool = False):
        """
        Record prediction metrics
        
        Args:
            response_time: Response time in seconds
            has_feedback: Whether actual value is available
            error: Whether prediction resulted in error
        """
        self.metrics['predictions_made'] += 1
        self.metrics['total_response_time'] += response_time
        
        if has_feedback:
            self.metrics['predictions_with_feedback'] += 1
        
        if error:
            self.metrics['errors'] += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary
        
        Returns:
            Performance summary dictionary
        """
        uptime = datetime.now() - self.metrics['start_time']
        avg_response_time = (self.metrics['total_response_time'] / 
                           self.metrics['predictions_made'] if self.metrics['predictions_made'] > 0 else 0)
        
        return {
            'uptime_seconds': uptime.total_seconds(),
            'predictions_made': self.metrics['predictions_made'],
            'predictions_with_feedback': self.metrics['predictions_with_feedback'],
            'average_response_time': avg_response_time,
            'error_rate': self.metrics['errors'] / self.metrics['predictions_made'] if self.metrics['predictions_made'] > 0 else 0,
            'feedback_rate': self.metrics['predictions_with_feedback'] / self.metrics['predictions_made'] if self.metrics['predictions_made'] > 0 else 0
        }


def create_monitoring_config() -> Dict[str, Any]:
    """Create default monitoring configuration"""
    return {
        'max_predictions': 10000,
        'drift_threshold': 0.1,
        'performance_threshold': 0.8,
        'alert_webhook': None,  # Set to your webhook URL
        'monitoring_interval': 300,  # 5 minutes
        'save_interval': 3600,  # 1 hour
        'enable_alerts': True,
        'enable_drift_detection': True,
        'enable_performance_tracking': True
    }


if __name__ == "__main__":
    # Example usage
    config = create_monitoring_config()
    
    # Create monitor
    monitor = ModelMonitor(config)
    
    # Simulate some predictions
    for i in range(100):
        record = PredictionRecord(
            timestamp=datetime.now().isoformat(),
            features={'feature1': np.random.normal(0, 1), 'feature2': np.random.normal(0, 1)},
            prediction=np.random.normal(100, 10),
            actual=np.random.normal(100, 10) if i % 2 == 0 else None,
            response_time=np.random.exponential(0.1)
        )
        monitor.record_prediction(record)
    
    # Generate report
    report = monitor.generate_monitoring_report()
    print("Monitoring Report:")
    print(json.dumps(report, indent=2))
    
    # Create dashboard
    monitor.plot_monitoring_dashboard("monitoring_dashboard.png") 