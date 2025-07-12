"""
Model Training Module for Linear Regression
Handles model training, hyperparameter tuning, and evaluation
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import logging
from typing import Dict, Any, Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and evaluate linear regression models"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ModelTrainer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models = {}
        self.best_model = None
        self.best_score = -np.inf
        self.training_history = {}
        
    def get_models(self) -> Dict[str, Any]:
        """Get dictionary of models to train"""
        models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(),
            'lasso': Lasso(),
            'elastic_net': ElasticNet(),
            'random_forest': RandomForestRegressor(random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42)
        }
        
        # Filter based on config
        if 'models' in self.config:
            models = {k: v for k, v in models.items() if k in self.config['models']}
        
        return models
    
    def get_hyperparameter_grids(self) -> Dict[str, Dict[str, List]]:
        """Get hyperparameter grids for each model"""
        grids = {
            'linear_regression': {},
            'ridge': {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            },
            'lasso': {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            },
            'elastic_net': {
                'alpha': [0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
        
        return grids
    
    def evaluate_model(self, model, X_train: np.ndarray, X_val: np.ndarray, 
                      y_train: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """
        Evaluate a model using multiple metrics
        
        Args:
            model: Trained model
            X_train: Training features
            X_val: Validation features
            y_train: Training targets
            y_val: Validation targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Calculate metrics
        metrics = {
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'val_mse': mean_squared_error(y_val, y_val_pred),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'val_mae': mean_absolute_error(y_val, y_val_pred),
            'val_r2': r2_score(y_val, y_val_pred)
        }
        
        return metrics
    
    def train_models(self, X_train: np.ndarray, X_val: np.ndarray, 
                    y_train: np.ndarray, y_val: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Train multiple models and evaluate them
        
        Args:
            X_train: Training features
            X_val: Validation features
            y_train: Training targets
            y_val: Validation targets
            
        Returns:
            Dictionary of model evaluation results
        """
        models = self.get_models()
        hyperparameter_grids = self.get_hyperparameter_grids()
        results = {}
        
        logger.info("Starting model training...")
        
        for model_name, model in models.items():
            logger.info(f"Training {model_name}...")
            
            try:
                if model_name in hyperparameter_grids and hyperparameter_grids[model_name]:
                    # Perform hyperparameter tuning
                    grid_search = GridSearchCV(
                        model, 
                        hyperparameter_grids[model_name],
                        cv=5,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1
                    )
                    grid_search.fit(X_train, y_train)
                    
                    # Get best model
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    
                    logger.info(f"Best parameters for {model_name}: {best_params}")
                    
                else:
                    # No hyperparameter tuning needed
                    best_model = model
                
                # Evaluate the model
                metrics = self.evaluate_model(best_model, X_train, X_val, y_train, y_val)
                results[model_name] = metrics
                
                # Store the model
                self.models[model_name] = best_model
                
                # Update best model if this one is better
                if metrics['val_r2'] > self.best_score:
                    self.best_score = metrics['val_r2']
                    self.best_model = best_model
                
                logger.info(f"{model_name} - Val R²: {metrics['val_r2']:.4f}, Val RMSE: {metrics['val_rmse']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        self.training_history = results
        return results
    
    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray, 
                           cv_folds: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation on a model
        
        Args:
            model: Model to cross-validate
            X: Features
            y: Targets
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary of cross-validation results
        """
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
        
        return {
            'cv_mean_r2': cv_scores.mean(),
            'cv_std_r2': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
    
    def plot_results(self, save_path: Optional[str] = None):
        """Plot training results"""
        if not self.training_history:
            logger.warning("No training history available for plotting")
            return
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract metrics
        model_names = list(self.training_history.keys())
        val_r2_scores = [self.training_history[name].get('val_r2', 0) for name in model_names]
        val_rmse_scores = [self.training_history[name].get('val_rmse', 0) for name in model_names]
        train_r2_scores = [self.training_history[name].get('train_r2', 0) for name in model_names]
        train_rmse_scores = [self.training_history[name].get('train_rmse', 0) for name in model_names]
        
        # R² comparison
        axes[0, 0].bar(model_names, val_r2_scores, alpha=0.7, label='Validation')
        axes[0, 0].bar(model_names, train_r2_scores, alpha=0.5, label='Training')
        axes[0, 0].set_title('R² Score Comparison')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # RMSE comparison
        axes[0, 1].bar(model_names, val_rmse_scores, alpha=0.7, label='Validation')
        axes[0, 1].bar(model_names, train_rmse_scores, alpha=0.5, label='Training')
        axes[0, 1].set_title('RMSE Comparison')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Training vs Validation R²
        axes[1, 0].scatter(train_r2_scores, val_r2_scores, s=100, alpha=0.7)
        for i, name in enumerate(model_names):
            axes[1, 0].annotate(name, (train_r2_scores[i], val_r2_scores[i]))
        axes[1, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
        axes[1, 0].set_xlabel('Training R²')
        axes[1, 0].set_ylabel('Validation R²')
        axes[1, 0].set_title('Training vs Validation R²')
        
        # Model performance ranking
        sorted_models = sorted(zip(model_names, val_r2_scores), key=lambda x: x[1], reverse=True)
        names, scores = zip(*sorted_models)
        axes[1, 1].barh(names, scores, alpha=0.7)
        axes[1, 1].set_title('Model Performance Ranking')
        axes[1, 1].set_xlabel('Validation R² Score')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Results plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, model_name: str, filepath: str):
        """Save a trained model"""
        if model_name in self.models:
            joblib.dump(self.models[model_name], filepath)
            logger.info(f"Model {model_name} saved to {filepath}")
        else:
            logger.error(f"Model {model_name} not found")
    
    def save_best_model(self, filepath: str):
        """Save the best performing model"""
        if self.best_model is not None:
            joblib.dump(self.best_model, filepath)
            logger.info(f"Best model saved to {filepath}")
        else:
            logger.error("No best model available")
    
    def load_model(self, model_name: str, filepath: str):
        """Load a trained model"""
        self.models[model_name] = joblib.load(filepath)
        logger.info(f"Model {model_name} loaded from {filepath}")


class MLflowTracker:
    """Track experiments using MLflow"""
    
    def __init__(self, experiment_name: str = "linear_regression"):
        """
        Initialize MLflow tracker
        
        Args:
            experiment_name: Name of the MLflow experiment
        """
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
    
    def log_experiment(self, model_name: str, model, metrics: Dict[str, float], 
                      params: Dict[str, Any] = None):
        """
        Log an experiment to MLflow
        
        Args:
            model_name: Name of the model
            model: Trained model
            metrics: Evaluation metrics
            params: Model parameters
        """
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            if params:
                mlflow.log_params(params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            logger.info(f"Experiment logged to MLflow: {model_name}")


def create_model_config() -> Dict[str, Any]:
    """Create default model configuration"""
    return {
        'models': ['linear_regression', 'ridge', 'lasso', 'elastic_net', 'random_forest', 'gradient_boosting'],
        'cv_folds': 5,
        'random_state': 42,
        'n_jobs': -1,
        'mlflow_tracking': True,
        'experiment_name': 'linear_regression_experiment'
    }


if __name__ == "__main__":
    # Example usage
    from data_generator import DataGenerator
    from preprocessing import PreprocessingPipeline, create_preprocessing_config
    
    # Generate and preprocess data
    generator = DataGenerator(random_state=42)
    data = generator.generate_house_data(n_samples=1000)
    
    preprocessor = PreprocessingPipeline(create_preprocessing_config())
    X_transformed, y = preprocessor.fit_transform(data)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X_transformed, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Train models
    config = create_model_config()
    trainer = ModelTrainer(config)
    results = trainer.train_models(X_train, X_val, y_train, y_val)
    
    # Plot results
    trainer.plot_results("model_comparison.png")
    
    # Save best model
    trainer.save_best_model("models/best_model.pkl") 