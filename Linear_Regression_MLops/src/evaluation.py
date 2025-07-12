"""
Model Evaluation Module for Linear Regression
Handles comprehensive model evaluation, residual analysis, and model interpretation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import explained_variance_score, max_error
from scipy import stats
import logging
from typing import Dict, Any, Tuple, Optional, List
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation and analysis"""
    
    def __init__(self, model, feature_names: Optional[List[str]] = None):
        """
        Initialize ModelEvaluator
        
        Args:
            model: Trained model
            feature_names: Names of features (for interpretation)
        """
        self.model = model
        self.feature_names = feature_names
        self.evaluation_results = {}
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'explained_variance': explained_variance_score(y_true, y_pred),
            'max_error': max_error(y_true, y_pred),
            'mean_absolute_percentage_error': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'median_absolute_error': np.median(np.abs(y_true - y_pred))
        }
        
        # Additional statistical metrics
        residuals = y_true - y_pred
        metrics.update({
            'residual_mean': np.mean(residuals),
            'residual_std': np.std(residuals),
            'residual_skewness': stats.skew(residuals),
            'residual_kurtosis': stats.kurtosis(residuals)
        })
        
        return metrics
    
    def analyze_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive residual analysis
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            save_path: Path to save residual plots
            
        Returns:
            Dictionary of residual analysis results
        """
        residuals = y_true - y_pred
        
        # Statistical tests
        shapiro_test = stats.shapiro(residuals)
        durbin_watson = self._calculate_durbin_watson(residuals)
        
        # Create residual plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        
        # 2. Q-Q Plot
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot of Residuals')
        
        # 3. Histogram of Residuals
        axes[0, 2].hist(residuals, bins=30, alpha=0.7, density=True)
        axes[0, 2].set_xlabel('Residuals')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].set_title('Distribution of Residuals')
        
        # Add normal distribution curve
        x = np.linspace(residuals.min(), residuals.max(), 100)
        normal_curve = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
        axes[0, 2].plot(x, normal_curve, 'r-', linewidth=2)
        
        # 4. Residuals vs Index
        axes[1, 0].plot(residuals, alpha=0.6)
        axes[1, 0].axhline(y=0, color='red', linestyle='--')
        axes[1, 0].set_xlabel('Observation Index')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Residuals vs Index')
        
        # 5. Scale-Location Plot
        standardized_residuals = residuals / np.std(residuals)
        axes[1, 1].scatter(y_pred, np.abs(standardized_residuals), alpha=0.6)
        axes[1, 1].set_xlabel('Predicted Values')
        axes[1, 1].set_ylabel('Standardized Residuals')
        axes[1, 1].set_title('Scale-Location Plot')
        
        # 6. Leverage Plot (if feature names available)
        if self.feature_names and hasattr(self.model, 'coef_'):
            feature_importance = np.abs(self.model.coef_)
            axes[1, 2].bar(range(len(feature_importance)), feature_importance)
            axes[1, 2].set_xlabel('Features')
            axes[1, 2].set_ylabel('Coefficient Magnitude')
            axes[1, 2].set_title('Feature Importance')
            axes[1, 2].set_xticks(range(len(self.feature_names)))
            axes[1, 2].set_xticklabels(self.feature_names, rotation=45)
        else:
            axes[1, 2].text(0.5, 0.5, 'Feature importance\nnot available', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Feature Importance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Residual analysis plots saved to {save_path}")
        
        plt.show()
        
        # Return analysis results
        analysis_results = {
            'shapiro_test_statistic': shapiro_test.statistic,
            'shapiro_test_pvalue': shapiro_test.pvalue,
            'durbin_watson_statistic': durbin_watson,
            'residual_normality': shapiro_test.pvalue > 0.05,
            'no_autocorrelation': 1.5 < durbin_watson < 2.5,
            'residual_statistics': {
                'mean': np.mean(residuals),
                'std': np.std(residuals),
                'min': np.min(residuals),
                'max': np.max(residuals),
                'skewness': stats.skew(residuals),
                'kurtosis': stats.kurtosis(residuals)
            }
        }
        
        return analysis_results
    
    def _calculate_durbin_watson(self, residuals: np.ndarray) -> float:
        """Calculate Durbin-Watson statistic for autocorrelation"""
        diff_residuals = np.diff(residuals)
        dw_statistic = np.sum(diff_residuals**2) / np.sum(residuals**2)
        return dw_statistic
    
    def plot_prediction_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               save_path: Optional[str] = None):
        """
        Create comprehensive prediction analysis plots
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            save_path: Path to save plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Actual vs Predicted')
        
        # Add R² text
        r2 = r2_score(y_true, y_pred)
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0, 0].transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 2. Prediction Error Distribution
        errors = y_true - y_pred
        axes[0, 1].hist(errors, bins=30, alpha=0.7, density=True)
        axes[0, 1].set_xlabel('Prediction Error')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Prediction Error Distribution')
        
        # 3. Residuals vs Predicted (with confidence bands)
        axes[1, 0].scatter(y_pred, errors, alpha=0.6)
        axes[1, 0].axhline(y=0, color='red', linestyle='--')
        
        # Add confidence bands
        std_error = np.std(errors)
        axes[1, 0].axhline(y=2*std_error, color='red', linestyle=':', alpha=0.5)
        axes[1, 0].axhline(y=-2*std_error, color='red', linestyle=':', alpha=0.5)
        
        axes[1, 0].set_xlabel('Predicted Values')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Residuals vs Predicted (with 2σ bands)')
        
        # 4. Cumulative Distribution of Errors
        sorted_errors = np.sort(errors)
        cumulative_prob = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        axes[1, 1].plot(sorted_errors, cumulative_prob, linewidth=2)
        axes[1, 1].set_xlabel('Prediction Error')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].set_title('Cumulative Distribution of Errors')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction analysis plots saved to {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 dataset_name: str = "Test") -> str:
        """
        Generate a comprehensive evaluation report
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            dataset_name: Name of the dataset being evaluated
            
        Returns:
            Formatted evaluation report
        """
        metrics = self.calculate_metrics(y_true, y_pred)
        residual_analysis = self.analyze_residuals(y_true, y_pred)
        
        report = f"""
        ========================================
        MODEL EVALUATION REPORT - {dataset_name.upper()}
        ========================================
        
        PERFORMANCE METRICS:
        -------------------
        R² Score:                    {metrics['r2']:.4f}
        Explained Variance:          {metrics['explained_variance']:.4f}
        Mean Squared Error:          {metrics['mse']:.4f}
        Root Mean Squared Error:     {metrics['rmse']:.4f}
        Mean Absolute Error:         {metrics['mae']:.4f}
        Median Absolute Error:       {metrics['median_absolute_error']:.4f}
        Max Error:                   {metrics['max_error']:.4f}
        Mean Absolute % Error:       {metrics['mean_absolute_percentage_error']:.2f}%
        
        RESIDUAL ANALYSIS:
        ------------------
        Residual Mean:               {metrics['residual_mean']:.4f}
        Residual Standard Deviation: {metrics['residual_std']:.4f}
        Residual Skewness:           {metrics['residual_skewness']:.4f}
        Residual Kurtosis:           {metrics['residual_kurtosis']:.4f}
        
        STATISTICAL TESTS:
        ------------------
        Shapiro-Wilk Test (Normality):
            Statistic:               {residual_analysis['shapiro_test_statistic']:.4f}
            P-value:                 {residual_analysis['shapiro_test_pvalue']:.4f}
            Residuals Normal:        {residual_analysis['residual_normality']}
        
        Durbin-Watson Test (Autocorrelation):
            Statistic:               {residual_analysis['durbin_watson_statistic']:.4f}
            No Autocorrelation:      {residual_analysis['no_autocorrelation']}
        
        MODEL ASSUMPTIONS:
        ------------------
        ✓ Linear Relationship:       {metrics['r2'] > 0.7}
        ✓ Homoscedasticity:         {residual_analysis['no_autocorrelation']}
        ✓ Normality of Residuals:   {residual_analysis['residual_normality']}
        ✓ Independence of Errors:   {residual_analysis['no_autocorrelation']}
        
        RECOMMENDATIONS:
        ----------------
        """
        
        # Add recommendations based on analysis
        if metrics['r2'] < 0.7:
            report += "- Consider feature engineering or non-linear models\n"
        
        if not residual_analysis['residual_normality']:
            report += "- Residuals are not normally distributed, consider transformations\n"
        
        if not residual_analysis['no_autocorrelation']:
            report += "- Autocorrelation detected, consider time series models\n"
        
        if metrics['mean_absolute_percentage_error'] > 10:
            report += "- High percentage error, consider log transformation\n"
        
        report += "\n" + "=" * 40
        
        return report
    
    def save_evaluation_results(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              filepath: str):
        """
        Save evaluation results to file
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            filepath: Path to save results
        """
        metrics = self.calculate_metrics(y_true, y_pred)
        residual_analysis = self.analyze_residuals(y_true, y_pred)
        
        results = {
            'metrics': metrics,
            'residual_analysis': residual_analysis,
            'predictions': {
                'y_true': y_true.tolist(),
                'y_pred': y_pred.tolist(),
                'residuals': (y_true - y_pred).tolist()
            }
        }
        
        joblib.dump(results, filepath)
        logger.info(f"Evaluation results saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    from data_generator import DataGenerator
    from preprocessing import PreprocessingPipeline, create_preprocessing_config
    from model import ModelTrainer, create_model_config
    
    # Generate and preprocess data
    generator = DataGenerator(random_state=42)
    data = generator.generate_house_data(n_samples=1000)
    
    preprocessor = PreprocessingPipeline(create_preprocessing_config())
    X_transformed, y = preprocessor.fit_transform(data)
    
    # Train a simple model
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate model
    evaluator = ModelEvaluator(model)
    y_pred = model.predict(X_test)
    
    # Generate report
    report = evaluator.generate_evaluation_report(y_test, y_pred, "Test")
    print(report)
    
    # Save results
    evaluator.save_evaluation_results(y_test, y_pred, "evaluation_results.pkl") 