"""
Unit Tests for MLOps Pipeline Components
Tests for data generation, preprocessing, model training, and evaluation
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import sys
import joblib
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_generator import DataGenerator
from preprocessing import PreprocessingPipeline, create_preprocessing_config
from model import ModelTrainer, create_model_config
from evaluation import ModelEvaluator


class TestDataGenerator(unittest.TestCase):
    """Test data generation functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.generator = DataGenerator(random_state=42)
    
    def test_generate_house_data(self):
        """Test house data generation"""
        data = self.generator.generate_house_data(n_samples=100, noise_level=0.1)
        
        # Check data shape
        self.assertEqual(data.shape[0], 100)
        self.assertEqual(data.shape[1], 6)  # 5 features + 1 target
        
        # Check column names
        expected_columns = ['square_feet', 'bedrooms', 'bathrooms', 'age', 'distance_to_city', 'price']
        self.assertListEqual(list(data.columns), expected_columns)
        
        # Check data types
        self.assertTrue(data['square_feet'].dtype in ['float64', 'float32'])
        self.assertTrue(data['bedrooms'].dtype in ['int64', 'int32'])
        self.assertTrue(data['price'].dtype in ['float64', 'float32'])
        
        # Check value ranges
        self.assertTrue(data['bedrooms'].min() >= 1)
        self.assertTrue(data['bedrooms'].max() <= 5)
        self.assertTrue(data['bathrooms'].min() >= 1)
        self.assertTrue(data['bathrooms'].max() <= 3)
        self.assertTrue(data['price'].min() > 0)
    
    def test_generate_sales_data(self):
        """Test sales data generation"""
        data = self.generator.generate_sales_data(n_samples=100, noise_level=0.1)
        
        # Check data shape
        self.assertEqual(data.shape[0], 100)
        self.assertEqual(data.shape[1], 6)  # 5 features + 1 target
        
        # Check column names
        expected_columns = ['advertising_budget', 'price', 'competitor_price', 'season', 'store_size', 'sales']
        self.assertListEqual(list(data.columns), expected_columns)
        
        # Check value ranges
        self.assertTrue(data['advertising_budget'].min() > 0)
        self.assertTrue(data['price'].min() > 0)
        self.assertTrue(data['sales'].min() >= 0)
        self.assertTrue(data['season'].min() >= 1)
        self.assertTrue(data['season'].max() <= 4)
    
    def test_split_data(self):
        """Test data splitting functionality"""
        data = self.generator.generate_house_data(n_samples=1000, noise_level=0.1)
        train_df, val_df, test_df = self.generator.split_data(data, 'price')
        
        # Check split sizes
        total_samples = len(data)
        expected_test_size = int(total_samples * 0.2)
        expected_val_size = int((total_samples - expected_test_size) * 0.2)
        expected_train_size = total_samples - expected_test_size - expected_val_size
        
        self.assertEqual(len(train_df), expected_train_size)
        self.assertEqual(len(val_df), expected_val_size)
        self.assertEqual(len(test_df), expected_test_size)
        
        # Check no overlap between splits
        train_indices = set(train_df.index)
        val_indices = set(val_df.index)
        test_indices = set(test_df.index)
        
        self.assertTrue(len(train_indices.intersection(val_indices)) == 0)
        self.assertTrue(len(train_indices.intersection(test_indices)) == 0)
        self.assertTrue(len(val_indices.intersection(test_indices)) == 0)
    
    def test_save_data(self):
        """Test data saving functionality"""
        data = self.generator.generate_house_data(n_samples=100, noise_level=0.1)
        train_df, val_df, test_df = self.generator.split_data(data, 'price')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            self.generator.save_data(train_df, val_df, test_df, temp_dir)
            
            # Check files exist
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'train.csv')))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'val.csv')))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'test.csv')))
            
            # Check data integrity
            loaded_train = pd.read_csv(os.path.join(temp_dir, 'train.csv'))
            self.assertEqual(len(loaded_train), len(train_df))


class TestPreprocessingPipeline(unittest.TestCase):
    """Test preprocessing pipeline functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = create_preprocessing_config()
        self.pipeline = PreprocessingPipeline(self.config)
        
        # Generate test data
        generator = DataGenerator(random_state=42)
        self.data = generator.generate_house_data(n_samples=100, noise_level=0.1)
    
    def test_preprocessing_config(self):
        """Test preprocessing configuration"""
        config = create_preprocessing_config()
        
        self.assertIn('target_column', config)
        self.assertIn('feature_engineering', config)
        self.assertEqual(config['target_column'], 'price')
    
    def test_pipeline_creation(self):
        """Test pipeline creation"""
        self.assertIsNotNone(self.pipeline)
        self.assertIsNotNone(self.pipeline.config)
    
    def test_fit_transform(self):
        """Test fit and transform functionality"""
        X_transformed, y = self.pipeline.fit_transform(self.data)
        
        # Check output shapes
        self.assertEqual(X_transformed.shape[0], len(self.data))
        self.assertEqual(len(y), len(self.data))
        
        # Check that X_transformed is numpy array
        self.assertIsInstance(X_transformed, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
    
    def test_transform_only(self):
        """Test transform functionality after fitting"""
        # First fit the pipeline
        X_transformed, y = self.pipeline.fit_transform(self.data)
        
        # Then transform new data
        new_data = self.data.head(10)
        X_new_transformed = self.pipeline.transform(new_data)
        
        # Check output shape
        self.assertEqual(X_new_transformed.shape[0], 10)
        self.assertEqual(X_new_transformed.shape[1], X_transformed.shape[1])
    
    def test_pipeline_save_load(self):
        """Test pipeline save and load functionality"""
        # Fit pipeline
        X_transformed, y = self.pipeline.fit_transform(self.data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline_path = os.path.join(temp_dir, 'pipeline.pkl')
            
            # Save pipeline
            self.pipeline.save_pipeline(pipeline_path)
            self.assertTrue(os.path.exists(pipeline_path))
            
            # Load pipeline
            new_pipeline = PreprocessingPipeline(self.config)
            new_pipeline.load_pipeline(pipeline_path)
            
            # Test loaded pipeline
            X_new_transformed = new_pipeline.transform(self.data.head(5))
            self.assertEqual(X_new_transformed.shape[0], 5)


class TestModelTrainer(unittest.TestCase):
    """Test model training functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = create_model_config()
        self.trainer = ModelTrainer(self.config)
        
        # Generate test data
        generator = DataGenerator(random_state=42)
        data = generator.generate_house_data(n_samples=200, noise_level=0.1)
        
        # Preprocess data
        preprocessor = PreprocessingPipeline(create_preprocessing_config())
        X_transformed, y = preprocessor.fit_transform(data)
        
        # Split data
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_temp, self.y_train, self.y_temp = train_test_split(
            X_transformed, y, test_size=0.3, random_state=42
        )
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
            self.X_temp, self.y_temp, test_size=0.5, random_state=42
        )
    
    def test_model_config(self):
        """Test model configuration"""
        config = create_model_config()
        
        self.assertIn('models', config)
        self.assertIn('cv_folds', config)
        self.assertIn('random_state', config)
    
    def test_get_models(self):
        """Test model retrieval"""
        models = self.trainer.get_models()
        
        self.assertIn('linear_regression', models)
        self.assertIn('ridge', models)
        self.assertIn('lasso', models)
        
        # Check that models are sklearn estimators
        from sklearn.base import BaseEstimator
        for model in models.values():
            self.assertIsInstance(model, BaseEstimator)
    
    def test_evaluate_model(self):
        """Test model evaluation"""
        from sklearn.linear_model import LinearRegression
        
        model = LinearRegression()
        metrics = self.trainer.evaluate_model(
            model, self.X_train, self.X_val, self.y_train, self.y_val
        )
        
        # Check metrics structure
        expected_metrics = [
            'train_mse', 'train_rmse', 'train_mae', 'train_r2',
            'val_mse', 'val_rmse', 'val_mae', 'val_r2'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
    
    def test_train_models(self):
        """Test model training"""
        # Use only a subset of models for faster testing
        self.trainer.config['models'] = ['linear_regression', 'ridge']
        
        results = self.trainer.train_models(
            self.X_train, self.X_val, self.y_train, self.y_val
        )
        
        # Check results structure
        self.assertIn('linear_regression', results)
        self.assertIn('ridge', results)
        
        # Check that best model is set
        self.assertIsNotNone(self.trainer.best_model)
        self.assertIsInstance(self.trainer.best_score, (int, float))
    
    def test_cross_validate_model(self):
        """Test cross-validation"""
        from sklearn.linear_model import LinearRegression
        
        model = LinearRegression()
        cv_results = self.trainer.cross_validate_model(
            model, self.X_train, self.y_train, cv_folds=3
        )
        
        # Check CV results structure
        self.assertIn('cv_mean_r2', cv_results)
        self.assertIn('cv_std_r2', cv_results)
        self.assertIn('cv_scores', cv_results)
        
        self.assertIsInstance(cv_results['cv_mean_r2'], float)
        self.assertIsInstance(cv_results['cv_std_r2'], float)
        self.assertIsInstance(cv_results['cv_scores'], list)
    
    def test_save_load_model(self):
        """Test model save and load functionality"""
        # Train a simple model
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'model.pkl')
            
            # Save model
            self.trainer.models['test_model'] = model
            self.trainer.save_model('test_model', model_path)
            self.assertTrue(os.path.exists(model_path))
            
            # Load model
            self.trainer.load_model('test_model', model_path)
            self.assertIn('test_model', self.trainer.models)


class TestModelEvaluator(unittest.TestCase):
    """Test model evaluation functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        from sklearn.linear_model import LinearRegression
        
        # Generate test data
        generator = DataGenerator(random_state=42)
        data = generator.generate_house_data(n_samples=100, noise_level=0.1)
        
        # Preprocess data
        preprocessor = PreprocessingPipeline(create_preprocessing_config())
        X_transformed, y = preprocessor.fit_transform(data)
        
        # Train a simple model
        self.model = LinearRegression()
        self.model.fit(X_transformed, y)
        
        # Create evaluator
        self.evaluator = ModelEvaluator(self.model)
        
        # Generate predictions
        self.y_pred = self.model.predict(X_transformed)
        self.y_true = y
    
    def test_calculate_metrics(self):
        """Test metrics calculation"""
        metrics = self.evaluator.calculate_metrics(self.y_true, self.y_pred)
        
        # Check required metrics
        required_metrics = [
            'mse', 'rmse', 'mae', 'r2', 'explained_variance', 'max_error'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
        
        # Check metric values are reasonable
        self.assertGreaterEqual(metrics['r2'], 0)
        self.assertLessEqual(metrics['r2'], 1)
        self.assertGreaterEqual(metrics['rmse'], 0)
    
    def test_analyze_residuals(self):
        """Test residual analysis"""
        analysis = self.evaluator.analyze_residuals(self.y_true, self.y_pred)
        
        # Check analysis structure
        self.assertIn('shapiro_test_statistic', analysis)
        self.assertIn('shapiro_test_pvalue', analysis)
        self.assertIn('durbin_watson_statistic', analysis)
        self.assertIn('residual_normality', analysis)
        self.assertIn('no_autocorrelation', analysis)
        
        # Check data types
        self.assertIsInstance(analysis['shapiro_test_statistic'], float)
        self.assertIsInstance(analysis['shapiro_test_pvalue'], float)
        self.assertIsInstance(analysis['durbin_watson_statistic'], float)
        self.assertIsInstance(analysis['residual_normality'], bool)
        self.assertIsInstance(analysis['no_autocorrelation'], bool)
    
    def test_generate_evaluation_report(self):
        """Test evaluation report generation"""
        report = self.evaluator.generate_evaluation_report(
            self.y_true, self.y_pred, "Test"
        )
        
        # Check report is a string
        self.assertIsInstance(report, str)
        
        # Check report contains expected sections
        self.assertIn("MODEL EVALUATION REPORT", report)
        self.assertIn("PERFORMANCE METRICS", report)
        self.assertIn("RESIDUAL ANALYSIS", report)
        self.assertIn("STATISTICAL TESTS", report)
    
    def test_save_evaluation_results(self):
        """Test evaluation results saving"""
        with tempfile.TemporaryDirectory() as temp_dir:
            results_path = os.path.join(temp_dir, 'evaluation_results.pkl')
            
            self.evaluator.save_evaluation_results(
                self.y_true, self.y_pred, results_path
            )
            
            self.assertTrue(os.path.exists(results_path))
            
            # Load and check results
            results = joblib.load(results_path)
            self.assertIn('metrics', results)
            self.assertIn('residual_analysis', results)
            self.assertIn('predictions', results)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline"""
        # Generate data
        generator = DataGenerator(random_state=42)
        data = generator.generate_house_data(n_samples=100, noise_level=0.1)
        
        # Preprocess data
        preprocessor = PreprocessingPipeline(create_preprocessing_config())
        X_transformed, y = preprocessor.fit_transform(data)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, test_size=0.2, random_state=42
        )
        
        # Train model
        trainer = ModelTrainer(create_model_config())
        trainer.config['models'] = ['linear_regression']  # Use only one model for speed
        
        results = trainer.train_models(X_train, X_test, y_train, y_test)
        
        # Evaluate model
        evaluator = ModelEvaluator(trainer.best_model)
        y_pred = trainer.best_model.predict(X_test)
        metrics = evaluator.calculate_metrics(y_test, y_pred)
        
        # Check that pipeline completed successfully
        self.assertIsNotNone(trainer.best_model)
        self.assertGreater(metrics['r2'], 0)
        self.assertIn('linear_regression', results)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2) 