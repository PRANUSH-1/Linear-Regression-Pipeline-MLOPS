"""
Preprocessing Module for Linear Regression Model
Handles data cleaning, feature engineering, and preprocessing pipelines
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import List, Dict, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator(BaseEstimator, TransformerMixin):
    """Validate and clean input data"""
    
    def __init__(self, expected_columns: List[str], target_column: str):
        """
        Initialize DataValidator
        
        Args:
            expected_columns: List of expected feature columns
            target_column: Name of target column
        """
        self.expected_columns = expected_columns
        self.target_column = target_column
        self.feature_columns = [col for col in expected_columns if col != target_column]
    
    def fit(self, X, y=None):
        """Fit the validator"""
        return self
    
    def transform(self, X):
        """
        Validate and clean input data
        
        Args:
            X: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            raise ValueError("Input must be a pandas DataFrame")
        
        # Check for required columns
        missing_cols = set(self.expected_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_rows:
            logger.warning(f"Removed {initial_rows - len(df)} duplicate rows")
        
        # Handle outliers for numerical columns
        for col in self.feature_columns:
            if df[col].dtype in ['int64', 'float64']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                if len(outliers) > 0:
                    logger.warning(f"Found {len(outliers)} outliers in column {col}")
                    # Cap outliers instead of removing
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        logger.info(f"Data validation completed. Shape: {df.shape}")
        return df


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Engineer new features from existing ones"""
    
    def __init__(self, feature_config: Dict[str, Any]):
        """
        Initialize FeatureEngineer
        
        Args:
            feature_config: Configuration for feature engineering
        """
        self.feature_config = feature_config
        self.feature_names_ = None
    
    def fit(self, X, y=None):
        """Fit the feature engineer"""
        return self
    
    def transform(self, X):
        """
        Engineer features
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df = X.copy()
        
        # Square footage per bedroom (for house data)
        if 'square_feet' in df.columns and 'bedrooms' in df.columns:
            df['sqft_per_bedroom'] = df['square_feet'] / df['bedrooms'].replace(0, 1)
        
        # Price ratio (for sales data)
        if 'price' in df.columns and 'competitor_price' in df.columns:
            df['price_ratio'] = df['price'] / df['competitor_price']
        
        # Age categories (for house data)
        if 'age' in df.columns:
            df['age_category'] = pd.cut(df['age'], 
                                      bins=[0, 10, 25, 50, 100], 
                                      labels=['new', 'young', 'mature', 'old'])
        
        # Season categories (for sales data)
        if 'season' in df.columns:
            df['season_name'] = df['season'].map({
                1: 'spring', 2: 'summer', 3: 'fall', 4: 'winter'
            })
        
        # Log transformations for skewed features
        for col in self.feature_config.get('log_transform', []):
            if col in df.columns and df[col].min() > 0:
                df[f'log_{col}'] = np.log1p(df[col])
        
        # Interaction features
        for interaction in self.feature_config.get('interactions', []):
            if len(interaction) == 2 and all(col in df.columns for col in interaction):
                col1, col2 = interaction
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        
        self.feature_names_ = df.columns.tolist()
        logger.info(f"Feature engineering completed. New shape: {df.shape}")
        return df


class PreprocessingPipeline:
    """Complete preprocessing pipeline for linear regression"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preprocessing pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.pipeline = None
        self.feature_columns = None
        self.target_column = config.get('target_column', 'target')
        
    def build_pipeline(self, data: pd.DataFrame) -> Pipeline:
        """
        Build the preprocessing pipeline
        
        Args:
            data: Training data to determine column types
            
        Returns:
            Fitted preprocessing pipeline
        """
        # Determine feature columns
        self.feature_columns = [col for col in data.columns if col != self.target_column]
        
        # Separate numerical and categorical columns
        numerical_cols = data[self.feature_columns].select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = data[self.feature_columns].select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Numerical columns: {numerical_cols}")
        logger.info(f"Categorical columns: {categorical_cols}")
        
        # Define preprocessing steps
        numerical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', pd.get_dummies)  # Simple one-hot encoding
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='passthrough'
        )
        
        # Build complete pipeline
        self.pipeline = Pipeline([
            ('validator', DataValidator(self.feature_columns + [self.target_column], self.target_column)),
            ('engineer', FeatureEngineer(self.config.get('feature_engineering', {}))),
            ('preprocessor', preprocessor)
        ])
        
        return self.pipeline
    
    def fit_transform(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit and transform the data
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (X_transformed, y)
        """
        if self.pipeline is None:
            self.build_pipeline(data)
        
        # Separate features and target
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]
        
        # Fit and transform
        X_transformed = self.pipeline.fit_transform(X)
        y_transformed = y.values
        
        logger.info(f"Preprocessing completed. X shape: {X_transformed.shape}, y shape: {y_transformed.shape}")
        
        return X_transformed, y_transformed
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted pipeline
        
        Args:
            data: Input DataFrame
            
        Returns:
            Transformed features
        """
        if self.pipeline is None:
            raise ValueError("Pipeline must be fitted first")
        
        X = data.drop(columns=[self.target_column])
        X_transformed = self.pipeline.transform(X)
        
        return X_transformed
    
    def save_pipeline(self, filepath: str):
        """Save the fitted pipeline"""
        import joblib
        joblib.dump(self.pipeline, filepath)
        logger.info(f"Pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath: str):
        """Load a fitted pipeline"""
        import joblib
        self.pipeline = joblib.load(filepath)
        logger.info(f"Pipeline loaded from {filepath}")


def create_preprocessing_config() -> Dict[str, Any]:
    """Create default preprocessing configuration"""
    return {
        'target_column': 'price',  # or 'sales'
        'feature_engineering': {
            'log_transform': ['square_feet', 'advertising_budget', 'store_size'],
            'interactions': [
                ['square_feet', 'bedrooms'],
                ['price', 'competitor_price']
            ]
        },
        'scaling': 'standard',  # 'standard', 'robust', 'minmax'
        'imputation': 'median',  # 'mean', 'median', 'most_frequent'
        'outlier_handling': 'clip'  # 'clip', 'remove', 'ignore'
    }


if __name__ == "__main__":
    # Example usage
    from data_generator import DataGenerator
    
    # Generate sample data
    generator = DataGenerator(random_state=42)
    data = generator.generate_house_data(n_samples=1000)
    
    # Create and fit preprocessing pipeline
    config = create_preprocessing_config()
    preprocessor = PreprocessingPipeline(config)
    
    X_transformed, y = preprocessor.fit_transform(data)
    print(f"Transformed data shape: {X_transformed.shape}")
    print(f"Target shape: {y.shape}") 