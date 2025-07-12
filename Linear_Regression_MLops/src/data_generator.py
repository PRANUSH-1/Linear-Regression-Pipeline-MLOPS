"""
Data Generator Module for Linear Regression Model
Generates synthetic data for training and testing
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataGenerator:
    """Generate synthetic data for linear regression training"""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize DataGenerator
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        logger.info(f"DataGenerator initialized with random_state={random_state}")
    
    def generate_house_data(self, n_samples: int = 1000, noise_level: float = 0.1) -> pd.DataFrame:
        """
        Generate synthetic house price data
        
        Args:
            n_samples: Number of samples to generate
            noise_level: Level of noise to add to the target variable
            
        Returns:
            DataFrame with features and target
        """
        logger.info(f"Generating {n_samples} house price samples with noise_level={noise_level}")
        
        # Generate features
        square_feet = np.random.normal(2000, 500, n_samples)
        bedrooms = np.random.randint(1, 6, n_samples)
        bathrooms = np.random.randint(1, 4, n_samples)
        age = np.random.randint(0, 50, n_samples)
        distance_to_city = np.random.exponential(5, n_samples)
        
        # Create feature matrix
        features = np.column_stack([square_feet, bedrooms, bathrooms, age, distance_to_city])
        
        # Define true coefficients (realistic house pricing)
        true_coefficients = np.array([150, 25000, 35000, -1000, -5000])
        true_intercept = 50000
        
        # Generate target with noise
        target = np.dot(features, true_coefficients) + true_intercept
        noise = np.random.normal(0, noise_level * np.std(target), n_samples)
        target += noise
        
        # Create DataFrame
        df = pd.DataFrame({
            'square_feet': square_feet,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'age': age,
            'distance_to_city': distance_to_city,
            'price': target
        })
        
        logger.info(f"Generated data shape: {df.shape}")
        logger.info(f"Price range: ${df['price'].min():.0f} - ${df['price'].max():.0f}")
        
        return df
    
    def generate_sales_data(self, n_samples: int = 1000, noise_level: float = 0.1) -> pd.DataFrame:
        """
        Generate synthetic sales data
        
        Args:
            n_samples: Number of samples to generate
            noise_level: Level of noise to add to the target variable
            
        Returns:
            DataFrame with features and target
        """
        logger.info(f"Generating {n_samples} sales samples with noise_level={noise_level}")
        
        # Generate features
        advertising_budget = np.random.exponential(1000, n_samples)
        price = np.random.normal(50, 15, n_samples)
        competitor_price = price + np.random.normal(0, 5, n_samples)
        season = np.random.randint(1, 5, n_samples)
        store_size = np.random.normal(1000, 200, n_samples)
        
        # Create feature matrix
        features = np.column_stack([advertising_budget, price, competitor_price, season, store_size])
        
        # Define true coefficients (realistic sales modeling)
        true_coefficients = np.array([0.8, -2.5, 1.2, 500, 0.3])
        true_intercept = 1000
        
        # Generate target with noise
        target = np.dot(features, true_coefficients) + true_intercept
        noise = np.random.normal(0, noise_level * np.std(target), n_samples)
        target += noise
        target = np.maximum(target, 0)  # Sales can't be negative
        
        # Create DataFrame
        df = pd.DataFrame({
            'advertising_budget': advertising_budget,
            'price': price,
            'competitor_price': competitor_price,
            'season': season,
            'store_size': store_size,
            'sales': target
        })
        
        logger.info(f"Generated data shape: {df.shape}")
        logger.info(f"Sales range: {df['sales'].min():.0f} - {df['sales'].max():.0f}")
        
        return df
    
    def split_data(self, df: pd.DataFrame, target_col: str, 
                   test_size: float = 0.2, val_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            test_size: Proportion of data for test set
            val_size: Proportion of remaining data for validation set
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"Splitting data with test_size={test_size}, val_size={val_size}")
        
        # First split: train+val vs test
        test_idx = np.random.choice(df.index, size=int(len(df) * test_size), replace=False)
        test_df = df.loc[test_idx]
        train_val_df = df.drop(test_idx)
        
        # Second split: train vs val
        val_idx = np.random.choice(train_val_df.index, 
                                 size=int(len(train_val_df) * val_size), replace=False)
        val_df = train_val_df.loc[val_idx]
        train_df = train_val_df.drop(val_idx)
        
        logger.info(f"Train set: {len(train_df)} samples")
        logger.info(f"Validation set: {len(val_df)} samples")
        logger.info(f"Test set: {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def save_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                  test_df: pd.DataFrame, output_dir: str = "data"):
        """
        Save datasets to CSV files
        
        Args:
            train_df: Training dataset
            val_df: Validation dataset
            test_df: Test dataset
            output_dir: Directory to save files
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        train_df.to_csv(f"{output_dir}/train.csv", index=False)
        val_df.to_csv(f"{output_dir}/val.csv", index=False)
        test_df.to_csv(f"{output_dir}/test.csv", index=False)
        
        logger.info(f"Data saved to {output_dir}/")


if __name__ == "__main__":
    # Example usage
    generator = DataGenerator(random_state=42)
    
    # Generate house price data
    house_data = generator.generate_house_data(n_samples=2000, noise_level=0.1)
    train_house, val_house, test_house = generator.split_data(house_data, 'price')
    generator.save_data(train_house, val_house, test_house, "data/house_prices")
    
    # Generate sales data
    sales_data = generator.generate_sales_data(n_samples=2000, noise_level=0.1)
    train_sales, val_sales, test_sales = generator.split_data(sales_data, 'sales')
    generator.save_data(train_sales, val_sales, test_sales, "data/sales") 