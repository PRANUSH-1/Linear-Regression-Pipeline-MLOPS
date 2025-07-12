"""
Main Orchestration Script for Linear Regression MLOps Pipeline
Coordinates data generation, preprocessing, training, evaluation, and deployment
"""

import os
import sys
import logging
import argparse
import json
from datetime import datetime
from typing import Dict, Any, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_generator import DataGenerator
from preprocessing import PreprocessingPipeline, create_preprocessing_config
from model import ModelTrainer, create_model_config
from evaluation import ModelEvaluator
from api import app
from monitoring import ModelMonitor, create_monitoring_config
from deployment import DockerDeployer, CICDPipeline, ProductionDeployer, create_deployment_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mlops_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MLOpsPipeline:
    """Complete MLOps pipeline orchestration"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize MLOps pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self.load_config(config_path)
        self.setup_directories()
        
        # Initialize components
        self.data_generator = DataGenerator(random_state=self.config.get('random_state', 42))
        self.preprocessor = PreprocessingPipeline(create_preprocessing_config())
        self.model_trainer = ModelTrainer(create_model_config())
        self.monitor = ModelMonitor(create_monitoring_config())
        
    def load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
        else:
            config = {
                'random_state': 42,
                'data': {
                    'n_samples': 2000,
                    'noise_level': 0.1,
                    'dataset_type': 'house_prices'  # or 'sales'
                },
                'model': {
                    'test_size': 0.2,
                    'val_size': 0.2,
                    'cv_folds': 5
                },
                'deployment': {
                    'enable_docker': True,
                    'enable_cicd': True,
                    'enable_monitoring': True
                }
            }
            logger.info("Using default configuration")
        
        return config
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            'data',
            'models',
            'logs',
            'reports',
            'deployment',
            'tests'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def generate_data(self) -> str:
        """Generate training data"""
        logger.info("Starting data generation...")
        
        data_config = self.config['data']
        n_samples = data_config['n_samples']
        noise_level = data_config['noise_level']
        dataset_type = data_config['dataset_type']
        
        if dataset_type == 'house_prices':
            data = self.data_generator.generate_house_data(n_samples, noise_level)
            target_col = 'price'
        elif dataset_type == 'sales':
            data = self.data_generator.generate_sales_data(n_samples, noise_level)
            target_col = 'sales'
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Split data
        train_df, val_df, test_df = self.data_generator.split_data(
            data, target_col, 
            test_size=self.config['model']['test_size'],
            val_size=self.config['model']['val_size']
        )
        
        # Save data
        output_dir = f"data/{dataset_type}"
        self.data_generator.save_data(train_df, val_df, test_df, output_dir)
        
        # Set reference data for monitoring
        self.monitor.set_reference_data(train_df)
        
        logger.info(f"Data generation completed. Saved to {output_dir}")
        return output_dir
    
    def preprocess_data(self, data_dir: str):
        """Preprocess the data"""
        logger.info("Starting data preprocessing...")
        
        # Load training data
        train_df = pd.read_csv(f"{data_dir}/train.csv")
        
        # Fit and transform training data
        X_train_transformed, y_train = self.preprocessor.fit_transform(train_df)
        
        # Transform validation and test data
        val_df = pd.read_csv(f"{data_dir}/val.csv")
        test_df = pd.read_csv(f"{data_dir}/test.csv")
        
        X_val_transformed = self.preprocessor.transform(val_df)
        y_val = val_df.iloc[:, -1].values  # Assuming target is last column
        
        X_test_transformed = self.preprocessor.transform(test_df)
        y_test = test_df.iloc[:, -1].values
        
        # Save preprocessed data
        preprocessed_data = {
            'X_train': X_train_transformed,
            'y_train': y_train,
            'X_val': X_val_transformed,
            'y_val': y_val,
            'X_test': X_test_transformed,
            'y_test': y_test
        }
        
        joblib.dump(preprocessed_data, 'data/preprocessed_data.pkl')
        self.preprocessor.save_pipeline('models/preprocessor.pkl')
        
        logger.info("Data preprocessing completed")
        return preprocessed_data
    
    def train_models(self, preprocessed_data: Dict[str, Any]):
        """Train and evaluate models"""
        logger.info("Starting model training...")
        
        # Train models
        results = self.model_trainer.train_models(
            preprocessed_data['X_train'],
            preprocessed_data['X_val'],
            preprocessed_data['y_train'],
            preprocessed_data['y_val']
        )
        
        # Evaluate best model on test set
        evaluator = ModelEvaluator(self.model_trainer.best_model)
        y_test_pred = self.model_trainer.best_model.predict(preprocessed_data['X_test'])
        
        # Generate evaluation report
        report = evaluator.generate_evaluation_report(
            preprocessed_data['y_test'], y_test_pred, "Test"
        )
        
        # Save evaluation results
        evaluator.save_evaluation_results(
            preprocessed_data['y_test'], y_test_pred, 
            'reports/evaluation_results.pkl'
        )
        
        # Save best model
        self.model_trainer.save_best_model('models/best_model.pkl')
        
        # Create model metadata
        metadata = {
            'model_version': '1.0.0',
            'training_date': datetime.now().isoformat(),
            'model_type': type(self.model_trainer.best_model).__name__,
            'performance_metrics': {
                'test_r2': evaluator.calculate_metrics(preprocessed_data['y_test'], y_test_pred)['r2'],
                'test_rmse': evaluator.calculate_metrics(preprocessed_data['y_test'], y_test_pred)['rmse']
            },
            'feature_names': list(self.preprocessor.feature_columns) if hasattr(self.preprocessor, 'feature_columns') else []
        }
        
        with open('models/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Plot results
        self.model_trainer.plot_results('reports/model_comparison.png')
        evaluator.plot_prediction_analysis(
            preprocessed_data['y_test'], y_test_pred, 
            'reports/prediction_analysis.png'
        )
        
        logger.info("Model training and evaluation completed")
        return results
    
    def setup_monitoring(self):
        """Setup model monitoring"""
        logger.info("Setting up model monitoring...")
        
        # Save monitoring configuration
        monitoring_config = create_monitoring_config()
        with open('models/monitoring_config.json', 'w') as f:
            json.dump(monitoring_config, f, indent=2)
        
        # Create monitoring dashboard
        self.monitor.plot_monitoring_dashboard('reports/monitoring_dashboard.png')
        
        logger.info("Model monitoring setup completed")
    
    def setup_deployment(self):
        """Setup deployment configuration"""
        logger.info("Setting up deployment configuration...")
        
        deployment_config = create_deployment_config()
        
        if deployment_config['deployment']['enable_docker']:
            docker_deployer = DockerDeployer(deployment_config['docker'])
            docker_deployer.create_dockerfile('deployment/Dockerfile')
            docker_deployer.create_docker_compose('deployment/docker-compose.yml')
            docker_deployer.create_nginx_config('deployment/nginx.conf')
        
        if deployment_config['ci_cd']['enable_cicd']:
            cicd = CICDPipeline(deployment_config['ci_cd'])
            cicd.create_github_actions('.github/workflows/deploy.yml')
            cicd.create_jenkins_pipeline('deployment/Jenkinsfile')
        
        if deployment_config['production']['enable_production']:
            prod_deployer = ProductionDeployer(deployment_config['production'])
            prod_deployer.create_production_compose('deployment/docker-compose.prod.yml')
            prod_deployer.create_monitoring_config('deployment/prometheus.yml')
        
        logger.info("Deployment configuration setup completed")
    
    def run_full_pipeline(self):
        """Run the complete MLOps pipeline"""
        logger.info("Starting complete MLOps pipeline...")
        
        try:
            # Step 1: Generate data
            data_dir = self.generate_data()
            
            # Step 2: Preprocess data
            preprocessed_data = self.preprocess_data(data_dir)
            
            # Step 3: Train models
            training_results = self.train_models(preprocessed_data)
            
            # Step 4: Setup monitoring
            self.setup_monitoring()
            
            # Step 5: Setup deployment
            self.setup_deployment()
            
            logger.info("MLOps pipeline completed successfully!")
            
            # Print summary
            self.print_pipeline_summary(training_results)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def print_pipeline_summary(self, training_results: Dict[str, Any]):
        """Print pipeline summary"""
        print("\n" + "="*60)
        print("MLOPS PIPELINE SUMMARY")
        print("="*60)
        
        print(f"Pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Data generated: {self.config['data']['n_samples']} samples")
        print(f"Dataset type: {self.config['data']['dataset_type']}")
        
        # Best model performance
        if self.model_trainer.best_model:
            best_model_name = type(self.model_trainer.best_model).__name__
            best_score = self.model_trainer.best_score
            print(f"Best model: {best_model_name} (R² = {best_score:.4f})")
        
        print("\nFiles created:")
        print("- data/: Training, validation, and test datasets")
        print("- models/: Trained model and preprocessor")
        print("- reports/: Evaluation reports and visualizations")
        print("- deployment/: Docker and CI/CD configuration")
        
        print("\nNext steps:")
        print("1. Review model performance in reports/")
        print("2. Test the API: python -m uvicorn src.api:app --reload")
        print("3. Deploy using Docker: docker-compose up -d")
        print("4. Monitor model performance in production")
        
        print("="*60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='MLOps Pipeline for Linear Regression')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data-only', action='store_true', help='Generate data only')
    parser.add_argument('--train-only', action='store_true', help='Train models only')
    parser.add_argument('--deploy-only', action='store_true', help='Setup deployment only')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = MLOpsPipeline(args.config)
    
    if args.data_only:
        logger.info("Running data generation only...")
        pipeline.generate_data()
    elif args.train_only:
        logger.info("Running model training only...")
        data_dir = f"data/{pipeline.config['data']['dataset_type']}"
        preprocessed_data = pipeline.preprocess_data(data_dir)
        pipeline.train_models(preprocessed_data)
    elif args.deploy_only:
        logger.info("Setting up deployment only...")
        pipeline.setup_deployment()
    else:
        # Run full pipeline
        pipeline.run_full_pipeline()


if __name__ == "__main__":
    # Import required modules
    import pandas as pd
    import joblib
    
    main() 