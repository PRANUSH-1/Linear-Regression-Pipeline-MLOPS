# Linear-Regression-Pipeline-MLOPS
This is the code for the Linear Regression Pipeline MLOps 
# Production-Ready Linear Regression MLOps Pipeline

A complete, production-ready MLOps pipeline for linear regression models with data generation, preprocessing, training, evaluation, deployment, and monitoring.

## 🚀 Features

- **Data Generation**: Synthetic data generation for house prices and sales prediction
- **Preprocessing Pipeline**: Automated data cleaning, feature engineering, and preprocessing
- **Model Training**: Multiple linear regression algorithms with hyperparameter tuning
- **Model Evaluation**: Comprehensive evaluation metrics and residual analysis
- **API Service**: FastAPI-based REST API for model serving
- **Monitoring**: Real-time model performance tracking and drift detection
- **Deployment**: Docker containerization and CI/CD pipeline setup
- **Testing**: Comprehensive unit tests and integration tests

## 📁 Project Structure

```
├── src/
│   ├── data_generator.py      # Synthetic data generation
│   ├── preprocessing.py       # Data preprocessing pipeline
│   ├── model.py              # Model training and evaluation
│   ├── evaluation.py         # Model evaluation and analysis
│   ├── api.py                # FastAPI REST API
│   ├── monitoring.py         # Model monitoring and drift detection
│   └── deployment.py         # Deployment configuration
├── tests/
│   └── test_pipeline.py      # Unit and integration tests
├── data/                     # Generated datasets
├── models/                   # Trained models and metadata
├── reports/                  # Evaluation reports and visualizations
├── deployment/               # Deployment configuration files
├── main.py                   # Main orchestration script
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose configuration
└── README.md                # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.9+
- Docker (for deployment)
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd linear-regression-mlops
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import sklearn, pandas, numpy, fastapi; print('Installation successful!')"
   ```

## 🚀 Quick Start

### 1. Run Complete Pipeline

```bash
python main.py
```

This will:
- Generate synthetic data (house prices or sales)
- Preprocess the data
- Train multiple models
- Evaluate and select the best model
- Generate reports and visualizations
- Setup deployment configuration

### 2. Run Individual Components

**Generate data only:**
```bash
python main.py --data-only
```

**Train models only:**
```bash
python main.py --train-only
```

**Setup deployment only:**
```bash
python main.py --deploy-only
```

### 3. Start API Server

```bash
# Start the FastAPI server
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

**API Endpoints:**
- `GET /` - API information
- `GET /health` - Health check
- `GET /model/info` - Model information
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `GET /metrics` - Model performance metrics

### 4. Make Predictions

**Single prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "features": {
         "square_feet": 2000.0,
         "bedrooms": 3,
         "bathrooms": 2,
         "age": 10,
         "distance_to_city": 5.0
       }
     }'
```

**Batch prediction:**
```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "features_list": [
         {
           "square_feet": 2000.0,
           "bedrooms": 3,
           "bathrooms": 2,
           "age": 10,
           "distance_to_city": 5.0
         },
         {
           "square_feet": 1500.0,
           "bedrooms": 2,
           "bathrooms": 1,
           "age": 5,
           "distance_to_city": 3.0
         }
       ]
     }'
```

## 🐳 Docker Deployment

### 1. Build and Run with Docker

```bash
# Build Docker image
docker build -t ml-api .

# Run container
docker run -p 8000:8000 ml-api
```

### 2. Deploy with Docker Compose

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 3. Production Deployment

```bash
# Deploy to production
docker-compose -f deployment/docker-compose.prod.yml up -d
```

## 📊 Monitoring and Analytics

### Model Performance Dashboard

The pipeline generates comprehensive reports in the `reports/` directory:

- `model_comparison.png` - Model performance comparison
- `prediction_analysis.png` - Prediction vs actual analysis
- `monitoring_dashboard.png` - Real-time monitoring dashboard

### Drift Detection

The monitoring system automatically detects:
- Data drift in feature distributions
- Model performance degradation
- Response time anomalies

### Alerts

Configure alerts by setting the `alert_webhook` in the monitoring configuration:

```python
config = {
    'alert_webhook': 'https://your-webhook-url.com/alerts',
    'drift_threshold': 0.1,
    'performance_threshold': 0.8
}
```

## 🧪 Testing

### Run All Tests

```bash
python -m pytest tests/ -v
```

### Run Specific Test Categories

```bash
# Unit tests only
python -m pytest tests/ -k "unit" -v

# Integration tests only
python -m pytest tests/ -k "integration" -v

# Data generation tests
python -m pytest tests/ -k "data" -v
```

### Test Coverage

```bash
pip install pytest-cov
python -m pytest tests/ --cov=src --cov-report=html
```

## ⚙️ Configuration

### Data Configuration

```python
data_config = {
    'n_samples': 2000,
    'noise_level': 0.1,
    'dataset_type': 'house_prices'  # or 'sales'
}
```

### Model Configuration

```python
model_config = {
    'models': ['linear_regression', 'ridge', 'lasso', 'elastic_net'],
    'cv_folds': 5,
    'random_state': 42
}
```

### Deployment Configuration

```python
deployment_config = {
    'docker': {
        'base_image': 'python:3.9-slim',
        'port': 8000
    },
    'ci_cd': {
        'platform': 'github_actions',
        'auto_deploy': True
    }
}
```

## 📈 Model Performance

### Supported Algorithms

1. **Linear Regression** - Basic linear regression
2. **Ridge Regression** - L2 regularization
3. **Lasso Regression** - L1 regularization with feature selection
4. **Elastic Net** - Combined L1 and L2 regularization
5. **Random Forest** - Ensemble method
6. **Gradient Boosting** - Advanced ensemble method

### Evaluation Metrics

- **R² Score** - Coefficient of determination
- **RMSE** - Root Mean Square Error
- **MAE** - Mean Absolute Error
- **Explained Variance** - Explained variance score
- **Max Error** - Maximum prediction error

## 🔧 Customization

### Adding New Data Types

1. Create a new method in `DataGenerator` class
2. Update the configuration in `main.py`
3. Add corresponding preprocessing steps

### Adding New Models

1. Import the model in `model.py`
2. Add to the `get_models()` method
3. Define hyperparameter grid in `get_hyperparameter_grids()`

### Custom Preprocessing

1. Extend the `FeatureEngineer` class
2. Add new transformations to the pipeline
3. Update the configuration

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure you're in the correct directory
   cd linear-regression-mlops
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ```

2. **Docker Build Failures**
   ```bash
   # Clean Docker cache
   docker system prune -a
   docker build --no-cache -t ml-api .
   ```

3. **API Connection Issues**
   ```bash
   # Check if API is running
   curl http://localhost:8000/health
   
   # Check Docker logs
   docker logs <container_id>
   ```

4. **Model Loading Issues**
   ```bash
   # Ensure model files exist
   ls -la models/
   
   # Regenerate models if needed
   python main.py --train-only
   ```

### Performance Optimization

1. **Increase Training Speed**
   - Reduce dataset size for testing
   - Use fewer models in configuration
   - Reduce cross-validation folds

2. **Memory Optimization**
   - Use smaller batch sizes
   - Reduce feature engineering complexity
   - Use data streaming for large datasets

## 📚 API Documentation

Once the API is running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pre-commit

# Setup pre-commit hooks
pre-commit install

# Run linting
flake8 src/ tests/
black src/ tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [scikit-learn](https://scikit-learn.org/) for machine learning algorithms
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Docker](https://www.docker.com/) for containerization
- [MLflow](https://mlflow.org/) for experiment tracking

## 📞 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the troubleshooting section
- Review the API documentation

---

**Happy Modeling! 🎯** 
