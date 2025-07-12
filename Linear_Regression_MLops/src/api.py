"""
FastAPI REST API for Linear Regression Model Serving
Provides prediction endpoints, health checks, and model management
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib
import logging
import os
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Linear Regression Model API",
    description="Production-ready API for linear regression model serving",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL_PATH = "models/best_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
MODEL_METADATA_PATH = "models/metadata.json"
model = None
preprocessor = None
model_metadata = {}


class PredictionRequest(BaseModel):
    """Request model for predictions"""
    features: Dict[str, float] = Field(..., description="Input features for prediction")
    
    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "square_feet": 2000.0,
                    "bedrooms": 3,
                    "bathrooms": 2,
                    "age": 10,
                    "distance_to_city": 5.0
                }
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    features_list: List[Dict[str, float]] = Field(..., description="List of input features")
    
    class Config:
        schema_extra = {
            "example": {
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
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: float
    confidence: Optional[float] = None
    timestamp: str
    model_version: str
    features_used: List[str]


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[float]
    confidence_scores: Optional[List[float]] = None
    timestamp: str
    model_version: str
    total_predictions: int


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    timestamp: str
    model_loaded: bool
    model_version: str
    uptime: str


class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    model_type: str
    model_version: str
    training_date: str
    performance_metrics: Dict[str, float]
    feature_names: List[str]
    model_size: str


def load_model():
    """Load the trained model and preprocessor"""
    global model, preprocessor, model_metadata
    
    try:
        # Load model
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded from {MODEL_PATH}")
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}")
            model = None
        
        # Load preprocessor
        if os.path.exists(PREPROCESSOR_PATH):
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            logger.info(f"Preprocessor loaded from {PREPROCESSOR_PATH}")
        else:
            logger.warning(f"Preprocessor file not found at {PREPROCESSOR_PATH}")
            preprocessor = None
        
        # Load metadata
        if os.path.exists(MODEL_METADATA_PATH):
            with open(MODEL_METADATA_PATH, 'r') as f:
                model_metadata = json.load(f)
            logger.info(f"Model metadata loaded from {MODEL_METADATA_PATH}")
        else:
            model_metadata = {
                "model_version": "1.0.0",
                "training_date": datetime.now().isoformat(),
                "model_type": "Linear Regression"
            }
            logger.warning("Model metadata file not found, using defaults")
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model = None
        preprocessor = None


def get_model():
    """Dependency to get the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


def get_preprocessor():
    """Dependency to get the loaded preprocessor"""
    if preprocessor is None:
        raise HTTPException(status_code=503, detail="Preprocessor not loaded")
    return preprocessor


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Linear Regression Model API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    from datetime import datetime, timedelta
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None,
        model_version=model_metadata.get("model_version", "unknown"),
        uptime=str(datetime.now() - datetime.fromisoformat(model_metadata.get("training_date", datetime.now().isoformat())))
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Get model size
    model_size = "unknown"
    if os.path.exists(MODEL_PATH):
        model_size = f"{os.path.getsize(MODEL_PATH) / 1024:.2f} KB"
    
    # Get feature names
    feature_names = []
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_.tolist()
    elif hasattr(model, 'coef_'):
        feature_names = [f"feature_{i}" for i in range(len(model.coef_))]
    
    return ModelInfoResponse(
        model_type=model_metadata.get("model_type", "Unknown"),
        model_version=model_metadata.get("model_version", "unknown"),
        training_date=model_metadata.get("training_date", "unknown"),
        performance_metrics=model_metadata.get("performance_metrics", {}),
        feature_names=feature_names,
        model_size=model_size
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, model=Depends(get_model)):
    """Make a single prediction"""
    try:
        # Convert features to DataFrame
        features_df = pd.DataFrame([request.features])
        
        # Preprocess features if preprocessor is available
        if preprocessor is not None:
            features_processed = preprocessor.transform(features_df)
        else:
            features_processed = features_df.values
        
        # Make prediction
        prediction = model.predict(features_processed)[0]
        
        # Get feature names
        feature_names = list(request.features.keys())
        
        logger.info(f"Prediction made: {prediction:.2f}")
        
        return PredictionResponse(
            prediction=float(prediction),
            confidence=None,  # Could be implemented for ensemble models
            timestamp=datetime.now().isoformat(),
            model_version=model_metadata.get("model_version", "unknown"),
            features_used=feature_names
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest, model=Depends(get_model)):
    """Make batch predictions"""
    try:
        # Convert features to DataFrame
        features_df = pd.DataFrame(request.features_list)
        
        # Preprocess features if preprocessor is available
        if preprocessor is not None:
            features_processed = preprocessor.transform(features_df)
        else:
            features_processed = features_df.values
        
        # Make predictions
        predictions = model.predict(features_processed)
        
        logger.info(f"Batch prediction made for {len(predictions)} samples")
        
        return BatchPredictionResponse(
            predictions=[float(p) for p in predictions],
            confidence_scores=None,  # Could be implemented for ensemble models
            timestamp=datetime.now().isoformat(),
            model_version=model_metadata.get("model_version", "unknown"),
            total_predictions=len(predictions)
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/model/reload")
async def reload_model(background_tasks: BackgroundTasks):
    """Reload the model in the background"""
    background_tasks.add_task(load_model)
    return {"message": "Model reload initiated"}


@app.get("/model/features")
async def get_expected_features():
    """Get expected feature names and types"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    feature_info = {}
    if hasattr(model, 'feature_names_in_'):
        for i, name in enumerate(model.feature_names_in_):
            feature_info[name] = "float"
    elif hasattr(model, 'coef_'):
        for i in range(len(model.coef_)):
            feature_info[f"feature_{i}"] = "float"
    
    return {
        "expected_features": feature_info,
        "model_version": model_metadata.get("model_version", "unknown")
    }


@app.get("/metrics")
async def get_model_metrics():
    """Get model performance metrics"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "performance_metrics": model_metadata.get("performance_metrics", {}),
        "model_version": model_metadata.get("model_version", "unknown"),
        "last_updated": model_metadata.get("training_date", "unknown")
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "detail": str(exc)}


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "detail": str(exc)}


if __name__ == "__main__":
    import uvicorn
    
    # Load model before starting server
    load_model()
    
    # Start server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 