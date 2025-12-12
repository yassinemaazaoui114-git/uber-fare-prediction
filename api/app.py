"""
FastAPI backend for Uber Fare Prediction
Serves predictions and static HTML pages
"""
from datetime import datetime
from typing import Optional
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.data_loader import add_enhanced_features

app = FastAPI(
    title="Uber Fare Prediction API",
    description="Predict NYC Uber fares using ML models",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files path
static_path = os.path.join(parent_dir, "static")


# Request/Response models
class RideInput(BaseModel):
    pickup_datetime: datetime
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    passenger_count: int


class PredictionResponse(BaseModel):
    predicted_fare: float
    trip_distance_km: float
    is_rush_hour: bool
    is_weekend: bool
    is_late_night: bool


# Load best model (CatBoost)
BEST_MODEL = None
MODEL_NAME = "CatBoost"


def load_model():
    """Load the best trained model"""
    global BEST_MODEL
    model_path = os.path.join(parent_dir, "models", "catboost_model.joblib")
    try:
        loaded = joblib.load(model_path)
        
        # Check if it's a dict with 'model' key
        if isinstance(loaded, dict) and 'model' in loaded:
            BEST_MODEL = loaded['model']
            print(f"✅ Loaded {MODEL_NAME} model from dict")
        elif hasattr(loaded, 'predict'):
            BEST_MODEL = loaded
            print(f"✅ Loaded {MODEL_NAME} model directly")
        else:
            BEST_MODEL = loaded
            print(f"⚠️  Loaded unknown type: {type(loaded)}")
            
        print(f"   Model type: {type(BEST_MODEL)}")
        
        # Verify it has predict method
        if hasattr(BEST_MODEL, 'predict'):
            print(f"   ✅ Model has predict method")
        else:
            print(f"   ❌ ERROR: Model doesn't have predict method!")
                
    except Exception as e:
        print(f"⚠️  Could not load model: {e}")
        import traceback
        traceback.print_exc()



def haversine(lon1, lat1, lon2, lat2):
    """Calculate distance between two points"""
    R = 6371  # Earth radius in km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(
        delta_lambda / 2
    ) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def prepare_features(ride: RideInput) -> pd.DataFrame:
    """Transform ride input into model features"""
    # Create dataframe
    df = pd.DataFrame(
        [
            {
                "pickup_datetime": ride.pickup_datetime,
                "pickup_longitude": ride.pickup_longitude,
                "pickup_latitude": ride.pickup_latitude,
                "dropoff_longitude": ride.dropoff_longitude,
                "dropoff_latitude": ride.dropoff_latitude,
                "passenger_count": ride.passenger_count,
            }
        ]
    )

    # Extract time features
    df["hour"] = df["pickup_datetime"].dt.hour
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
    df["month"] = df["pickup_datetime"].dt.month

    # Calculate distance
    df["trip_km"] = haversine(
        df["pickup_longitude"],
        df["pickup_latitude"],
        df["dropoff_longitude"],
        df["dropoff_latitude"],
    )

    # Add enhanced features
    df = add_enhanced_features(df)

    # Select features in correct order
    feature_cols = [
        "hour",
        "day_of_week",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "trip_km",
        "is_rush_hour",
        "is_weekend",
        "is_late_night",
        "distance_category",
        "hour_distance_interaction",
    ]

    return df[feature_cols]


@app.on_event("startup")
async def startup_event():
    """Load model when API starts"""
    load_model()
    if not BEST_MODEL:
        print("⚠️  WARNING: No model loaded!")


# Serve HTML pages
@app.get("/")
async def serve_home():
    """Serve home page"""
    return FileResponse(os.path.join(static_path, "index.html"))


@app.get("/predict-page")
async def serve_predict():
    """Serve prediction page"""
    return FileResponse(os.path.join(static_path, "predict.html"))


@app.get("/retrain-page")
async def serve_retrain():
    """Serve retrain page"""
    return FileResponse(os.path.join(static_path, "retrain.html"))


# API endpoints
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": BEST_MODEL is not None,
        "model_name": MODEL_NAME,
        "model_type": str(type(BEST_MODEL)),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_fare(ride: RideInput):
    """
    Predict fare for a given ride using the best model (CatBoost)
    """
    if not BEST_MODEL:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Prepare features
        X = prepare_features(ride)
        
        print(f"Features shape: {X.shape}")
        print(f"Features: {X.columns.tolist()}")
        print(f"Model type: {type(BEST_MODEL)}")

        # Make prediction
        prediction = BEST_MODEL.predict(X)[0]
        
        print(f"Prediction: {prediction}")

        return PredictionResponse(
            predicted_fare=round(float(prediction), 2),
            trip_distance_km=round(float(X["trip_km"].values[0]), 2),
            is_rush_hour=bool(X["is_rush_hour"].values[0]),
            is_weekend=bool(X["is_weekend"].values[0]),
            is_late_night=bool(X["is_late_night"].values[0]),
        )

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ============ RETRAIN ENDPOINTS ============

class RetrainRequest(BaseModel):
    model_type: str = Field(..., description="Model to train: catboost, lightgbm, xgboost, etc.")
    hyperparameters: dict = Field(default_factory=dict, description="Model hyperparameters")
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)


class RetrainResponse(BaseModel):
    success: bool
    message: str
    model_name: str
    metrics: dict
    hyperparameters: dict
    training_time: float


@app.post("/retrain", response_model=RetrainResponse)
async def retrain_model(request: RetrainRequest):
    """
    Retrain a model with custom hyperparameters
    """
    import time
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    
    start_time = time.time()
    
    try:
        # Load data
        data_path = os.path.join(parent_dir, "data", "datauber.csv")
        if not os.path.exists(data_path):
            raise HTTPException(status_code=404, detail="Training data not found")
        
        # Import data loader
        from utils.data_loader import load_and_preprocess_data
        
        # Load and preprocess
        X_train, X_test, y_train, y_test = load_and_preprocess_data(
            filepath=data_path,
            test_size=request.test_size,
            random_state=42
        )
        
        # Select and configure model
        model = None
        model_name = request.model_type.lower()
        params = request.hyperparameters
        
        if model_name == "catboost":
            from catboost import CatBoostRegressor
            default_params = {
                "iterations": 100,
                "learning_rate": 0.1,
                "depth": 6,
                "verbose": False,
                "random_state": 42
            }
            default_params.update(params)
            model = CatBoostRegressor(**default_params)
            
        elif model_name == "lightgbm":
            from lightgbm import LGBMRegressor
            default_params = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "verbose": -1,
                "random_state": 42
            }
            default_params.update(params)
            model = LGBMRegressor(**default_params)
            
        elif model_name == "xgboost":
            from xgboost import XGBRegressor
            default_params = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "random_state": 42
            }
            default_params.update(params)
            model = XGBRegressor(**default_params)
            
        elif model_name == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            default_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42,
                "n_jobs": -1
            }
            default_params.update(params)
            model = RandomForestRegressor(**default_params)
            
        elif model_name == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingRegressor
            default_params = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 5,
                "random_state": 42
            }
            default_params.update(params)
            model = GradientBoostingRegressor(**default_params)
            
        elif model_name == "linear_regression":
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            
        elif model_name == "knn":
            from sklearn.neighbors import KNeighborsRegressor
            default_params = {
                "n_neighbors": 5,
                "weights": "uniform",
                "n_jobs": -1
            }
            default_params.update(params)
            model = KNeighborsRegressor(**default_params)
            
        elif model_name == "decision_tree":
            from sklearn.tree import DecisionTreeRegressor
            default_params = {
                "max_depth": 10,
                "random_state": 42
            }
            default_params.update(params)
            model = DecisionTreeRegressor(**default_params)
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model type: {model_name}")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # Save model
        model_save_path = os.path.join(parent_dir, "models", f"{model_name}_model_retrained.joblib")
        joblib.dump({"model": model, "preprocessor": None}, model_save_path)
        
        training_time = time.time() - start_time
        
        return RetrainResponse(
            success=True,
            message=f"Model trained successfully in {training_time:.2f} seconds",
            model_name=model_name.replace("_", " ").title(),
            metrics={
                "r2_score": round(float(r2), 4),
                "rmse": round(float(rmse), 4),
                "mae": round(float(mae), 4),
                "training_samples": int(len(X_train)),
                "test_samples": int(len(X_test))
            },
            hyperparameters=params if params else {"default": "used"},
            training_time=round(training_time, 2)
        )
        
    except Exception as e:
        print(f"Retraining error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.get("/models/hyperparameters/{model_type}")
async def get_model_hyperparameters(model_type: str):
    """
    Get default hyperparameters for a specific model
    """
    hyperparams = {
        "catboost": {
            "iterations": {"type": "int", "default": 100, "min": 50, "max": 500, "step": 50},
            "learning_rate": {"type": "float", "default": 0.1, "min": 0.01, "max": 0.3, "step": 0.01},
            "depth": {"type": "int", "default": 6, "min": 3, "max": 10, "step": 1}
        },
        "lightgbm": {
            "n_estimators": {"type": "int", "default": 100, "min": 50, "max": 500, "step": 50},
            "learning_rate": {"type": "float", "default": 0.1, "min": 0.01, "max": 0.3, "step": 0.01},
            "max_depth": {"type": "int", "default": 6, "min": 3, "max": 15, "step": 1}
        },
        "xgboost": {
            "n_estimators": {"type": "int", "default": 100, "min": 50, "max": 500, "step": 50},
            "learning_rate": {"type": "float", "default": 0.1, "min": 0.01, "max": 0.3, "step": 0.01},
            "max_depth": {"type": "int", "default": 6, "min": 3, "max": 15, "step": 1}
        },
        "random_forest": {
            "n_estimators": {"type": "int", "default": 100, "min": 50, "max": 300, "step": 50},
            "max_depth": {"type": "int", "default": 10, "min": 5, "max": 30, "step": 5}
        },
        "gradient_boosting": {
            "n_estimators": {"type": "int", "default": 100, "min": 50, "max": 300, "step": 50},
            "learning_rate": {"type": "float", "default": 0.1, "min": 0.01, "max": 0.3, "step": 0.01},
            "max_depth": {"type": "int", "default": 5, "min": 3, "max": 10, "step": 1}
        },
        "knn": {
            "n_neighbors": {"type": "int", "default": 5, "min": 1, "max": 50, "step": 1},
            "weights": {"type": "select", "default": "uniform", "options": ["uniform", "distance"]}
        },
        "decision_tree": {
            "max_depth": {"type": "int", "default": 10, "min": 3, "max": 30, "step": 1}
        },
        "linear_regression": {}
    }
    
    if model_type.lower() not in hyperparams:
        raise HTTPException(status_code=404, detail=f"Model type '{model_type}' not found")
    
    return {
        "model_type": model_type,
        "hyperparameters": hyperparams[model_type.lower()]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
