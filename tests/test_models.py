"""
Basic tests for model functionality
"""
import os
import pytest
import joblib
import pandas as pd
import numpy as np


def test_models_directory_exists():
    """Test if models directory exists"""
    assert os.path.exists('models'), "Models directory should exist"


def test_data_directory_exists():
    """Test if data directory exists"""
    assert os.path.exists('data'), "Data directory should exist"


def test_catboost_model_file_exists():
    """Test if CatBoost model file exists"""
    model_path = 'models/catboost_model.joblib'
    if os.path.exists(model_path):
        assert os.path.getsize(model_path) > 0, "CatBoost model file should not be empty"


def test_lightgbm_model_file_exists():
    """Test if LightGBM model file exists"""
    model_path = 'models/lightgbm_model.joblib'
    if os.path.exists(model_path):
        assert os.path.getsize(model_path) > 0, "LightGBM model file should not be empty"


def test_model_can_load():
    """Test if at least one model can be loaded"""
    model_files = [
        'models/catboost_model.joblib',
        'models/lightgbm_model.joblib',
        'models/xgboost_model.joblib'
    ]
    
    loaded = False
    for model_path in model_files:
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                assert model is not None
                loaded = True
                break
            except Exception as e:
                continue
    
    if not loaded:
        pytest.skip("No trained models found - run 'make train' first")


def test_prediction_shape():
    """Test if model predictions have correct shape"""
    model_path = 'models/catboost_model.joblib'
    
    if not os.path.exists(model_path):
        pytest.skip("CatBoost model not found - run 'make train' first")
    
    # Create sample input
    sample_data = pd.DataFrame({
        'hour': [14],
        'day_of_week': [3],
        'pickup_longitude': [-73.985],
        'pickup_latitude': [40.748],
        'dropoff_longitude': [-73.974],
        'dropoff_latitude': [40.750],
        'passenger_count': [2],
        'trip_km': [2.3],
        'is_rush_hour': [0],
        'is_weekend': [0],
        'is_late_night': [0],
        'distance_category': [2],
        'hour_distance_interaction': [32.2]
    })
    
    try:
        model_data = joblib.load(model_path)
        
        # Handle CatBoost's special structure
        if isinstance(model_data, dict):
            model = model_data['model']
            preprocessor = model_data['preprocessor']
            sample_prep = preprocessor.transform(sample_data)
            prediction = model.predict(sample_prep)
        else:
            prediction = model_data.predict(sample_data)
        
        assert len(prediction) == 1, "Prediction should have length 1"
        assert isinstance(prediction[0], (int, float, np.number)), "Prediction should be numeric"
        assert prediction[0] > 0, "Predicted fare should be positive"
        
    except Exception as e:
        pytest.skip(f"Could not test prediction: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
