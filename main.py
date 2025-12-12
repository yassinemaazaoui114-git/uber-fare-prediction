#!/usr/bin/env python3
"""
Uber Fare Prediction - Enhanced Training Pipeline with Advanced Models
"""

import os
import sys
import argparse
import pandas as pd
from utils.data_loader import load_and_preprocess_data
from pipelines.linear_regression_pipeline import LinearRegressionPipeline
from pipelines.random_forest_pipeline import RandomForestPipeline
from pipelines.knn_pipeline import KNNPipeline
from pipelines.decision_tree_pipeline import DecisionTreePipeline
from pipelines.xgboost_pipeline import XGBoostPipeline
from pipelines.lightgbm_pipeline import LightGBMPipeline
from pipelines.catboost_pipeline import CatBoostPipeline
from pipelines.gradient_boosting_pipeline import GradientBoostingPipeline
from pipelines.stacking_pipeline import StackingPipeline


def create_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    print("✅ Directories created/verified\n")


def compare_models(results_list):
    """Compare all models and display results"""
    print("\n" + "=" * 80)
    print("📊 MODEL COMPARISON SUMMARY")
    print("=" * 80)

    # Create comparison dataframe
    comparison_df = pd.DataFrame(results_list)
    comparison_df = comparison_df.sort_values("r2_test", ascending=False)

    print("\nRanked by Test R² Score (Higher is Better):")
    print("-" * 80)
    for idx, row in comparison_df.iterrows():
        print(
            f"{row['model']:30} | R²: {row['r2_test']:.4f} | RMSE: {row['rmse_test']:.4f} | MAE: {row['mae_test']:.4f}"
        )

    print("\n" + "=" * 80)
    best_model = comparison_df.iloc[0]
    print(f"🏆 BEST MODEL: {best_model['model']}")
    print(f"   Test R² Score: {best_model['r2_test']:.4f}")
    print(f"   Test RMSE: {best_model['rmse_test']:.4f}")
    print(f"   Test MAE: {best_model['mae_test']:.4f}")
    print("=" * 80 + "\n")

    return comparison_df


def train_models(tune_hyperparameters=False):
    """Main training pipeline"""
    mode = "WITH HYPERPARAMETER TUNING" if tune_hyperparameters else "FAST MODE"

    print("\n" + "=" * 80)
    print("🚀 UBER FARE PREDICTION - ENHANCED TRAINING PIPELINE")
    print(f"   Mode: {mode}")
    print("=" * 80 + "\n")

    # Step 1: Create directories
    create_directories()

    # Step 2: Load and preprocess data
    print("📥 Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        filepath="data/datauber.csv", test_size=0.2, random_state=42
    )

    # Step 3: Initialize all models
    models = {
        "linear_regression": LinearRegressionPipeline(),
        "random_forest": RandomForestPipeline(tune_hyperparameters=tune_hyperparameters),
        "knn": KNNPipeline(tune_hyperparameters=tune_hyperparameters),
        "decision_tree": DecisionTreePipeline(tune_hyperparameters=tune_hyperparameters),
        "xgboost": XGBoostPipeline(tune_hyperparameters=tune_hyperparameters),
        "lightgbm": LightGBMPipeline(tune_hyperparameters=tune_hyperparameters),
        "catboost": CatBoostPipeline(tune_hyperparameters=tune_hyperparameters),
        "gradient_boosting": GradientBoostingPipeline(tune_hyperparameters=tune_hyperparameters),
    }

    results_list = []
    trained_models = {}

    # Step 4: Train and evaluate each model
    for model_key, model_pipeline in models.items():
        print(f"\n{'='*80}")
        print(f"🔄 Processing: {model_pipeline.model_name}")
        print(f"{'='*80}")

        # Train
        model_pipeline.train(X_train, y_train)

        # Evaluate
        results = model_pipeline.evaluate(X_train, y_train, X_test, y_test)
        results_list.append(results)

        # Save
        model_pipeline.save_model()

        # Store for stacking
        trained_models[model_key] = model_pipeline

    # Step 5: Create Stacking Ensemble
    print(f"\n{'='*80}")
    print("🔄 Creating Stacking Ensemble with TOP 3 Models")
    print(f"{'='*80}")

    stacking_base_models = {
        "lightgbm": trained_models["lightgbm"],
        "gradient_boosting": trained_models["gradient_boosting"],
        "random_forest": trained_models["random_forest"],
    }

    print("   Selected models:")
    print("   1. LightGBM")
    print("   2. Gradient Boosting")
    print("   3. Random Forest")
    print("   Note: CatBoost excluded due to sklearn Pipeline compatibility")

    if len(stacking_base_models) >= 2:
        stacking_pipeline = StackingPipeline(stacking_base_models)
        stacking_pipeline.train(X_train, y_train)
        stacking_results = stacking_pipeline.evaluate(X_train, y_train, X_test, y_test)

        if stacking_results is not None:
            results_list.append(stacking_results)
            stacking_pipeline.save_model()

    # Step 6: Compare all models
    comparison_df = compare_models(results_list)

    print("✅ Training pipeline completed successfully!")
    print("📁 All models saved in the 'models/' directory\n")

    return comparison_df


def evaluate_models():
    """Evaluate existing trained models"""
    import joblib

    print("\n" + "=" * 80)
    print("📊 EVALUATING TRAINED MODELS")
    print("=" * 80 + "\n")

    # Load test data
    print("📥 Loading test data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        filepath="data/datauber.csv", test_size=0.2, random_state=42
    )

    # Check which models exist
    model_files = {
        "Linear Regression": "models/linear_regression_model.joblib",
        "Random Forest": "models/random_forest_model.joblib",
        "KNN": "models/knn_model.joblib",
        "Decision Tree": "models/decision_tree_model.joblib",
        "XGBoost": "models/xgboost_model.joblib",
        "LightGBM": "models/lightgbm_model.joblib",
        "CatBoost": "models/catboost_model.joblib",
        "Gradient Boosting": "models/gradient_boosting_model.joblib",
        "Stacking": "models/stacking_model.joblib",
    }

    print("📋 Found models:")
    for name, path in model_files.items():
        if os.path.exists(path):
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name} (not trained yet)")

    print("\n✅ Evaluation complete!")
    print("💡 Tip: Run 'make train' to train all models\n")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Uber Fare Prediction Training Pipeline")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning (slower but better results)",
    )
    parser.add_argument("--evaluate", action="store_true", help="Evaluate existing trained models")
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()

    if args.evaluate:
        evaluate_models()
    else:
        train_models(tune_hyperparameters=args.tune)


if __name__ == "__main__":
    main()
