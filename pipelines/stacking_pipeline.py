import numpy as np
import joblib
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class StackingPipeline:
    def __init__(self, base_models):
        """
        Initialize stacking with trained base models

        Parameters:
        -----------
        base_models : dict
            Dictionary of trained model pipelines
            Example: {'rf': rf_pipeline, 'xgb': xgb_pipeline}
        """
        self.model_name = "Stacking Ensemble"
        self.base_models = base_models
        self.stacking_model = None

    def build_stacking_model(self):
        """Build the stacking ensemble"""
        estimators = []

        # Filter out models that don't have proper sklearn pipelines
        for name, model in self.base_models.items():
            # Skip CatBoost due to sklearn compatibility issues
            if "catboost" in name.lower():
                print(f"   ⚠️  Skipping {name} (compatibility issue with sklearn Stacking)")
                continue

            # Check if it has a proper pipeline
            if hasattr(model, "pipeline") and model.pipeline is not None:
                estimators.append((name, model.pipeline))

        if len(estimators) < 2:
            print("❌ Not enough compatible models for stacking")
            return None

        self.stacking_model = StackingRegressor(
            estimators=estimators, final_estimator=Ridge(), cv=5
        )

        print(f"✅ {self.model_name} built with {len(estimators)} base models!")
        print(f"   Models used: {[name for name, _ in estimators]}")
        return self.stacking_model

    def train(self, X_train, y_train):
        """Train the stacking model"""
        if self.stacking_model is None:
            self.build_stacking_model()

        if self.stacking_model is None:
            print("❌ Cannot train stacking - model build failed")
            return

        print(f"\n🚀 Training {self.model_name}...")

        self.stacking_model.fit(X_train, y_train)
        print(f"✅ {self.model_name} trained successfully!")

    def evaluate(self, X_train, y_train, X_test, y_test):
        """Evaluate the stacking model"""
        if self.stacking_model is None:
            print("❌ Cannot evaluate - model not trained")
            return None

        # Training predictions
        y_pred_train = self.stacking_model.predict(X_train)
        r2_train = r2_score(y_train, y_pred_train)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        mae_train = mean_absolute_error(y_train, y_pred_train)

        # Testing predictions
        y_pred_test = self.stacking_model.predict(X_test)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae_test = mean_absolute_error(y_test, y_pred_test)

        results = {
            "model": self.model_name,
            "r2_train": r2_train,
            "r2_test": r2_test,
            "rmse_train": rmse_train,
            "rmse_test": rmse_test,
            "mae_train": mae_train,
            "mae_test": mae_test,
            "base_models": [name for name, _ in self.stacking_model.estimators],
        }

        self._print_results(results)
        return results

    def _print_results(self, results):
        """Print evaluation results"""
        print(f"\n{'='*60}")
        print(f"📊 {self.model_name} - Evaluation Results")
        print(f"{'='*60}")
        print(f"Base Models: {', '.join(results['base_models'])}")
        print(f"\nTraining Set:")
        print(f"  R² Score:  {results['r2_train']:.4f}")
        print(f"  RMSE:      {results['rmse_train']:.4f}")
        print(f"  MAE:       {results['mae_train']:.4f}")
        print(f"\nTesting Set:")
        print(f"  R² Score:  {results['r2_test']:.4f}")
        print(f"  RMSE:      {results['rmse_test']:.4f}")
        print(f"  MAE:       {results['mae_test']:.4f}")
        print(f"{'='*60}\n")

    def save_model(self, filepath="models/stacking_model.joblib"):
        """Save the stacking model"""
        if self.stacking_model is not None:
            joblib.dump(self.stacking_model, filepath)
            print(f"✅ {self.model_name} saved to {filepath}")

    def load_model(self, filepath="models/stacking_model.joblib"):
        """Load a trained stacking model"""
        self.stacking_model = joblib.load(filepath)
        print(f"✅ {self.model_name} loaded from {filepath}")

    def predict(self, X):
        """Make predictions"""
        if self.stacking_model is None:
            raise ValueError("Model not trained yet")
        return self.stacking_model.predict(X)
