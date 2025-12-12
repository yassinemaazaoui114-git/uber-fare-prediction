import numpy as np
import joblib
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from pipelines.preprocessing import get_preprocessor


class CatBoostPipeline:
    def __init__(self, tune_hyperparameters=False):
        self.model_name = "CatBoost"
        self.model = None
        self.preprocessor = None
        self.tune_hyperparameters = tune_hyperparameters
        self.best_params = None

    def build_pipeline(self):
        """Build the CatBoost model (without sklearn Pipeline)"""
        self.preprocessor = get_preprocessor()
        self.model = CatBoostRegressor(random_state=42, verbose=0)

        print(f"✅ {self.model_name} pipeline built successfully!")

    def train(self, X_train, y_train):
        """Train the model with optional hyperparameter tuning"""
        if self.model is None:
            self.build_pipeline()

        print(f"\n🚀 Training {self.model_name}...")

        # Preprocess data
        X_train_prep = self.preprocessor.fit_transform(X_train)

        if self.tune_hyperparameters:
            print("🔍 Performing hyperparameter tuning...")
            self._tune_hyperparameters(X_train_prep, y_train)
        else:
            self.model.fit(X_train_prep, y_train)
            print(f"✅ {self.model_name} trained successfully!")

    def _tune_hyperparameters(self, X_train_prep, y_train):
        """Perform hyperparameter tuning using RandomizedSearchCV"""
        param_grid = {
            "iterations": [100, 200, 300, 500],
            "depth": [4, 6, 8, 10],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "l2_leaf_reg": [1, 3, 5, 7],
            "border_count": [32, 64, 128],
        }

        cat_search = RandomizedSearchCV(
            CatBoostRegressor(random_state=42, verbose=0),
            param_grid,
            n_iter=20,
            scoring="r2",
            cv=5,
            n_jobs=-1,
            verbose=1,
            random_state=42,
        )

        cat_search.fit(X_train_prep, y_train)
        self.best_params = cat_search.best_params_
        self.model = cat_search.best_estimator_

        print(f"✅ Best parameters found: {self.best_params}")
        print(f"✅ {self.model_name} trained with tuned hyperparameters!")

    def evaluate(self, X_train, y_train, X_test, y_test):
        """Evaluate the model"""
        # Preprocess data
        X_train_prep = self.preprocessor.transform(X_train)
        X_test_prep = self.preprocessor.transform(X_test)

        # Training predictions
        y_pred_train = self.model.predict(X_train_prep)
        r2_train = r2_score(y_train, y_pred_train)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        mae_train = mean_absolute_error(y_train, y_pred_train)

        # Testing predictions
        y_pred_test = self.model.predict(X_test_prep)
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
            "best_params": self.best_params,
        }

        self._print_results(results)
        return results

    def _print_results(self, results):
        """Print evaluation results"""
        print(f"\n{'='*60}")
        print(f"📊 {self.model_name} - Evaluation Results")
        print(f"{'='*60}")
        if results["best_params"]:
            print(f"Best Parameters: {results['best_params']}")
        print(f"Training Set:")
        print(f"  R² Score:  {results['r2_train']:.4f}")
        print(f"  RMSE:      {results['rmse_train']:.4f}")
        print(f"  MAE:       {results['mae_train']:.4f}")
        print(f"\nTesting Set:")
        print(f"  R² Score:  {results['r2_test']:.4f}")
        print(f"  RMSE:      {results['rmse_test']:.4f}")
        print(f"  MAE:       {results['mae_test']:.4f}")
        print(f"{'='*60}\n")

    def save_model(self, filepath="models/catboost_model.joblib"):
        """Save the trained model"""
        save_data = {"model": self.model, "preprocessor": self.preprocessor}
        joblib.dump(save_data, filepath)
        print(f"✅ {self.model_name} saved to {filepath}")

    def load_model(self, filepath="models/catboost_model.joblib"):
        """Load a trained model"""
        save_data = joblib.load(filepath)
        self.model = save_data["model"]
        self.preprocessor = save_data["preprocessor"]
        print(f"✅ {self.model_name} loaded from {filepath}")

    def predict(self, X):
        """Make predictions"""
        X_prep = self.preprocessor.transform(X)
        return self.model.predict(X_prep)

    # Add this property to make it compatible with stacking
    @property
    def pipeline(self):
        """Return a dummy pipeline-like object for compatibility"""

        class DummyPipeline:
            def __init__(self, preprocessor, model):
                self.preprocessor = preprocessor
                self.model = model

            def predict(self, X):
                X_prep = self.preprocessor.transform(X)
                return self.model.predict(X_prep)

        return DummyPipeline(self.preprocessor, self.model)
