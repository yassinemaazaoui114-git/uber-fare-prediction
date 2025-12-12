import numpy as np
import joblib
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pipelines.preprocessing import get_preprocessor


class LightGBMPipeline:
    def __init__(self, tune_hyperparameters=False):
        self.model_name = "LightGBM"
        self.pipeline = None
        self.tune_hyperparameters = tune_hyperparameters
        self.best_params = None

    def build_pipeline(self):
        """Build the LightGBM pipeline"""
        preprocessor = get_preprocessor()

        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)),
            ]
        )

        print(f"✅ {self.model_name} pipeline built successfully!")
        return self.pipeline

    def train(self, X_train, y_train):
        """Train the model with optional hyperparameter tuning"""
        if self.pipeline is None:
            self.build_pipeline()

        print(f"\n🚀 Training {self.model_name}...")

        if self.tune_hyperparameters:
            print("🔍 Performing hyperparameter tuning...")
            self._tune_hyperparameters(X_train, y_train)
        else:
            self.pipeline.fit(X_train, y_train)
            print(f"✅ {self.model_name} trained successfully!")

    def _tune_hyperparameters(self, X_train, y_train):
        """Perform hyperparameter tuning using RandomizedSearchCV"""
        # Preprocess data first
        X_train_prep = self.pipeline.named_steps["preprocessor"].fit_transform(X_train)

        param_grid = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 5, 7, 10, -1],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "num_leaves": [31, 50, 100],
            "min_child_samples": [20, 30, 50],
        }

        lgb_search = RandomizedSearchCV(
            lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
            param_grid,
            n_iter=20,
            scoring="r2",
            cv=5,
            n_jobs=-1,
            verbose=1,
            random_state=42,
        )

        lgb_search.fit(X_train_prep, y_train)
        self.best_params = lgb_search.best_params_

        # Update pipeline with best model
        self.pipeline.named_steps["model"] = lgb_search.best_estimator_

        print(f"✅ Best parameters found: {self.best_params}")
        print(f"✅ {self.model_name} trained with tuned hyperparameters!")

    def evaluate(self, X_train, y_train, X_test, y_test):
        """Evaluate the model"""
        # Training predictions
        y_pred_train = self.pipeline.predict(X_train)
        r2_train = r2_score(y_train, y_pred_train)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        mae_train = mean_absolute_error(y_train, y_pred_train)

        # Testing predictions
        y_pred_test = self.pipeline.predict(X_test)
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

    def save_model(self, filepath="models/lightgbm_model.joblib"):
        """Save the trained model"""
        joblib.dump(self.pipeline, filepath)
        print(f"✅ {self.model_name} saved to {filepath}")

    def load_model(self, filepath="models/lightgbm_model.joblib"):
        """Load a trained model"""
        self.pipeline = joblib.load(filepath)
        print(f"✅ {self.model_name} loaded from {filepath}")
