import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pipelines.preprocessing import get_preprocessor


class LinearRegressionPipeline:
    def __init__(self):
        self.model_name = "Linear Regression"
        self.pipeline = None

    def build_pipeline(self):
        """Build the Linear Regression pipeline"""
        preprocessor = get_preprocessor()

        self.pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", LinearRegression())]
        )

        print(f"✅ {self.model_name} pipeline built successfully!")
        return self.pipeline

    def train(self, X_train, y_train):
        """Train the model"""
        if self.pipeline is None:
            self.build_pipeline()

        print(f"\n🚀 Training {self.model_name}...")
        self.pipeline.fit(X_train, y_train)
        print(f"✅ {self.model_name} trained successfully!")

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
        }

        self._print_results(results)
        return results

    def _print_results(self, results):
        """Print evaluation results"""
        print(f"\n{'='*60}")
        print(f"📊 {self.model_name} - Evaluation Results")
        print(f"{'='*60}")
        print(f"Training Set:")
        print(f"  R² Score:  {results['r2_train']:.4f}")
        print(f"  RMSE:      {results['rmse_train']:.4f}")
        print(f"  MAE:       {results['mae_train']:.4f}")
        print(f"\nTesting Set:")
        print(f"  R² Score:  {results['r2_test']:.4f}")
        print(f"  RMSE:      {results['rmse_test']:.4f}")
        print(f"  MAE:       {results['mae_test']:.4f}")
        print(f"{'='*60}\n")

    def save_model(self, filepath="models/linear_regression_model.joblib"):
        """Save the trained model"""
        joblib.dump(self.pipeline, filepath)
        print(f"✅ {self.model_name} saved to {filepath}")

    def load_model(self, filepath="models/linear_regression_model.joblib"):
        """Load a trained model"""
        self.pipeline = joblib.load(filepath)
        print(f"✅ {self.model_name} loaded from {filepath}")
