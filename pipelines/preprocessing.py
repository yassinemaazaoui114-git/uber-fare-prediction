from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def get_preprocessor():
    """
    Create and return the preprocessing pipeline

    Returns:
    --------
    ColumnTransformer with StandardScaler for all numeric features
    """
    # Updated to include new features
    numeric_features = [
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

    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_features)], remainder="passthrough"
    )

    return preprocessor
