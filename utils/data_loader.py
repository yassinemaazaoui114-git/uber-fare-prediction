import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def add_enhanced_features(df):
    """
    Add enhanced time-based features for better predictions
    """
    # 1. Rush hour flag (weekday 7-9am, 5-7pm)
    df["is_rush_hour"] = df.apply(
        lambda row: (
            1
            if (row["day_of_week"] < 5 and ((7 <= row["hour"] <= 9) or (17 <= row["hour"] <= 19)))
            else 0
        ),
        axis=1,
    )

    # 2. Weekend flag
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # 3. Late night flag (10pm - 4am - surge pricing time)
    df["is_late_night"] = ((df["hour"] >= 22) | (df["hour"] <= 4)).astype(int)

    # 4. Distance category (short/medium/long trips)
    # Handle 0 and NaN values first
    df["distance_category"] = pd.cut(
        df["trip_km"], bins=[-0.1, 2, 5, 10, 200], labels=[1, 2, 3, 4]  # Include 0 values
    )
    # Fill any remaining NaN with 1 (short distance)
    df["distance_category"] = df["distance_category"].fillna(1).astype(int)

    # 5. Hour x Distance interaction (rush hour + long distance = expensive)
    df["hour_distance_interaction"] = df["hour"] * df["trip_km"]

    return df


def load_and_preprocess_data(filepath="data/datauber.csv", test_size=0.2, random_state=42):
    """
    Load and preprocess the Uber dataset

    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    test_size : float
        Proportion of test set
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    # Load data
    df = pd.read_csv(filepath)

    # Convert pickup_datetime to datetime
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")

    # Feature engineering - Basic features
    df["hour"] = df["pickup_datetime"].dt.hour
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
    df["month"] = df["pickup_datetime"].dt.month

    # Calculate trip distance (Haversine formula)
    def haversine(lon1, lat1, lon2, lat2):
        R = 6371  # Earth radius in km
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)

        a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return R * c

    df["trip_km"] = haversine(
        df["pickup_longitude"],
        df["pickup_latitude"],
        df["dropoff_longitude"],
        df["dropoff_latitude"],
    )

    # Add enhanced features
    df = add_enhanced_features(df)

    # Select features (now includes new ones!)
    feature_cols = [
        "hour",
        "day_of_week",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "trip_km",
        # NEW FEATURES:
        "is_rush_hour",
        "is_weekend",
        "is_late_night",
        "distance_category",
        "hour_distance_interaction",
    ]

    # Remove missing values and outliers
    df_clean = df[feature_cols + ["fare_amount"]].dropna().drop_duplicates()

    # Remove outliers using IQR method
    Q1 = df_clean.quantile(0.25)
    Q3 = df_clean.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mask = ((df_clean >= lower_bound) & (df_clean <= upper_bound)).all(axis=1)
    df_clean = df_clean[mask]

    # Separate features and target
    y = df_clean["fare_amount"]
    X = df_clean[feature_cols]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Data loaded successfully!")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]} (including 5 new enhanced features)")

    return X_train, X_test, y_train, y_test
