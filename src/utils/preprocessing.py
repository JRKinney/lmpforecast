"""Preprocessing utilities for the ERCOT price forecasting project.

This module provides functions for data preprocessing, feature engineering,
and preparation of data for model training and prediction.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import DatetimeIndex
from pandera.typing import DataFrame
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.utils.schemas import PriceDataSchema, WeatherDataSchema


def align_time_series(
    price_data: DataFrame[PriceDataSchema], weather_data: DataFrame[WeatherDataSchema]
) -> Tuple[DataFrame[PriceDataSchema], DataFrame[WeatherDataSchema]]:
    """Align price and weather data to ensure they have the same time index.

    Args:
        price_data: Price data with datetime index
        weather_data: Weather data with datetime index

    Returns:
        Tuple of aligned price and weather dataframes
    """
    # Find the overlapping time period
    start_date = max(price_data.index.min(), weather_data.index.min())
    end_date = min(price_data.index.max(), weather_data.index.max())

    # Extract data for the overlapping period
    aligned_price = price_data.loc[start_date:end_date].copy()
    aligned_weather = weather_data.loc[start_date:end_date].copy()

    # Check if there are any missing time points and handle them
    all_timestamps = pd.date_range(start=start_date, end=end_date, freq="H")

    # Reindex both dataframes to ensure they have the same timestamps
    aligned_price = aligned_price.reindex(all_timestamps)
    aligned_weather = aligned_weather.reindex(all_timestamps)

    # Handle missing values (interpolate for reasonable gaps)
    aligned_price["price"] = aligned_price["price"].interpolate(method="time", limit=6)

    for col in ["temperature", "humidity", "wind_speed", "solar_irradiance"]:
        aligned_weather[col] = aligned_weather[col].interpolate(method="time", limit=6)

    # Fill any remaining NaNs with forward/backward fill
    aligned_price = aligned_price.fillna(method="ffill").fillna(method="bfill")
    aligned_weather = aligned_weather.fillna(method="ffill").fillna(method="bfill")

    # Ensure price_node and location columns are filled
    if (
        "price_node" in aligned_price.columns
        and aligned_price["price_node"].isna().any()
    ):
        aligned_price["price_node"] = aligned_price["price_node"].fillna(
            price_data["price_node"].iloc[0]
        )

    if (
        "location" in aligned_weather.columns
        and aligned_weather["location"].isna().any()
    ):
        aligned_weather["location"] = aligned_weather["location"].fillna(
            weather_data["location"].iloc[0]
        )

    return aligned_price, aligned_weather


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features from a dataframe with datetime index.

    Args:
        df: DataFrame with datetime index

    Returns:
        DataFrame with additional time features
    """
    # Make a copy to avoid modifying the original
    result = df.copy()

    dt_index = DatetimeIndex(df.index)

    # Extract datetime components
    result["hour"] = dt_index.hour / 23.0  # Scaled to [0,1]
    result["day_of_week"] = dt_index.dayofweek
    result["month"] = dt_index.month
    result["quarter"] = dt_index.quarter
    result["year"] = dt_index.year

    # Create cyclical features for hour, day of week, and month
    result["hour_sin"] = np.sin(2 * np.pi * dt_index.hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * dt_index.hour / 24)

    result["day_sin"] = np.sin(2 * np.pi * dt_index.dayofweek / 7)
    result["day_cos"] = np.cos(2 * np.pi * dt_index.dayofweek / 7)

    result["month_sin"] = np.sin(2 * np.pi * dt_index.month / 12)
    result["month_cos"] = np.cos(2 * np.pi * dt_index.month / 12)

    # Indicator features
    result["is_weekend"] = dt_index.dayofweek >= 5
    result["is_business_hour"] = (
        (dt_index.hour >= 8) & (dt_index.hour < 18) & (~result["is_weekend"])
    )
    result["is_peak_hour"] = (dt_index.hour >= 14) & (dt_index.hour < 20)

    return result


def create_lag_features(
    df: pd.DataFrame, column: str, lag_hours: Optional[List[int]] = None
) -> pd.DataFrame:
    """Create lagged features for a specified column.

    Args:
        df: DataFrame with datetime index
        column: Name of column to create lags for
        lag_hours: List of hourly lags to create

    Returns:
        DataFrame with additional lag features
    """
    if lag_hours is None:
        lag_hours = [1, 24, 48, 168]

    result = df.copy()

    for lag in lag_hours:
        result[f"{column}_lag_{lag}"] = result[column].shift(lag)

    return result


def create_rolling_features(
    df: pd.DataFrame,
    column: str,
    windows: Optional[List[int]] = None,
    functions: Optional[Dict[str, Callable]] = None,
) -> pd.DataFrame:
    """Create rolling window features for a specified column.

    Args:
        df: DataFrame with datetime index
        column: Name of column to create rolling features for
        windows: List of window sizes (in hours)
        functions: Dictionary mapping function names to functions

    Returns:
        DataFrame with additional rolling features
    """
    if windows is None:
        windows = [24, 48, 168]

    if functions is None:
        functions = {
            "mean": np.mean,
            "std": np.std,
            "max": np.max,
            "min": np.min,
        }

    result = df.copy()

    for window in windows:
        for func_name, func in functions.items():
            result[f"{column}_{func_name}_{window}h"] = (
                result[column].rolling(window=window, min_periods=1).apply(func)
            )

    return result


def prepare_data_for_model(
    price_data: DataFrame[PriceDataSchema],
    weather_data: DataFrame[WeatherDataSchema],
    seq_length: int = 24,
    forecast_horizon: int = 24,
    include_time_features: bool = True,
    include_lags: bool = True,
    include_rolling: bool = True,
    scale_features: bool = True,
) -> Dict[str, Any]:
    """Prepare data for model training or prediction.

    Args:
        price_data: Price data with datetime index
        weather_data: Weather data with datetime index
        seq_length: Number of time steps to use as input sequence
        forecast_horizon: Number of time steps to forecast
        include_time_features: Whether to include time-based features
        include_lags: Whether to include lagged features
        include_rolling: Whether to include rolling statistics
        scale_features: Whether to scale numerical features

    Returns:
        Dictionary containing X (features), y (targets), and metadata
    """
    # Step 1: Align time series
    aligned_price, aligned_weather = align_time_series(price_data, weather_data)

    # Step 2: Merge data
    merged_data = pd.merge(
        aligned_price, aligned_weather, left_index=True, right_index=True, how="inner"
    )

    # Step 3: Create features
    # Time features
    if include_time_features:
        merged_data = create_time_features(merged_data)

    # Lag features
    if include_lags:
        merged_data = create_lag_features(
            merged_data, "price", lag_hours=[1, 24, 48, 168]
        )

        # Also create lags for important weather variables
        for weather_var in ["temperature", "solar_irradiance"]:
            merged_data = create_lag_features(
                merged_data, weather_var, lag_hours=[1, 24]
            )

    # Rolling statistics
    if include_rolling:
        merged_data = create_rolling_features(
            merged_data,
            "price",
            windows=[24, 48, 168],
            functions={"mean": np.mean, "std": np.std},
        )

    # Step 4: Drop missing values resulting from lag/rolling calculations
    merged_data = merged_data.dropna()

    # Step 5: Scale features if requested
    feature_scaler = None
    target_scaler = None

    if scale_features:
        # Select numeric columns, excluding the target (price)
        numeric_columns = merged_data.select_dtypes(
            include=["float", "int"]
        ).columns.tolist()
        if "price" in numeric_columns:
            numeric_columns.remove("price")

        # Scale features
        feature_scaler = MinMaxScaler(feature_range=(-1, 1))
        merged_data[numeric_columns] = feature_scaler.fit_transform(
            merged_data[numeric_columns]
        )

        # Scale target separately to easily inverse transform later
        target_scaler = MinMaxScaler(feature_range=(-1, 1))
        merged_data[["price"]] = target_scaler.fit_transform(merged_data[["price"]])

    # Step 6: Create sequences for training
    X_sequences = []
    y_sequences = []

    # Exclude non-numeric columns for model input
    feature_columns = merged_data.select_dtypes(
        include=["float", "int"]
    ).columns.tolist()

    # Exclude categorical columns and the target variable from features
    if "price" in feature_columns:
        feature_columns.remove("price")

    for i in range(len(merged_data) - seq_length - forecast_horizon + 1):
        # Input sequence (features for the past seq_length time steps)
        X_seq = merged_data.iloc[i : i + seq_length][feature_columns].values

        # Target sequence (prices for the next forecast_horizon time steps)
        y_seq = merged_data.iloc[i + seq_length : i + seq_length + forecast_horizon][
            "price"
        ].values

        X_sequences.append(X_seq)
        y_sequences.append(y_seq)

    # Convert to numpy arrays
    X = np.array(X_sequences)
    y = np.array(y_sequences)

    # Return dictionary with prepared data and metadata
    result = {
        "X": X,
        "y": y,
        "feature_columns": feature_columns,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "seq_length": seq_length,
        "forecast_horizon": forecast_horizon,
        "data_shape": {"X_shape": X.shape, "y_shape": y.shape},
        "timestamps": {
            "sequence_start_dates": merged_data.index[
                : -seq_length - forecast_horizon + 1
            ].tolist(),
            "forecast_start_dates": merged_data.index[
                seq_length : -forecast_horizon + 1
            ].tolist(),
        },
    }

    return result


def preprocess_data(
    price_data: DataFrame[PriceDataSchema],
    weather_data: Optional[DataFrame[WeatherDataSchema]] = None,
    scale: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Preprocess and combine price and weather data for modeling.

    Parameters:
    -----------
    price_data : DataFrame[PriceDataSchema]
        DataFrame containing price data with datetime index
    weather_data : Optional[DataFrame[WeatherDataSchema]], default=None
        DataFrame containing weather data with datetime index
    scale : bool, default=True
        Whether to standardize the data

    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, Any]]
        (data, scalers) where data is the preprocessed DataFrame and
        scalers is a dict of fitted scalers for each column group
    """
    # Start with price data
    data = price_data[["price"]].copy()

    # Add weather data if provided
    if weather_data is not None:
        # Align time series
        price_aligned, weather_aligned = align_time_series(price_data, weather_data)

        # Drop redundant columns
        weather_cols = [
            col for col in weather_aligned.columns if col not in ["location"]
        ]

        # Join weather data to price data
        data = price_aligned[["price"]].join(weather_aligned[weather_cols])

    # Create features for time of day and day of week
    data["hour"] = data.index.hour
    data["day_of_week"] = data.index.dayofweek

    # Convert cyclical time features to sine and cosine components
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    data["dow_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
    data["dow_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)

    # Drop original time columns
    data.drop(["hour", "day_of_week"], axis=1, inplace=True)

    # Scale the data if requested
    scalers: Dict[str, Any] = {}
    if scale:
        # Scale price
        price_scaler = StandardScaler()
        data[["price"]] = price_scaler.fit_transform(data[["price"]])
        scalers["price"] = price_scaler

        # Scale weather features if present
        if weather_data is not None:
            weather_cols = [
                col
                for col in data.columns
                if col not in ["price", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
            ]
            if weather_cols:
                weather_scaler = StandardScaler()
                data[weather_cols] = weather_scaler.fit_transform(data[weather_cols])
                scalers["weather"] = weather_scaler

    return data, scalers


def add_lagged_features(
    data: pd.DataFrame,
    lag_cols: List[str],
    lag_periods: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Add lagged values of specified columns as new features.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with datetime index
    lag_cols : List[str]
        List of column names to create lags for
    lag_periods : List[int], default=[1, 2, 3, 6, 12, 24]
        List of lag periods to create

    Returns:
    --------
    pd.DataFrame
        DataFrame with added lag features
    """
    if lag_periods is None:
        lag_periods = [1, 2, 3, 6, 12, 24]

    df = data.copy()

    for col in lag_cols:
        for lag in lag_periods:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    # Drop rows with NaN values from lagging
    df.dropna(inplace=True)

    return df


def calculate_rolling_statistics(
    data: pd.DataFrame,
    window_sizes: Optional[List[int]] = None,
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Calculate rolling statistics for specified features.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with datetime index
    window_sizes : List[int], default=[24, 48, 168]
        List of window sizes for rolling statistics (in periods)
    features : List[str], default=['price']
        List of features to calculate rolling statistics for

    Returns:
    --------
    pd.DataFrame
        DataFrame with added rolling statistics
    """
    if window_sizes is None:
        window_sizes = [24, 48, 168]

    if features is None:
        features = ["price"]

    df = data.copy()

    for feature in features:
        for window in window_sizes:
            # Calculate rolling mean
            df[f"{feature}_rolling_mean_{window}"] = (
                df[feature].rolling(window=window).mean()
            )

            # Calculate rolling standard deviation (for volatility)
            df[f"{feature}_rolling_std_{window}"] = (
                df[feature].rolling(window=window).std()
            )

            # Calculate rolling min and max
            df[f"{feature}_rolling_min_{window}"] = (
                df[feature].rolling(window=window).min()
            )
            df[f"{feature}_rolling_max_{window}"] = (
                df[feature].rolling(window=window).max()
            )

    # Drop rows with NaN values from rolling calculations
    df.dropna(inplace=True)

    return df


def train_test_split_time_series(
    data: pd.DataFrame, train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split time series data into training and testing sets based on time.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with datetime index
    train_ratio : float, default=0.8
        Proportion of data to use for training

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]
        (train_data, test_data, split_date) tuple with the respective datasets and split point
    """
    # Calculate split point
    split_idx = int(len(data) * train_ratio)
    split_date = data.index[split_idx]

    # Split data
    train_data = data.loc[:split_date]
    test_data = data.loc[split_date:]

    return train_data, test_data, split_date
