"""Hybrid model combining neural networks for mean price forecasting and GARCH models for volatility.

This module implements a hybrid approach that leverages the strengths of both neural networks
and GARCH models to provide accurate price forecasts with confidence intervals.
"""

import logging
import os
import pickle
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from arch import arch_model
from pandera.typing import DataFrame
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model

from src.utils.schemas import ForecastSchema, PriceDataSchema, WeatherDataSchema

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HybridModel:
    """Hybrid model combining neural networks for mean prediction and GARCH for volatility.

    This class implements a hybrid approach that uses:
    1. A neural network (LSTM) to predict the mean price movements
    2. A GARCH model to predict the variance/volatility around the mean
    """

    def __init__(
        self,
        seq_length: int = 24,
        forecast_horizon: int = 24,
        p: int = 1,
        q: int = 1,
        mean: Literal["Zero", "Constant", "AR"] = "Zero",
        vol: Literal["ARCH", "GARCH", "EGARCH", "FIGARCH"] = "GARCH",
        dist: Literal["normal", "studentst", "skewstudent"] = "normal",
    ) -> None:
        """Initialize the hybrid model.

        Args:
            seq_length: Number of previous time steps to use as input for the neural network
            forecast_horizon: Number of time steps to forecast ahead
            p: GARCH model autoregressive lag order
            q: GARCH model moving average lag order
            mean: GARCH mean model specification
            vol: GARCH volatility model specification
            dist: Error distribution for GARCH model
        """
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.p = p
        self.q = q
        self.mean = mean
        self.vol = vol
        self.dist = dist

        # Initialize models to None (will be created during fit)
        self.nn_model: Optional[Sequential] = None
        self.garch_model: Optional[arch_model] = None

        # Store the features used for the neural network model
        self.feature_columns: List[str] = []

        logger.info(
            f"Initialized HybridModel with seq_length={seq_length}, forecast_horizon={forecast_horizon}"
        )

    def _preprocess_data(
        self,
        price_data: DataFrame[PriceDataSchema],
        weather_data: DataFrame[WeatherDataSchema],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Preprocess the price and weather data for model training.

        Args:
            price_data: Historical price data
            weather_data: Historical weather data

        Returns:
            Tuple containing:
            - X: Features for neural network (n_samples, seq_length, n_features)
            - y: Target price values (n_samples, forecast_horizon)
            - preprocessing_info: Dictionary with preprocessing metadata
        """
        # Align the time series
        aligned_price = price_data.copy()
        aligned_weather = weather_data.copy()

        # Merge price and weather data
        merged_data = pd.merge(
            aligned_price,
            aligned_weather,
            left_index=True,
            right_index=True,
            how="inner",
        )
        merged_data_index = pd.DatetimeIndex(merged_data.index)

        # Create time features
        merged_data["hour"] = merged_data_index.hour / 23.0  # Scale to [0, 1]
        merged_data["day_of_week"] = merged_data_index.dayofweek

        # Create one-hot encoding for day of week
        day_dummies = pd.get_dummies(merged_data["day_of_week"], prefix="day_of_week")
        merged_data = pd.concat([merged_data, day_dummies], axis=1)

        # Create lagged price features
        for lag in [1, 24, 48, 168]:  # 1-hour, 1-day, 2-day, 1-week lags
            if lag < len(merged_data):
                merged_data[f"price_lag_{lag}"] = merged_data["price"].shift(lag)

        # Drop rows with NaN values after creating lag features
        merged_data = merged_data.dropna()

        # Select features for the model
        self.feature_columns = [
            "hour",
            "price_lag_1",
            "price_lag_24",
            "temperature",
            "humidity",
            "wind_speed",
            "solar_irradiance",
        ] + [col for col in merged_data.columns if col.startswith("day_of_week_")]

        # Create sequences for LSTM training
        X, y = self._create_sequences(merged_data, self.feature_columns)

        # Store preprocessing information
        preprocessing_info = {
            "feature_columns": self.feature_columns,
            "data_shape": merged_data.shape,
            "X_shape": X.shape,
            "y_shape": y.shape,
        }

        return X, y, preprocessing_info

    def _create_sequences(
        self, data: pd.DataFrame, feature_columns: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences and target values for the neural network.

        Args:
            data: Preprocessed dataframe
            feature_columns: List of column names to use as features

        Returns:
            X: Input sequences (n_samples, seq_length, n_features)
            y: Target values (n_samples, forecast_horizon)
        """
        X_list = []
        y_list = []

        # Get the number of samples
        n_samples = len(data) - self.seq_length - self.forecast_horizon + 1

        for i in range(n_samples):
            # Input sequence (seq_length time steps of features)
            X_sequence = data[feature_columns].iloc[i : i + self.seq_length].values

            # Target is the next forecast_horizon price values
            y_sequence = (
                data["price"]
                .iloc[i + self.seq_length : i + self.seq_length + self.forecast_horizon]
                .values
            )

            X_list.append(X_sequence)
            y_list.append(y_sequence)

        return np.array(X_list), np.array(y_list)

    def _build_nn_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build the neural network model architecture.

        Args:
            input_shape: Shape of input data (seq_length, n_features)

        Returns:
            Compiled Keras model
        """
        model = Sequential(
            [
                LSTM(64, input_shape=input_shape, return_sequences=True),
                Dropout(0.2),
                LSTM(32),
                Dropout(0.1),
                Dense(self.forecast_horizon),
            ]
        )

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        return model

    def fit(
        self,
        price_data: DataFrame[PriceDataSchema],
        weather_data: DataFrame[WeatherDataSchema],
        nn_epochs: int = 50,
        nn_batch_size: int = 32,
        nn_validation_split: float = 0.2,
        verbose: int = 1,
    ) -> Dict[str, Any]:
        """Train both the neural network and GARCH components of the hybrid model.

        Args:
            price_data: Historical price data
            weather_data: Historical weather data
            nn_epochs: Number of epochs for neural network training
            nn_batch_size: Batch size for neural network training
            nn_validation_split: Validation split for neural network training
            verbose: Verbosity level (0, 1, or 2)

        Returns:
            Dictionary with training metrics
        """
        logger.info("Preprocessing data for model training")
        X, y, preprocessing_info = self._preprocess_data(price_data, weather_data)

        # Build and train the neural network model
        logger.info("Building and training neural network model")
        input_shape = (X.shape[0], X.shape[1])
        self.nn_model = self._build_nn_model(input_shape)

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        # Train the neural network
        nn_history = self.nn_model.fit(
            X,
            y,
            epochs=nn_epochs,
            batch_size=nn_batch_size,
            validation_split=nn_validation_split,
            callbacks=[early_stopping],
            verbose=verbose,
        )

        # Get predictions for training data to compute residuals
        y_pred = self.nn_model.predict(X)

        # Calculate residuals (actual - predicted prices)
        # For GARCH, we'll use the residuals from the first forecast step
        residuals = y[:, 0] - y_pred[:, 0]

        # Train GARCH model on the residuals
        logger.info(f"Training GARCH({self.p},{self.q}) model on residuals")
        self.garch_model = arch_model(
            residuals, p=self.p, q=self.q, mean=self.mean, vol=self.vol, dist=self.dist
        )
        garch_result = self.garch_model.fit(disp="off")

        # Return training metrics
        metrics = {
            "nn_history": nn_history.history,
            "garch_summary": garch_result.summary(),
            "preprocessing_info": preprocessing_info,
            "residuals_mean": float(np.mean(residuals)),
            "residuals_std": float(np.std(residuals)),
        }

        logger.info("Model training completed successfully")
        return metrics

    def predict(
        self,
        price_data: DataFrame[PriceDataSchema],
        weather_data: DataFrame[WeatherDataSchema],
        confidence_level: float = 0.95,
    ) -> DataFrame[ForecastSchema]:
        """Generate price forecasts with confidence intervals.

        Args:
            price_data: Recent price data (must include at least seq_length historical points)
            weather_data: Recent weather data (matching the price data timeframe)
            confidence_level: Confidence level for prediction intervals (0.0 to 1.0)

        Returns:
            DataFrame containing price forecasts and confidence bounds
        """
        if self.nn_model is None or self.garch_model is None:
            raise ValueError("Model has not been trained. Call fit() before predict().")

        # Ensure we have enough historical data
        if len(price_data) < self.seq_length:
            raise ValueError(
                f"Not enough historical price data. Need at least {self.seq_length} points."
            )

        # Preprocess the input data (similar to _preprocess_data but for a single sample)
        # Align the time series
        aligned_price = price_data.copy()
        aligned_weather = weather_data.copy()

        # Merge price and weather data
        merged_data = pd.merge(
            aligned_price,
            aligned_weather,
            left_index=True,
            right_index=True,
            how="inner",
        )
        merged_data_index = pd.DatetimeIndex(merged_data.index)
        # Create time features
        merged_data["hour"] = merged_data_index.hour / 23.0
        merged_data["day_of_week"] = merged_data_index.dayofweek

        # One-hot encode day of week
        day_dummies = pd.get_dummies(merged_data["day_of_week"], prefix="day_of_week")
        merged_data = pd.concat([merged_data, day_dummies], axis=1)

        # Create lagged features
        for lag in [1, 24, 48, 168]:
            if lag < len(merged_data):
                merged_data[f"price_lag_{lag}"] = merged_data["price"].shift(lag)

        # Fill any missing columns that were in the training data
        for col in self.feature_columns:
            if col not in merged_data.columns:
                merged_data[col] = 0.0

        # Handle missing lag features at the beginning
        merged_data = merged_data.bfill()

        # Get the most recent sequence for prediction
        input_sequence = (
            merged_data[self.feature_columns].iloc[-self.seq_length :].values
        )
        input_sequence = np.expand_dims(input_sequence, axis=0)  # Add batch dimension

        # Generate mean price forecast using neural network
        price_forecast = self.nn_model.predict(input_sequence)[0]

        # Generate variance forecast using GARCH model
        garch_forecast = self.garch_model.forecast(
            horizon=self.forecast_horizon, reindex=False
        )
        variance_forecast = garch_forecast.variance.values[-1]

        # Create forecast dates (starting after the last historical date)
        last_date = merged_data.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(hours=1),
            periods=self.forecast_horizon,
            freq="H",
        )

        # Calculate confidence intervals
        from scipy.stats import norm

        z_value = norm.ppf(1 - (1 - confidence_level) / 2)

        # Create forecast dataframe
        forecast_df = pd.DataFrame(
            {
                "price_forecast": price_forecast,
                "variance": variance_forecast,
                "lower_bound": price_forecast - z_value * np.sqrt(variance_forecast),
                "upper_bound": price_forecast + z_value * np.sqrt(variance_forecast),
            },
            index=forecast_dates,
        )

        # Ensure non-negative prices for lower bounds
        forecast_df["lower_bound"] = forecast_df["lower_bound"].clip(lower=0)

        # Validate against the schema
        try:
            validated_forecast = ForecastSchema.validate(forecast_df)
            return validated_forecast
        except Exception as e:
            logger.warning(f"Forecast validation failed: {e}")
            logger.warning("Returning unvalidated forecast")
            return forecast_df

    def save_models(self, save_path: str) -> None:
        """Save the trained models to disk.

        Args:
            save_path: Directory path where models will be saved
        """
        if self.nn_model is None or self.garch_model is None:
            raise ValueError("Models have not been trained yet.")

        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Save neural network model
        nn_path = os.path.join(save_path, "nn_model")
        self.nn_model.save(nn_path)

        # Save GARCH model using pickle
        garch_path = os.path.join(save_path, "garch_model.pkl")
        with open(garch_path, "wb") as f:
            pickle.dump(self.garch_model, f)

        # Save model parameters and feature columns
        params_path = os.path.join(save_path, "model_params.pkl")
        model_params = {
            "seq_length": self.seq_length,
            "forecast_horizon": self.forecast_horizon,
            "p": self.p,
            "q": self.q,
            "mean": self.mean,
            "vol": self.vol,
            "dist": self.dist,
            "feature_columns": self.feature_columns,
        }
        with open(params_path, "wb") as f:
            pickle.dump(model_params, f)

        logger.info(f"Models saved to {save_path}")

    def load_models(self, load_path: str) -> None:
        """Load trained models from disk.

        Args:
            load_path: Directory path where models are saved
        """
        try:
            # Load neural network model
            nn_path = os.path.join(load_path, "nn_model")
            self.nn_model = load_model(nn_path)

            # Load GARCH model
            garch_path = os.path.join(load_path, "garch_model.pkl")
            with open(garch_path, "rb") as f:
                self.garch_model = pickle.load(f)

            # Load model parameters
            params_path = os.path.join(load_path, "model_params.pkl")
            with open(params_path, "rb") as f:
                model_params = pickle.load(f)

            # Update model parameters
            self.seq_length = model_params["seq_length"]
            self.forecast_horizon = model_params["forecast_horizon"]
            self.p = model_params["p"]
            self.q = model_params["q"]
            self.mean = model_params["mean"]
            self.vol = model_params["vol"]
            self.dist = model_params["dist"]
            self.feature_columns = model_params["feature_columns"]

            logger.info(f"Models loaded from {load_path}")
        except Exception as e:
            error_msg = f"Error loading models from {load_path}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
