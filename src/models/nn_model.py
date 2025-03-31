from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pandera.typing import DataFrame
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

from src.utils.schemas import PriceDataSchema, WeatherDataSchema


class NNModel:
    """Neural Network model for forecasting the mean of energy prices."""

    def __init__(self, seq_length: int = 24, forecast_horizon: int = 24) -> None:
        """Initialize the neural network model.

        Parameters:
        -----------
        seq_length : int, default=24
            Number of time steps to use as input features (lookback window)
        forecast_horizon : int, default=24
            Number of time steps to predict ahead
        """
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.model: Optional[keras.Model] = None
        self.price_scaler = StandardScaler()
        self.weather_scaler = StandardScaler()
        self.feature_names: Optional[List[str]] = None

    def _build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build the neural network model architecture.

        Parameters:
        -----------
        input_shape : Tuple[int, int]
            Shape of the input data (sequence length, number of features)

        Returns:
        --------
        tf.keras.Model
            Compiled Keras model
        """
        inputs = keras.Input(shape=input_shape)

        # First LSTM layer with return sequences
        x = layers.LSTM(64, return_sequences=True)(inputs)
        x = layers.Dropout(0.2)(x)

        # Second LSTM layer
        x = layers.LSTM(32)(x)
        x = layers.Dropout(0.2)(x)

        # Dense layers
        x = layers.Dense(32, activation="relu")(x)

        # Output layer
        outputs = layers.Dense(self.forecast_horizon)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"],
        )

        return model

    def _create_sequences(
        self, data: np.ndarray, seq_length: int, forecast_horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences and target values from time series data.

        Parameters:
        -----------
        data : np.ndarray
            Input data array
        seq_length : int
            Length of input sequences
        forecast_horizon : int
            Number of steps to predict ahead

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (X, y) where X contains the input sequences and y contains the target values
        """
        X, y = [], []
        for i in range(len(data) - seq_length - forecast_horizon + 1):
            X.append(data[i : (i + seq_length)])
            y.append(
                data[i + seq_length : i + seq_length + forecast_horizon, 0]
            )  # Only price column for target
        return np.array(X), np.array(y)

    def _prepare_data(
        self,
        price_data: DataFrame[PriceDataSchema],
        weather_data: Optional[DataFrame[WeatherDataSchema]] = None,
        train_split: float = 0.8,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare the data for training and testing.

        Parameters:
        -----------
        price_data : DataFrame[PriceDataSchema]
            DataFrame containing price data with datetime index
        weather_data : Optional[DataFrame[WeatherDataSchema]], default=None
            DataFrame containing weather data with datetime index
        train_split : float, default=0.8
            Proportion of data to use for training

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (X_train, y_train, X_test, y_test) arrays for training and testing
        """
        # Ensure data is aligned by datetime
        data = price_data[["price"]].copy()

        if weather_data is not None:
            # Merge with weather data
            data = data.join(weather_data.drop(columns=["location"], errors="ignore"))

        # Fill any missing values
        data.fillna(method="ffill", inplace=True)
        data.fillna(method="bfill", inplace=True)

        # Store feature names (excluding the index)
        self.feature_names = data.columns.tolist()

        # Scale the data
        price_values = data[["price"]].values
        self.price_scaler.fit(price_values)
        data[["price"]] = self.price_scaler.transform(price_values)

        if weather_data is not None:
            weather_cols = [col for col in data.columns if col != "price"]
            if weather_cols:
                weather_values = data[weather_cols].values
                self.weather_scaler.fit(weather_values)
                data[weather_cols] = self.weather_scaler.transform(weather_values)

        # Convert to numpy array
        values = data.values

        # Create sequences
        X, y = self._create_sequences(values, self.seq_length, self.forecast_horizon)

        # Split into train and test sets
        train_size = int(len(X) * train_split)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        return X_train, y_train, X_test, y_test

    def fit(
        self,
        price_data: DataFrame[PriceDataSchema],
        weather_data: Optional[DataFrame[WeatherDataSchema]] = None,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: int = 1,
    ) -> keras.callbacks.History:
        """Fit the neural network model to the data.

        Parameters:
        -----------
        price_data : DataFrame[PriceDataSchema]
            DataFrame containing price data with datetime index
        weather_data : Optional[DataFrame[WeatherDataSchema]], default=None
            DataFrame containing weather data with datetime index
        epochs : int, default=50
            Number of epochs to train for
        batch_size : int, default=32
            Batch size for training
        validation_split : float, default=0.2
            Proportion of training data to use for validation
        verbose : int, default=1
            Verbosity mode

        Returns:
        --------
        keras.callbacks.History
            Training history object
        """
        # Prepare the data
        X_train, y_train, X_test, y_test = self._prepare_data(price_data, weather_data)

        # Build the model if it doesn't exist
        if self.model is None:
            self.model = self._build_model(X_train.shape)

        # Train the model
        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ],
        )

        return history

    def predict(
        self,
        price_data: DataFrame[PriceDataSchema],
        weather_data: Optional[DataFrame[WeatherDataSchema]] = None,
    ) -> pd.DataFrame:
        """Generate price forecasts.

        Parameters:
        -----------
        price_data : DataFrame[PriceDataSchema]
            DataFrame containing recent price data with datetime index
        weather_data : Optional[DataFrame[WeatherDataSchema]], default=None
            DataFrame containing recent weather data with datetime index

        Returns:
        --------
        pd.DataFrame
            DataFrame with datetime index and forecasted prices
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        # Ensure data is aligned by datetime
        data = price_data[["price"]].copy()

        if weather_data is not None:
            # Merge with weather data
            data = data.join(weather_data.drop(columns=["location"], errors="ignore"))

        # Fill any missing values
        data.fillna(method="ffill", inplace=True)
        data.fillna(method="bfill", inplace=True)

        # Scale the data
        data[["price"]] = self.price_scaler.transform(data[["price"]].values)

        if weather_data is not None:
            weather_cols = [col for col in data.columns if col != "price"]
            if weather_cols:
                data[weather_cols] = self.weather_scaler.transform(
                    data[weather_cols].values
                )

        # Ensure we have at least seq_length points
        if len(data) < self.seq_length:
            raise ValueError(
                f"Input data must have at least {self.seq_length} time points"
            )

        # Get the last sequence
        last_sequence = data.values[-self.seq_length :].reshape(1, self.seq_length, -1)

        # Make prediction
        prediction = self.model.predict(last_sequence)

        # Inverse transform to get actual prices
        prediction_reshaped = prediction.reshape(-1, 1)
        prediction_inv = self.price_scaler.inverse_transform(prediction_reshaped)

        # Create forecast DataFrame
        last_date = price_data.index[-1]
        forecast_index = pd.date_range(
            start=last_date + pd.Timedelta(hours=1),
            periods=self.forecast_horizon,
            freq="H",
        )

        forecast_df = pd.DataFrame(
            prediction_inv, index=forecast_index, columns=["price_forecast"]
        )

        return forecast_df
