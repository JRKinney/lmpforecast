"""ERCOT weather data loading and preprocessing module.

This module provides functionality to load, validate, and preprocess weather data
for various locations in the ERCOT region, ensuring type safety and data integrity
through Pandera schemas.
"""

import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from pandera.decorators import check_types
from pandera.typing import DataFrame

from src.utils.schemas import WeatherDataSchema


class ErcotWeatherData:
    """Class for loading and preprocessing weather data for the ERCOT region.

    This class provides methods to:
    - Load historical weather data from files or APIs
    - Clean and validate the data using Pandera schemas
    - Apply transformations for analysis and modeling
    """

    def __init__(self, data_dir: Optional[str] = None):
        """Initialize the ERCOT weather data loader.

        Args:
            data_dir: Directory containing weather data files (defaults to '../data/weather')
        """
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "weather",
        )

        # Create the directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)

        # Dictionary of available locations with coordinates (lat, lon)
        self.available_locations = {
            "Houston": (29.7604, -95.3698),
            "Dallas": (32.7767, -96.7970),
            "Austin": (30.2672, -97.7431),
            "San Antonio": (29.4241, -98.4936),
            "Corpus Christi": (27.8006, -97.3964),
            "Midland": (31.9973, -102.0779),
            "El Paso": (31.7619, -106.4850),
        }

    @check_types
    def load_data(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        location: str = "Houston",
        resample_freq: Optional[str] = None,
    ) -> DataFrame[WeatherDataSchema]:
        """Load weather data for a specified date range and location.

        Args:
            start_date: Start date for the data
            end_date: End date for the data
            location: Location name (city) in the ERCOT region
            resample_freq: Optional frequency to resample data to (e.g., 'H' for hourly)

        Returns:
            DataFrame validated against WeatherDataSchema

        Raises:
            ValueError: If the location is not recognized or data cannot be loaded
        """
        # Validate location
        if location not in self.available_locations:
            valid_locations = ", ".join(self.available_locations.keys())
            raise ValueError(
                f"Invalid location: '{location}'. Valid options are: {valid_locations}"
            )

        # Convert date strings to datetime objects if needed
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Load data from file or API (example implementation)
        # In a real implementation, this would fetch from a weather API or read from files
        # For this example, we'll generate synthetic data

        # Generate synthetic data
        date_range = pd.date_range(start=start_date, end=end_date, freq="H")

        # Create weather patterns with daily, seasonal variations and some randomness
        hour_of_day = date_range.hour
        day_of_year = date_range.dayofyear

        # Get location coordinates
        lat, lon = self.available_locations[location]

        # Temperature: daily cycle + seasonal cycle + random noise
        # Base temperature depends on latitude (lower latitudes = warmer)
        base_temp = 25 - 0.4 * (lat - 30)  # Base temperature in Celsius

        # Daily temperature cycle (cooler at night, warmer during day)
        daily_cycle = 5 * np.sin(np.pi * (hour_of_day - 4) / 12)

        # Seasonal cycle (warmer in summer, cooler in winter)
        # Assuming day 1 is January 1, day 183 is July 1
        seasonal_cycle = 10 * np.sin(2 * np.pi * (day_of_year - 15) / 365)

        # Random component
        np.random.seed(42)  # For reproducibility
        temp_noise = np.random.normal(0, 2, size=len(date_range))

        # Combine components
        temperature = base_temp + daily_cycle + seasonal_cycle + temp_noise

        # Humidity: higher when cooler, lower when warmer, with some randomness
        # Base humidity is higher near the coast (negative longitude = east)
        base_humidity = 60 + 0.5 * (-lon - 95)
        humidity = (
            base_humidity
            - 0.5 * (temperature - base_temp)
            + np.random.normal(0, 10, size=len(date_range))
        )
        humidity = np.clip(humidity, 0, 100)  # Ensure between 0-100%

        # Wind speed: some daily pattern with randomness
        wind_speed = (
            5
            + 3 * np.sin(np.pi * hour_of_day / 12)
            + np.random.exponential(2, size=len(date_range))
        )

        # Solar irradiance: follows daylight hours
        solar_hours = (hour_of_day >= 6) & (hour_of_day <= 18)
        solar_peak = (
            np.sin(np.pi * (hour_of_day - 6) / 12) * solar_hours
        )  # Peak at noon
        seasonal_factor = 0.7 + 0.3 * np.sin(
            2 * np.pi * (day_of_year - 15) / 365
        )  # Stronger in summer
        solar_irradiance = 1000 * solar_peak * seasonal_factor + np.random.normal(
            0, 50, size=len(date_range)
        )
        solar_irradiance = np.maximum(solar_irradiance, 0)  # No negative irradiance

        # Create DataFrame
        weather_data = pd.DataFrame(
            {
                "temperature": temperature,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "solar_irradiance": solar_irradiance,
                "location": location,
            },
            index=date_range,
        )

        # Resample if requested
        if resample_freq:
            # For resampling, we need to handle non-numeric columns separately
            numeric_data = weather_data.drop(columns=["location"])
            resampled_data = numeric_data.resample(resample_freq).mean()
            resampled_data["location"] = location
            weather_data = resampled_data

        # Return data (validation handled by decorator)
        return weather_data

    def get_available_locations(self) -> Dict[str, Tuple[float, float]]:
        """Get a dictionary of available locations with their coordinates.

        Returns:
            Dictionary mapping location names to (latitude, longitude) tuples
        """
        return self.available_locations.copy()

    @check_types
    def clean_outliers(
        self,
        data: DataFrame[WeatherDataSchema],
        method: str = "iqr",
        threshold: float = 3.0,
    ) -> DataFrame[WeatherDataSchema]:
        """Clean outliers in weather data.

        Args:
            data: Weather data to clean
            method: Method to use for outlier detection ('iqr' or 'zscore')
            threshold: Threshold for outlier detection

        Returns:
            Cleaned weather data
        """
        cleaned_data = data.copy()

        # Numerical columns to check for outliers
        numeric_columns = ["temperature", "humidity", "wind_speed", "solar_irradiance"]

        for column in numeric_columns:
            if method == "iqr":
                # IQR method
                Q1 = cleaned_data[column].quantile(0.25)
                Q3 = cleaned_data[column].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                # Cap outliers rather than removing them
                cleaned_data.loc[cleaned_data[column] < lower_bound, column] = (
                    lower_bound
                )
                cleaned_data.loc[cleaned_data[column] > upper_bound, column] = (
                    upper_bound
                )

            elif method == "zscore":
                # Z-score method
                mean = cleaned_data[column].mean()
                std = cleaned_data[column].std()

                # Cap outliers based on z-score
                cleaned_data.loc[
                    (cleaned_data[column] - mean).abs() > threshold * std, column
                ] = cleaned_data[column].clip(
                    lower=mean - threshold * std, upper=mean + threshold * std
                )
            else:
                raise ValueError(f"Unknown outlier cleaning method: {method}")

        # Special handling for physical constraints
        # Humidity must be between 0-100%
        cleaned_data["humidity"] = cleaned_data["humidity"].clip(0, 100)

        # Wind speed and solar irradiance cannot be negative
        cleaned_data["wind_speed"] = cleaned_data["wind_speed"].clip(lower=0)
        cleaned_data["solar_irradiance"] = cleaned_data["solar_irradiance"].clip(
            lower=0
        )

        # Return cleaned data (validation handled by decorator)
        return cleaned_data

    @check_types
    def calculate_summary_statistics(
        self, data: DataFrame[WeatherDataSchema]
    ) -> Dict[str, Any]:
        """Calculate summary statistics for the weather data.

        Args:
            data: Weather data to analyze

        Returns:
            Dictionary containing:
            - Statistical metrics for each weather variable (mean, median, etc.)
            - Metadata about the dataset (location, time range, count)
        """
        # Variables to analyze
        variables = ["temperature", "humidity", "wind_speed", "solar_irradiance"]

        stats: Dict[str, Any] = {}
        for var in variables:
            stats[var] = {
                "mean": float(data[var].mean()),
                "median": float(data[var].median()),
                "std": float(data[var].std()),
                "min": float(data[var].min()),
                "max": float(data[var].max()),
                "q25": float(data[var].quantile(0.25)),
                "q75": float(data[var].quantile(0.75)),
                "missing": int(data[var].isna().sum()),
            }

        # Add overall metadata
        stats["metadata"] = {
            "location": data["location"].iloc[0],
            "time_range": {
                "start": data.index.min().strftime("%Y-%m-%d %H:%M:%S"),
                "end": data.index.max().strftime("%Y-%m-%d %H:%M:%S"),
                "duration_days": (data.index.max() - data.index.min()).days,
            },
            "count": len(data),
        }

        return stats

    @check_types
    def get_extreme_weather_events(
        self, data: DataFrame[WeatherDataSchema], percentile_threshold: float = 0.95
    ) -> pd.DataFrame:
        """Identify extreme weather events in the data.

        Args:
            data: Weather data to analyze
            percentile_threshold: Percentile threshold for extreme events

        Returns:
            DataFrame with identified extreme weather events
        """
        # Make a copy to avoid modifying the original
        df = data.copy()

        # Calculate thresholds for extreme events
        high_temp_threshold = df["temperature"].quantile(percentile_threshold)
        high_wind_threshold = df["wind_speed"].quantile(percentile_threshold)
        low_humidity_threshold = df["humidity"].quantile(1 - percentile_threshold)

        # Create flags for extreme events
        df["extreme_heat"] = df["temperature"] > high_temp_threshold
        df["extreme_wind"] = df["wind_speed"] > high_wind_threshold
        df["extreme_dry"] = df["humidity"] < low_humidity_threshold

        # Overall extreme flag (any extreme condition)
        df["any_extreme_condition"] = (
            df["extreme_heat"] | df["extreme_wind"] | df["extreme_dry"]
        )

        # Return only periods with extreme events
        return cast(pd.DataFrame, df[df["any_extreme_condition"]].copy())
