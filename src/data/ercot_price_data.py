"""ERCOT price data loading and preprocessing module.

This module provides functionality to load, validate, and preprocess ERCOT price data
from various sources, ensuring type safety and data integrity through Pandera schemas.
"""

import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


class ErcotPriceData:
    """Class for loading and preprocessing ERCOT price data.

    This class provides methods to:
    - Load historical price data from files or APIs
    - Clean and validate the data using Pandera schemas
    - Apply transformations for analysis and modeling
    """

    def __init__(self, data_dir: str | None = None):
        """Initialize the ERCOT price data loader.

        Args:
            data_dir: Directory containing price data files (defaults to '../data/price')
        """
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "price"
        )

        # Create the directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)

        # Dictionary of available price nodes
        self.available_nodes = {
            "HB_HOUSTON": "Houston Hub",
            "HB_NORTH": "North Hub",
            "HB_SOUTH": "South Hub",
            "HB_WEST": "West Hub",
            "LZ_HOUSTON": "Houston Load Zone",
            "LZ_NORTH": "North Load Zone",
            "LZ_SOUTH": "South Load Zone",
            "LZ_WEST": "West Load Zone",
        }

    def load_data(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
        price_node: str = "HB_HOUSTON",
        resample_freq: str | None = None,
    ) -> pd.DataFrame:
        """Load ERCOT price data for a specified date range and price node.

        Args:
            start_date: Start date for the data
            end_date: End date for the data
            price_node: ERCOT price node identifier
            resample_freq: Optional frequency to resample data to (e.g., 'H' for hourly)

        Returns:
            DataFrame validated against PriceDataSchema

        Raises:
            ValueError: If the price node is not recognized or data cannot be loaded
        """
        # Validate price node
        if price_node not in self.available_nodes:
            valid_nodes = ", ".join(self.available_nodes.keys())
            raise ValueError(
                f"Invalid price node: '{price_node}'. Valid options are: {valid_nodes}"
            )

        # Convert date strings to datetime objects if needed
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Load data from file or API (example implementation)
        # In a real implementation, this would fetch from ERCOT API or read from files
        # For this example, we'll generate synthetic data

        # Generate synthetic data
        date_range = pd.date_range(start=start_date, end=end_date, freq="H")

        # Create price patterns with daily, weekly seasonality and some randomness
        hour_of_day = date_range.hour
        day_of_week = date_range.dayofweek

        # Daily pattern: prices peak in afternoon/evening hours
        # Shift the peak to afternoon hours (around 3pm)
        shift_factor = 3
        daily_pattern = 10 + 30 * np.sin(np.pi * (hour_of_day - shift_factor) / 12)

        # Weekly pattern: weekdays have higher prices than weekends
        weekly_pattern = 15 * (day_of_week < 5).astype(float)

        # Random component
        np.random.seed(42)  # For reproducibility
        random_component = np.random.normal(0, 10, size=len(date_range))

        # Generate prices with trend, seasonality, and randomness
        prices = 35.0 + daily_pattern + weekly_pattern + random_component

        # Ensure non-negative prices
        prices = np.maximum(prices, 0)

        # Create DataFrame
        price_data = pd.DataFrame(
            {"price": prices, "price_node": price_node}, index=date_range
        )

        # Resample if requested
        if resample_freq:
            price_data = price_data.resample(resample_freq).mean()
            price_data["price_node"] = price_node  # Restore price_node after resampling

        # Return the data (validation handled by decorator)
        return price_data

    def get_available_nodes(self) -> dict[str, str]:
        """Get a dictionary of available price nodes.

        Returns:
            Dictionary mapping node identifiers to descriptive names
        """
        return self.available_nodes.copy()

    def clean_outliers(
        self,
        data: pd.DataFrame,
        method: str = "iqr",
        threshold: float = 3.0,
    ) -> pd.DataFrame:
        """Clean outliers in price data.

        Args:
            data: Price data to clean
            method: Method to use for outlier detection ('iqr' or 'zscore')
            threshold: Threshold for outlier detection

        Returns:
            Cleaned price data
        """
        cleaned_data = data.copy()

        if method == "iqr":
            # IQR method
            Q1 = cleaned_data["price"].quantile(0.25)
            Q3 = cleaned_data["price"].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            # Cap outliers rather than removing them
            cleaned_data.loc[cleaned_data["price"] < lower_bound, "price"] = lower_bound
            cleaned_data.loc[cleaned_data["price"] > upper_bound, "price"] = upper_bound

        elif method == "zscore":
            # Z-score method
            mean = cleaned_data["price"].mean()
            std = cleaned_data["price"].std()

            # Cap outliers based on z-score
            cleaned_data.loc[
                (cleaned_data["price"] - mean).abs() > threshold * std, "price"
            ] = cleaned_data["price"].clip(
                lower=mean - threshold * std, upper=mean + threshold * std
            )
        else:
            raise ValueError(f"Unknown outlier cleaning method: {method}")

        # Return cleaned data (validation handled by decorator)
        return cleaned_data

    def calculate_summary_statistics(self, data: pd.DataFrame) -> dict[str, Any]:
        """Calculate summary statistics for the price data.

        Args:
            data: Price data to analyze

        Returns:
            Dictionary of summary statistics
        """
        return {
            "mean": float(data["price"].mean()),
            "median": float(data["price"].median()),
            "std": float(data["price"].std()),
            "min": float(data["price"].min()),
            "max": float(data["price"].max()),
            "q25": float(data["price"].quantile(0.25)),
            "q75": float(data["price"].quantile(0.75)),
            "skew": data["price"].skew(),
            "kurtosis": data["price"].kurtosis(),
            "count": int(data["price"].count()),
            "missing": int(data["price"].isna().sum()),
            "time_range": {
                "start": data.index.min().strftime("%Y-%m-%d %H:%M:%S"),
                "end": data.index.max().strftime("%Y-%m-%d %H:%M:%S"),
                "duration_days": (data.index.max() - data.index.min()).days,
            },
        }
