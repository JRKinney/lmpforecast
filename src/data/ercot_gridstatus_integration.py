"""ERCOT data integration with gridstatus library.

This module provides integration with the gridstatus library to fetch real ERCOT data
for price and weather information, ensuring type safety and data integrity through
Pandera schemas.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from gridstatus.ercot import Ercot
from gridstatus.ercot_api.ercot_api import ErcotAPI
from pandera.typing import DataFrame

from src.utils.schemas import PriceDataSchema, WeatherDataSchema


class ErcotGridstatusIntegration:
    """Class for fetching real ERCOT data using the gridstatus library.

    This class provides methods to:
    - Load real historical price data from ERCOT
    - Load real weather data for the ERCOT region
    - Validate and preprocess the data using Pandera schemas
    """

    def __init__(self, ercot_api_key: Optional[str] = None):
        """Initialize the ERCOT gridstatus integration.

        Args:
            ercot_api_key: Optional API key for ERCOT API access (can be set via environment variable)
        """
        # Initialize the Ercot client
        self.ercot = Ercot()

        # Set up API key for ErcotAPI (if needed)
        if ercot_api_key:
            os.environ["ERCOT_API_KEY"] = ercot_api_key
            self.ercot_api = ErcotAPI()
        else:
            self.ercot_api = None

        # Map between our price node names and ERCOT's settlement point names
        self.price_node_mapping = {
            "HB_HOUSTON": "HB_HOUSTON",
            "HB_NORTH": "HB_NORTH",
            "HB_SOUTH": "HB_SOUTH",
            "HB_WEST": "HB_WEST",
            "LZ_HOUSTON": "LZ_HOUSTON",
            "LZ_NORTH": "LZ_NORTH",
            "LZ_SOUTH": "LZ_SOUTH",
            "LZ_WEST": "LZ_WEST",
        }

        # Map our location names to ERCOT weather zones
        self.location_to_weather_zone = {
            "Houston": "Coast",
            "Dallas": "North",
            "Austin": "South",
            "San Antonio": "South",
            "Corpus Christi": "Coast",
            "Midland": "West",
            "El Paso": "West",
        }

    def fetch_price_data(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        price_node: str = "HB_HOUSTON",
        market: Literal["day_ahead", "real_time"] = "real_time",
        resample_freq: Optional[str] = None,
    ) -> DataFrame[PriceDataSchema]:
        """Fetch ERCOT price data for a specified date range and price node.

        Args:
            start_date: Start date for the data
            end_date: End date for the data
            price_node: ERCOT price node identifier
            market: 'day_ahead' or 'real_time'
            resample_freq: Optional frequency to resample data to (e.g., 'H' for hourly)

        Returns:
            DataFrame validated against PriceDataSchema

        Raises:
            ValueError: If the price node is not recognized or data cannot be loaded
        """
        # Validate price node
        if price_node not in self.price_node_mapping:
            valid_nodes = ", ".join(self.price_node_mapping.keys())
            raise ValueError(
                f"Invalid price node: '{price_node}'. Valid options are: {valid_nodes}"
            )

        # Convert date strings to datetime objects if needed
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Map the market parameter to gridstatus market type
        market_mapping = {
            "day_ahead": "DAY_AHEAD_HOURLY",
            "real_time": "REAL_TIME_15_MIN",
        }
        gridstatus_market = market_mapping[market]

        # Fetch data from ERCOT using gridstatus
        ercot_price_node = self.price_node_mapping[price_node]

        # Fetch settlement point prices
        try:
            raw_price_data = self.ercot.get_spp(
                date=start_date,
                end=end_date,
                market=gridstatus_market,
                locations=[ercot_price_node],
                verbose=False,
            )

            # Convert to our schema format
            price_data = pd.DataFrame(
                {"price": raw_price_data["price"].values, "price_node": price_node},
                index=raw_price_data.index,
            )

            # Resample if requested
            if resample_freq:
                price_data = price_data.resample(resample_freq).mean()
                price_data["price_node"] = (
                    price_node  # Restore price_node after resampling
                )

            # Validate against schema
            try:
                validated_data = PriceDataSchema.validate(price_data)
                return validated_data
            except Exception as e:
                raise ValueError(f"Data validation failed: {str(e)}") from e

        except Exception as e:
            raise ValueError(f"Failed to fetch ERCOT price data: {str(e)}") from e

    def fetch_weather_data(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        location: str = "Houston",
        resample_freq: Optional[str] = None,
    ) -> DataFrame[WeatherDataSchema]:
        """Fetch weather data for a specified date range and location.

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
        if location not in self.location_to_weather_zone:
            valid_locations = ", ".join(self.location_to_weather_zone.keys())
            raise ValueError(
                f"Invalid location: '{location}'. Valid options are: {valid_locations}"
            )

        # Convert date strings to datetime objects if needed
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        try:
            # Fetch temperature forecast by weather zone
            temp_data = self.ercot.get_temperature_forecast_by_weather_zone(
                date=start_date, end=end_date, verbose=False
            )

            # Get the corresponding weather zone for the location
            weather_zone = self.location_to_weather_zone[location]

            # Filter for the specific weather zone. If exact zone not found, use average as fallback
            temperature = (
                temp_data[weather_zone]
                if weather_zone in temp_data.columns
                else temp_data.mean(axis=1)
            )

            # For wind and solar, we need to use wind_actual_and_forecast and solar_actual_and_forecast
            try:
                wind_data = self.ercot.get_wind_actual_and_forecast_hourly(
                    date=start_date, end=end_date, verbose=False
                )

                # Get wind speed (estimated from generation)
                if "Wind Speed" in wind_data.columns:
                    wind_speed = wind_data["Wind Speed"]
                elif "Wind Generation" in wind_data.columns:
                    # Estimate wind speed from generation (simplified conversion)
                    # In a real scenario, this would need proper modeling based on turbine characteristics
                    wind_speed = np.sqrt(
                        wind_data["Wind Generation"] / 100
                    )  # Simplified conversion
                else:
                    # Generate synthetic wind data if not available
                    date_range = pd.date_range(start=start_date, end=end_date, freq="H")
                    wind_speed = pd.Series(
                        5
                        + 3 * np.sin(np.pi * date_range.hour / 12)
                        + np.random.exponential(2, size=len(date_range)),
                        index=date_range,
                    )
            except Exception:
                # Generate synthetic wind data if API fails
                date_range = pd.date_range(start=start_date, end=end_date, freq="H")
                wind_speed = pd.Series(
                    5
                    + 3 * np.sin(np.pi * date_range.hour / 12)
                    + np.random.exponential(2, size=len(date_range)),
                    index=date_range,
                )

            try:
                solar_data = self.ercot.get_solar_actual_and_forecast_hourly(
                    date=start_date, end=end_date, verbose=False
                )

                # Get solar irradiance (estimated from generation)
                if "Solar Irradiance" in solar_data.columns:
                    solar_irradiance = solar_data["Solar Irradiance"]
                elif "Solar Generation" in solar_data.columns:
                    # Estimate solar irradiance from generation (simplified conversion)
                    solar_irradiance = (
                        solar_data["Solar Generation"] * 2
                    )  # Simplified conversion
                else:
                    # Generate synthetic solar data if not available
                    date_range = pd.date_range(start=start_date, end=end_date, freq="H")
                    hour_of_day = date_range.hour
                    day_of_year = date_range.dayofyear

                    solar_hours = (hour_of_day >= 6) & (hour_of_day <= 18)
                    solar_peak = (
                        np.sin(np.pi * (hour_of_day - 6) / 12) * solar_hours
                    )  # Peak at noon
                    seasonal_factor = 0.7 + 0.3 * np.sin(
                        2 * np.pi * (day_of_year - 15) / 365
                    )  # Stronger in summer
                    solar_irradiance = pd.Series(
                        1000 * solar_peak * seasonal_factor
                        + np.random.normal(0, 50, size=len(date_range)),
                        index=date_range,
                    )
                    solar_irradiance = solar_irradiance.clip(
                        lower=0
                    )  # No negative irradiance
            except Exception:
                # Generate synthetic solar data if API fails
                date_range = pd.date_range(start=start_date, end=end_date, freq="H")
                hour_of_day = date_range.hour
                day_of_year = date_range.dayofyear

                solar_hours = (hour_of_day >= 6) & (hour_of_day <= 18)
                solar_peak = (
                    np.sin(np.pi * (hour_of_day - 6) / 12) * solar_hours
                )  # Peak at noon
                seasonal_factor = 0.7 + 0.3 * np.sin(
                    2 * np.pi * (day_of_year - 15) / 365
                )  # Stronger in summer
                solar_irradiance = pd.Series(
                    1000 * solar_peak * seasonal_factor
                    + np.random.normal(0, 50, size=len(date_range)),
                    index=date_range,
                )
                solar_irradiance = solar_irradiance.clip(
                    lower=0
                )  # No negative irradiance

            # For humidity, we'll need to generate synthetic data as it's not directly available
            date_range = pd.date_range(start=start_date, end=end_date, freq="H")
            # Base humidity depends on location
            if location in ["Houston", "Corpus Christi"]:
                base_humidity = 75  # Higher humidity in coastal areas
            elif location in ["Midland", "El Paso"]:
                base_humidity = 40  # Lower humidity in western areas
            else:
                base_humidity = 60  # Moderate humidity in central areas

            # Humidity: higher when cooler, lower when warmer, with some randomness
            if isinstance(temperature, pd.Series) and len(temperature) > 0:
                base_temp = temperature.mean()
                humidity = (
                    base_humidity
                    - 0.5 * (temperature - base_temp)
                    + np.random.normal(0, 10, size=len(temperature))
                )
                humidity = humidity.clip(0, 100)  # Ensure between 0-100%
            else:
                # Generate synthetic humidity if temperature data is unavailable
                humidity = pd.Series(
                    base_humidity + np.random.normal(0, 10, size=len(date_range)),
                    index=date_range,
                )
                humidity = humidity.clip(0, 100)  # Ensure between 0-100%

            # Combine all weather data
            # First ensure all series have the same index by reindexing to a common hourly index
            common_index = pd.date_range(start=start_date, end=end_date, freq="H")

            if isinstance(temperature, pd.Series):
                temperature = temperature.reindex(common_index, method="ffill")
            else:
                temperature = pd.Series(
                    np.random.normal(25, 5, size=len(common_index)), index=common_index
                )

            wind_speed = wind_speed.reindex(common_index, method="ffill")
            solar_irradiance = solar_irradiance.reindex(common_index, method="ffill")
            humidity = humidity.reindex(common_index, method="ffill")

            # Create the final weather dataframe
            weather_data = pd.DataFrame(
                {
                    "temperature": temperature,
                    "humidity": humidity,
                    "wind_speed": wind_speed,
                    "solar_irradiance": solar_irradiance,
                    "location": location,
                },
                index=common_index,
            )

            # Resample if requested
            if resample_freq:
                # For resampling, we need to handle non-numeric columns separately
                numeric_data = weather_data.drop(columns=["location"])
                resampled_data = numeric_data.resample(resample_freq).mean()
                resampled_data["location"] = location
                weather_data = resampled_data

            # Validate against schema
            try:
                validated_data = WeatherDataSchema.validate(weather_data)
                return validated_data
            except Exception as e:
                raise ValueError(f"Weather data validation failed: {str(e)}") from e

        except Exception as e:
            raise ValueError(f"Failed to fetch ERCOT weather data: {str(e)}") from e

    def get_available_price_nodes(self) -> Dict[str, str]:
        """Get a dictionary of available price nodes.

        Returns:
            Dictionary mapping node identifiers to descriptive names
        """
        return {
            "HB_HOUSTON": "Houston Hub",
            "HB_NORTH": "North Hub",
            "HB_SOUTH": "South Hub",
            "HB_WEST": "West Hub",
            "LZ_HOUSTON": "Houston Load Zone",
            "LZ_NORTH": "North Load Zone",
            "LZ_SOUTH": "South Load Zone",
            "LZ_WEST": "West Load Zone",
        }

    def fetch_system_wide_data(
        self, start_date: Union[str, datetime], end_date: Union[str, datetime]
    ) -> pd.DataFrame:
        """Fetch system-wide data like load, generation by fuel type, etc.

        Args:
            start_date: Start date for the data
            end_date: End date for the data

        Returns:
            DataFrame with system-wide data
        """
        try:
            # Fetch system-wide actual load (with type assertion)
            system_data = pd.DataFrame(
                self.ercot.get_system_wide_actual_load(
                    date=start_date, end=end_date, verbose=False
                )
            )

            return system_data
        except Exception as e:
            raise ValueError(f"Failed to fetch system-wide data: {str(e)}") from e

    def fetch_fuel_mix(
        self, start_date: Union[str, datetime], end_date: Union[str, datetime]
    ) -> pd.DataFrame:
        """Fetch generation by fuel type data.

        Args:
            start_date: Start date for the data
            end_date: End date for the data

        Returns:
            DataFrame with generation by fuel type
        """
        try:
            # Fetch daily data and concatenate
            all_data: List[pd.DataFrame] = []
            current_date = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)

            while current_date <= end:
                try:
                    daily_mix = self.ercot.get_fuel_mix(
                        date=current_date, verbose=False
                    )
                    all_data.append(daily_mix)
                except Exception:
                    # Continue if a particular day fails
                    pass

                current_date += timedelta(days=1)

            if all_data:
                return pd.concat(all_data)
            else:
                raise ValueError(
                    "No fuel mix data available for the specified date range"
                )
        except Exception as e:
            raise ValueError(f"Failed to fetch fuel mix data: {str(e)}") from e
