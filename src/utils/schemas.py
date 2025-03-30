"""
Pandera schemas for data validation across the ERCOT price forecasting project.

This module defines schemas for price data, weather data, and forecast outputs
to ensure data consistency and validity throughout the application.
"""

from typing import Optional, List, Dict, Any, Union, Literal, TypeVar, Generic
import pandas as pd
import numpy as np
import pandera as pa
from pandera.typing import Series, DataFrame, Index


# Schema for price data
class PriceDataSchema(pa.SchemaModel):
    """
    Schema for ERCOT price data.
    
    Validates:
    - Price values are non-negative
    - Required columns are present
    - Price node information is included
    - Index is a datetime index
    """
    # Price column (must be non-negative)
    price: Series[float] = pa.Field(
        ge=0.0,  # Greater than or equal to zero
        description="Price in $/MWh"
    )
    
    # Price node column (categorical)
    price_node: Series[str] = pa.Field(
        description="ERCOT price node identifier"
    )
    
    # Index must be a datetime
    index: Index[pd.DatetimeIndex] = pa.Field(
        description="Timestamp of the price data point"
    )
    
    # Additional schema-level checks
    class Config:
        """Configuration for the price data schema"""
        strict = True
        coerce = True


# Schema for weather data
class WeatherDataSchema(pa.SchemaModel):
    """
    Schema for weather data.
    
    Validates:
    - Temperature ranges are plausible
    - Required columns are present
    - Location information is included
    - Index is a datetime index aligned with price data
    """
    # Temperature columns with physical constraints
    temperature: Series[float] = pa.Field(
        ge=-50.0,  # Greater than or equal to -50°C (extreme cold)
        le=60.0,   # Less than or equal to 60°C (extreme heat)
        description="Temperature in degrees Celsius"
    )
    
    # Wind speed (non-negative)
    wind_speed: Series[float] = pa.Field(
        ge=0.0,
        description="Wind speed in m/s"
    )
    
    # Humidity (between 0-100%)
    humidity: Series[float] = pa.Field(
        ge=0.0,
        le=100.0,
        description="Relative humidity percentage"
    )
    
    # Solar irradiance (non-negative)
    solar_irradiance: Series[float] = pa.Field(
        ge=0.0,
        description="Solar irradiance in W/m²"
    )
    
    # Location column
    location: Series[str] = pa.Field(
        description="Weather measurement location"
    )
    
    # Index must be a datetime
    index: Index[pd.DatetimeIndex] = pa.Field(
        description="Timestamp of the weather data point"
    )
    
    # Additional schema-level checks
    class Config:
        """Configuration for the weather data schema"""
        strict = True
        coerce = True


# Schema for price forecasts
class PriceForecastSchema(pa.SchemaModel):
    """
    Schema for price forecast outputs.
    
    Validates:
    - Forecast values are non-negative
    - Confidence bounds are properly ordered (lower <= forecast <= upper)
    - Required columns are present
    - Index is a datetime index
    """
    # Price forecast column (non-negative)
    price_forecast: Series[float] = pa.Field(
        ge=0.0,
        description="Forecasted price in $/MWh"
    )
    
    # Lower confidence bound (non-negative)
    lower_bound: Series[float] = pa.Field(
        ge=0.0,
        description="Lower bound of forecast confidence interval"
    )
    
    # Upper confidence bound (non-negative)
    upper_bound: Series[float] = pa.Field(
        ge=0.0,
        description="Upper bound of forecast confidence interval"
    )
    
    # Index must be a datetime
    index: Index[pd.DatetimeIndex] = pa.Field(
        description="Timestamp of the forecasted data point"
    )
    
    @pa.check("upper_bound", "lower_bound")
    def upper_bound_greater_than_lower(cls, upper_bound: Series, lower_bound: Series) -> Series[bool]:
        """Check that upper bound is greater than or equal to lower bound."""
        return upper_bound >= lower_bound
    
    @pa.check("price_forecast", "upper_bound", "lower_bound")
    def forecast_within_bounds(cls, forecast: Series, upper: Series, lower: Series) -> Series[bool]:
        """Check that forecast is within the confidence bounds."""
        return (forecast <= upper) & (forecast >= lower)
    
    # Additional schema-level checks
    class Config:
        """Configuration for the forecast schema"""
        strict = True
        coerce = True


# Schema for neural network inputs
class NeuralNetworkInputSchema(pa.SchemaModel):
    """
    Schema for data prepared for neural network inputs.
    
    Validates:
    - Features are properly scaled
    - Required columns are present
    - Index is a datetime index
    """
    # Price lags (scaled between -1 and 1)
    price_lag_1: Series[float] = pa.Field(
        ge=-10.0, le=10.0,
        description="1-hour lagged price (scaled)"
    )
    
    price_lag_24: Series[float] = pa.Field(
        ge=-10.0, le=10.0,
        description="24-hour lagged price (scaled)"
    )
    
    # Day of week (one-hot encoded)
    day_of_week_0: Series[int] = pa.Field(
        isin=[0, 1],
        description="Monday indicator (one-hot encoded)"
    )
    
    day_of_week_1: Series[int] = pa.Field(
        isin=[0, 1],
        description="Tuesday indicator (one-hot encoded)"
    )
    
    day_of_week_2: Series[int] = pa.Field(
        isin=[0, 1],
        description="Wednesday indicator (one-hot encoded)"
    )
    
    day_of_week_3: Series[int] = pa.Field(
        isin=[0, 1],
        description="Thursday indicator (one-hot encoded)"
    )
    
    day_of_week_4: Series[int] = pa.Field(
        isin=[0, 1],
        description="Friday indicator (one-hot encoded)"
    )
    
    day_of_week_5: Series[int] = pa.Field(
        isin=[0, 1],
        description="Saturday indicator (one-hot encoded)"
    )
    
    day_of_week_6: Series[int] = pa.Field(
        isin=[0, 1],
        description="Sunday indicator (one-hot encoded)"
    )
    
    # Hour of day (scaled)
    hour: Series[float] = pa.Field(
        ge=0.0, le=1.0,
        description="Hour of day (scaled between 0 and 1)"
    )
    
    # Weather features (scaled)
    temperature: Series[float] = pa.Field(
        ge=-10.0, le=10.0,
        description="Temperature (scaled)"
    )
    
    wind_speed: Series[float] = pa.Field(
        ge=-10.0, le=10.0,
        description="Wind speed (scaled)"
    )
    
    humidity: Series[float] = pa.Field(
        ge=-10.0, le=10.0,
        description="Humidity (scaled)"
    )
    
    solar_irradiance: Series[float] = pa.Field(
        ge=-10.0, le=10.0,
        description="Solar irradiance (scaled)"
    )
    
    # Index must be a datetime
    index: Index[pd.DatetimeIndex] = pa.Field(
        description="Timestamp of the input data point"
    )
    
    # Additional schema-level checks
    class Config:
        """Configuration for the neural network input schema"""
        strict = False  # Allow extra columns
        coerce = True


# Schema for GARCH model inputs
class GarchInputSchema(pa.SchemaModel):
    """
    Schema for data prepared for GARCH model inputs.
    
    Validates:
    - Residuals are properly calculated
    - Required columns are present
    - Index is a datetime index
    """
    # Residuals from neural network prediction
    residual: Series[float] = pa.Field(
        description="Residual (actual - predicted) from neural network"
    )
    
    # Index must be a datetime
    index: Index[pd.DatetimeIndex] = pa.Field(
        description="Timestamp of the input data point"
    )
    
    # Additional schema-level checks
    class Config:
        """Configuration for the GARCH input schema"""
        strict = False  # Allow extra columns
        coerce = True


# Function to validate dataframes against their schemas
def validate_dataframe(
    df: pd.DataFrame, 
    schema: Union[type[PriceDataSchema], type[WeatherDataSchema], type[PriceForecastSchema]]
) -> DataFrame:
    """
    Validate a dataframe against a schema.
    
    Args:
        df: DataFrame to validate
        schema: Schema to validate against
        
    Returns:
        The validated DataFrame
        
    Raises:
        pa.errors.SchemaError: If validation fails
    """
    try:
        return schema.validate(df)
    except pa.errors.SchemaError as e:
        # Add more context to the error message
        print(f"Schema validation failed: {e}")
        print(f"DataFrame overview:\n{df.head()}")
        print(f"DataFrame info:\n{df.info()}")
        raise 