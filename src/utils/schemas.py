"""Pandera schemas for data validation in the ERCOT price forecasting project.

This module defines schemas for validating price and weather data structures
to ensure data consistency throughout the forecasting pipeline.
"""

from typing import cast

import pandas as pd
import pandera as pa

# Define the price data schema
PriceDataSchema = pa.DataFrameSchema(
    columns={
        "price": pa.Column(
            float,
            checks=pa.Check.ge(0.0),
            description="Electricity price in $/MWh",
        ),
        "price_node": pa.Column(
            str,
            nullable=True,
            required=False,
            description="Price node identifier (e.g., HB_HOUSTON)",
        ),
    },
    index=pa.Index(pd.DatetimeIndex, description="Time index of price data"),
    strict=False,  # Allow extra columns
    coerce=True,  # Attempt to coerce data types
)


# Define the weather data schema
WeatherDataSchema = pa.DataFrameSchema(
    columns={
        "temperature": pa.Column(
            float,
            description="Temperature in degrees Celsius",
        ),
        "humidity": pa.Column(
            float,
            checks=[pa.Check.ge(0.0), pa.Check.le(100.0)],
            description="Relative humidity percentage",
        ),
        "wind_speed": pa.Column(
            float,
            checks=pa.Check.ge(0.0),
            description="Wind speed in meters per second",
        ),
        "solar_irradiance": pa.Column(
            float,
            checks=pa.Check.ge(0.0),
            description="Solar irradiance in W/m²",
        ),
        "location": pa.Column(
            str,
            nullable=True,
            required=False,
            description="Weather location identifier",
        ),
    },
    index=pa.Index(pd.DatetimeIndex, description="Time index of weather data"),
    strict=False,  # Allow extra columns
    coerce=True,  # Attempt to coerce data types
)


# Define the forecast data schema
ForecastSchema = pa.DataFrameSchema(
    columns={
        "price_forecast": pa.Column(
            float,
            description="Forecasted electricity price in $/MWh",
        ),
        "variance_forecast": pa.Column(
            float,
            checks=pa.Check.ge(0.0),
            description="Forecasted price variance",
        ),
        "lower_bound": pa.Column(
            float,
            description="Lower confidence interval bound",
        ),
        "upper_bound": pa.Column(
            float,
            description="Upper confidence interval bound",
        ),
    },
    index=pa.Index(pd.DatetimeIndex, description="Time index of forecast data"),
    checks=[
        # Ensure upper bound is greater than or equal to lower bound
        pa.Check(
            lambda df: df["upper_bound"] >= df["lower_bound"],
            element_wise=True,
            error="Upper bound must be greater than or equal to lower bound",
        ),
        # Ensure forecast is within bounds
        pa.Check(
            lambda df: (df["price_forecast"] >= df["lower_bound"])
            & (df["price_forecast"] <= df["upper_bound"]),
            element_wise=True,
            error="Forecast must be within confidence bounds",
        ),
    ],
    strict=False,  # Allow extra columns
    coerce=True,  # Attempt to coerce data types
)


# Schema validation functions
def validate_price_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate price data against the schema.

    Args:
        data: Price data DataFrame

    Returns:
        Validated DataFrame

    Raises:
        pa.errors.SchemaError: If data doesn't conform to schema
    """
    return cast(pd.DataFrame, PriceDataSchema.validate(data))


def validate_weather_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate weather data against the schema.

    Args:
        data: Weather data DataFrame

    Returns:
        Validated DataFrame

    Raises:
        pa.errors.SchemaError: If data doesn't conform to schema
    """
    return cast(pd.DataFrame, WeatherDataSchema.validate(data))


def validate_forecast(data: pd.DataFrame) -> pd.DataFrame:
    """Validate forecast data against the schema.

    Args:
        data: Forecast DataFrame

    Returns:
        Validated DataFrame

    Raises:
        pa.errors.SchemaError: If data doesn't conform to schema
    """
    return cast(pd.DataFrame, ForecastSchema.validate(data))
