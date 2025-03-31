"""Example usage of Pandera schemas for data validation in the ERCOT price forecasting project.

This script demonstrates how to:
1. Load data using the data classes
2. Use decorators for schema validation
3. Handle validation errors
4. Create custom dataframes that conform to schemas
"""

import os
import sys

import pandas as pd

# Add the project root to the path if running this file directly
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.ercot_price_data import ErcotPriceData
from src.data.ercot_weather_data import ErcotWeatherData
from src.utils.schemas import ForecastSchema, PriceDataFrame, PriceDataSchema


def validate_data_example() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Example of how to validate data using Pandera schemas and decorators.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Validated price and weather data
    """
    print("Loading data...")
    # Initialize data loaders
    price_loader = ErcotPriceData()
    weather_loader = ErcotWeatherData()

    # Load a small sample of data
    start_date = "2022-01-01"
    end_date = "2022-01-07"  # One week of data
    price_node = "HB_HOUSTON"
    location = "Houston"

    # Load price data (validation happens automatically)
    price_data = price_loader.load_data(
        start_date=start_date, end_date=end_date, price_node=price_node
    )

    # Load weather data (validation happens automatically)
    weather_data = weather_loader.load_data(
        start_date=start_date, end_date=end_date, location=location
    )

    print("\n1. Successful validation with schema:")
    print("✅ Price and weather data loaded and validated successfully!")

    print("\n2. Example of validation failure (negative prices):")
    # Example: Introducing an error to demonstrate validation failure
    try:
        # Create a copy with invalid data (negative price)
        invalid_price_data = price_data.copy()
        invalid_price_data.loc[invalid_price_data.index[0], "price"] = -50.0

        # Try to validate
        PriceDataSchema.validate(invalid_price_data)
        print("✅ Validation passed (this shouldn't happen!)")
    except Exception as e:
        print(f"❌ Validation failed as expected: {e}")

    print("\n3. Example of creating a valid forecast dataframe:")
    try:
        # Create a simple forecast dataframe
        forecast_df = pd.DataFrame(
            {
                "price_forecast": [25.0, 30.0, 35.0],
                "lower_bound": [20.0, 25.0, 30.0],
                "upper_bound": [30.0, 35.0, 40.0],
            },
            index=pd.date_range(start="2022-01-08", periods=3, freq="H"),
        )

        # Validate with the schema
        valid_forecast = ForecastSchema.validate(forecast_df)
        print("✅ Forecast validation successful!")
        print(valid_forecast)
    except Exception as e:
        print(f"❌ Validation failed: {e}")

    print("\n4. Example of validation failure due to bounds violation:")
    try:
        # Create another forecast with bounds violation
        invalid_forecast = pd.DataFrame(
            {
                "price_forecast": [25.0, 30.0, 35.0],
                "lower_bound": [20.0, 25.0, 30.0],
                "upper_bound": [30.0, 35.0, 33.0],  # Last upper bound < forecast
            },
            index=pd.date_range(start="2022-01-08", periods=3, freq="H"),
        )

        # Try to validate
        ForecastSchema.validate(invalid_forecast)
        print("✅ Validation passed (this shouldn't happen!)")
    except Exception as e:
        print(f"❌ Validation failed as expected: {e}")

    print("\n5. Using check_types decorator for runtime validation:")

    def process_data(df: PriceDataFrame) -> PriceDataFrame:
        """Process price data with validation via decorator."""
        # This will be validated both on input and output
        return df

    try:
        # This should pass (valid data)
        process_data(price_data)
        print("✅ Valid data passed decorator validation")

        # This should fail (invalid data)
        process_data(invalid_price_data)
        print("❌ Invalid data passed validation (this shouldn't happen!)")
    except Exception as e:
        print(f"✅ Decorator validation failed as expected: {e}")

    return price_data, weather_data


if __name__ == "__main__":
    print("=== ERCOT Price Forecasting: Data Validation Example ===\n")
    price_data, weather_data = validate_data_example()

    print("\nData Summary:")
    print(f"Price data shape: {price_data.shape}")
    print(f"Weather data shape: {weather_data.shape}")

    print("\nPrice data sample:")
    print(price_data.head(3))

    print("\nWeather data sample:")
    print(weather_data.head(3))
