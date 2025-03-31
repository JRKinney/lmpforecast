# Gridstatus Integration Guide

This guide explains how to use the gridstatus library integration to access real ERCOT data for energy price forecasting.

## Overview

The gridstatus library provides a uniform API for accessing electricity supply, demand, and pricing data for major Independent System Operators (ISOs) in the United States, including ERCOT. Our integration allows you to:

- Fetch real historical price data from ERCOT
- Access weather data and forecast information
- Get system-wide data such as load and generation by fuel type
- Use this real data for price forecasting

## Installation

The gridstatus library is automatically installed when you install the project requirements:

```bash
pip install -r requirements.txt
```

## Configuration

Some gridstatus features require an ERCOT API key. To use these features:

1. Register for an API key at the [ERCOT website](https://www.ercot.com/services/api)
2. Copy the `.env.template` file to `.env`:
   ```bash
   cp .env.template .env
   ```
3. Edit the `.env` file and add your API key:
   ```
   ERCOT_API_KEY=your_api_key_here
   ```

Note: Basic functionality will still work without an API key.

## Using the Integration

### From Python Code

To use the integration in your Python code:

```python
from datetime import datetime, timedelta
from src.data.ercot_gridstatus_integration import ErcotGridstatusIntegration

# Initialize the integration
gridstatus = ErcotGridstatusIntegration()

# Define date range
end_date = datetime.now()
start_date = end_date - timedelta(days=7)

# Fetch price data
price_data = gridstatus.fetch_price_data(
    start_date=start_date,
    end_date=end_date,
    price_node='HB_HOUSTON',
    market='real_time',
    resample_freq='H'
)

# Fetch weather data
weather_data = gridstatus.fetch_weather_data(
    start_date=start_date,
    end_date=end_date,
    location='Houston',
    resample_freq='H'
)

# Fetch additional data types
fuel_mix = gridstatus.fetch_fuel_mix(
    start_date=start_date,
    end_date=end_date
)

system_data = gridstatus.fetch_system_wide_data(
    start_date=start_date,
    end_date=end_date
)
```

### Using the Command Line Script

We provide a command-line script that can use real data from gridstatus for forecasting:

```bash
python run_forecast_with_gridstatus.py --price-node=HB_HOUSTON --location=Houston
```

By default, the script will try to fetch real data, but it will fall back to synthetic data if real data cannot be accessed.

To explicitly use synthetic data:

```bash
python run_forecast_with_gridstatus.py --use-synthetic
```

Other useful options:

```bash
# Change the forecast horizon
python run_forecast_with_gridstatus.py --forecast-horizon=48

# Save the plot and model
python run_forecast_with_gridstatus.py --save-plot --save-model

# Change the amount of historical data used
python run_forecast_with_gridstatus.py --days-history=60

# Get help with all options
python run_forecast_with_gridstatus.py --help
```

## Exploring with Jupyter Notebook

For an interactive exploration of the gridstatus integration, check out the demonstration notebook:

```bash
jupyter lab notebooks/gridstatus_integration_demo.ipynb
```

This notebook shows how to fetch and visualize various types of data from ERCOT using the gridstatus integration.

## Available Data Types

### Price Data

The integration can fetch price data for various ERCOT nodes:

- `HB_HOUSTON`: Houston Hub
- `HB_NORTH`: North Hub
- `HB_SOUTH`: South Hub
- `HB_WEST`: West Hub
- `LZ_HOUSTON`: Houston Load Zone
- `LZ_NORTH`: North Load Zone
- `LZ_SOUTH`: South Load Zone
- `LZ_WEST`: West Load Zone

### Weather Data

Weather data is available for these locations, which map to ERCOT weather zones:

- `Houston`: Coast
- `Dallas`: North
- `Austin`: South
- `San Antonio`: South
- `Corpus Christi`: Coast
- `Midland`: West
- `El Paso`: West

## Fallback Mechanism

The integration includes a fallback mechanism that automatically reverts to synthetic data generation if real data cannot be accessed. This ensures that the forecasting pipeline can always run, even if there are issues with the ERCOT API or network connectivity.

## Troubleshooting

If you encounter issues with the gridstatus integration:

1. Check your API key and environment variables
2. Ensure you have internet connectivity
3. Verify the date range (ERCOT may have limits on how far back you can query)
4. Check the gridstatus GitHub repository for updates or known issues

If all else fails, the system will automatically fall back to synthetic data generation.

## Further Resources

- [Gridstatus GitHub Repository](https://github.com/gridstatus/gridstatus)
- [ERCOT Website](https://www.ercot.com/)
- [GridStatus.io](https://www.gridstatus.io/) - Preview of data provided by gridstatus
