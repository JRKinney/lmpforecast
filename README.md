# Hybrid GARCH-NN Model for ERCOT Energy Price Forecasting

This project implements a hybrid model for forecasting energy prices in the ERCOT (Electric Reliability Council of Texas) market. The model combines the strengths of neural networks for mean price forecasting with GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models for variance/volatility forecasting.

## Features

- **Neural Network Forecasting**: LSTM-based model for predicting mean price movements
- **GARCH Volatility Modeling**: Captures price volatility patterns for accurate confidence intervals
- **Weather Data Integration**: Includes weather variables as exogenous inputs
- **Interactive Visualizations**: Plotly-based visualizations of forecasts and model performance
- **Confidence Intervals**: Provides uncertainty bounds around price forecasts
- **Extensible Architecture**: Easily add new feature sets or model components
- **Static Type Checking**: Full type hints with `typing` and `pandera` for better code quality
- **Data Validation**: Pandera schemas for validating DataFrame structures and data constraints
- **Real ERCOT Data Integration**: Connects to real ERCOT data through the gridstatus library

## Project Structure

```
/
├── .envrc                     # direnv configuration
├── README.md                  # This documentation
├── requirements.txt           # Python dependencies
├── run_forecast.py            # Command-line interface for forecasting
├── notebooks/                 # Jupyter notebooks
│   ├── model_visualization.ipynb    # Example usage and visualization
│   └── gridstatus_integration_demo.ipynb  # Demo of gridstatus integration
├── models/                    # Saved model files
│   └── saved/                 # Directory for saved models
└── src/                       # Source code
    ├── data/                  # Data handling modules
    │   ├── ercot_price_data.py         # ERCOT price data class
    │   ├── ercot_weather_data.py       # Weather data class
    │   └── ercot_gridstatus_integration.py  # Integration with gridstatus
    ├── models/                # Model implementations
    │   ├── garch_model.py     # GARCH model implementation
    │   ├── nn_model.py        # Neural network model implementation
    │   └── hybrid_model.py    # Hybrid model combining GARCH and NN
    ├── visualization/         # Visualization tools
    │   └── plotting.py        # Plotting functions using Plotly
    └── utils/                 # Utility functions
        ├── preprocessing.py   # Data preprocessing functions
        ├── schemas.py         # Pandera schemas for data validation
        └── validation_example.py  # Example of schema validation
```

## Installation

1. Clone this repository:

```bash
git clone <repository-url>
cd ercot-price-forecasting
```

2. Ensure you have `direnv` installed:

```bash
# On macOS with Homebrew
brew install direnv

# On Ubuntu
sudo apt-get install direnv
```

3. Add the following to your shell configuration file (e.g., `.bashrc` or `.zshrc`):

```bash
eval "$(direnv hook bash)"  # or zsh if you use zsh
```

4. Allow direnv to create and use the virtual environment:

```bash
direnv allow
```

This will automatically create a Python virtual environment and install the required dependencies.

## Usage

### Data Classes

The project includes three main data classes:

1. `ErcotPriceData`: For loading synthetic ERCOT price data
2. `ErcotWeatherData`: For loading synthetic weather data
3. `ErcotGridstatusIntegration`: For fetching real ERCOT data using the gridstatus library

#### Using Synthetic Data

```python
from src.data.ercot_price_data import ErcotPriceData
from src.data.ercot_weather_data import ErcotWeatherData

# Initialize data loaders
price_loader = ErcotPriceData()
weather_loader = ErcotWeatherData()

# Load historical data
price_data = price_loader.load_data(
    start_date='2022-01-01',
    end_date='2022-12-31',
    price_node='HB_HOUSTON'
)

weather_data = weather_loader.load_data(
    start_date='2022-01-01',
    end_date='2022-12-31',
    location='Houston'
)
```

#### Using Real ERCOT Data (with gridstatus)

```python
from src.data.ercot_gridstatus_integration import ErcotGridstatusIntegration

# Initialize the gridstatus integration
gridstatus_loader = ErcotGridstatusIntegration()

# Fetch real price data
real_price_data = gridstatus_loader.fetch_price_data(
    start_date='2023-01-01',
    end_date='2023-01-07',
    price_node='HB_HOUSTON',
    market='real_time'  # or 'day_ahead'
)

# Fetch real weather data
real_weather_data = gridstatus_loader.fetch_weather_data(
    start_date='2023-01-01',
    end_date='2023-01-07',
    location='Houston'
)

# Fetch additional data like fuel mix or system-wide data
fuel_mix = gridstatus_loader.fetch_fuel_mix(
    start_date='2023-01-01',
    end_date='2023-01-07'
)

system_data = gridstatus_loader.fetch_system_wide_data(
    start_date='2023-01-01',
    end_date='2023-01-07'
)
```

### Training and Using the Hybrid Model

```python
from src.models.hybrid_model import HybridModel

# Initialize the model
model = HybridModel(
    seq_length=24,           # Use 24 hours of data for input
    forecast_horizon=24,     # Forecast 24 hours ahead
    p=1, q=1,                # GARCH(1,1) model
    mean='Zero',             # Zero mean for GARCH
    vol='GARCH',             # Standard GARCH volatility model
    dist='normal'            # Normal distribution for errors
)

# Train the model
model.fit(
    price_data=train_price,
    weather_data=train_weather,
    nn_epochs=50,
    nn_batch_size=32,
    nn_validation_split=0.2,
    verbose=1
)

# Generate forecast with confidence intervals
forecast = model.predict(
    price_data=recent_price,
    weather_data=recent_weather,
    confidence_level=0.95
)
```

### Visualization

The project includes several visualization functions:

```python
from src.visualization.plotting import plot_price_forecast

# Plot price forecast with confidence intervals
fig = plot_price_forecast(
    forecasts=forecast,
    historical_data=historical_price_data,
    title="ERCOT Price Forecast"
)
fig.show()
```

### Data Schema Validation

The project uses Pandera schemas to validate data structures:

```python
from src.utils.schemas import PriceDataSchema, WeatherDataSchema
import pandas as pd

# Validate a price DataFrame
my_price_data = pd.DataFrame({
    'price': [25.0, 30.0, 35.0],
    'price_node': ['HB_HOUSTON'] * 3
}, index=pd.date_range('2023-01-01', periods=3))

# Validate according to schema
validated_data = PriceDataSchema.validate(my_price_data)

# The schema will enforce:
# - datetime index
# - non-negative price values
# - appropriate column names and types
```

## Command Line Interface

You can also run the forecasting model from the command line:

```bash
python run_forecast.py --price-node=HB_HOUSTON --location=Houston --forecast-horizon=24 --save-plot
```

## Example Notebooks

For interactive examples of the model in action, check out the Jupyter notebooks in the `notebooks` directory:

```bash
# General model visualization
jupyter lab notebooks/model_visualization.ipynb

# Gridstatus integration demonstration
jupyter lab notebooks/gridstatus_integration_demo.ipynb
```

## Gridstatus Integration

The project integrates with the [gridstatus](https://github.com/gridstatus/gridstatus) library to fetch real ERCOT data. This provides several advantages:

- **Real Historical Data**: Access to actual historical ERCOT price and weather data
- **Multiple Data Sources**: Fetch price data, weather data, fuel mix, and system-wide metrics
- **Market Types**: Support for both real-time and day-ahead markets
- **Multiple Nodes**: Data for different price nodes (hubs, load zones)
- **Fallback Mechanism**: Graceful degradation to synthetic data when API access fails

### Setup

To use the gridstatus integration with the ERCOT API, you may need to:

1. Register for an ERCOT API key (if required)
2. Set the API key as an environment variable:

```
export ERCOT_API_KEY=your_api_key_here
```

or create a `.env` file with the following content:

```
ERCOT_API_KEY=your_api_key_here
```

### Example Usage

For a complete demonstration of the gridstatus integration, refer to the notebook at `notebooks/gridstatus_integration_demo.ipynb`.

## Extending the Model

To add new feature sets:

1. Create a new data class in `src/data/` similar to the existing ones
2. Define a schema for the new data type in `src/utils/schemas.py`
3. Modify the preprocessing functions in `src/utils/preprocessing.py` to incorporate the new features
4. Update the hybrid model to use these features during training and prediction

## Type Checking

This project supports static type checking with mypy. To check types, run:

```bash
mypy --ignore-missing-imports src
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 