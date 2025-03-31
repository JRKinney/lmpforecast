# ERCOT Price Forecasting

A hybrid model for forecasting electricity prices in the ERCOT market, combining neural networks and GARCH volatility modeling.

## Project Overview

This project implements a hybrid forecasting model for ERCOT electricity prices, combining:

1. Neural networks to capture complex patterns in historical price and weather data
2. GARCH models to forecast volatility and provide uncertainty estimates

The model is designed to generate forecasts with confidence intervals, making it useful for risk management and trading decisions in the electricity market.

## Features

- Data loading and preprocessing for ERCOT price data and weather data
- Feature engineering optimized for time series forecasting
- Hybrid model combining neural networks and GARCH
- Visualization tools for exploring data and model results
- Evaluation metrics for forecast accuracy and volatility modeling
- Integration with real data from the gridstatus API

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ercot_price_forecasting.git
cd ercot_price_forecasting
```

2. Create and activate a virtual environment:
```bash
# Using venv
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e ".[dev,notebook]"
```

## Development Setup

### Code Quality Tools

This project uses pre-commit hooks to maintain code quality. To set up:

1. Install pre-commit:
```bash
pip install pre-commit
```

2. Install the git hooks:
```bash
pre-commit install
```

3. (Optional) Run against all files:
```bash
pre-commit run --all-files
```

### Code Quality Checks

The following tools are configured for this project:

- **Black**: Code formatter
- **isort**: Import sorter
- **Ruff**: Fast Python linter
- **mypy**: Static type checker
- **nbQA**: Apply quality tools to Jupyter notebooks
- **nbstripout**: Clean notebook output before commits

## Usage

### Data Exploration

Use the data exploration notebook to analyze ERCOT price and weather data:

```bash
jupyter lab notebooks/data_exploration_visualization.py
```

### Model Training and Evaluation

Train and evaluate the hybrid model:

```bash
python notebooks/model_building_visualization_demo.py
```

### Feature Analysis and Model Comparison

Compare different model configurations and analyze feature importance:

```bash
python notebooks/feature_analysis_model_comparison.py
```

### Gridstatus Integration

Use real data from the gridstatus API:

```bash
python notebooks/gridstatus_integration_demo.py
```

## License

MIT License

## Acknowledgements

- ERCOT for providing electricity market data
- The gridstatus project for API access to real-time grid data
