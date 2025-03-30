"""
ERCOT Price Forecasting Package

This package implements a hybrid neural network + GARCH model for forecasting
ERCOT electricity prices, with comprehensive type checking and data validation.

The key components include:
- Data loading and processing modules for price and weather data
- Neural network model for mean price forecasting
- GARCH model for volatility/variance forecasting
- Hybrid model combining neural network and GARCH approaches
- Visualization utilities for model outputs
- Integration with gridstatus library for real ERCOT data
"""

__version__ = '0.2.0'

# Main module initialization 