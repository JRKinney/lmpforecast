#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for running ERCOT price forecasts using the hybrid GARCH-NN model.
Provides a command-line interface for generating and visualizing forecasts.
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Literal

import pandas as pd
import numpy as np
from pandera.typing import DataFrame

from src.data.ercot_price_data import ErcotPriceData
from src.data.ercot_weather_data import ErcotWeatherData
from src.models.hybrid_model import HybridModel
from src.visualization.plotting import plot_price_forecast
from src.utils.schemas import PriceDataSchema, WeatherDataSchema

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the forecast runner.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Run ERCOT price forecasts')
    
    parser.add_argument('--price-node', type=str, default='HB_HOUSTON',
                        help='ERCOT price node to forecast (default: HB_HOUSTON)')
    
    parser.add_argument('--location', type=str, default='Houston',
                        help='Weather location to use (default: Houston)')
    
    parser.add_argument('--forecast-horizon', type=int, default=24,
                        help='Forecast horizon in hours (default: 24)')
    
    parser.add_argument('--seq-length', type=int, default=24,
                        help='Sequence length for the NN model in hours (default: 24)')
    
    parser.add_argument('--start-date', type=str, 
                        default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                        help='Start date for historical data (default: 1 year ago)')
    
    parser.add_argument('--end-date', type=str, 
                        default=datetime.now().strftime('%Y-%m-%d'),
                        help='End date for historical data (default: today)')
    
    parser.add_argument('--confidence-level', type=float, default=0.95,
                        help='Confidence level for prediction intervals (default: 0.95)')
    
    parser.add_argument('--save-plot', action='store_true',
                        help='Save the forecast plot as HTML file')
    
    parser.add_argument('--save-forecast', action='store_true',
                        help='Save the forecast data as CSV file')
    
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Directory to save outputs (default: ./output)')
    
    return parser.parse_args()

def main() -> None:
    """
    Main function to run the forecasting process.
    
    Loads data, trains the model, generates forecasts, and outputs results.
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    if args.save_plot or args.save_forecast:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize data loaders
    price_loader = ErcotPriceData()
    weather_loader = ErcotWeatherData()
    
    # Load historical data
    print(f"Loading price data for node {args.price_node}...")
    price_data: DataFrame[PriceDataSchema] = price_loader.load_data(
        start_date=args.start_date,
        end_date=args.end_date,
        price_node=args.price_node
    )
    
    print(f"Loading weather data for location {args.location}...")
    weather_data: DataFrame[WeatherDataSchema] = weather_loader.load_data(
        start_date=args.start_date,
        end_date=args.end_date,
        location=args.location
    )
    
    # Initialize the hybrid model
    print("Initializing hybrid model...")
    model = HybridModel(
        seq_length=args.seq_length,
        forecast_horizon=args.forecast_horizon,
        p=1, q=1,
        mean='Zero',
        vol='GARCH',
        dist='normal'
    )
    
    # Train/test split (use last forecast_horizon hours for testing)
    split_idx = len(price_data) - args.forecast_horizon
    train_price = price_data.iloc[:split_idx].copy()
    test_price = price_data.iloc[split_idx:].copy()
    train_weather = weather_data.iloc[:split_idx].copy()
    test_weather = weather_data.iloc[split_idx:].copy()
    
    # Train the model
    print("Training model...")
    model.fit(
        price_data=train_price,
        weather_data=train_weather,
        nn_epochs=50,
        nn_batch_size=32,
        nn_validation_split=0.2,
        verbose=1
    )
    
    # Generate forecast
    print("Generating forecast...")
    forecast = model.predict(
        price_data=test_price,
        weather_data=test_weather,
        confidence_level=args.confidence_level
    )
    
    # Display forecast summary
    print("\nForecast Summary:")
    print(f"Mean forecast: ${forecast['price_forecast'].mean():.2f}/MWh")
    print(f"Min forecast: ${forecast['price_forecast'].min():.2f}/MWh")
    print(f"Max forecast: ${forecast['price_forecast'].max():.2f}/MWh")
    
    # Create and show plot
    fig = plot_price_forecast(
        forecasts=forecast,
        historical_data=price_data.iloc[-args.forecast_horizon*2:],
        title=f"ERCOT {args.price_node} Price Forecast"
    )
    
    # Save outputs if requested
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.save_plot:
        plot_path = os.path.join(args.output_dir, f"forecast_plot_{timestamp}.html")
        fig.write_html(plot_path)
        print(f"Plot saved to {plot_path}")
    
    if args.save_forecast:
        csv_path = os.path.join(args.output_dir, f"forecast_data_{timestamp}.csv")
        forecast.to_csv(csv_path)
        print(f"Forecast data saved to {csv_path}")
    
    print("Forecast complete!")

if __name__ == "__main__":
    main() 