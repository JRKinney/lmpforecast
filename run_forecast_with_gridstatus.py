#!/usr/bin/env python
"""
ERCOT Price Forecasting with Real Data from Gridstatus

This script uses the gridstatus library integration to fetch real ERCOT data
and run the hybrid neural network + GARCH model for price forecasting.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from dotenv import load_dotenv

# Import our modules
from src.data.ercot_gridstatus_integration import ErcotGridstatusIntegration
from src.data.ercot_price_data import ErcotPriceData
from src.data.ercot_weather_data import ErcotWeatherData
from src.models.hybrid_model import HybridModel
from src.visualization.plotting import plot_price_forecast

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if present
load_dotenv()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run ERCOT price forecasting with real data from gridstatus')
    
    # Data source arguments
    parser.add_argument('--use-synthetic', action='store_true',
                      help='Use synthetic data instead of real data from gridstatus')
    
    # Data parameters
    parser.add_argument('--price-node', type=str, default='HB_HOUSTON',
                      help='ERCOT price node to forecast (default: HB_HOUSTON)')
    parser.add_argument('--location', type=str, default='Houston',
                      help='Location for weather data (default: Houston)')
    parser.add_argument('--days-history', type=int, default=30,
                      help='Number of days of historical data to use (default: 30)')
    
    # Model parameters
    parser.add_argument('--seq-length', type=int, default=24,
                      help='Sequence length for the neural network (default: 24)')
    parser.add_argument('--forecast-horizon', type=int, default=24,
                      help='Forecast horizon in hours (default: 24)')
    parser.add_argument('--confidence-level', type=float, default=0.95,
                      help='Confidence level for prediction intervals (default: 0.95)')
    
    # Training parameters
    parser.add_argument('--nn-epochs', type=int, default=50,
                      help='Number of epochs for neural network training (default: 50)')
    parser.add_argument('--nn-batch-size', type=int, default=32,
                      help='Batch size for neural network training (default: 32)')
    
    # Output options
    parser.add_argument('--save-plot', action='store_true',
                      help='Save forecast plot to file')
    parser.add_argument('--save-model', action='store_true',
                      help='Save trained model to file')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                      help='Directory to save outputs (default: ./outputs)')
    
    return parser.parse_args()


def main():
    """Main function to run the forecast with real or synthetic data."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    if args.save_plot or args.save_model:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up date ranges
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days_history)
    
    logger.info(f"Setting up forecast for {args.price_node} with data from {start_date} to {end_date}")
    logger.info(f"Forecast horizon: {args.forecast_horizon} hours")
    
    try:
        # Initialize data sources
        if args.use_synthetic:
            # Use synthetic data
            logger.info("Using synthetic data")
            price_loader = ErcotPriceData()
            weather_loader = ErcotWeatherData()
            
            # Load synthetic data
            price_data = price_loader.load_data(
                start_date=start_date,
                end_date=end_date,
                price_node=args.price_node,
                resample_freq='H'
            )
            
            weather_data = weather_loader.load_data(
                start_date=start_date,
                end_date=end_date,
                location=args.location,
                resample_freq='H'
            )
        else:
            # Use real data from gridstatus
            logger.info("Using real data from gridstatus")
            gridstatus_loader = ErcotGridstatusIntegration()
            
            try:
                # Try to fetch real price data
                price_data = gridstatus_loader.fetch_price_data(
                    start_date=start_date,
                    end_date=end_date,
                    price_node=args.price_node,
                    market='real_time',
                    resample_freq='H'
                )
                logger.info(f"Successfully fetched real price data: {len(price_data)} hours")
                
                # Try to fetch real weather data
                weather_data = gridstatus_loader.fetch_weather_data(
                    start_date=start_date,
                    end_date=end_date,
                    location=args.location,
                    resample_freq='H'
                )
                logger.info(f"Successfully fetched real weather data: {len(weather_data)} hours")
                
            except Exception as e:
                logger.error(f"Failed to fetch real data: {e}")
                logger.info("Falling back to synthetic data")
                
                # Fall back to synthetic data
                price_loader = ErcotPriceData()
                weather_loader = ErcotWeatherData()
                
                price_data = price_loader.load_data(
                    start_date=start_date,
                    end_date=end_date,
                    price_node=args.price_node,
                    resample_freq='H'
                )
                
                weather_data = weather_loader.load_data(
                    start_date=start_date,
                    end_date=end_date,
                    location=args.location,
                    resample_freq='H'
                )
        
        # Split data into training and recent periods
        train_end = end_date - timedelta(hours=args.forecast_horizon)
        
        train_price = price_data[price_data.index <= train_end]
        train_weather = weather_data[weather_data.index <= train_end]
        
        recent_price = price_data[price_data.index > (train_end - timedelta(hours=args.seq_length))]
        recent_weather = weather_data[weather_data.index > (train_end - timedelta(hours=args.seq_length))]
        
        logger.info(f"Training data: {len(train_price)} hours")
        logger.info(f"Recent data for prediction: {len(recent_price)} hours")
        
        # Initialize and train the model
        logger.info("Initializing hybrid model")
        model = HybridModel(
            seq_length=args.seq_length,
            forecast_horizon=args.forecast_horizon,
            p=1, q=1,  # GARCH(1,1) model
            mean='Zero',  # Zero mean for GARCH
            vol='GARCH',  # Standard GARCH volatility model
            dist='normal'  # Normal distribution for errors
        )
        
        logger.info("Training model...")
        model.fit(
            price_data=train_price,
            weather_data=train_weather,
            nn_epochs=args.nn_epochs,
            nn_batch_size=args.nn_batch_size,
            nn_validation_split=0.2,
            verbose=1
        )
        
        # Generate forecast
        logger.info("Generating forecast...")
        forecast = model.predict(
            price_data=recent_price,
            weather_data=recent_weather,
            confidence_level=args.confidence_level
        )
        
        # Display forecast summary
        last_historical = recent_price.index.max()
        forecast_start = last_historical + timedelta(hours=1)
        forecast_end = forecast_start + timedelta(hours=args.forecast_horizon - 1)
        
        logger.info(f"Forecast period: {forecast_start} to {forecast_end}")
        logger.info(f"Average forecasted price: ${forecast['price_forecast'].mean():.2f} per MWh")
        logger.info(f"Price range: ${forecast['price_forecast'].min():.2f} to ${forecast['price_forecast'].max():.2f} per MWh")
        
        # Create and display visualization
        fig = plot_price_forecast(
            forecasts=forecast,
            historical_data=price_data,
            title=f"ERCOT {args.price_node} Price Forecast"
        )
        
        # Save plot if requested
        if args.save_plot:
            plot_file = os.path.join(
                args.output_dir, 
                f"forecast_{args.price_node}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            )
            fig.write_html(plot_file)
            logger.info(f"Forecast plot saved to {plot_file}")
        
        # Save model if requested
        if args.save_model:
            model_dir = os.path.join(args.output_dir, 'models')
            os.makedirs(model_dir, exist_ok=True)
            model_file = os.path.join(
                model_dir,
                f"model_{args.price_node}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            model.save_models(model_file)
            logger.info(f"Model saved to {model_file}")
            
        logger.info("Forecast completed successfully")
        
        # Show the interactive plot (blocks execution)
        fig.show()
        
    except Exception as e:
        logger.error(f"Error in forecasting process: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main()) 