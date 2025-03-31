"""Visualization functions for the ERCOT price forecasting project.

This module contains functions for creating interactive visualizations of
price forecasts, volatility forecasts, and model performance evaluations.
"""

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandera.typing import DataFrame
from plotly.subplots import make_subplots

from src.utils.schemas import ForecastSchema, PriceDataSchema, WeatherDataSchema


def plot_price_forecast(
    forecasts: DataFrame[ForecastSchema],
    historical_data: DataFrame[PriceDataSchema],
    title: Optional[str] = "ERCOT Price Forecast",
) -> go.Figure:
    """Plot price forecasts with confidence intervals.

    Args:
        forecasts: DataFrame containing forecasted prices and confidence bounds
        historical_data: DataFrame containing historical price data
        title: Plot title

    Returns:
        Plotly figure object
    """
    # Create figure
    fig = go.Figure()

    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=historical_data.index,
            y=historical_data["price"],
            mode="lines",
            name="Historical",
            line={"color": "royalblue", "width": 2},
        )
    )

    # Add forecasted price
    fig.add_trace(
        go.Scatter(
            x=forecasts.index,
            y=forecasts["price_forecast"],
            mode="lines",
            name="Forecast",
            line={"color": "firebrick", "width": 2},
        )
    )

    # Add confidence interval as a filled area
    fig.add_trace(
        go.Scatter(
            x=forecasts.index,
            y=forecasts["upper_bound"],
            mode="lines",
            name="Upper Bound",
            line={"width": 0},
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecasts.index,
            y=forecasts["lower_bound"],
            mode="lines",
            name="Lower Bound",
            fill="tonexty",
            fillcolor="rgba(231, 76, 60, 0.2)",
            line={"width": 0},
            showlegend=False,
        )
    )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price ($/MWh)",
        legend={"x": 0.01, "y": 0.99},
        hovermode="x unified",
        template="plotly_white",
    )

    return fig


def plot_volatility_forecast(
    variance_forecast: DataFrame,
    historical_volatility: DataFrame,
    title: Optional[str] = "ERCOT Price Volatility Forecast",
) -> go.Figure:
    """Plot volatility forecasts with historical volatility.

    Args:
        variance_forecast: DataFrame containing forecasted variance
        historical_volatility: DataFrame containing historical volatility
        title: Plot title

    Returns:
        Plotly figure object
    """
    # Create figure
    fig = go.Figure()

    # If variance is provided, convert to volatility (standard deviation)
    if "variance" in variance_forecast.columns:
        volatility_forecast = np.sqrt(variance_forecast["variance"])
    else:
        volatility_forecast = variance_forecast["volatility"]

    # Add historical volatility
    fig.add_trace(
        go.Scatter(
            x=historical_volatility.index,
            y=historical_volatility["volatility"],
            mode="lines",
            name="Historical Volatility",
            line={"color": "royalblue", "width": 2},
        )
    )

    # Add forecasted volatility
    fig.add_trace(
        go.Scatter(
            x=variance_forecast.index,
            y=volatility_forecast,
            mode="lines",
            name="Forecasted Volatility",
            line={"color": "firebrick", "width": 2},
        )
    )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Volatility ($/MWh)",
        legend={"x": 0.01, "y": 0.99},
        hovermode="x unified",
        template="plotly_white",
    )

    return fig


def plot_price_components(
    price_data: DataFrame[PriceDataSchema],
    weather_data: DataFrame[WeatherDataSchema],
    lookback_window: int = 168,
    forecast_window: int = 24,
    title: Optional[str] = "ERCOT Price Components",
) -> go.Figure:
    """Plot price components including price vs. temperature, hourly patterns, etc.

    Args:
        price_data: DataFrame containing price data
        weather_data: DataFrame containing weather data
        lookback_window: Number of hours to look back for patterns
        forecast_window: Number of hours to forecast
        title: Plot title

    Returns:
        Plotly figure object
    """
    # Create subplot figure
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Price vs. Time",
            "Price vs. Temperature",
            "Hourly Price Pattern",
            "Daily Price Pattern",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # Filter to the lookback window
    end_date = price_data.index.max()
    start_date = end_date - pd.Timedelta(hours=lookback_window)

    price_subset = price_data[
        (price_data.index >= start_date) & (price_data.index <= end_date)
    ]
    weather_subset = weather_data[
        (weather_data.index >= start_date) & (weather_data.index <= end_date)
    ]

    # Merge data
    merged_data = pd.merge(
        price_subset, weather_subset, left_index=True, right_index=True, how="inner"
    )
    # For linting
    merged_data_index = pd.DatetimeIndex(merged_data.index)

    # 1. Price vs. Time
    fig.add_trace(
        go.Scatter(
            x=merged_data_index,
            y=merged_data["price"],
            mode="lines",
            name="Price",
            line={"color": "royalblue"},
        ),
        row=1,
        col=1,
    )

    # 2. Price vs. Temperature
    fig.add_trace(
        go.Scatter(
            x=merged_data["temperature"],
            y=merged_data["price"],
            mode="markers",
            name="Price vs. Temp",
            marker={"color": "firebrick", "size": 6, "opacity": 0.6},
        ),
        row=1,
        col=2,
    )

    # 3. Hourly Price Pattern
    hourly_price = merged_data.groupby(merged_data_index.hour)["price"].mean()
    fig.add_trace(
        go.Bar(
            x=hourly_price.index,
            y=hourly_price.values,
            name="Hourly Pattern",
            marker_color="royalblue",
        ),
        row=2,
        col=1,
    )

    # 4. Daily Price Pattern
    daily_price = merged_data.groupby(merged_data_index.dayofweek)["price"].mean()
    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    fig.add_trace(
        go.Bar(
            x=[days[i] for i in daily_price.index],
            y=daily_price.values,
            name="Daily Pattern",
            marker_color="firebrick",
        ),
        row=2,
        col=2,
    )

    # Update layout
    fig.update_layout(
        title=title, height=800, showlegend=False, template="plotly_white"
    )

    # Update axes
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Price ($/MWh)", row=1, col=1)

    fig.update_xaxes(title_text="Temperature (°C)", row=1, col=2)
    fig.update_yaxes(title_text="Price ($/MWh)", row=1, col=2)

    fig.update_xaxes(title_text="Hour of Day", row=2, col=1)
    fig.update_yaxes(title_text="Avg Price ($/MWh)", row=2, col=1)

    fig.update_xaxes(title_text="Day of Week", row=2, col=2)
    fig.update_yaxes(title_text="Avg Price ($/MWh)", row=2, col=2)

    return fig


def plot_model_performance(
    actual_prices: pd.Series,
    forecasted_prices: pd.Series,
    train_test_split_date: Optional[str] = None,
    title: Optional[str] = "Model Performance Evaluation",
) -> go.Figure:
    """Evaluate and visualize model performance.

    Args:
        actual_prices: Series of actual price values
        forecasted_prices: Series of forecasted price values with datetime index
        train_test_split_date: Date string marking train/test split point
        title: Plot title

    Returns:
        Plotly figure object with performance metrics and visualizations
    """
    # Create subplot figure
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Actual vs. Predicted",
            "Residuals",
            "Error Distribution",
            "Metrics",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "histogram"}, {"type": "table"}],
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # Convert to Series if lists
    dates = forecasted_prices.index
    actual_prices = pd.Series(data=actual_prices, index=dates)
    forecasted_prices = pd.Series(data=forecasted_prices, index=dates)

    # Calculate errors and metrics
    errors = actual_prices - forecasted_prices
    mae = np.mean(np.abs(errors))
    mape = np.mean(np.abs(errors / actual_prices)) * 100
    rmse = np.sqrt(np.mean(errors**2))
    r2 = 1 - (np.sum(errors**2) / np.sum((actual_prices - actual_prices.mean()) ** 2))

    # 1. Actual vs. Predicted
    fig.add_trace(
        go.Scatter(
            x=actual_prices.index,
            y=actual_prices,
            mode="lines",
            name="Actual",
            line={"color": "royalblue"},
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=forecasted_prices.index,
            y=forecasted_prices,
            mode="lines",
            name="Predicted",
            line={"color": "firebrick"},
        ),
        row=1,
        col=1,
    )

    # Add vertical line at train/test split if provided
    if train_test_split_date:
        split_date = pd.to_datetime(train_test_split_date)
        fig.add_vline(x=split_date, line_dash="dash", line_color="green", row=1, col=1)

        # Add annotation
        fig.add_annotation(
            x=split_date,
            y=max(actual_prices.max(), forecasted_prices.max()),
            text="Train/Test Split",
            showarrow=True,
            arrowhead=1,
            row=1,
            col=1,
        )

    # 2. Residuals
    fig.add_trace(
        go.Scatter(
            x=errors.index,
            y=errors,
            mode="lines",
            name="Residuals",
            line={"color": "forestgreen"},
        ),
        row=1,
        col=2,
    )

    # Add horizontal line at zero
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)

    # 3. Error Distribution
    fig.add_trace(
        go.Histogram(
            x=errors,
            nbinsx=30,
            name="Error Distribution",
            marker_color="forestgreen",
            opacity=0.7,
        ),
        row=2,
        col=1,
    )

    # 4. Metrics Table
    fig.add_trace(
        go.Table(
            header={
                "values": ["Metric", "Value"],
                "fill_color": "royalblue",
                "align": "center",
                "font": {"color": "white", "size": 12},
            },
            cells={
                "values": [
                    ["MAE", "MAPE", "RMSE", "R²"],
                    [
                        f"${mae:.2f}/MWh",
                        f"{mape:.2f}%",
                        f"${rmse:.2f}/MWh",
                        f"{r2:.4f}",
                    ],
                ],
                "fill_color": "lavender",
                "align": "center",
            },
        ),
        row=2,
        col=2,
    )

    # Update layout
    fig.update_layout(
        title=title,
        height=800,
        showlegend=True,
        legend={"x": 0.01, "y": 0.99},
        template="plotly_white",
    )

    # Update axes
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Price ($/MWh)", row=1, col=1)

    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_yaxes(title_text="Error ($/MWh)", row=1, col=2)

    fig.update_xaxes(title_text="Error ($/MWh)", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)

    return fig
