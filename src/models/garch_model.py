import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import ARCHModelResult
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from typing import Tuple, Dict, List, Optional, Union, Any, Literal
from pandera.typing import DataFrame

from src.utils.schemas import PriceDataSchema, WeatherDataSchema, VarianceForecastSchema, PriceForecastSchema


class GARCHModel:
    """
    GARCH model for forecasting the variance of energy prices.
    Can be extended to GARCH-X by including exogenous variables.
    """
    
    def __init__(
        self, 
        p: int = 1, 
        q: int = 1, 
        mean: Literal['Zero', 'Constant', 'AR', 'ARX'] = 'Zero', 
        vol: Literal['GARCH', 'EGARCH', 'FIGARCH'] = 'GARCH', 
        dist: Literal['normal', 'studentst', 'skewstudent'] = 'normal'
    ) -> None:
        """
        Initialize the GARCH model.
        
        Parameters:
        -----------
        p : int, default=1
            Order of the symmetric innovation
        q : int, default=1
            Order of lagged volatility
        mean : Literal['Zero', 'Constant', 'AR', 'ARX'], default='Zero'
            Mean model, options: 'Constant', 'Zero', 'ARX', etc.
        vol : Literal['GARCH', 'EGARCH', 'FIGARCH'], default='GARCH'
            Volatility model, options: 'GARCH', 'EGARCH', 'FIGARCH', etc.
        dist : Literal['normal', 'studentst', 'skewstudent'], default='normal'
            Error distribution, options: 'normal', 'studentst', 'skewstudent', etc.
        """
        self.p = p
        self.q = q
        self.mean = mean
        self.vol = vol
        self.dist = dist
        self.model = None
        self.result: Optional[ARCHModelResult] = None
        self.scaler = StandardScaler()
        self.last_prices: Optional[np.ndarray] = None
        
    def _prepare_data(
        self, 
        price_data: DataFrame[PriceDataSchema], 
        weather_data: Optional[DataFrame[WeatherDataSchema]] = None
    ) -> Tuple[pd.Series, Optional[np.ndarray]]:
        """
        Prepare data for GARCH modeling.
        
        Parameters:
        -----------
        price_data : DataFrame[PriceDataSchema]
            DataFrame containing price data with datetime index
        weather_data : Optional[DataFrame[WeatherDataSchema]], default=None
            DataFrame containing weather data with datetime index, for GARCH-X
            
        Returns:
        --------
        Tuple[pd.Series, Optional[np.ndarray]]
            (returns, X) where returns are the price returns and X are exogenous variables
        """
        # Calculate returns
        self.last_prices = price_data['price'].values
        returns = 100 * price_data['price'].pct_change().dropna()
        
        # Prepare exogenous variables if available (for GARCH-X)
        X = None
        if weather_data is not None and self.mean == 'ARX':
            # Align weather data with returns data
            aligned_weather = weather_data.reindex(returns.index)
            
            # Fill missing values
            aligned_weather.fillna(method='ffill', inplace=True)
            aligned_weather.fillna(method='bfill', inplace=True)
            
            # Scale weather data
            self.scaler.fit(aligned_weather.values)
            weather_scaled = self.scaler.transform(aligned_weather.values)
            
            # Create exogenous variable matrix
            X = weather_scaled
        
        return returns, X
    
    def fit(
        self, 
        price_data: DataFrame[PriceDataSchema], 
        weather_data: Optional[DataFrame[WeatherDataSchema]] = None, 
        update: bool = True
    ) -> 'GARCHModel':
        """
        Fit the GARCH model to the data.
        
        Parameters:
        -----------
        price_data : DataFrame[PriceDataSchema]
            DataFrame containing price data with datetime index
        weather_data : Optional[DataFrame[WeatherDataSchema]], default=None
            DataFrame containing weather data with datetime index, for GARCH-X
        update : bool, default=True
            Whether to update an existing model or fit a new one
            
        Returns:
        --------
        GARCHModel
            Fitted model instance
        """
        # Prepare the data
        returns, X = self._prepare_data(price_data, weather_data)
        
        # Create and fit the model
        if not update or self.model is None:
            self.model = arch_model(
                returns,
                x=X,
                p=self.p,
                q=self.q,
                mean=self.mean,
                vol=self.vol,
                dist=self.dist
            )
        
        self.result = self.model.fit(disp='off')
        
        return self
    
    def predict(
        self, 
        forecast_horizon: int = 24, 
        weather_data: Optional[DataFrame[WeatherDataSchema]] = None
    ) -> DataFrame[VarianceForecastSchema]:
        """
        Generate variance forecasts.
        
        Parameters:
        -----------
        forecast_horizon : int, default=24
            Number of steps to forecast ahead
        weather_data : Optional[DataFrame[WeatherDataSchema]], default=None
            DataFrame containing future weather data for GARCH-X
            
        Returns:
        --------
        DataFrame[VarianceForecastSchema]
            DataFrame with datetime index and forecasted variance/volatility
        """
        if self.result is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Prepare exogenous variables for forecasting
        X_pred = None
        if weather_data is not None and self.mean == 'ARX':
            # Scale the weather data for prediction
            X_pred = self.scaler.transform(weather_data.values)
        
        # Generate forecasts
        forecast = self.result.forecast(
            horizon=forecast_horizon,
            reindex=False,
            x=X_pred
        )
        
        # Extract variance forecasts
        variance = forecast.variance.iloc[-1].values
        
        # Create forecast index
        last_date = self.model.index[-1]
        forecast_index = pd.date_range(
            start=last_date + pd.Timedelta(hours=1),
            periods=forecast_horizon,
            freq='H'
        )
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame(
            variance,
            index=forecast_index,
            columns=['variance_forecast']
        )
        
        # Add standard deviation forecast (volatility)
        forecast_df['volatility_forecast'] = np.sqrt(forecast_df['variance_forecast'])
        
        # Validate against schema
        return VarianceForecastSchema.validate(forecast_df)
    
    def calculate_confidence_intervals(
        self, 
        mean_forecast: pd.DataFrame, 
        variance_forecast: DataFrame[VarianceForecastSchema], 
        confidence_level: float = 0.95
    ) -> DataFrame[PriceForecastSchema]:
        """
        Calculate confidence intervals for the price forecasts based on GARCH volatility.
        
        Parameters:
        -----------
        mean_forecast : pd.DataFrame
            DataFrame with forecasted mean prices
        variance_forecast : DataFrame[VarianceForecastSchema]
            DataFrame with forecasted variance
        confidence_level : float, default=0.95
            Confidence level for the intervals
            
        Returns:
        --------
        DataFrame[PriceForecastSchema]
            DataFrame with mean forecast and lower/upper confidence bounds
        """
        # Ensure dataframes are aligned
        common_index = mean_forecast.index.intersection(variance_forecast.index)
        mean_values = mean_forecast.loc[common_index, 'price_forecast'].values
        volatility_values = variance_forecast.loc[common_index, 'volatility_forecast'].values
        
        # Calculate the z-score for the given confidence level
        z_score = abs(sm.stats.norm.ppf((1 - confidence_level) / 2))
        
        # Calculate confidence intervals
        lower_bound = mean_values - z_score * volatility_values
        upper_bound = mean_values + z_score * volatility_values
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'price_forecast': mean_values,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }, index=common_index)
        
        # Validate against schema
        return PriceForecastSchema.validate(result_df) 