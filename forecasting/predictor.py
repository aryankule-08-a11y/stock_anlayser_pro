"""
Forecast Engine Module
Unified interface for generating future predictions
"""

import logging
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models.prophet_model import ProphetModel
from models.xgboost_model import XGBoostModel
try:
    from models.lstm_model import LSTMModel
except ImportError:
    LSTMModel = None
from models.base import BaseModel
from config import DEFAULT_PREDICTION_DAYS, CHART_HEIGHT, CHART_TEMPLATE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForecastEngine:
    """
    Unified forecasting engine that works with any model.
    Generates predictions with confidence intervals and visualization.
    """
    
    def __init__(self, model: BaseModel):
        """
        Initialize forecast engine with a trained model.
        
        Args:
            model: Trained model instance
        """
        self.model = model
        self.forecast = None
        self.historical_df = None
    
    def generate_forecast(
        self,
        df: pd.DataFrame,
        periods: int = DEFAULT_PREDICTION_DAYS,
        retrain: bool = True
    ) -> pd.DataFrame:
        """
        Generate future price forecast.
        
        Args:
            df: Historical data DataFrame
            periods: Number of days to forecast
            retrain: Whether to retrain model on full data
            
        Returns:
            DataFrame with forecast
        """
        logger.info(f"Generating {periods}-day forecast using {self.model.name}")
        
        self.historical_df = df.copy()
        
        # Retrain on full data if requested
        if retrain or not self.model.is_fitted:
            self.model.fit(df)
        
        # Generate predictions
        self.forecast = self.model.predict(periods=periods, return_confidence=True)
        
        logger.info(f"Forecast generated: {len(self.forecast)} days")
        
        return self.forecast
    
    def create_forecast_chart(
        self,
        symbol: str,
        show_historical: int = 90
    ) -> go.Figure:
        """
        Create interactive forecast visualization.
        
        Args:
            symbol: Stock symbol for title
            show_historical: Number of historical days to show
            
        Returns:
            Plotly figure
        """
        if self.forecast is None:
            raise ValueError("Generate forecast first using generate_forecast()")
        
        # Prepare historical data
        hist_df = self.historical_df.tail(show_historical).copy()
        
        fig = go.Figure()
        
        # Historical prices
        fig.add_trace(go.Scatter(
            x=hist_df['date'],
            y=hist_df['close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='#2196f3', width=2)
        ))
        
        # Forecast line
        fig.add_trace(go.Scatter(
            x=self.forecast['date'],
            y=self.forecast['predicted'],
            mode='lines',
            name='Forecast',
            line=dict(color='#ff9800', width=2, dash='dash')
        ))
        
        # Confidence interval
        if 'upper_bound' in self.forecast.columns:
            fig.add_trace(go.Scatter(
                x=self.forecast['date'],
                y=self.forecast['upper_bound'],
                mode='lines',
                name='Upper Bound (95% CI)',
                line=dict(color='rgba(255, 152, 0, 0.3)', width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=self.forecast['date'],
                y=self.forecast['lower_bound'],
                mode='lines',
                name='Confidence Interval',
                line=dict(color='rgba(255, 152, 0, 0.3)', width=0),
                fill='tonexty',
                fillcolor='rgba(255, 152, 0, 0.2)'
            ))
        
        # Add vertical line at forecast start
        last_hist_date = hist_df['date'].iloc[-1]
        fig.add_vline(
            x=last_hist_date,
            line_dash="dot",
            line_color="gray",
            annotation_text="Forecast Start"
        )
        
        fig.update_layout(
            title=f'{symbol} Price Forecast ({self.model.name})',
            yaxis_title='Price',
            template=CHART_TEMPLATE,
            height=CHART_HEIGHT,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def get_forecast_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the forecast.
        
        Returns:
            Dictionary with forecast summary
        """
        if self.forecast is None:
            raise ValueError("Generate forecast first")
        
        last_price = self.historical_df['close'].iloc[-1]
        final_pred = self.forecast['predicted'].iloc[-1]
        
        summary = {
            'model': self.model.name,
            'forecast_days': len(self.forecast),
            'start_date': self.forecast['date'].iloc[0].strftime('%Y-%m-%d'),
            'end_date': self.forecast['date'].iloc[-1].strftime('%Y-%m-%d'),
            'current_price': last_price,
            'final_prediction': final_pred,
            'predicted_change_pct': ((final_pred - last_price) / last_price) * 100,
            'predicted_direction': 'Bullish' if final_pred > last_price else 'Bearish',
            'min_prediction': self.forecast['predicted'].min(),
            'max_prediction': self.forecast['predicted'].max()
        }
        
        if 'upper_bound' in self.forecast.columns:
            summary['confidence_range'] = {
                'lower': self.forecast['lower_bound'].iloc[-1],
                'upper': self.forecast['upper_bound'].iloc[-1]
            }
        
        return summary
    
    def get_prediction_table(self) -> pd.DataFrame:
        """
        Get formatted prediction table.
        
        Returns:
            DataFrame with formatted predictions
        """
        if self.forecast is None:
            raise ValueError("Generate forecast first")
        
        table = self.forecast.copy()
        table['date'] = table['date'].dt.strftime('%Y-%m-%d')
        table['predicted'] = table['predicted'].round(2)
        
        if 'upper_bound' in table.columns:
            table['lower_bound'] = table['lower_bound'].round(2)
            table['upper_bound'] = table['upper_bound'].round(2)
        
        return table


def create_combined_forecast_chart(
    df: pd.DataFrame,
    forecasts: Dict[str, pd.DataFrame],
    symbol: str,
    show_historical: int = 60
) -> go.Figure:
    """
    Create chart comparing multiple model forecasts.
    
    Args:
        df: Historical data
        forecasts: Dictionary of model_name -> forecast DataFrame
        symbol: Stock symbol
        show_historical: Days of historical data to show
        
    Returns:
        Plotly figure
    """
    hist_df = df.tail(show_historical)
    
    fig = go.Figure()
    
    # Historical prices
    fig.add_trace(go.Scatter(
        x=hist_df['date'],
        y=hist_df['close'],
        mode='lines',
        name='Historical',
        line=dict(color='#2196f3', width=2)
    ))
    
    # Forecast from each model
    colors = ['#ff9800', '#e91e63', '#4caf50', '#9c27b0']
    
    for i, (model_name, forecast) in enumerate(forecasts.items()):
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=forecast['date'],
            y=forecast['predicted'],
            mode='lines',
            name=f'{model_name} Forecast',
            line=dict(color=color, width=2, dash='dash')
        ))
    
    # Vertical line at forecast start
    fig.add_vline(
        x=hist_df['date'].iloc[-1],
        line_dash="dot",
        line_color="gray"
    )
    
    fig.update_layout(
        title=f'{symbol} - Model Comparison',
        yaxis_title='Price',
        template=CHART_TEMPLATE,
        height=CHART_HEIGHT,
        hovermode='x unified'
    )
    
    return fig
