"""
LSTM Model for Stock Price Prediction
Deep learning approach using recurrent neural networks
"""

import logging
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .base import BaseModel
from config import LSTM_SEQUENCE_LENGTH, LSTM_EPOCHS, LSTM_BATCH_SIZE, LSTM_UNITS

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    HAS_TF = True
except ImportError:
    HAS_TF = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMModel(BaseModel):
    """
    LSTM neural network for stock price prediction.
    Uses sequence of historical prices to predict future values.
    """
    
    def __init__(
        self,
        sequence_length: int = LSTM_SEQUENCE_LENGTH,
        epochs: int = LSTM_EPOCHS,
        batch_size: int = LSTM_BATCH_SIZE,
        units: int = LSTM_UNITS
    ):
        """
        Initialize LSTM model.
        
        Args:
            sequence_length: Number of time steps to look back
            epochs: Training epochs
            batch_size: Batch size for training
            units: Number of LSTM units
        """
        super().__init__(name="LSTM")
        
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.units = units
        
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.last_sequence = None
        self.last_date = None
    
    def _create_sequences(
        self,
        data: np.ndarray
    ) -> tuple:
        """
        Create sequences for LSTM training.
        
        Args:
            data: Scaled price data
            
        Returns:
            Tuple of (X, y) sequences
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i, 0])
            y.append(data[i, 0])
        
        return np.array(X), np.array(y)
    
    def _build_model(self, input_shape: tuple):
        """
        Build LSTM architecture.
        
        Args:
            input_shape: Shape of input sequences
        """
        if not HAS_TF:
            raise ImportError("TensorFlow not installed. Run: pip install tensorflow")
            
        self.model = Sequential([
            LSTM(units=self.units, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=self.units, return_sequences=True),
            Dropout(0.2),
            LSTM(units=self.units, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error'
        )
    
    def fit(self, df: pd.DataFrame, target_col: str = 'close') -> 'LSTMModel':
        """
        Train LSTM model on historical data.
        
        Args:
            df: DataFrame with price data
            target_col: Column to predict
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training LSTM model on {len(df)} data points")
        
        # Suppress TensorFlow warnings
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Prepare data
        prices = df[target_col].values.reshape(-1, 1)
        
        # Scale data
        scaled_data = self.scaler.fit_transform(prices)
        
        # Create sequences
        X, y = self._create_sequences(scaled_data)
        
        if len(X) == 0:
            raise ValueError(f"Not enough data for sequence length {self.sequence_length}")
        
        # Reshape for LSTM [samples, time steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
        
        # Build model
        self._build_model(input_shape=(X.shape[1], 1))
        
        # Train model
        self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=0
        )
        
        # Store last sequence for prediction
        self.last_sequence = scaled_data[-self.sequence_length:]
        self.last_date = df['date'].iloc[-1]
        
        self.is_fitted = True
        self.train_dates = (df['date'].min(), df['date'].max())
        
        logger.info("LSTM model training complete")
        
        return self
    
    def predict(
        self,
        periods: int = 30,
        return_confidence: bool = True
    ) -> pd.DataFrame:
        """
        Generate future predictions using recursive forecasting.
        
        Args:
            periods: Number of future days to predict
            return_confidence: Include confidence intervals
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        logger.info(f"Generating {periods}-day forecast with LSTM")
        
        predictions = []
        dates = []
        
        # Start with last known sequence
        current_sequence = self.last_sequence.copy()
        
        for i in range(periods):
            # Reshape for prediction
            X_pred = current_sequence.reshape((1, self.sequence_length, 1))
            
            # Predict next value
            next_pred_scaled = self.model.predict(X_pred, verbose=0)[0, 0]
            
            # Inverse transform to get actual price
            next_pred = self.scaler.inverse_transform([[next_pred_scaled]])[0, 0]
            predictions.append(next_pred)
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1, 0] = next_pred_scaled
            
            # Next business day
            next_date = self.last_date + pd.Timedelta(days=i+1)
            while next_date.weekday() >= 5:
                next_date += pd.Timedelta(days=1)
            dates.append(next_date)
        
        predictions = np.array(predictions)
        
        result = pd.DataFrame({
            'date': dates,
            'predicted': predictions
        })
        
        if return_confidence:
            # Estimate uncertainty
            base_uncertainty = np.std(predictions) * 0.15 if len(predictions) > 1 else predictions[0] * 0.03
            uncertainties = [base_uncertainty * np.sqrt(i+1) for i in range(periods)]
            
            result['lower_bound'] = predictions - 1.96 * np.array(uncertainties)
            result['upper_bound'] = predictions + 1.96 * np.array(uncertainties)
        
        return result.reset_index(drop=True)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'model_type': 'LSTM',
            'sequence_length': self.sequence_length,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'units': self.units
        }
