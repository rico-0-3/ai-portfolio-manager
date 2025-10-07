"""
Transformer model with attention mechanism for stock prediction.
State-of-the-art for time series forecasting.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Transformer model will not work.")


class TimeSeriesDataset(Dataset):
    """Dataset for time series prediction."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]


class TransformerPredictor(nn.Module):
    """
    Transformer model for stock price prediction.
    Uses multi-head attention to capture temporal dependencies.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer
        self.fc_out = nn.Linear(d_model, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch, seq_len, features)

        # Project input to d_model dimension
        x = self.input_proj(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # (batch, seq_len, d_model)

        # Apply transformer
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)

        # Use last time step for prediction
        x = x[:, -1, :]  # (batch, d_model)

        # Output
        out = self.fc_out(self.dropout(x))  # (batch, 1)

        return out


class TransformerTrainer:
    """Trainer for Transformer model."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Transformer trainer.

        Args:
            input_dim: Number of input features
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            learning_rate: Learning rate
            device: Device to use ('cpu' or 'cuda')
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install: pip install torch")

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = TransformerPredictor(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        logger.info(f"Transformer model initialized on {self.device}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True
    ):
        """
        Train the model.

        Args:
            X_train: Training features (samples, seq_len, features)
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of epochs
            batch_size: Batch size
            verbose: Whether to print progress
        """
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            val_dataset = TimeSeriesDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        else:
            val_loader = None

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(X_batch).squeeze()
                loss = self.criterion(output, y_batch)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            if val_loader:
                self.model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)

                        output = self.model(X_batch).squeeze()
                        loss = self.criterion(output, y_batch)
                        val_loss += loss.item()

                val_loss /= len(val_loader)

                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}")

        logger.info("Transformer training completed")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        self.model.eval()

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            output = self.model(X_tensor).squeeze()
            predictions = output.cpu().numpy()

            # Ensure we always return an array, even for single predictions
            if predictions.ndim == 0:  # Scalar
                predictions = np.array([float(predictions)])

        return predictions

    def save(self, path: str):
        """Save model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        logger.info(f"Transformer model saved to {path}")

    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Transformer model loaded from {path}")
