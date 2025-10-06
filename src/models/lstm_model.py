"""
LSTM and GRU models for time series prediction.
Implements state-of-the-art deep learning architectures for stock price forecasting.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Install: pip install torch")


class StockDataset(Dataset):
    """Dataset for stock time series data."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize dataset.

        Args:
            X: Input features (samples, sequence_length, features)
            y: Target values (samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """LSTM model for stock prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [128, 64, 32],
        dropout: float = 0.2,
        output_size: int = 1
    ):
        """
        Initialize LSTM model.

        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout rate
            output_size: Output dimension (usually 1 for regression)
        """
        super(LSTMModel, self).__init__()

        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)

        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        # First LSTM layer
        self.lstm_layers.append(
            nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        )
        self.dropout_layers.append(nn.Dropout(dropout))

        # Additional LSTM layers
        for i in range(1, len(hidden_sizes)):
            self.lstm_layers.append(
                nn.LSTM(hidden_sizes[i-1], hidden_sizes[i], batch_first=True)
            )
            self.dropout_layers.append(nn.Dropout(dropout))

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_sizes[-1], 32)
        self.fc2 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass."""
        # Pass through LSTM layers
        for i, (lstm, dropout) in enumerate(zip(self.lstm_layers, self.dropout_layers)):
            x, _ = lstm(x)
            x = dropout(x)

        # Take output from last time step
        x = x[:, -1, :]

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)

        return x


class GRUModel(nn.Module):
    """GRU model for stock prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [100, 50],
        dropout: float = 0.2,
        output_size: int = 1
    ):
        """
        Initialize GRU model.

        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout rate
            output_size: Output dimension
        """
        super(GRUModel, self).__init__()

        self.hidden_sizes = hidden_sizes

        # GRU layers
        self.gru_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        # First GRU layer
        self.gru_layers.append(
            nn.GRU(input_size, hidden_sizes[0], batch_first=True)
        )
        self.dropout_layers.append(nn.Dropout(dropout))

        # Additional GRU layers
        for i in range(1, len(hidden_sizes)):
            self.gru_layers.append(
                nn.GRU(hidden_sizes[i-1], hidden_sizes[i], batch_first=True)
            )
            self.dropout_layers.append(nn.Dropout(dropout))

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_sizes[-1], 32)
        self.fc2 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass."""
        # Pass through GRU layers
        for gru, dropout in zip(self.gru_layers, self.dropout_layers):
            x, _ = gru(x)
            x = dropout(x)

        # Take output from last time step
        x = x[:, -1, :]

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)

        return x


class LSTMTrainer:
    """Trainer for LSTM/GRU models."""

    def __init__(
        self,
        model_type: str = 'lstm',
        input_size: int = 50,
        hidden_sizes: List[int] = [128, 64, 32],
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        device: Optional[str] = None
    ):
        """
        Initialize trainer.

        Args:
            model_type: 'lstm' or 'gru'
            input_size: Number of input features
            hidden_sizes: Hidden layer sizes
            dropout: Dropout rate
            learning_rate: Learning rate
            device: Device to use ('cuda' or 'cpu')
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install: pip install torch")

        self.model_type = model_type
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize model
        if model_type == 'lstm':
            self.model = LSTMModel(input_size, hidden_sizes, dropout)
        elif model_type == 'gru':
            self.model = GRUModel(input_size, hidden_sizes, dropout)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model.to(self.device)

        # Optimizer and loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Training history
        self.train_losses = []
        self.val_losses = []

        logger.info(f"{model_type.upper()} model initialized on {self.device}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of epochs
            batch_size: Batch size
            early_stopping_patience: Patience for early stopping
            verbose: Whether to print progress

        Returns:
            Training history dictionary
        """
        # Create datasets
        train_dataset = StockDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = StockDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(X_batch).squeeze()
                loss = self.criterion(outputs, y_batch)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)

            # Validation
            val_loss = 0.0
            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)

                        outputs = self.model(X_batch).squeeze()
                        loss = self.criterion(outputs, y_batch)
                        val_loss += loss.item()

                val_loss /= len(val_loader)
                self.val_losses.append(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                if val_loader is not None:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                else:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}")

        logger.info("Training completed")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features
            batch_size: Batch size

        Returns:
            Predictions array
        """
        self.model.eval()

        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(dataset, batch_size=batch_size)

        predictions = []

        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch).squeeze()
                predictions.extend(outputs.cpu().numpy())

        return np.array(predictions)

    def save_model(self, path: str):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_type': self.model_type,
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'dropout': self.dropout,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)

        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])

        logger.info(f"Model loaded from {path}")

    def get_model_summary(self) -> Dict:
        """Get model summary."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'model_type': self.model_type,
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device)
        }
