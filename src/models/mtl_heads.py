# src/models/mtl_heads.py
# Multi-task learning heads: Anomaly Detection + Forecasting
import tensorflow as tf
from tensorflow.keras import layers
from typing import List


class AnomalyDetectionHead(layers.Layer):
    """Binary classification head for anomaly detection."""

    def __init__(self, hidden_dims: List[int] = [64, 32],
                 dropout: float = 0.3, activation: str = 'relu', **kwargs):
        super().__init__(**kwargs)

        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout

        # Global pooling to aggregate sequence information
        self.global_avg_pool = layers.GlobalAveragePooling1D()
        self.global_max_pool = layers.GlobalMaxPooling1D()

        # MLP layers
        self.dense_layers = []
        for hidden_dim in hidden_dims:
            self.dense_layers.append(layers.Dense(hidden_dim, activation=activation))
            self.dense_layers.append(layers.Dropout(dropout))

        # Output layer: binary classification
        self.output_layer = layers.Dense(1, activation='sigmoid', name='anomaly_output')

    def call(self, encoder_output, training=False):
        """Predict anomaly score from encoder output.

        Args:
            encoder_output: Transformer encoder output [batch, seq_len, d_model]
            training: Training mode flag

        Returns:
            Anomaly probability [batch, 1] (0=normal, 1=anomaly)
        """
        # Aggregate temporal information using both avg and max pooling
        avg_pool = self.global_avg_pool(encoder_output)
        max_pool = self.global_max_pool(encoder_output)

        # Concatenate pooled features
        x = tf.concat([avg_pool, max_pool], axis=-1)

        # Pass through MLP
        for layer in self.dense_layers:
            x = layer(x, training=training)

        # Output anomaly score
        anomaly_score = self.output_layer(x)

        return anomaly_score

    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout_rate
        })
        return config


class ForecastingHead(layers.Layer):
    """Regression head for multi-step forecasting."""

    def __init__(self, forecast_horizon: int = 24,
                 hidden_dims: List[int] = [128, 64],
                 dropout: float = 0.2, activation: str = 'relu',
                 use_decoder: bool = False, **kwargs):
        super().__init__(**kwargs)

        self.forecast_horizon = forecast_horizon
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout
        self.use_decoder = use_decoder

        if use_decoder:
            # Option 1: Use transformer decoder (autoregressive)
            # More complex but can model dependencies between forecast steps
            self.decoder = TransformerDecoder(
                num_layers=2,
                d_model=hidden_dims[0] if hidden_dims else 128,
                num_heads=4,
                ff_dim=256,
                dropout=dropout
            )
            self.output_projection = layers.Dense(forecast_horizon, name='forecast_output')
        else:
            # Option 2: Simple MLP projection (faster, works well for short horizons)
            # Use only the last timestep from encoder
            self.dense_layers = []
            for hidden_dim in hidden_dims:
                self.dense_layers.append(layers.Dense(hidden_dim, activation=activation))
                self.dense_layers.append(layers.Dropout(dropout))

            # Direct projection to forecast horizon
            self.output_layer = layers.Dense(forecast_horizon, name='forecast_output')

    def call(self, encoder_output, training=False):
        """Predict future values from encoder output.

        Args:
            encoder_output: Transformer encoder output [batch, seq_len, d_model]
            training: Training mode flag

        Returns:
            Forecast values [batch, forecast_horizon]
        """
        if self.use_decoder:
            # Use transformer decoder
            # Create target sequence (shifted by 1 for autoregressive)
            decoder_output = self.decoder(encoder_output, training=training)
            # Take last timestep and project to forecast horizon
            last_output = decoder_output[:, -1, :]
            forecast = self.output_projection(last_output)
        else:
            # Simple MLP approach: use last timestep from encoder
            x = encoder_output[:, -1, :]  # [batch, d_model]

            # Pass through MLP
            for layer in self.dense_layers:
                x = layer(x, training=training)

            # Output forecast
            forecast = self.output_layer(x)

        return forecast

    def get_config(self):
        config = super().get_config()
        config.update({
            'forecast_horizon': self.forecast_horizon,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout_rate,
            'use_decoder': self.use_decoder
        })
        return config


class TransformerDecoder(layers.Layer):
    """Simple transformer decoder for autoregressive forecasting (optional)."""

    def __init__(self, num_layers: int, d_model: int, num_heads: int,
                 ff_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)

        self.num_layers = num_layers
        self.d_model = d_model

        # Decoder layers with masked self-attention
        self.decoder_layers = [
            TransformerDecoderBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ]

        self.layernorm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, encoder_output, training=False):
        """Decode future values autoregressively.

        Args:
            encoder_output: Encoder output [batch, seq_len, d_model]
            training: Training mode flag

        Returns:
            Decoder output [batch, seq_len, d_model]
        """
        x = encoder_output

        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, encoder_output, training=training)

        x = self.layernorm(x)
        return x


class TransformerDecoderBlock(layers.Layer):
    """Single transformer decoder block (simplified)."""

    def __init__(self, d_model: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)

        # Import MultiHeadAttention from transformer module
        from .transformer import MultiHeadAttention, PositionWiseFeedForward

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, ff_dim, dropout)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        self.dropout3 = layers.Dropout(dropout)

    def call(self, x, encoder_output, training=False):
        """Forward pass through decoder block.

        Args:
            x: Decoder input [batch, seq_len, d_model]
            encoder_output: Encoder output for cross-attention [batch, seq_len, d_model]
            training: Training mode flag

        Returns:
            Decoder output [batch, seq_len, d_model]
        """
        # Masked self-attention (look only at past)
        attn1 = self.self_attention(x, training=training)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        # Cross-attention with encoder output
        attn2 = self.cross_attention(out1, training=training)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        # Feed-forward
        ffn_out = self.ffn(out2, training=training)
        ffn_out = self.dropout3(ffn_out, training=training)
        out3 = self.layernorm3(out2 + ffn_out)

        return out3


class MTLModel(tf.keras.Model):
    """Complete Multi-Task Learning model: Encoder + Dual Heads."""

    def __init__(self, encoder, anomaly_head, forecasting_head, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.anomaly_head = anomaly_head
        self.forecasting_head = forecasting_head

    def call(self, inputs, training=False):
        """Forward pass through MTL model.

        Args:
            inputs: Input sequences [batch, seq_len, features]
            training: Training mode flag

        Returns:
            Tuple of (anomaly_scores, forecasts)
                - anomaly_scores: [batch, 1]
                - forecasts: [batch, forecast_horizon]
        """
        # Encode input sequence
        encoder_output = self.encoder(inputs, training=training)

        # Anomaly detection head
        anomaly_scores = self.anomaly_head(encoder_output, training=training)

        # Forecasting head
        forecasts = self.forecasting_head(encoder_output, training=training)

        return anomaly_scores, forecasts

    def get_config(self):
        return {
            'encoder': self.encoder,
            'anomaly_head': self.anomaly_head,
            'forecasting_head': self.forecasting_head
        }


if __name__ == "__main__":
    # Test MTL heads
    print("Testing MTL Heads...")

    # Create sample encoder output: [batch=2, seq_len=168, d_model=128]
    batch_size = 2
    seq_len = 168
    d_model = 128

    encoder_output = tf.random.normal((batch_size, seq_len, d_model))

    # Test anomaly head
    anomaly_head = AnomalyDetectionHead(hidden_dims=[64, 32])
    anomaly_scores = anomaly_head(encoder_output, training=False)
    print(f"Anomaly scores shape: {anomaly_scores.shape}")  # [2, 1]

    # Test forecasting head
    forecast_head = ForecastingHead(forecast_horizon=24, hidden_dims=[128, 64])
    forecasts = forecast_head(encoder_output, training=False)
    print(f"Forecasts shape: {forecasts.shape}")  # [2, 24]

    print("✅ MTL heads work correctly!")
