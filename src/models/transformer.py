# src/models/transformer.py
# Transformer Encoder for time series processing
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from typing import Optional


class PositionalEncoding(layers.Layer):
    """Sinusoidal or learnable positional encoding for transformer."""

    def __init__(self, d_model: int, max_seq_length: int = 512,
                 encoding_type: str = 'sinusoidal', **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.encoding_type = encoding_type

        if encoding_type == 'sinusoidal':
            # Pre-compute sinusoidal positional encodings
            self.pos_encoding = self._create_sinusoidal_encoding()
        elif encoding_type == 'learnable':
            # Learnable positional embeddings
            self.pos_encoding = self.add_weight(
                name='pos_encoding',
                shape=(1, max_seq_length, d_model),
                initializer='glorot_uniform',
                trainable=True
            )
        else:
            raise ValueError(f"Unknown encoding_type: {encoding_type}")

    def _create_sinusoidal_encoding(self):
        """Create sinusoidal positional encoding matrix."""
        position = np.arange(self.max_seq_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))

        pos_encoding = np.zeros((1, self.max_seq_length, self.d_model))
        pos_encoding[0, :, 0::2] = np.sin(position * div_term)
        pos_encoding[0, :, 1::2] = np.cos(position * div_term)

        return tf.constant(pos_encoding, dtype=tf.float32)

    def call(self, x):
        """Add positional encoding to input.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            x + positional encoding [batch, seq_len, d_model]
        """
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'max_seq_length': self.max_seq_length,
            'encoding_type': self.encoding_type
        })
        return config


class MultiHeadAttention(layers.Layer):
    """Multi-head self-attention mechanism."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        # Linear projections for Q, K, V
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)

        self.dropout = layers.Dropout(dropout)
        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x, mask=None, training=False):
        """Apply multi-head attention.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask
            training: Training mode flag

        Returns:
            Attention output [batch, seq_len, d_model]
        """
        batch_size = tf.shape(x)[0]

        # Linear projections
        q = self.wq(x)  # [batch, seq_len, d_model]
        k = self.wk(x)
        v = self.wv(x)

        # Split into multiple heads
        q = self.split_heads(q, batch_size)  # [batch, num_heads, seq_len, depth]
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # [batch, num_heads, seq_len, seq_len]
        dk = tf.cast(self.depth, tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # Apply mask if provided
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # Softmax
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)

        # Apply attention to values
        attention_output = tf.matmul(attention_weights, v)  # [batch, num_heads, seq_len, depth]

        # Concatenate heads
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.d_model))

        # Final linear projection
        output = self.dense(concat_attention)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads
        })
        return config


class PositionWiseFeedForward(layers.Layer):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.1,
                 activation: str = 'gelu', **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.ff_dim = ff_dim

        self.dense1 = layers.Dense(ff_dim, activation=activation)
        self.dense2 = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout)

    def call(self, x, training=False):
        """Apply feed-forward network.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            training: Training mode flag

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'ff_dim': self.ff_dim
        })
        return config


class TransformerEncoderBlock(layers.Layer):
    """Single transformer encoder block with attention + FFN."""

    def __init__(self, d_model: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.1, activation: str = 'gelu', **kwargs):
        super().__init__(**kwargs)

        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, ff_dim, dropout, activation)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x, mask=None, training=False):
        """Forward pass through encoder block.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask
            training: Training mode flag

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Multi-head attention with residual connection and layer norm
        attn_output = self.attention(x, mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # Feed-forward network with residual connection and layer norm
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class TransformerEncoder(layers.Layer):
    """Transformer encoder: stack of encoder blocks."""

    def __init__(self, num_layers: int, d_model: int, num_heads: int,
                 ff_dim: int, input_dim: int, dropout: float = 0.1,
                 activation: str = 'gelu', use_positional_encoding: bool = True,
                 positional_encoding_type: str = 'sinusoidal',
                 max_seq_length: int = 512, **kwargs):
        super().__init__(**kwargs)

        self.num_layers = num_layers
        self.d_model = d_model

        # Input projection to d_model
        self.input_projection = layers.Dense(d_model)

        # Positional encoding
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(
                d_model, max_seq_length, positional_encoding_type
            )

        self.dropout = layers.Dropout(dropout)

        # Stack of encoder blocks
        self.encoder_blocks = [
            TransformerEncoderBlock(d_model, num_heads, ff_dim, dropout, activation)
            for _ in range(num_layers)
        ]

        # Final layer normalization
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, mask=None, training=False):
        """Forward pass through transformer encoder.

        Args:
            x: Input tensor [batch, seq_len, input_dim]
            mask: Optional attention mask
            training: Training mode flag

        Returns:
            Encoded representation [batch, seq_len, d_model]
        """
        # Project input to d_model dimension
        x = self.input_projection(x)

        # Add positional encoding
        if self.use_positional_encoding:
            x = self.pos_encoding(x)

        x = self.dropout(x, training=training)

        # Pass through encoder blocks
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, mask=mask, training=training)

        # Final layer normalization
        x = self.layernorm(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_layers': self.num_layers,
            'd_model': self.d_model
        })
        return config


if __name__ == "__main__":
    # Test transformer encoder
    print("Testing Transformer Encoder...")

    # Create sample input: [batch=2, seq_len=168, features=8]
    batch_size = 2
    seq_len = 168
    input_dim = 8

    x = tf.random.normal((batch_size, seq_len, input_dim))

    # Create encoder
    encoder = TransformerEncoder(
        num_layers=4,
        d_model=128,
        num_heads=8,
        ff_dim=512,
        input_dim=input_dim,
        dropout=0.1
    )

    # Forward pass
    output = encoder(x, training=False)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"✅ Transformer encoder works correctly!")
