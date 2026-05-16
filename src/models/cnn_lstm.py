from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow.keras import Model, layers

from ..config import CNNConfig, ForecastHeadConfig, LSTMConfig
from ..exceptions import ModelBuildError


class CNNFeatureExtractor(layers.Layer):
    """1D CNN for local pattern extraction and noise filtering in time series."""

    def __init__(
        self,
        filters: tuple[int, ...] = (64, 128, 128),
        kernel_sizes: tuple[int, ...] = (3, 3, 3),
        activation: str = "relu",
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if len(filters) != len(kernel_sizes):
            raise ModelBuildError(
                f"filters and kernel_sizes must have same length, got {len(filters)} and {len(kernel_sizes)}"
            )

        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.dropout_rate = dropout
        self.use_batch_norm = use_batch_norm

        self.conv_blocks: list[list[layers.Layer]] = []

        for i, (n_filters, kernel_size) in enumerate(zip(filters, kernel_sizes, strict=False)):
            block: list[layers.Layer] = []

            block.append(
                layers.Conv1D(
                    filters=n_filters,
                    kernel_size=kernel_size,
                    padding="same",
                    activation=None,
                    name=f"conv1d_{i}",
                )
            )

            if use_batch_norm:
                block.append(layers.BatchNormalization(name=f"batch_norm_{i}"))

            if activation == "relu":
                block.append(layers.ReLU(name=f"relu_{i}"))
            elif activation == "gelu":
                block.append(layers.Activation("gelu", name=f"gelu_{i}"))
            elif activation == "swish":
                block.append(layers.Activation(tf.nn.swish, name=f"swish_{i}"))

            if i < len(filters) - 1:
                block.append(layers.MaxPooling1D(pool_size=2, name=f"maxpool_{i}"))

            if dropout > 0:
                block.append(layers.Dropout(dropout, name=f"dropout_{i}"))

            self.conv_blocks.append(block)

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        x = inputs
        for block in self.conv_blocks:
            for layer in block:
                if isinstance(layer, (layers.Dropout, layers.BatchNormalization)):
                    x = layer(x, training=training)
                else:
                    x = layer(x)
        return x

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_sizes": self.kernel_sizes,
                "activation": self.activation,
                "dropout": self.dropout_rate,
                "use_batch_norm": self.use_batch_norm,
            }
        )
        return config  # type: ignore[no-any-return]


class LSTMTemporalEncoder(layers.Layer):
    """Bidirectional LSTM for capturing long-term temporal dependencies."""

    def __init__(
        self,
        units: tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
        recurrent_dropout: float = 0.0,  # 0.0 forces generic GPU kernel for cuDNN 9.0+ compat
        return_sequences: bool = False,
        use_bidirectional: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.units_list = units
        self.dropout_rate = dropout
        self.recurrent_dropout_rate = recurrent_dropout
        self.return_sequences = return_sequences
        self.use_bidirectional = use_bidirectional

        self.lstm_layers: list[layers.Layer] = []

        for i, n_units in enumerate(units):
            return_seq = True if i < len(units) - 1 else return_sequences

            lstm = layers.LSTM(
                units=n_units,
                return_sequences=return_seq,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                name=f"lstm_{i}",
            )

            if use_bidirectional:
                lstm = layers.Bidirectional(lstm, name=f"bidirectional_lstm_{i}")

            self.lstm_layers.append(lstm)

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        x = inputs
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x, training=training)
        return x

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "units": self.units_list,
                "dropout": self.dropout_rate,
                "recurrent_dropout": self.recurrent_dropout_rate,
                "return_sequences": self.return_sequences,
                "use_bidirectional": self.use_bidirectional,
            }
        )
        return config  # type: ignore[no-any-return]


class ForecastingHead(layers.Layer):
    """Dense head for multi-step forecasting."""

    def __init__(
        self,
        horizon: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
        activation: str = "relu",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.horizon = horizon
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout
        self.activation = activation

        self.dense_layers: list[layers.Layer] = []

        for i, dim in enumerate(hidden_dims):
            self.dense_layers.append(layers.Dense(dim, activation=activation, name=f"forecast_dense_{i}"))
            if dropout > 0:
                self.dense_layers.append(layers.Dropout(dropout, name=f"forecast_dropout_{i}"))

        self.output_layer = layers.Dense(horizon, activation=None, name="forecast_output")

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        x = inputs
        for layer in self.dense_layers:
            if isinstance(layer, layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return self.output_layer(x)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "horizon": self.horizon,
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout_rate,
                "activation": self.activation,
            }
        )
        return config  # type: ignore[no-any-return]


class HybridCNNLSTM(Model):
    """Hybrid CNN-LSTM model for energy consumption forecasting.

    Architecture: Input -> CNN (local features) -> LSTM (temporal) -> Dense -> Forecast
    """

    def __init__(
        self,
        forecast_horizon: int,
        cnn_filters: tuple[int, ...] = (64, 128, 128),
        cnn_kernel_sizes: tuple[int, ...] = (3, 3, 3),
        cnn_dropout: float = 0.2,
        lstm_units: tuple[int, ...] = (128, 64),
        lstm_dropout: float = 0.2,
        lstm_recurrent_dropout: float = 0.0,  # cuDNN 9.0+ compat
        use_bidirectional: bool = True,
        forecast_hidden_dims: tuple[int, ...] = (128, 64),
        forecast_dropout: float = 0.2,
        activation: str = "relu",
        use_batch_norm: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.forecast_horizon = forecast_horizon

        self.cnn_extractor = CNNFeatureExtractor(
            filters=cnn_filters,
            kernel_sizes=cnn_kernel_sizes,
            activation=activation,
            dropout=cnn_dropout,
            use_batch_norm=use_batch_norm,
            name="cnn_feature_extractor",
        )

        self.lstm_encoder = LSTMTemporalEncoder(
            units=lstm_units,
            dropout=lstm_dropout,
            recurrent_dropout=lstm_recurrent_dropout,
            return_sequences=False,
            use_bidirectional=use_bidirectional,
            name="lstm_temporal_encoder",
        )

        self.forecast_head = ForecastingHead(
            horizon=forecast_horizon,
            hidden_dims=forecast_hidden_dims,
            dropout=forecast_dropout,
            activation=activation,
            name="forecasting_head",
        )

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        cnn_features = self.cnn_extractor(inputs, training=training)
        lstm_output = self.lstm_encoder(cnn_features, training=training)
        return self.forecast_head(lstm_output, training=training)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"forecast_horizon": self.forecast_horizon})
        return config  # type: ignore[no-any-return]


def build_cnn_lstm_model(
    input_shape: tuple[int, ...],
    forecast_horizon: int,
    cnn_config: CNNConfig | None = None,
    lstm_config: LSTMConfig | None = None,
    forecast_config: ForecastHeadConfig | None = None,
) -> HybridCNNLSTM:
    """Build a hybrid CNN-LSTM model (uncompiled).

    Returns an uncompiled model. For custom training loops, the trainer owns
    the optimizer. For inference, call model.compile(loss=...) before predict().

    Raises:
        ModelBuildError: If model construction fails.
    """
    cnn = cnn_config or CNNConfig()
    lstm = lstm_config or LSTMConfig()
    head = forecast_config or ForecastHeadConfig()

    try:
        model = HybridCNNLSTM(
            forecast_horizon=forecast_horizon,
            cnn_filters=tuple(cnn.filters),
            cnn_kernel_sizes=tuple(cnn.kernel_sizes),
            cnn_dropout=cnn.dropout,
            lstm_units=tuple(lstm.units),
            lstm_dropout=lstm.dropout,
            lstm_recurrent_dropout=lstm.recurrent_dropout,
            use_bidirectional=lstm.use_bidirectional,
            forecast_hidden_dims=tuple(head.hidden_dims),
            forecast_dropout=head.dropout,
            activation=cnn.activation,
            use_batch_norm=cnn.use_batch_norm,
        )

        model.build(input_shape=(None,) + input_shape)

        return model

    except (ValueError, TypeError) as e:
        raise ModelBuildError(f"Failed to build CNN-LSTM model: {e}") from e
