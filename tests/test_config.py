import math

import pytest

from src.config import Config
from src.exceptions import ConfigurationError


class TestConfig:
    def test_defaults_are_valid(self):
        config = Config()
        assert config.data.lookback_window > 0
        assert config.data.forecast_horizon > 0
        assert config.training.batch_size > 0
        assert config.training.epochs > 0

    def test_ratios_sum_to_one(self):
        config = Config()
        total = config.data.train_ratio + config.data.val_ratio + config.data.test_ratio
        assert math.isclose(total, 1.0, abs_tol=1e-9)

    def test_json_roundtrip(self, tmp_path):
        config = Config()
        config.cnn.filters = [32, 64]
        config.training.batch_size = 4

        json_path = str(tmp_path / "test_config.json")
        config.to_json(json_path)

        loaded = Config.from_json(json_path)
        assert loaded.cnn.filters == [32, 64]
        assert loaded.training.batch_size == 4
        assert loaded.data.forecast_horizon == config.data.forecast_horizon

    def test_str_representation(self):
        config = Config()
        text = str(config)
        assert "CONFIGURATION" in text
        assert "DATA" in text
        assert "CNN" in text

    def test_validate_passes_for_defaults(self):
        Config().validate()

    def test_validate_catches_mismatched_cnn_lengths(self):
        config = Config()
        config.cnn.filters = [64, 128]
        config.cnn.kernel_sizes = [3]
        with pytest.raises(ConfigurationError, match="same length"):
            config.validate()

    def test_validate_catches_bad_ratios(self):
        config = Config()
        config.data.train_ratio = 0.5
        config.data.val_ratio = 0.5
        config.data.test_ratio = 0.5
        with pytest.raises(ConfigurationError, match="sum to 1.0"):
            config.validate()

    def test_from_json_rejects_unknown_section(self, tmp_path):
        import json

        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            json.dump({"unknown_section": {"key": "val"}}, f)
        with pytest.raises(ConfigurationError, match="Unknown config sections"):
            Config.from_json(path)

    def test_from_json_rejects_unknown_key(self, tmp_path):
        import json

        path = str(tmp_path / "bad_key.json")
        with open(path, "w") as f:
            json.dump({"cnn": {"nonexistent_param": 42}}, f)
        with pytest.raises(ConfigurationError, match="Unknown keys"):
            Config.from_json(path)
