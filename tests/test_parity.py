import numpy as np
import pandas as pd
import pytest

from src.analysis.clean_building_data_parallel import BasicPreprocessor
from src.analysis.data_quality import calculate_building_quality_metrics, compute_quality_score


def _make_test_df(n_hours=48, freq="1h", building_id="test_b1"):
    timestamps = pd.date_range("2020-01-01", periods=n_hours, freq=freq)
    return pd.DataFrame(
        {
            "timestamp_local": timestamps,
            "building_id": building_id,
            "meter": "electricity",
            "value": np.random.uniform(10, 100, n_hours).astype(np.float32),
        }
    )


class TestPreprocessingParity:
    def _cpu_processor(self):
        proc = BasicPreprocessor(verbose=False)
        proc.gpu_available = False
        return proc

    def test_time_features_columns_and_types(self):
        proc = self._cpu_processor()
        df = _make_test_df(n_hours=24)
        result = proc.add_time_features(df.copy())

        expected_int8 = ["hour", "day_of_week", "month", "is_weekend", "is_working_hours", "quarter"]
        expected_int16 = ["day_of_year"]

        for col in expected_int8:
            assert col in result.columns, f"Missing column: {col}"
            assert result[col].dtype == np.int8, f"{col} dtype is {result[col].dtype}, expected int8"

        for col in expected_int16:
            assert col in result.columns, f"Missing column: {col}"
            assert result[col].dtype == np.int16, f"{col} dtype is {result[col].dtype}, expected int16"

    def test_time_features_values_are_correct(self):
        proc = self._cpu_processor()
        df = _make_test_df(n_hours=24)
        result = proc.add_time_features(df.copy())

        assert result["hour"].iloc[0] == 0
        assert result["hour"].iloc[8] == 8
        assert result["month"].iloc[0] == 1
        assert result["quarter"].iloc[0] == 1
        assert result["day_of_year"].iloc[0] == 1

    def test_resample_3h_aggregation(self):
        proc = self._cpu_processor()
        df = _make_test_df(n_hours=24, freq="1h")
        result = proc.resample_to_3h(df.copy())

        assert len(result) == 8, f"Expected 8 rows after 3h resampling of 24h data, got {len(result)}"

        original_first_3 = df["value"].iloc[:3].mean()
        resampled_first = result.sort_values("timestamp_local")["value"].iloc[0]
        np.testing.assert_allclose(resampled_first, original_first_3, rtol=1e-5)

    def test_resample_3h_preserves_building_id(self):
        proc = self._cpu_processor()
        df = _make_test_df(n_hours=24, building_id="bldg_42")
        result = proc.resample_to_3h(df.copy())

        assert (result["building_id"] == "bldg_42").all()

    def test_deduplication_removes_exact_duplicates(self):
        proc = self._cpu_processor()
        df = _make_test_df(n_hours=10)
        dup_row = df.iloc[[3]].copy()
        df_with_dups = pd.concat([df, dup_row], ignore_index=True)

        assert len(df_with_dups) == 11
        result = proc.remove_duplicates(df_with_dups)
        assert len(result) == 10

    def test_deduplication_preserves_different_buildings(self):
        proc = self._cpu_processor()
        df1 = _make_test_df(n_hours=5, building_id="b1")
        df2 = _make_test_df(n_hours=5, building_id="b2")
        combined = pd.concat([df1, df2], ignore_index=True)

        result = proc.remove_duplicates(combined)
        assert len(result) == 10

    def test_full_process_pipeline(self):
        proc = self._cpu_processor()
        df = _make_test_df(n_hours=48, freq="1h")
        result = proc.process(df.copy(), resample_3h=True)

        assert len(result) < 48
        time_feature_cols = ["hour", "day_of_week", "month", "is_weekend", "is_working_hours", "quarter", "day_of_year"]
        for col in time_feature_cols:
            assert col in result.columns

    def test_full_process_without_resample(self):
        proc = self._cpu_processor()
        df = _make_test_df(n_hours=24)
        result = proc.process(df.copy(), resample_3h=False)

        assert len(result) == 24


class TestQualityScoreParity:
    def test_quality_score_formula_perfect(self):
        score = compute_quality_score(1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
        assert score == pytest.approx(1.0)

    def test_quality_score_formula_worst(self):
        score = compute_quality_score(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        assert score == pytest.approx(0.0)

    def test_quality_score_formula_weights_sum_to_one(self):
        score = compute_quality_score(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        assert score == pytest.approx(0.5)

    def test_quality_score_monotonic_in_completeness(self):
        low = compute_quality_score(0.5, 0.8, 0.8, 0.0, 0.0, 0.0)
        high = compute_quality_score(1.0, 0.8, 0.8, 0.0, 0.0, 0.0)
        assert high > low

    def test_cpu_quality_metrics_complete_data(self):
        df = _make_test_df(n_hours=168, freq="1h")
        metrics = calculate_building_quality_metrics(df, "test_b1", granularity_hours=1)

        assert metrics is not None
        assert metrics["quality_score"] > 0.9
        assert metrics["completeness_ratio"] == pytest.approx(1.0)
        assert metrics["duplicate_ratio"] == pytest.approx(0.0)
        assert metrics["total_records"] == 168

    def test_cpu_quality_metrics_with_gaps(self):
        ts1 = pd.date_range("2020-01-01", periods=48, freq="1h")
        ts2 = pd.date_range("2020-01-04", periods=48, freq="1h")
        timestamps = ts1.union(ts2)
        df = pd.DataFrame(
            {
                "timestamp_local": timestamps,
                "building_id": "test_b1",
                "meter": "electricity",
                "value": np.random.uniform(10, 100, len(timestamps)).astype(np.float32),
            }
        )
        metrics = calculate_building_quality_metrics(df, "test_b1", granularity_hours=1)

        assert metrics["continuity_ratio"] < 1.0
        assert metrics["gaps_24h"] >= 1

    def test_cpu_quality_metrics_with_missing_values(self):
        df = _make_test_df(n_hours=100)
        df.loc[df.index[:10], "value"] = np.nan
        metrics = calculate_building_quality_metrics(df, "test_b1", granularity_hours=1)

        assert metrics["completeness_ratio"] == pytest.approx(0.9)

    def test_cpu_quality_metrics_empty_building(self):
        df = _make_test_df(n_hours=10, building_id="other")
        metrics = calculate_building_quality_metrics(df, "nonexistent", granularity_hours=1)
        assert metrics is None
