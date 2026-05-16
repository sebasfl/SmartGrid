import numpy as np
import pandas as pd

from src.analysis.data_quality import (
    calculate_building_quality_metrics,
    compute_quality_score,
)


def _make_building_df(building_id: str, n_hours: int = 500, freq: str = '1h',
                      missing_ratio: float = 0.0, constant_hours: int = 0) -> pd.DataFrame:
    """Helper to create a single-building DataFrame."""
    timestamps = pd.date_range('2020-01-01', periods=n_hours, freq=freq)
    values = np.random.uniform(10, 100, n_hours).astype(np.float32)

    if constant_hours > 0:
        values[:constant_hours] = 42.0

    df = pd.DataFrame({
        'timestamp_local': timestamps,
        'building_id': building_id,
        'value': values,
    })

    if missing_ratio > 0:
        mask = np.random.random(n_hours) < missing_ratio
        df.loc[mask, 'value'] = np.nan

    return df


class TestComputeQualityScore:
    def test_perfect_data(self):
        score = compute_quality_score(1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
        assert abs(score - 1.0) < 1e-9

    def test_worst_data(self):
        score = compute_quality_score(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        assert abs(score - 0.0) < 1e-9

    def test_partial_quality(self):
        score = compute_quality_score(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        assert 0.0 < score < 1.0


class TestBuildingQualityMetrics:
    def test_perfect_data_high_score(self):
        df = _make_building_df('b1', n_hours=500)
        metrics = calculate_building_quality_metrics(df, 'b1', granularity_hours=1)

        assert metrics is not None
        assert metrics['quality_score'] > 0.8
        assert metrics['completeness_ratio'] == 1.0

    def test_missing_data_reduces_completeness(self):
        df = _make_building_df('b1', n_hours=500, missing_ratio=0.3)
        metrics = calculate_building_quality_metrics(df, 'b1', granularity_hours=1)

        assert metrics['completeness_ratio'] < 0.8

    def test_constant_sequences_detected(self):
        df = _make_building_df('b1', n_hours=500, constant_hours=100)
        metrics = calculate_building_quality_metrics(df, 'b1', granularity_hours=1)

        assert metrics['constant_ratio'] > 0

    def test_empty_building_returns_none(self):
        df = pd.DataFrame(columns=['timestamp_local', 'building_id', 'value'])
        metrics = calculate_building_quality_metrics(df, 'nonexistent', granularity_hours=1)
        assert metrics is None

    def test_3h_granularity(self):
        df = _make_building_df('b1', n_hours=200, freq='3h')
        metrics = calculate_building_quality_metrics(df, 'b1', granularity_hours=3)

        assert metrics is not None
        assert metrics['total_records'] == 200
