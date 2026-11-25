"""Testes das análises exploratórias."""

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from lab.eda_module import analysis


class AnalysisTest(unittest.TestCase):
    def test_descriptive_stats_ignores_missing_ratings(self):
        df = pd.DataFrame({"rating": [80.0, 90.0, None]})
        result = analysis.descriptive_stats(df)
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["mean"], 85)
        self.assertEqual(result["min"], 80)
        self.assertEqual(result["max"], 90)

    def test_outliers_use_strict_iqr_bounds_and_custom_column(self):
        df = pd.DataFrame({"value": [-4, -3, 0, 0, 1, 2, 2, 5, 6, None]})
        original = df.copy(deep=True)
        result = analysis.detect_outliers_iqr(df, column="value")

        assert_frame_equal(result, df.loc[[0, 8]])
        result.loc[0, "value"] = -100
        assert_frame_equal(df, original)

    def test_missing_values_report_counts_each_column(self):
        df = pd.DataFrame({"rating": [80, None], "variety": [None, None]})
        expected = pd.Series({"rating": 1, "variety": 2})
        assert_series_equal(analysis.missing_values_report(df), expected)

    def test_rating_stats_by_variety_excludes_nulls_and_sorts(self):
        df = pd.DataFrame({
            "variety": ["A", "A", "A", "B", None],
            "rating": [80.0, 90.0, None, 95.0, 100.0],
        })
        original = df.copy(deep=True)
        result = analysis.rating_stats_by_variety(df)

        self.assertEqual(result.columns.tolist(), ["count", "mean", "min", "max"])
        self.assertEqual(result.index.tolist(), ["B", "A"])
        self.assertEqual(result.loc["A"].tolist(), [2, 85, 80, 90])
        self.assertEqual(result.loc["B"].tolist(), [1, 95, 95, 95])
        assert_frame_equal(df, original)


if __name__ == "__main__":
    unittest.main()
