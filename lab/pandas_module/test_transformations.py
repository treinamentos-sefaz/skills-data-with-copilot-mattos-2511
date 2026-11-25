"""Testes das transformações de vinhos."""

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from lab.pandas_module import transformations as T


class TransformationsTest(unittest.TestCase):
    def test_clean_ratings_removes_invalid_without_mutating_input(self):
        source = pd.DataFrame({"rating": ["85", "inválido", None, "90.5"]})
        original = source.copy(deep=True)

        result = T.clean_ratings(source)

        self.assertEqual(result["rating"].tolist(), [85.0, 90.5])
        assert_frame_equal(source, original)

    def test_top_varieties_returns_sorted_means(self):
        source = pd.DataFrame(
            {"variety": ["A", "B", "A", "C"], "rating": [80, 90, 100, 95]}
        )
        original = source.copy(deep=True)

        result = T.top_varieties_by_avg_rating(source, n=2)

        self.assertEqual(result.columns.tolist(), ["variety", "avg_rating"])
        self.assertEqual(result["variety"].tolist(), ["C", "A"])
        self.assertEqual(result["avg_rating"].tolist(), [95.0, 90.0])
        assert_frame_equal(source, original)

    def test_rating_categories_include_boundaries_and_preserve_nulls(self):
        source = pd.DataFrame({"rating": [84.9, 85, 90, 90.1, None]})
        original = source.copy(deep=True)

        result = T.add_rating_category(source)

        self.assertEqual(result["rating_category"].iloc[:4].tolist(),
                         ["Baixa", "Média", "Média", "Alta"])
        self.assertTrue(pd.isna(result["rating_category"].iloc[4]))
        assert_frame_equal(source, original)

    def test_country_uses_last_segment_and_preserves_nulls(self):
        source = pd.DataFrame({"region": ["A, B, California ", " Portugal ", None]})
        original = source.copy(deep=True)

        result = T.enrich_with_country(source)

        self.assertEqual(result["country"].tolist(), ["California", "Portugal", None])
        assert_frame_equal(source, original)


if __name__ == "__main__":
    unittest.main()
