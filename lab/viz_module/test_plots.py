"""Testes dos gráficos de vinhos."""

import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pandas.testing import assert_frame_equal

from lab.viz_module import plots


class PlotsTest(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "variety": [f"V{i}" for i in range(12) for _ in range(12 - i)],
            "rating": [85 + i % 10 for i in range(78)],
        })
        self.original = self.df.copy(deep=True)

    def test_plots_save_pngs_and_show_expected_data(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, plot in [
                ("histograma", plots.plot_rating_distribution),
                ("barras", plots.plot_top_varieties),
                ("boxplot", plots.plot_boxplot_by_variety),
            ]:
                with self.subTest(plot=name):
                    path = Path(directory) / f"{name}.png"
                    fig = plot(self.df, str(path))
                    ax = fig.axes[0]
                    self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                    self.assertTrue(ax.get_title() and ax.get_xlabel() and ax.get_ylabel())
                    self.assertNotIn(fig.number, plt.get_fignums())
                    if name == "histograma":
                        self.assertEqual(len(ax.patches), 20)
                        self.assertEqual(sum(patch.get_height() for patch in ax.patches), len(self.df))
                    elif name == "barras":
                        self.assertEqual([tick.get_text() for tick in ax.get_yticklabels()],
                                         [f"V{i}" for i in range(10)])
                        self.assertEqual([bar.get_width() for bar in ax.patches], list(range(12, 2, -1)))
                    else:
                        self.assertEqual([tick.get_text() for tick in ax.get_xticklabels()],
                                         [f"V{i}" for i in range(5)])
            assert_frame_equal(self.df, self.original)

    def test_figure_closed_if_save_fails(self):
        before = plt.get_fignums()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                plots.plot_rating_distribution(self.df, str(Path(directory) / "missing" / "plot.png"))
        self.assertEqual(plt.get_fignums(), before)


if __name__ == "__main__":
    unittest.main()
