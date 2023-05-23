import unittest
from random import random

from matplotlib.pyplot import subplots
from numpy import linspace, ndarray, pi, sin

from graphinglib.data_plotting_1d import *
from graphinglib.graph_elements import *


class TestCurve(unittest.TestCase):
    def setUp(self):
        x = linspace(0, 3 * pi, 200)
        self.testCurve = Curve(x, sin(x), "Test Curve", color="k")

    def test_xdata_is_list_or_ndarray(self):
        self.assertIsInstance(self.testCurve.x_data, list | ndarray)

    def test_ydata_is_list_or_ndarray(self):
        self.assertIsInstance(self.testCurve.y_data, list | ndarray)

    def test_default_value(self):
        self.assertEqual(self.testCurve.line_width, "default")

    def test_color_is_str(self):
        self.assertIsInstance(self.testCurve.color, str)

    def test_label_is_str(self):
        self.assertIsInstance(self.testCurve.label, str)

    def test_curve_is_plotted(self):
        x = linspace(0, 3 * pi, 200)
        self.testCurve = Curve(
            x, sin(x), "Test Curve", color="k", line_width=3, line_style="--"
        )
        _, self.testAxes = subplots()
        self.testCurve._plot_element(self.testAxes, 0)
        self.assertEqual(len(self.testAxes.get_lines()), 1)


class TestScatter(unittest.TestCase):
    def setUp(self):
        x = linspace(0, 3 * pi, 200)
        self.testScatter = Scatter(x, sin(x), "Test Curve")

    def test_scatter_is_scatter(self):
        self.assertIsInstance(self.testScatter, Scatter)

    def test_xdata_is_list_or_ndarray(self):
        self.assertIsInstance(self.testScatter.x_data, list | ndarray)

    def test_color_is_str(self):
        self.assertIsInstance(self.testScatter.face_color, str)
        self.assertIsInstance(self.testScatter.edge_color, str)

    def test_label_is_str(self):
        self.assertIsInstance(self.testScatter.label, str)

    def test_curve_is_plotted(self):
        x = linspace(0, 3 * pi, 200)
        self.testScatter = Scatter(
            x,
            sin(x),
            "test scatter",
            face_color="k",
            edge_color="k",
            marker_size=30,
            marker_style="o",
        )
        _, self.testAxes = subplots()
        self.testScatter._plot_element(self.testAxes, 0)
        self.assertEqual(len(self.testAxes.collections), 1)


class TestHistogram(unittest.TestCase):
    def setUp(self):
        self.testHist = Histogram(
            [random() for _ in range(100)],
            20,
            label="Random Distribution",
            face_color="silver",
            edge_color="k",
        )
        # _, self.testAxes = subplots()
        # self.testHist.plot_curve(self.testAxes)
        # plt.close('all')

    def test_label_is_str(self):
        self.assertEqual(self.testHist.label[:19], "Random Distribution")

    def test_xdata_is_list_or_ndarray(self):
        self.assertIsInstance(self.testHist.x_data, list | ndarray)

    def test_face_color_is_str(self):
        self.assertEqual(self.testHist.face_color, "silver")

    def test_edge_color_is_str(self):
        self.assertEqual(self.testHist.edge_color, "k")

    def test_bins_is_int(self):
        self.assertEqual(self.testHist.number_of_bins, 20)

    def test_alpha_is_default(self):
        self.assertEqual(self.testHist.alpha, "default")

    def test_hist_type_is_str(self):
        self.assertEqual(self.testHist.hist_type, "default")


class TestHlines(unittest.TestCase):
    def setUp(self):
        self.testHlines = Hlines(1, 0, 1, "Test Hlines")
        # _, self.testAxes = subplots()
        # self.testHlines.plot_curve(self.testAxes)
        # plt.close('all')

    def test_y_is_list_ndarray_float_int(self):
        self.assertIsInstance(self.testHlines.y, list | ndarray | float | int)

    def test_xmin_is_list_ndarray_float_int(self):
        self.assertIsInstance(self.testHlines.x_min, list | ndarray | float | int)

    def test_ymin_is_list_ndarray_float_int(self):
        self.assertIsInstance(self.testHlines.x_min, list | ndarray | float | int)

    def test_colors_is_str_list_or_none(self):
        self.assertIsInstance(self.testHlines.colors, list | str | None)

    def test_linestyles_is_str_list_or_none(self):
        self.assertIsInstance(self.testHlines.line_styles, list | str | None)

    def test_label_is_str(self):
        self.assertIsInstance(self.testHlines.label, str)


class TestVlines(unittest.TestCase):
    def setUp(self):
        self.testVlines = Vlines(x=[4, 5, 6, 7], y_min=0, y_max=1, label="Test Vlines")
        # _, self.testAxes = subplots()
        # self.testVlines.plot_curve(self.testAxes)
        # plt.close('all')

    def test_x_is_list_ndarray_float_int(self):
        self.assertListEqual(self.testVlines.x, [4, 5, 6, 7])

    def test_ymin_is_list_ndarray_float_int(self):
        self.assertEqual(self.testVlines.y_min, 0)

    def test_ymax_is_list_ndarray_float_int(self):
        self.assertEqual(self.testVlines.y_max, 1)

    def test_colors_is_default(self):
        self.assertEqual(self.testVlines.colors, "default")

    def test_linestyles_is_str_list_or_none(self):
        self.assertIsInstance(self.testVlines.line_styles, list | str | None)

    def test_label_is_str(self):
        self.assertEqual(self.testVlines.label, "Test Vlines")


class TestPoint(unittest.TestCase):
    pass
