import unittest
from random import random

from GraphingLib.Graph_elements import *
from numpy import ndarray, linspace, pi, sin
from matplotlib.pyplot import subplots


class TestCurve(unittest.TestCase):
    def setUp(self):
        x = linspace(0, 3*pi, 200)
        self.testCurve = Curve(x, sin(x), 'Test Curve', color='k')

    def test_xdata_is_list_or_ndarray(self):
        self.assertIsInstance(self.testCurve.xdata, list | ndarray)

    def test_ydata_is_list_or_ndarray(self):
        self.assertIsInstance(self.testCurve.ydata, list | ndarray)
    
    def test_default_value(self):
        self.assertEqual(self.testCurve.line_width, "default")

    def test_color_is_str(self):
        self.assertIsInstance(self.testCurve.color, str)

    def test_label_is_str(self):
        self.assertIsInstance(self.testCurve.label, str)

    def test_color_is_changed(self):
        self.testCurve.set_color('r')
        self.assertEqual(self.testCurve.color, 'r')

    def test_curve_is_plotted(self):
        x = linspace(0, 3*pi, 200)
        self.testCurve = Curve(x, sin(x), 'Test Curve', color='k', line_width=3)
        _, self.testAxes = subplots()
        self.testCurve.plot_curve(self.testAxes)
        self.assertEqual(len(self.testAxes.get_lines()), 1)


class TestScatter(unittest.TestCase):
    def setUp(self):
        x = linspace(0, 3*pi, 200)
        self.testScatter = Scatter(x, sin(x), 'Test Curve', 'k')

    def test_scatter_is_curve(self):
        self.assertIsInstance(self.testScatter, Curve)

    def test_xdata_is_list_or_ndarray(self):
        self.assertIsInstance(self.testScatter.xdata, list | ndarray)

    def test_color_is_str(self):
        self.assertIsInstance(self.testScatter.color, str)

    def test_label_is_str(self):
        self.assertIsInstance(self.testScatter.label, str)

    def test_color_is_changed(self):
        self.testScatter.set_color('r')
        self.assertEqual(self.testScatter.color, 'r')

    def test_curve_is_plotted(self):
        x = linspace(0, 3*pi, 200)
        self.testScatter = Scatter(x, sin(x), 'Test Curve', 'k', line_width=1, marker_size=10)
        _, self.testAxes = subplots()
        self.testScatter.plot_curve(self.testAxes)
        self.assertEqual(len(self.testAxes.collections), 1)


class TestHistogram(unittest.TestCase):
    def setUp(self):
        self.testHist = Histogram([random() for _ in range(100)], 20,
                                    label='Random Distribution', face_color='silver', edge_color='k')
        # _, self.testAxes = subplots()
        # self.testHist.plot_curve(self.testAxes)
        # plt.close('all')

    def test_label_is_str(self):
        self.assertEqual(self.testHist.label[:19], "Random Distribution")

    def test_xdata_is_list_or_ndarray(self):
        self.assertIsInstance(self.testHist.xdata, list | ndarray)

    def test_face_color_is_str(self):
        self.assertEqual(self.testHist.face_color, 'silver')

    def test_edge_color_is_str(self):
        self.assertEqual(self.testHist.edge_color, 'k')

    def test_bins_is_int(self):
        self.assertEqual(self.testHist.number_of_bins, 20)

    def test_alpha_is_default(self):
        self.assertEqual(self.testHist.alpha, "default")

    def test_hist_type_is_str(self):
        self.assertEqual(self.testHist.hist_type, "default")


class TestHlines(unittest.TestCase):
    def setUp(self):
        self.testHlines = Hlines(1, 0, 1, 'Test Hlines')
        # _, self.testAxes = subplots()
        # self.testHlines.plot_curve(self.testAxes)
        # plt.close('all')

    def test_y_is_list_ndarray_float_int(self):
        self.assertIsInstance(self.testHlines.y, list | ndarray | float | int)

    def test_xmin_is_list_ndarray_float_int(self):
        self.assertIsInstance(self.testHlines.xmin, list | ndarray | float | int)

    def test_ymin_is_list_ndarray_float_int(self):
        self.assertIsInstance(self.testHlines.xmin, list | ndarray | float | int)

    def test_colors_is_str_list_or_none(self):
        self.assertIsInstance(self.testHlines.colors, list | str | None)

    def test_linestyles_is_str_list_or_none(self):
        self.assertIsInstance(self.testHlines.line_styles, list | str | None)

    def test_label_is_str(self):
        self.assertIsInstance(self.testHlines.label, str)


class TestVlines(unittest.TestCase):
    def setUp(self):
        self.testVlines = Vlines(x=[4,5,6,7], ymin=0, ymax=1, label="Test Vlines")
        # _, self.testAxes = subplots()
        # self.testVlines.plot_curve(self.testAxes)
        # plt.close('all')

    def test_x_is_list_ndarray_float_int(self):
        self.assertListEqual(self.testVlines.x, [4,5,6,7])

    def test_ymin_is_list_ndarray_float_int(self):
        self.assertEqual(self.testVlines.ymin, 0)

    def test_ymax_is_list_ndarray_float_int(self):
        self.assertEqual(self.testVlines.ymax, 1)

    def test_colors_is_default(self):
        self.assertEqual(self.testVlines.colors, "default")

    def test_linestyles_is_str_list_or_none(self):
        self.assertIsInstance(self.testVlines.line_styles, list | str | None)

    def test_label_is_str(self):
        self.assertEqual(self.testVlines.label, "Test Vlines")


class TestDashed(unittest.TestCase):
    def setUp(self):
        x = linspace(0, 3*pi, 200)
        self.testDashed = Dashed(x, sin(x), 'Test Dashed', color='k')

    def test_dashed_is_curve(self):
        self.assertIsInstance(self.testDashed, Curve)
    
    def test_color_init(self):
        self.assertEqual(self.testDashed.color, "k")
    
    def test_line_width_is_default(self):
        self.assertEqual(self.testDashed.line_width, "default")