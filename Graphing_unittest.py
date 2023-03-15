import unittest
from Graphing import *
from numpy import linspace, pi, sin, ndarray
from matplotlib.axes import Axes
from matplotlib.pyplot import subplots
from random import random


class TestFigure(unittest.TestCase):
    def setUp(self):
        x = linspace(0, 3*pi, 200)
        self.testFigure = Figure()
        self.testCurve = Curve(x, sin(x), 'k', 'Test Curve')

    def test_curves_is_list(self):
        self.assertIsInstance(self.testFigure.curves, list)

    def test_labels_is_list(self):
        self.assertIsInstance(self.testFigure.labels, list)

    def test_handles_is_list(self):
        self.assertIsInstance(self.testFigure.handles, list)

    def test_curve_is_added(self):
        self.testFigure.add_curve(self.testCurve)
        self.assertIs(self.testFigure.curves[0], self.testCurve)

    def test_axes_is_Axes(self):
        self.assertIsInstance(self.testFigure.axes, Axes)

    def test_all_curves_plotted(self):
        self.testFigure.add_curve(self.testCurve)
        self.testFigure.generate_figure(test=True)
        self.assertEqual(len(self.testFigure.axes.get_lines()), len(self.testFigure.curves))

    def test_raise_exception_if_no_curve_added(self):
        self.assertRaises(GraphingException, self.testFigure.generate_figure)


class TestCurve(unittest.TestCase):
    def setUp(self):
        x = linspace(0, 3*pi, 200)
        self.testCurve = Curve(x, sin(x), 'k', 'Test Curve')
        _, self.testAxes = subplots()
        self.testCurve.plot_curve(self.testAxes)

    def test_xdata_is_list_or_ndarray(self):
        self.assertIsInstance(self.testCurve.xdata, list | ndarray)

    def test_ydata_is_list_or_ndarray(self):
        self.assertIsInstance(self.testCurve.ydata, list | ndarray)

    def test_color_is_str(self):
        self.assertIsInstance(self.testCurve.color, str)

    def test_label_is_str(self):
        self.assertIsInstance(self.testCurve.label, str)

    def test_color_is_changed(self):
        self.testCurve.set_color('r')
        self.assertEqual(self.testCurve.color, 'r')

    def test_curve_is_plotted(self):
        self.assertEqual(len(self.testAxes.get_lines()), 1)


class TestScatter(unittest.TestCase):
    def setUp(self):
        x = linspace(0, 3*pi, 200)
        self.testScatter = Scatter(x, sin(x), 'k', 'Test Curve')
        _, self.testAxes = subplots()
        self.testScatter.plot_curve(self.testAxes)

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
        self.assertEqual(len(self.testAxes.collections), 1)


class TestHisttogram(unittest.TestCase):
    def setUp(self):
        self.testHist = Histogram([random() for _ in range(100)], 20,
                                    'Random Distribution', 'silver', 'k')
        _, self.testAxes = subplots()
        self.testHist.plot_curve(self.testAxes)
        plt.close('all')

    def test_label_is_str(self):
        self.assertIsInstance(self.testHist.label, str)

    def test_xdata_is_list_or_ndarray(self):
        self.assertIsInstance(self.testHist.xdata, list | ndarray)

    def test_face_color_is_str(self):
        self.assertIsInstance(self.testHist.face_color, str)

    def test_edge_color_is_str(self):
        self.assertIsInstance(self.testHist.edge_color, str)

    def test_bins_is_int(self):
        self.assertIsInstance(self.testHist.bins, int)

    def test_alpha_is_float(self):
        self.assertIsInstance(self.testHist.alpha, float)

    def test_hist_type_is_str(self):
        self.assertIsInstance(self.testHist.hist_type, str)


class TestHlines(unittest.TestCase):
    def setUp(self):
        self.testHlines = Hlines(1, 0, 1, 'Test Hlines')
        _, self.testAxes = subplots()
        self.testHlines.plot_curve(self.testAxes)
        plt.close('all')

    def test_y_is_list_ndarray_float_int(self):
        self.assertIsInstance(self.testHlines.y, list | ndarray | float | int)

    def test_xmin_is_list_ndarray_float_int(self):
        self.assertIsInstance(self.testHlines.xmin, list | ndarray | float | int)

    def test_ymin_is_list_ndarray_float_int(self):
        self.assertIsInstance(self.testHlines.xmin, list | ndarray | float | int)

    def test_colors_is_str_list_or_none(self):
        self.assertIsInstance(self.testHlines.colors, list | str | None)

    def test_linestyles_is_str_list_or_none(self):
        self.assertIsInstance(self.testHlines.linestyles, list | str | None)

    def test_label_is_str(self):
        self.assertIsInstance(self.testHlines.label, str)


class TestVlines(unittest.TestCase):
    def setUp(self):
        self.testVlines = Vlines(1, 0, 1, 'Test Vlines')
        _, self.testAxes = subplots()
        self.testVlines.plot_curve(self.testAxes)
        plt.close('all')

    def test_y_is_list_ndarray_float_int(self):
        self.assertIsInstance(self.testVlines.x, list | ndarray | float | int)

    def test_xmin_is_list_ndarray_float_int(self):
        self.assertIsInstance(self.testVlines.ymin, list | ndarray | float | int)

    def test_ymin_is_list_ndarray_float_int(self):
        self.assertIsInstance(self.testVlines.ymin, list | ndarray | float | int)

    def test_colors_is_str_list_or_none(self):
        self.assertIsInstance(self.testVlines.colors, list | str | None)

    def test_linestyles_is_str_list_or_none(self):
        self.assertIsInstance(self.testVlines.linestyles, list | str | None)

    def test_label_is_str(self):
        self.assertIsInstance(self.testVlines.label, str)


class TestDashed(unittest.TestCase):
    def setUp(self):
        x = linspace(0, 3*pi, 200)
        self.testDashed = Dashed(x, sin(x), 'k', 'Test Dashed')

    def test_dashed_is_curve(self):
        self.assertIsInstance(self.testDashed, Curve)


if __name__ == '__main__':
    unittest.main()