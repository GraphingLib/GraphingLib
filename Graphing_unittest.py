import unittest
from random import random

import numpy as np
from Fits import FitFromPolynomial, FitFromSine
from Graphing import *
from matplotlib.axes import Axes
from matplotlib.pyplot import subplots
from numpy import linspace, ndarray, pi, sin


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
        self.assertIsInstance(self.testHist.number_of_bins, int)

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


class TestFitFromPolynomial(unittest.TestCase):
    def setUp(self):
        x = linspace(0, 3*pi, 200)
        self.scatter_first_degree = Scatter(x, 3*x + 2, 'k', 'Test Curve')
        self.scatter_second_degree = Scatter(x, 4*x**2 - 3*x - 2, 'k', 'Test Curve')
        self.fit_first_degree = FitFromPolynomial(self.scatter_first_degree, 1, 'k', 'First degree fit')
        self.fit_second_degree = FitFromPolynomial(self.scatter_second_degree, 2, 'k', 'Second degree fit')
    
    def test_first_degree_coeffs(self):
        self.assertListEqual([round(i, 5) for i in list(self.fit_first_degree.coeffs)], [2,3])
    
    def test_first_degree_cov(self):
        self.assertIsInstance(self.fit_first_degree.cov_matrix, np.ndarray)
        self.assertEqual(self.fit_first_degree.cov_matrix.shape, (2,2))
    
    def test_first_degree_std_dev(self):
        self.assertIsInstance(self.fit_first_degree.standard_deviation, np.ndarray)
        self.assertEqual(self.fit_first_degree.standard_deviation.shape, (2,))
    
    def test_first_degree_string(self):
        self.assertEqual(str(self.fit_first_degree), "3.0x^1 + 2.0")
    
    def test_first_degree_functions(self):
        self.assertAlmostEqual(self.fit_first_degree.function(5), 17)
    
    def test_second_degree_coeffs(self):
        self.assertListEqual([round(i, 5) for i in list(self.fit_second_degree.coeffs)], [-2,-3,4])
    
    def test_second_degree_cov(self):
        self.assertIsInstance(self.fit_second_degree.cov_matrix, np.ndarray)
        self.assertEqual(self.fit_second_degree.cov_matrix.shape, (3,3))
    
    def test_second_degree_std_dev(self):
        self.assertIsInstance(self.fit_second_degree.standard_deviation, np.ndarray)
        self.assertEqual(self.fit_second_degree.standard_deviation.shape, (3,))
    
    def test_second_degree_string(self):
        self.assertEqual(str(self.fit_second_degree), "4.0x^2 - 3.0x^1 - 2.0")
    
    def test_second_degree_functions(self):
        self.assertAlmostEqual(round(self.fit_second_degree.function(5), 5), 83)


class TestFitFromSine(unittest.TestCase):
    def setUp(self):
        x = linspace(0, 3*pi, 200)
        self.data = Scatter(x, 2*sin(3*x + 4) + 5 , 'k', 'Data')
        self.fit = FitFromSine(self.data, 'k', 'Sinusoidal fit', guesses=[2.09,3.1,4.2,5.2])
    
    def test_parameters(self):
        self.assertListEqual([self.fit.amplitude, self.fit.frequency_rad, self.fit.phase, self.fit.vertical_shift], [2,3,4,5])
    
    def test_cov(self):
        self.assertIsInstance(self.fit.cov_matrix, np.ndarray)
        self.assertEqual(self.fit.cov_matrix.shape, (4,4))
    
    def test_std_dev(self):
        self.assertIsInstance(self.fit.standard_deviation, np.ndarray)
        self.assertEqual(self.fit.standard_deviation.shape, (4,))
    
    def test_string(self):
        self.assertEqual(str(self.fit), "2.000 sin(3.000x + 4.000) + 5.000")
    
    def test_function(self):
        self.assertAlmostEqual(self.fit.function(17), 3.00048965)


if __name__ == '__main__':
    unittest.main()