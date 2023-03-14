import unittest
from Graphing import Figure, Curve, Scatter, GraphingException, Histogram
from Fits import FitFromPolynomial, FitFromSine
from numpy import linspace, pi, sin, ndarray
from matplotlib.axes import Axes
from matplotlib.pyplot import subplots
from random import random
import numpy as np


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
        self.testHist = Histogram([random() for _ in range(100)], 20, 'Random Distribution', 'silver', 'k')
    
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
    
    def test_first_degree_cov(self):
        self.assertIsInstance(self.fit_first_degree.standard_deviation, np.ndarray)
        self.assertEqual(self.fit_first_degree.standard_deviation.shape, (2,))
    
    def test_first_degree_string(self):
        self.assertEqual(str(self.fit_first_degree), "f(x) = 3.0x^1 + 2.0")
    
    def test_first_degree_functions(self):
        self.assertEqual(round(self.fit_first_degree.function(5), 5), 17)
    
    def test_second_degree_cov(self):
        self.assertIsInstance(self.fit_second_degree.cov_matrix, np.ndarray)
        self.assertEqual(self.fit_second_degree.cov_matrix.shape, (3,3))
    
    def test_second_degree_cov(self):
        self.assertIsInstance(self.fit_second_degree.cov_matrix, np.ndarray)
        self.assertEqual(self.fit_second_degree.cov_matrix.shape, (3,3))
    
    def test_second_degree_cov(self):
        self.assertIsInstance(self.fit_second_degree.standard_deviation, np.ndarray)
        self.assertEqual(self.fit_second_degree.standard_deviation.shape, (3,))
    
    def test_second_degree_coeffs(self):
        self.assertListEqual([round(i, 5) for i in list(self.fit_second_degree.coeffs)], [-2,-3,4])
    
    def test_second_degree_string(self):
        self.assertEqual(str(self.fit_second_degree), "f(x) = 4.0x^2 - 3.0x^1 - 2.0")
    
    def test_second_degree_functions(self):
        self.assertEqual(round(self.fit_second_degree.function(5), 5), 83)

if __name__ == '__main__':
    unittest.main()