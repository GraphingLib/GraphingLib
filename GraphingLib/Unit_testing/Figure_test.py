import unittest
import sys

sys.path.insert(0,'..')

from Figure import *
from numpy import linspace, sin, pi
from matplotlib.pyplot import Axes


class TestFigure(unittest.TestCase):
    def setUp(self):
        x = linspace(0, 3*pi, 200)
        self.testFigure = Figure()
        self.testCurve = Curve(x, sin(x), 'Test Curve', color='k')

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
    
    def test_auto_assign_default_params(self):
        x = linspace(0, 3*pi, 200)
        a_curve = Curve(x, sin(x), label='Test Curve')
        a_figure = Figure()
        a_figure.add_curve(a_curve)
        a_figure.fill_in_missing_params(a_curve)
        self.assertEqual(a_curve.line_width, 1)
    
    def test_auto_assign_default_params_weird(self):
        x = linspace(0, 3*pi, 200)
        a_curve = Curve(x, sin(x), label='Test Curve')
        a_figure = Figure(figure_style='weird')
        a_figure.add_curve(a_curve)
        a_figure.fill_in_missing_params(a_curve)
        self.assertEqual(a_curve.line_width, 10)
    
    def test_auto_assign_default_params_skip_predefined(self):
        x = linspace(0, 3*pi, 200)
        a_curve = Curve(x, sin(x), label='Test Curve', line_width=3)
        a_figure = Figure()
        a_figure.add_curve(a_curve)
        a_figure.fill_in_missing_params(a_curve)
        self.assertEqual(a_curve.line_width, 3)
    
    def test_assign_figure_params_weird(self):
        a_figure = Figure(figure_style='weird')
        self.assertListEqual(a_figure.size, [10,7])
    
    def test_assign_figure_params_not_boxed(self):
        a_figure = Figure(figure_style='weird', legend_is_boxed=False)
        self.assertFalse(a_figure.legend_is_boxed)