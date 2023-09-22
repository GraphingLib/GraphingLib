import unittest

from matplotlib.pyplot import Axes
from numpy import linspace, pi, sin

from graphinglib.data_plotting_1d import Curve
from graphinglib.figure import Figure
from graphinglib.graph_elements import GraphingException


class TestFigure(unittest.TestCase):
    def setUp(self):
        x = linspace(0, 3 * pi, 200)
        self.testFigure = Figure()
        self.testCurve = Curve(x, sin(x), "Test Curve", color="k")

    def test_curves_is_list(self):
        self.assertIsInstance(self.testFigure._elements, list)

    def test_labels_is_list(self):
        self.assertIsInstance(self.testFigure._labels, list)

    def test_handles_is_list(self):
        self.assertIsInstance(self.testFigure._handles, list)

    def test_curve_is_added(self):
        self.testFigure.add_element(self.testCurve)
        self.assertIs(self.testFigure._elements[0], self.testCurve)

    def test_all_curves_plotted(self):
        self.testFigure.add_element(self.testCurve)
        self.testFigure._prepare_figure()
        self.assertEqual(
            len(self.testFigure._axes.get_lines()), len(self.testFigure._elements)
        )

    def test_raise_exception_if_no_curve_added(self):
        self.assertRaises(GraphingException, self.testFigure.display)

    def test_auto_assign_default_params(self):
        x = linspace(0, 3 * pi, 200)
        a_curve = Curve(x, sin(x), label="Test Curve")
        a_figure = Figure()
        a_figure.add_element(a_curve)
        a_figure._fill_in_missing_params(a_curve)
        self.assertEqual(a_curve.line_width, 1)

    def test_auto_assign_default_params_horrible(self):
        x = linspace(0, 3 * pi, 200)
        a_curve = Curve(x, sin(x), label="Test Curve")
        a_figure = Figure(figure_style="horrible")
        a_figure.add_element(a_curve)
        a_figure._fill_in_missing_params(a_curve)
        self.assertEqual(a_curve.line_width, 10)

    def test_auto_assign_default_params_skip_predefined(self):
        x = linspace(0, 3 * pi, 200)
        a_curve = Curve(x, sin(x), label="Test Curve", line_width=3)
        a_figure = Figure()
        a_figure.add_element(a_curve)
        a_figure._fill_in_missing_params(a_curve)
        self.assertEqual(a_curve.line_width, 3)

    def test_assign_figure_params_horrible(self):
        a_figure = Figure(figure_style="horrible")
        self.assertListEqual(list(a_figure.size), [10, 7])

    def test_assign_figure_params_not_boxed(self):
        a_figure = Figure(figure_style="horrible", legend_is_boxed=False)
        self.assertFalse(a_figure.legend_is_boxed)
