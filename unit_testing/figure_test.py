import unittest

from numpy import linspace, pi, sin

from graphinglib.data_plotting_1d import Curve
from graphinglib.figure import Figure
from graphinglib.file_manager import FileLoader
from graphinglib.graph_elements import GraphingException


class TestFigure(unittest.TestCase):
    def setUp(self):
        x = linspace(0, 3 * pi, 200)
        self.testFigure = Figure()
        self.testCurve = Curve(x, sin(x), "Test Curve", color="k")
        self.plainDefaults = FileLoader("plain").load()
        self.horribleDefaults = FileLoader("horrible").load()

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
        a_figure.default_params = self.plainDefaults
        a_figure._fill_in_missing_params(a_curve)
        self.assertEqual(a_curve.line_width, 2)

    def test_auto_assign_default_params_horrible(self):
        x = linspace(0, 3 * pi, 200)
        a_curve = Curve(x, sin(x), label="Test Curve")
        a_figure = Figure(figure_style="horrible")
        a_figure.add_element(a_curve)
        a_figure.default_params = self.horribleDefaults
        a_figure._fill_in_missing_params(a_curve)
        self.assertEqual(a_curve.line_width, 10)

    def test_auto_assign_default_params_skip_predefined(self):
        x = linspace(0, 3 * pi, 200)
        a_curve = Curve(x, sin(x), label="Test Curve", line_width=3)
        a_figure = Figure()
        a_figure.add_element(a_curve)
        a_figure.default_params = self.plainDefaults
        a_figure._fill_in_missing_params(a_curve)
        self.assertEqual(a_curve.line_width, 3)

    def test_assign_figure_params_horrible(self):
        a_figure = Figure(figure_style="horrible")
        a_figure.add_element(self.testCurve)
        a_figure.default_params = self.horribleDefaults
        a_figure._fill_in_missing_params(a_figure)
        self.assertListEqual(list(a_figure.size), [10, 7])

    def test_assign_figure_params_not_boxed(self):
        a_figure = Figure(figure_style="horrible", legend_is_boxed=False)
        self.assertFalse(a_figure.legend_is_boxed)

    def test_element_defaults_are_reset(self):
        self.testCurve.line_width = "default"
        self.testFigure.add_element(self.testCurve)
        self.testFigure._prepare_figure()
        self.assertEqual(self.testCurve.line_width, "default")
        self.testFigure._fill_in_missing_params(self.testCurve)
        self.assertEqual(self.testCurve.line_width, 2)

    def test_handles_and_labels_cleared(self):
        self.testFigure.add_element(self.testCurve)
        self.testFigure._prepare_figure()
        self.assertEqual(len(self.testFigure._handles), 0)
        self.assertEqual(len(self.testFigure._labels), 0)

    def test_handles_and_labels_added(self):
        self.testFigure.add_element(self.testCurve)
        other_curve = Curve(self.testCurve.x_data, self.testCurve.y_data, "Other Curve")
        self.testFigure.add_element(other_curve)
        self.testFigure._prepare_figure()
        handles, labels = self.testFigure._axes.get_legend_handles_labels()
        self.assertEqual(len(handles), 2)
        self.assertListEqual(labels, ["Test Curve", "Other Curve"])
        # test if still ok when replotting
        self.testFigure.figure_style = "horrible"
        self.testFigure._prepare_figure()
        handles, labels = self.testFigure._axes.get_legend_handles_labels()
        self.assertEqual(len(handles), 2)
        self.assertListEqual(labels, ["Test Curve", "Other Curve"])
