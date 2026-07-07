import unittest

from graphinglib import INHERIT
from graphinglib.inherit import resolved

from matplotlib import pyplot as plt
from numpy import linspace, pi, sin

from graphinglib.data_plotting_1d import Curve
from graphinglib.figure import Figure, TwinAxis
from graphinglib.file_manager import FileLoader
from graphinglib.graph_elements import GraphingException


class TestFigure(unittest.TestCase):
    def setUp(self):
        x = linspace(0, 3 * pi, 200)
        self.testFigure = Figure(figure_style="plain")
        self.testCurve = Curve(x, sin(x), "Test Curve", color="k")
        self.plainDefaults = FileLoader("plain").load()
        self.horribleDefaults = FileLoader("horrible").load()

    def test_curves_is_list(self):
        self.assertIsInstance(self.testFigure._elements, list)

    def test_labels_is_list(self):
        self.assertIsInstance(self.testFigure._labels, list)

    def test_handles_is_list(self):
        self.assertIsInstance(self.testFigure._handles, list)

    def test_add_element(self):
        self.testFigure.add_elements(self.testCurve)
        self.assertIs(self.testFigure._elements[0], self.testCurve)
        # Test if adding multiple elements works
        other_curve = Curve(
            self.testCurve._x_data, self.testCurve._y_data, "Other Curve"
        )
        other_other_curve = Curve(
            self.testCurve._x_data, self.testCurve._y_data, "Other Other Curve"
        )
        self.testFigure.add_elements(other_curve, other_other_curve)
        self.assertTrue(len(self.testFigure._elements), 3)

    def test_copy_and_copy_with(self):
        copied = self.testFigure.copy()
        self.assertIsNot(copied, self.testFigure)

        modified = self.testFigure.copy_with(title="New Title")
        self.assertEqual(modified.title, "New Title")
        self.assertEqual(self.testFigure.title, None)

    def test_copy_with_suggests_similar_property(self):
        with self.assertRaisesRegex(AttributeError, "Did you mean 'title'"):
            self.testFigure.copy_with(titel="New Title")

    def test_copy_with_rejects_private_property(self):
        with self.assertRaisesRegex(
            AttributeError, "has no public writable property '_title'"
        ):
            self.testFigure.copy_with(_title="New Title")

    def test_all_curves_plotted(self):
        self.testFigure.add_elements(self.testCurve)
        self.testFigure._prepare_figure()
        self.assertEqual(
            len(self.testFigure._axes.get_lines()), len(self.testFigure._elements)
        )
        plt.close("all")

    def test_raise_exception_if_no_curve_added(self):
        self.assertRaises(GraphingException, self.testFigure.show)

    def test_auto_assign_default_params(self):
        x = linspace(0, 3 * pi, 200)
        a_curve = Curve(x, sin(x), label="Test Curve")
        a_figure = Figure()
        a_figure.add_elements(a_curve)
        a_figure._default_params = self.plainDefaults
        a_figure._fill_in_missing_params(a_curve)
        self.assertEqual(a_curve._line_width, 2)

    def test_auto_assign_default_params_horrible(self):
        x = linspace(0, 3 * pi, 200)
        a_curve = Curve(x, sin(x), label="Test Curve")
        a_figure = Figure(figure_style="horrible")
        a_figure.add_elements(a_curve)
        a_figure._default_params = self.horribleDefaults
        a_figure._fill_in_missing_params(a_curve)
        self.assertEqual(a_curve._line_width, 10)

    def test_auto_assign_default_params_skip_predefined(self):
        x = linspace(0, 3 * pi, 200)
        a_curve = Curve(x, sin(x), label="Test Curve", line_width=3)
        a_figure = Figure()
        a_figure.add_elements(a_curve)
        a_figure._default_params = self.plainDefaults
        a_figure._fill_in_missing_params(a_curve)
        self.assertEqual(a_curve._line_width, 3)

    def test_assign_figure_params_horrible(self):
        a_figure = Figure(figure_style="horrible")
        a_figure.add_elements(self.testCurve)
        a_figure._default_params = self.horribleDefaults
        a_figure._fill_in_missing_params(a_figure)
        self.assertListEqual(list(resolved(a_figure._size)), [10, 7])

    def test_assign_figure_params_no_grid(self):
        a_figure = Figure(figure_style="plain")
        a_figure.set_grid()
        self.assertTrue(a_figure._show_grid)

    def test_xkcd_style_renders(self):
        a_figure = Figure(figure_style="xkcd")
        a_figure.add_elements(self.testCurve)
        a_figure._prepare_figure()
        self.assertEqual(plt.rcParams["path.sketch"], (1.0, 100.0, 2.0))
        plt.close("all")

    def test_element_defaults_are_reset(self):
        self.testCurve._line_width = INHERIT
        self.testFigure.add_elements(self.testCurve)
        self.testFigure._prepare_figure()
        self.assertEqual(self.testCurve._line_width, INHERIT)
        self.testFigure._fill_in_missing_params(self.testCurve)
        self.assertEqual(self.testCurve._line_width, 2)
        plt.close("all")

    def test_handles_and_labels_cleared(self):
        self.testFigure.add_elements(self.testCurve)
        self.testFigure._prepare_figure()
        self.assertEqual(len(self.testFigure._handles), 0)
        self.assertEqual(len(self.testFigure._labels), 0)
        plt.close("all")

    def test_handles_and_labels_added(self):
        self.testFigure.add_elements(self.testCurve)
        other_curve = Curve(
            self.testCurve._x_data, self.testCurve._y_data, "Other Curve"
        )
        self.testFigure.add_elements(other_curve)
        self.testFigure._prepare_figure()
        handles, labels = self.testFigure._axes.get_legend_handles_labels()
        self.assertEqual(len(handles), 2)
        self.assertListEqual(labels, ["Test Curve", "Other Curve"])
        # test if still ok when replotting
        self.testFigure._figure_style = "horrible"
        self.testFigure._prepare_figure()
        handles, labels = self.testFigure._axes.get_legend_handles_labels()
        self.assertEqual(len(handles), 2)
        self.assertListEqual(labels, ["Test Curve", "Other Curve"])
        plt.close("all")

    def test_fill_in_rc_params_gl(self):
        a_figure = Figure()
        a_figure.set_visual_params(legend_edge_color="red")
        # Get default params for dim style
        a_figure._default_params = {
            "rc_params": {
                "axes.grid": False,
                "axes.facecolor": "dimgrey",
                "legend.edgecolor": "blue",
            }
        }
        # Fill in rc params
        a_figure._fill_in_rc_params()
        # Check axes fill color is updated
        self.assertEqual(plt.rcParams["axes.grid"], False)
        self.assertEqual(plt.rcParams["axes.facecolor"], "dimgrey")
        self.assertEqual(plt.rcParams["legend.edgecolor"], "red")  # Overridden by user

    def test_update_rc_params(self):
        a_figure = Figure()
        params = {
            "lines.linewidth": 2,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
        }
        a_figure.set_rc_params(params)
        self.assertDictEqual(a_figure._user_rc_dict, params)
        more_params = {
            "lines.linewidth": 3,
            "axes.grid": True,
        }
        a_figure.set_rc_params(more_params)
        resulting_params = {
            "lines.linewidth": 3,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "axes.grid": True,
        }
        self.assertDictEqual(a_figure._user_rc_dict, resulting_params)

    def test_update_rc_params_reset(self):
        a_figure = Figure()
        params = {
            "lines.linewidth": 2,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
        }
        a_figure.set_rc_params(params)
        a_figure.set_rc_params({"lines.linewidth": 3}, reset=True)
        self.assertDictEqual(a_figure._user_rc_dict, {"lines.linewidth": 3})

    def test_customize_visual_style(self):
        a_figure = Figure()
        a_figure.set_visual_params(figure_face_color="red", axes_face_color="blue")
        self.assertDictEqual(
            a_figure._user_rc_dict,
            {
                "figure.facecolor": "red",
                "axes.facecolor": "blue",
            },
        )
        a_figure.set_visual_params(axes_face_color="yellow", x_tick_color="green")
        self.assertDictEqual(
            a_figure._user_rc_dict,
            {
                "figure.facecolor": "red",
                "axes.facecolor": "yellow",
                "xtick.color": "green",
            },
        )

    def test_customize_visual_style_reset(self):
        a_figure = Figure()
        a_figure.set_visual_params(figure_face_color="red", axes_face_color="blue")
        a_figure.set_visual_params(axes_face_color="yellow", x_tick_color="green")
        a_figure.set_visual_params(reset=True)
        self.assertDictEqual(a_figure._user_rc_dict, {})

    def test_matplotlib_style_functional(self):
        a_figure = Figure(figure_style="matplotlib")
        a_figure.add_elements(self.testCurve)
        a_figure._prepare_figure()
        plt.close("all")

    def test_matplotlib_style_renders_every_element_type(self):
        # Under a native matplotlib style, style resolution is skipped and elements
        # plot with INHERIT still stored in their attributes. This is a regression
        # test for a family of crashes on that path (Table, Scatter with intensity
        # colors, VectorField, capped Line, Contour, multi-value Hlines/Vlines).
        import numpy as np

        import graphinglib as gl

        x = linspace(0.5, 9.5, 40)
        error_curve = gl.Curve(x, sin(x))
        error_curve.add_error_curves(y_error=np.full(40, 0.1))
        area_curve = gl.Curve(x, sin(x))
        area_curve.get_area_between(1, 5, fill_between=True)
        errorbar_scatter = gl.Scatter(x, sin(x))
        errorbar_scatter.add_errorbars(x_error=0.1, y_error=0.1)
        pdf_histogram = gl.Histogram(np.random.normal(5, 1, 200), 12)
        pdf_histogram.add_pdf(show_mean=True, show_std=True)
        arrow_text = gl.Text(3, -0.5, "hello")
        arrow_text.add_arrow((4, 0.5))
        elements = [
            gl.Curve(x, sin(x)),
            error_curve,
            area_curve,
            gl.Scatter(x, sin(x), face_color=linspace(0, 1, 40).tolist()),
            errorbar_scatter,
            gl.Histogram(np.random.normal(5, 1, 200), 12),
            pdf_histogram,
            gl.Hlines([0.5, 1.0], 0, 10),
            gl.Vlines([4.0, 5.0], -1, 1),
            gl.Hlines(0.75, 0, 10),
            gl.Vlines(4.5, -1, 1),
            gl.Point(2, 0.5),
            gl.Text(3, -0.5, "hello"),
            arrow_text,
            gl.Table([["1", "2"], ["3", "4"]]),
            gl.Heatmap(np.random.rand(5, 5)),
            gl.Contour(*np.meshgrid(np.arange(8), np.arange(8)), np.random.rand(8, 8)),
            gl.VectorField(
                *np.meshgrid(np.arange(5), np.arange(5)),
                np.ones((5, 5)),
                np.ones((5, 5)),
            ),
            gl.Stream(
                *np.meshgrid(linspace(0, 10, 30), linspace(0, 10, 30)),
                np.ones((30, 30)),
                np.ones((30, 30)),
            ),
            gl.Circle(5, 0, 1),
            gl.Rectangle(1, 1, 2, 1),
            gl.Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
            gl.Arrow((0, 0), (1, 1)),
            gl.Line((0, 0), (2, 2), capped_line=True),
            gl.FitFromPolynomial(gl.Scatter(x, 2 * x + 1), 1),
        ]
        a_figure = Figure(figure_style="matplotlib")
        a_figure.add_elements(*elements)
        a_figure._prepare_figure()
        plt.close("all")

    def test_create_twin_axis(self):
        a_figure = Figure()
        a_figure.add_elements(self.testCurve)
        twin_axis = a_figure.create_twin_axis()
        self.assertEqual(twin_axis._is_y, True)
        self.assertEqual(twin_axis._label, None)
        self.assertEqual(twin_axis._log_scale, False)
        self.assertEqual(twin_axis, a_figure._twin_y_axis)

    def test_aspect_ratio(self):
        a_figure = Figure(aspect_ratio=1.5, figure_style="plain")
        self.assertEqual(a_figure._aspect_ratio, 1.5)
        a_figure.add_elements(self.testCurve)
        a_figure._prepare_figure()
        self.assertEqual(a_figure._axes.get_aspect(), 1.5)

        a_figure = Figure(aspect_ratio="equal", figure_style="plain")
        self.assertEqual(a_figure._aspect_ratio, "equal")
        a_figure.add_elements(self.testCurve)
        a_figure._prepare_figure()
        self.assertEqual(a_figure._axes.get_aspect(), 1)

    def test_aspect_ratio_invalid_init(self):
        with self.assertRaises(GraphingException):
            Figure(aspect_ratio="nope", figure_style="plain")
        with self.assertRaises(GraphingException):
            Figure(aspect_ratio=-1.0, figure_style="plain")

    def test_aspect_ratio_setter_validation(self):
        a_figure = Figure(figure_style="plain")
        with self.assertRaises(GraphingException):
            a_figure.aspect_ratio = "nope"
        with self.assertRaises(GraphingException):
            a_figure.aspect_ratio = -1.0


class TestTwinAxis(unittest.TestCase):
    def test_init(self):
        twin = TwinAxis()
        self.assertEqual(twin._is_y, True)
        self.assertEqual(twin._label, None)
        self.assertEqual(twin._log_scale, False)
        self.assertEqual(twin._elements, [])
        self.assertEqual(twin._labels, [])
        self.assertEqual(twin._handles, [])
        self.assertEqual(twin._custom_ticks, False)
        self.assertEqual(twin._figure_style, None)
        self.assertEqual(twin._default_params, None)
        self.assertEqual(twin._tick_color, None)
        self.assertEqual(twin._axes_edge_color, None)
        self.assertEqual(twin._axes_label_color, None)

        twin = TwinAxis(is_y=False, label="Test", log_scale=True)
        self.assertEqual(twin._is_y, False)
        self.assertEqual(twin._label, "Test")
        self.assertEqual(twin._log_scale, True)

    def test_add_element(self):
        twin = TwinAxis()
        curve = Curve([1, 2, 3], [1, 2, 3])
        twin.add_elements(curve)
        self.assertEqual(twin._elements[0], curve)

    def test_copy_and_copy_with(self):
        twin = TwinAxis(label="A", log_scale=False)
        copied = twin.copy()
        self.assertIsNot(copied, twin)

        modified = twin.copy_with(label="B")
        self.assertEqual(modified.label, "B")
        self.assertEqual(twin.label, "A")

    def test_copy_with_rejects_read_only_property(self):
        twin = TwinAxis()
        with self.assertRaisesRegex(AttributeError, "read-only property"):
            twin.copy_with(axis_lim=(0, 1))

    def test_customized_visual_style(self):
        twin = TwinAxis()
        twin.set_visual_params(tick_color="red", axes_edge_color="blue")
        self.assertEqual(twin._tick_color, "red")
        self.assertEqual(twin._axes_edge_color, "blue")

    def test_set_ticks(self):
        twin = TwinAxis()
        twin.set_ticks([1, 2, 3], ["a", "b", "c"])
        self.assertEqual(twin._custom_ticks, True)
        self.assertEqual(twin._ticks, [1, 2, 3])
        self.assertEqual(twin._ticklabels, ["a", "b", "c"])

    def test_prepare_twin_axes(self):
        twin = TwinAxis()
        axes = plt.axes()
        twin.add_elements(Curve([1, 2, 3], [1, 2, 3], label="Test"))
        labels, handles = twin._prepare_twin_axis(axes, True, {}, "plain")
        self.assertEqual(len(labels), 1)
        self.assertEqual(len(handles), 1)
        plt.close("all")

    def test_fill_in_missing_params(self):
        default_params = {
            "Curve": {
                "_line_width": 2,
                "_color": "k",
                "_line_style": "-",
                "_alpha": 1.0,
            }
        }
        twin = TwinAxis()
        twin._default_params = default_params

        curve = Curve([1, 2, 3], [1, 2, 3], label="Test")
        twin._fill_in_missing_params(curve)
        self.assertEqual(curve._line_width, 2)
        self.assertEqual(curve._color, "k")
        self.assertEqual(curve._line_style, "-")

    def test_reset_params_to_default(self):
        curve = Curve(
            [1, 2, 3], [1, 2, 3], label="Test", line_width=3, color="r", line_style="--"
        )
        params_to_reset = ["line_width", "color", "line_style"]
        twin = TwinAxis()
        twin._reset_params_to_default(curve, params_to_reset)


if __name__ == "__main__":
    unittest.main()
