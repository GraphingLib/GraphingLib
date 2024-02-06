import unittest

from matplotlib import pyplot as plt
from numpy import linspace, pi, sin

from graphinglib.data_plotting_1d import Curve
from graphinglib.figure import Figure, TwinAxis
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

    def test_add_element(self):
        self.testFigure.add_element(self.testCurve)
        self.assertIs(self.testFigure._elements[0], self.testCurve)
        # Test if adding multiple elements works
        other_curve = Curve(self.testCurve.x_data, self.testCurve.y_data, "Other Curve")
        other_other_curve = Curve(
            self.testCurve.x_data, self.testCurve.y_data, "Other Other Curve"
        )
        self.testFigure.add_element(other_curve, other_other_curve)
        self.assertTrue(len(self.testFigure._elements), 3)

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

    def test_assign_figure_params_no_grid(self):
        a_figure = Figure(figure_style="horrible", show_grid=False)
        self.assertFalse(a_figure.show_grid)

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

    def test_fill_in_rc_params_gl(self):
        a_figure = Figure()
        a_figure.customize_visual_style(legend_edge_color="red")
        # Get default params for dim style
        a_figure.default_params = {
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
        a_figure.update_rc_params(params)
        self.assertDictEqual(a_figure._user_rc_dict, params)
        more_params = {
            "lines.linewidth": 3,
            "axes.grid": True,
        }
        a_figure.update_rc_params(more_params)
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
        a_figure.update_rc_params(params)
        a_figure.update_rc_params({"lines.linewidth": 3}, reset=True)
        self.assertDictEqual(a_figure._user_rc_dict, {"lines.linewidth": 3})

    def test_customize_visual_style(self):
        a_figure = Figure()
        a_figure.customize_visual_style(
            figure_face_color="red", axes_face_color="blue", grid_line_style="dashed"
        )
        self.assertDictEqual(
            a_figure._user_rc_dict,
            {
                "figure.facecolor": "red",
                "axes.facecolor": "blue",
                "grid.linestyle": "dashed",
            },
        )
        a_figure.customize_visual_style(
            axes_face_color="yellow", grid_line_style="dotted", x_tick_color="green"
        )
        self.assertDictEqual(
            a_figure._user_rc_dict,
            {
                "figure.facecolor": "red",
                "axes.facecolor": "yellow",
                "grid.linestyle": "dotted",
                "xtick.color": "green",
            },
        )

    def test_customize_visual_style_reset(self):
        a_figure = Figure()
        a_figure.customize_visual_style(
            figure_face_color="red", axes_face_color="blue", grid_line_style="dashed"
        )
        a_figure.customize_visual_style(
            axes_face_color="yellow", grid_line_style="dotted", x_tick_color="green"
        )
        a_figure.customize_visual_style(reset=True)
        self.assertDictEqual(a_figure._user_rc_dict, {})

    def test_matplotlib_style_functional(self):
        a_figure = Figure(figure_style="matplotlib")
        a_figure.add_element(self.testCurve)
        a_figure._prepare_figure()

    def test_create_twin_axis(self):
        a_figure = Figure()
        a_figure.add_element(self.testCurve)
        twin_axis = a_figure.create_twin_axis()
        self.assertEqual(twin_axis.is_y, True)
        self.assertEqual(twin_axis.label, None)
        self.assertEqual(twin_axis.log_scale, False)
        self.assertEqual(twin_axis, a_figure._twin_y_axis)


class TestTwinAxis(unittest.TestCase):
    def test_init(self):
        twin = TwinAxis()
        self.assertEqual(twin.is_y, True)
        self.assertEqual(twin.label, None)
        self.assertEqual(twin.log_scale, False)
        self.assertEqual(twin._elements, [])
        self.assertEqual(twin._labels, [])
        self.assertEqual(twin._handles, [])
        self.assertEqual(twin._custom_ticks, False)
        self.assertEqual(twin.figure_style, None)
        self.assertEqual(twin.default_params, None)
        self.assertEqual(twin.tick_color, None)
        self.assertEqual(twin.axes_edge_color, None)
        self.assertEqual(twin.axes_label_color, None)

        twin = TwinAxis(is_y=False, label="Test", log_scale=True)
        self.assertEqual(twin.is_y, False)
        self.assertEqual(twin.label, "Test")
        self.assertEqual(twin.log_scale, True)

    def test_add_element(self):
        twin = TwinAxis()
        curve = Curve([1, 2, 3], [1, 2, 3])
        twin.add_element(curve)
        self.assertEqual(twin._elements[0], curve)

    def test_customized_visual_style(self):
        twin = TwinAxis()
        twin.customize_visual_style(tick_color="red", axes_edge_color="blue")
        self.assertEqual(twin.tick_color, "red")
        self.assertEqual(twin.axes_edge_color, "blue")

    def test_set_ticks(self):
        twin = TwinAxis()
        twin.set_ticks([1, 2, 3], ["a", "b", "c"])
        self.assertEqual(twin._custom_ticks, True)
        self.assertEqual(twin._ticks, [1, 2, 3])
        self.assertEqual(twin._ticklabels, ["a", "b", "c"])

    def test_prepare_twin_axes(self):
        twin = TwinAxis()
        axes = plt.axes()
        twin.add_element(Curve([1, 2, 3], [1, 2, 3], label="Test"))
        labels, handles = twin._prepare_twin_axis(axes, True, {}, "plain")
        self.assertEqual(len(labels), 1)
        self.assertEqual(len(handles), 1)

    def test_fill_in_missing_params(self):
        default_params = {
            "Curve": {
                "line_width": 2,
                "color": "k",
                "line_style": "-",
            }
        }
        twin = TwinAxis()
        twin.default_params = default_params

        curve = Curve([1, 2, 3], [1, 2, 3], label="Test")
        twin._fill_in_missing_params(curve)
        self.assertEqual(curve.line_width, 2)
        self.assertEqual(curve.color, "k")
        self.assertEqual(curve.line_style, "-")

    def test_reset_params_to_default(self):
        curve = Curve(
            [1, 2, 3], [1, 2, 3], label="Test", line_width=3, color="r", line_style="--"
        )
        params_to_reset = ["line_width", "color", "line_style"]
        twin = TwinAxis()
        twin._reset_params_to_default(curve, params_to_reset)


if __name__ == "__main__":
    unittest.main()
