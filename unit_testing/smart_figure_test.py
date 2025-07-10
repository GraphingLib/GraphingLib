import unittest
import os

import warnings

from matplotlib import use as matplotlib_use
matplotlib_use("Agg")  # Use non-GUI backend for tests

from matplotlib import pyplot as plt
from numpy import linspace, sin, pi, array

from graphinglib.file_manager import FileLoader
from graphinglib.smart_figure import SmartFigure
from graphinglib.data_plotting_1d import Curve, Scatter, Histogram
from graphinglib.data_plotting_2d import Heatmap, VectorField, Contour, Stream
from graphinglib.fits import FitFromFunction
from graphinglib.graph_elements import Plottable, GraphingException, Hlines, Vlines, Point, Text, Table
from graphinglib.shapes import Arrow, Line, Polygon, Circle, Rectangle


class DummyPlottable(Plottable):
    def __init__(self, label="dummy"):
        self.label = label
        self.handle = None
        self._x_data = [0, 1]
        self._y_data = [0, 1]

    def _plot_element(self, ax, z_order, cycle_color=None):
        self.handle, = ax.plot(self._x_data, self._y_data, label=self.label, color=cycle_color)

class TestSmartFigure(unittest.TestCase):
    def setUp(self):
        self.fig = SmartFigure()
        x = linspace(0, 3 * pi, 200)
        self.testCurve = Curve(x, sin(x), "Test Curve", color="k")
        self.plainDefaults = FileLoader("plain").load()
        self.horribleDefaults = FileLoader("horrible").load()

    def test_init_defaults(self):
        self.assertEqual(self.fig.num_rows, 1)
        self.assertEqual(self.fig.num_cols, 1)
        self.assertEqual(self.fig.figure_style, "default")
        self.assertIsInstance(self.fig, SmartFigure)

    def test_init_custom_args(self):
        elements = [[DummyPlottable(), DummyPlottable()], None, DummyPlottable(),
                    [None], [DummyPlottable(), DummyPlottable(), DummyPlottable()], []]
        fig = SmartFigure(num_rows=2, num_cols=3, x_label="X", y_label="Y", size=(8, 6), title="Test Figure",
                          x_lim=(0, 10), y_lim=(-5, 5), log_scale_x=True, log_scale_y=True, remove_axes=True,
                          aspect_ratio=1.5, remove_x_ticks=True, remove_y_ticks=True, reference_labels=False,
                          global_reference_label=True, reference_label_loc="inside", reference_label_start_index=3,
                          width_padding=0.5, height_padding=0, width_ratios=[1,2,3], height_ratios=[1,2], share_x=True,
                          share_y=True, projection="polar", general_legend=True, legend_loc="upper right",
                          legend_cols=2, show_legend=False, figure_style="dark", elements=elements)
        self.assertEqual(fig.num_rows, 2)
        self.assertEqual(fig.num_cols, 3)
        self.assertEqual(fig.x_label, "X")
        self.assertEqual(fig.y_label, "Y")
        self.assertEqual(fig.size, (8, 6))
        self.assertEqual(fig.title, "Test Figure")
        self.assertEqual(fig.x_lim, (0, 10))
        self.assertEqual(fig.y_lim, (-5, 5))
        self.assertTrue(fig.log_scale_x)
        self.assertTrue(fig.log_scale_y)
        self.assertTrue(fig.remove_axes)
        self.assertEqual(fig.aspect_ratio, 1.5)
        self.assertTrue(fig.remove_x_ticks)
        self.assertTrue(fig.remove_y_ticks)
        self.assertFalse(fig.reference_labels)
        self.assertTrue(fig.global_reference_label)
        self.assertEqual(fig.reference_label_loc, "inside")
        self.assertEqual(fig.reference_label_start_index, 3)
        self.assertEqual(fig.width_padding, 0.5)
        self.assertEqual(fig.height_padding, 0)
        self.assertEqual(fig.width_ratios, [1, 2, 3])
        self.assertEqual(fig.height_ratios, [1, 2])
        self.assertTrue(fig.share_x)
        self.assertTrue(fig.share_y)
        self.assertEqual(fig.projection, "polar")
        self.assertTrue(fig.general_legend)
        self.assertEqual(fig.legend_loc, "upper right")
        self.assertEqual(fig.legend_cols, 2)
        self.assertFalse(fig.show_legend)
        self.assertEqual(fig.figure_style, "dark")
        self.assertEqual(fig[0, 0], elements[0])
        self.assertEqual(fig[0, 1], [])
        self.assertEqual(fig[0, 2], [elements[2]])
        self.assertEqual(fig[1, 0], elements[3])
        self.assertEqual(fig[1, 1], elements[4])
        self.assertEqual(fig[1, 2], elements[5])

    def test_num_rows_and_num_cols(self):
        # Invalid types
        with self.assertRaises(TypeError):
            self.fig.num_rows = "a"
        with self.assertRaises(TypeError):
            self.fig.num_cols = "a"
        with self.assertRaises(TypeError):
            self.fig.num_rows = 0.5
        with self.assertRaises(TypeError):
            self.fig.num_cols = 0.5
        # Invalid values
        with self.assertRaises(ValueError):
            self.fig.num_rows = 0
        with self.assertRaises(ValueError):
            self.fig.num_cols = 0
        # Valid set/get
        self.fig.num_rows = 2
        self.fig.num_cols = 2
        self.assertEqual(self.fig.num_rows, 2)
        self.assertEqual(self.fig.num_cols, 2)
        # Changing shape with elements present
        new_fig = self.fig.copy_with(num_rows=2, num_cols=2)
        new_fig[:, :] = DummyPlottable()
        with self.assertRaises(GraphingException):
            new_fig.num_rows = 1
        with self.assertRaises(GraphingException):
            new_fig.num_cols = 1

    def test_size(self):
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.size = "big"
        with self.assertRaises(ValueError):
            self.fig.size = (1,)
        with self.assertRaises(ValueError):
            self.fig.size = (-1, 2)
        with self.assertRaises(ValueError):
            self.fig.size = (3, -2)
        # Valid
        self.fig.size = (5, 4)
        self.assertEqual(self.fig.size, (5, 4))

    def test_x_lim(self):
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.x_lim = 1
        with self.assertRaises(ValueError):
            self.fig.x_lim = (1, 2, 3)
        # Valid
        self.fig.x_lim = (0, 10)
        self.assertEqual(self.fig.x_lim, (0, 10))

    def test_y_lim(self):
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.y_lim = 1
        with self.assertRaises(ValueError):
            self.fig.y_lim = (1, 2, 3)
        # Valid
        self.fig.y_lim = (-5, 5)
        self.assertEqual(self.fig.y_lim, (-5, 5))

    def test_log_scale_x(self):
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.log_scale_x = "yes"
        with self.assertRaises(TypeError):
            self.fig.log_scale_x = 1
        # Valid
        self.fig.log_scale_x = True
        self.assertTrue(self.fig.log_scale_x)

    def test_log_scale_y(self):
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.log_scale_y = "yes"
        with self.assertRaises(TypeError):
            self.fig.log_scale_y = 1
        # Valid
        self.fig.log_scale_y = True
        self.assertTrue(self.fig.log_scale_y)

    def test_remove_axes(self):
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.remove_axes = "yes"
        with self.assertRaises(TypeError):
            self.fig.remove_axes = 1
        # Valid
        self.fig.remove_axes = True
        self.assertTrue(self.fig.remove_axes)

    def test_aspect_ratio(self):
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.aspect_ratio = "yes"
        with self.assertRaises(TypeError):
            self.fig.aspect_ratio = [1]
        with self.assertRaises(ValueError):
            self.fig.aspect_ratio = -1
        # Valid
        self.fig.aspect_ratio = "auto"
        self.assertEqual(self.fig.aspect_ratio, "auto")
        self.fig.aspect_ratio = "equal"
        self.assertEqual(self.fig.aspect_ratio, "equal")
        self.fig.aspect_ratio = 1.5
        self.assertEqual(self.fig.aspect_ratio, 1.5)

    def test_remove_x_ticks(self):
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.remove_x_ticks = "yes"
        with self.assertRaises(TypeError):
            self.fig.remove_x_ticks = 1
        # Valid
        self.fig.remove_x_ticks = True
        self.assertTrue(self.fig.remove_x_ticks)

    def test_remove_y_ticks(self):
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.remove_y_ticks = "yes"
        with self.assertRaises(TypeError):
            self.fig.remove_y_ticks = 1
        # Valid
        self.fig.remove_y_ticks = True
        self.assertTrue(self.fig.remove_y_ticks)

    def test_reference_labels(self):
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.reference_labels = "yes"
        with self.assertRaises(TypeError):
            self.fig.reference_labels = 1
        # Valid
        self.fig.reference_labels = True
        self.assertTrue(self.fig.reference_labels)

    def test_global_reference_label(self):
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.global_reference_label = "yes"
        with self.assertRaises(TypeError):
            self.fig.global_reference_label = 1
        # Valid
        self.fig.global_reference_label = True
        self.assertTrue(self.fig.global_reference_label)

    def test_reference_label_loc(self):
        # Invalid
        with self.assertRaises(ValueError):
            self.fig.reference_label_loc = "top"
        with self.assertRaises(ValueError):
            self.fig.reference_label_loc = True
        # Valid
        self.fig.reference_label_loc = "inside"
        self.assertEqual(self.fig.reference_label_loc, "inside")
        self.fig.reference_label_loc = "outside"
        self.assertEqual(self.fig.reference_label_loc, "outside")

    def test_reference_label_start_index(self):
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.reference_label_start_index = "a"
        with self.assertRaises(ValueError):
            self.fig.reference_label_start_index = -1
        # Valid
        self.fig.reference_label_start_index = 2
        self.assertEqual(self.fig.reference_label_start_index, 2)

    def test_width_padding(self):
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.width_padding = "a"
        with self.assertRaises(ValueError):
            self.fig.width_padding = -1
        # Valid
        self.fig.width_padding = 0.1
        self.assertEqual(self.fig.width_padding, 0.1)

    def test_height_padding(self):
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.height_padding = "b"
        with self.assertRaises(ValueError):
            self.fig.height_padding = -2
        # Valid
        self.fig.height_padding = 0.2
        self.assertEqual(self.fig.height_padding, 0.2)

    def test_width_and_height_ratios(self):
        new_fig = self.fig.copy_with(num_rows=3, num_cols=4)
        # Invalid
        with self.assertRaises(TypeError):
            new_fig.width_ratios = 1
        with self.assertRaises(TypeError):
            new_fig.height_ratios = 1
        with self.assertRaises(ValueError):
            new_fig.width_ratios = [1, 2, 3]
        with self.assertRaises(ValueError):
            new_fig.height_ratios = [1, 2, 3, 4]
        # Valid
        new_fig.width_ratios = [1, 2, 3, 4]
        new_fig.height_ratios = [1, 2, 3]
        self.assertEqual(list(new_fig.width_ratios), [1, 2, 3, 4])
        self.assertEqual(list(new_fig.height_ratios), [1, 2, 3])

    def test_share_x(self):
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.share_x = "yes"
        with self.assertRaises(TypeError):
            self.fig.share_x = 1
        # Valid
        self.fig.share_x = True
        self.assertTrue(self.fig.share_x)

    def test_share_y(self):
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.share_y = "yes"
        with self.assertRaises(TypeError):
            self.fig.share_y = 1
        # Valid
        self.fig.share_y = True
        self.assertTrue(self.fig.share_y)

    def test_projection(self):
        # Invalid
        with self.assertRaises(GraphingException):
            self.fig.projection = "3d"
        with self.assertRaises(ValueError):
            self.fig.projection = "not_a_projection"
        # Valid
        self.fig.projection = None
        self.assertIsNone(self.fig.projection)
        self.fig.projection = "polar"
        self.assertEqual(self.fig.projection, "polar")
        self.fig.projection = object()
        self.assertIsInstance(self.fig.projection, object)

    def test_general_legend(self):
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.general_legend = "yes"
        with self.assertRaises(TypeError):
            self.fig.general_legend = 1
        # Valid
        self.fig.general_legend = True
        self.assertTrue(self.fig.general_legend)

    def test_legend_loc(self):
        # Invalid
        with self.assertRaises(ValueError):
            self.fig.legend_loc = "not_a_loc"
        with self.assertRaises(ValueError):
            self.fig.legend_loc = (1,)
        with self.assertRaises(TypeError):
            self.fig.legend_loc = 1
        # Valid
        for pos in [
            "upper right", "upper left", "lower left", "lower right", "right",
            "center left", "center right", "lower center", "upper center", "center", "outside upper center",
            "outside center right", "outside lower center", "outside center left", (0.5, 0.5)
        ]:
            self.fig.legend_loc = pos
            self.assertEqual(self.fig.legend_loc, pos)

    def test_legend_cols(self):
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.legend_cols = "a"
        with self.assertRaises(ValueError):
            self.fig.legend_cols = 0
        with self.assertRaises(TypeError):
            self.fig.legend_cols = 1.5
        # Valid
        self.fig.legend_cols = 2
        self.assertEqual(self.fig.legend_cols, 2)

    def test_show_legend(self):
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.show_legend = "yes"
        with self.assertRaises(TypeError):
            self.fig.show_legend = 1
        # Valid
        self.fig.show_legend = True
        self.assertTrue(self.fig.show_legend)

    def test_figure_style(self):
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.figure_style = 1
        with self.assertRaises(ValueError):
            self.fig.figure_style = "not_a_style"
        # Valid
        self.fig.figure_style = "default"
        self.assertEqual(self.fig.figure_style, "default")
        self.fig.figure_style = "dark"
        self.assertEqual(self.fig.figure_style, "dark")

    def test_show_grid(self):
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.show_grid = "yes"
        with self.assertRaises(TypeError):
            self.fig.show_grid = 1
        # Valid
        self.fig.show_grid = True
        self.assertTrue(self.fig.show_grid)

    def test_hide_custom_legend_elements(self):
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.hide_custom_legend_elements = "yes"
        with self.assertRaises(TypeError):
            self.fig.hide_custom_legend_elements = 1
        # Valid
        self.fig.hide_custom_legend_elements = True
        self.assertTrue(self.fig.hide_custom_legend_elements)

    def test_hide_default_legend_elements(self):
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.hide_default_legend_elements = "yes"
        with self.assertRaises(TypeError):
            self.fig.hide_default_legend_elements = 1
        # Valid
        self.fig.hide_default_legend_elements = True
        self.assertTrue(self.fig.hide_default_legend_elements)

    def test_len_and_setitem_getitem(self):
        fig = SmartFigure(num_rows=2, num_cols=2)
        dummy = DummyPlottable()
        fig[0, 0] = dummy
        self.assertEqual(len(fig), 1)
        self.assertIsInstance(fig[0, 0][0], DummyPlottable)
        fig[0, 0] = None
        self.assertEqual(len(fig), 0)
        self.assertEqual(fig[0, 0], [])
        with self.assertRaises(TypeError):
            fig[0, 0] = 123  # not Plottable or SmartFigure
        fig[0, :] = dummy
        self.assertIsInstance(fig[0, :][0], DummyPlottable)
        fig[0, :] = None
        self.assertEqual(len(fig), 0)
        dummy1 = DummyPlottable("a")
        dummy2 = DummyPlottable("b")
        fig[0, 1] = [dummy1, dummy2]
        self.assertEqual(len(fig[0, 1]), 2)
        self.assertEqual(fig[0, 1][0].label, "a")
        self.assertEqual(fig[0, 1][1].label, "b")

    def test_1d_setitem(self):
        fig = SmartFigure(num_rows=1, num_cols=3)
        fig[1] = DummyPlottable()
        fig[0, 1] = DummyPlottable()
        fig[:] = DummyPlottable()
        fig_v = SmartFigure(num_rows=3, num_cols=1)
        fig_v[1] = DummyPlottable()
        fig_v[1, 0] = DummyPlottable()
        fig_v[:] = DummyPlottable()
        fig[1] = [DummyPlottable(), DummyPlottable()]
        fig[1] = fig_v
        with self.assertRaises(TypeError):
            fig[1] = "not a plottable"
        with self.assertRaises(IndexError):
            fig[1, 0] = DummyPlottable()
        with self.assertRaises(ValueError):
            fig[1, 1, 1] = DummyPlottable()
        with self.assertRaises(IndexError):
            fig[3] = DummyPlottable()
        with self.assertRaises(IndexError):
            fig[-1] = DummyPlottable()
        with self.assertRaises(TypeError):
            fig[1.5] = DummyPlottable()
        with self.assertRaises(IndexError):
            fig[0:4] = DummyPlottable()
        with self.assertRaises(IndexError):
            fig[-1:] = DummyPlottable()
        with self.assertRaises(IndexError):
            fig[1:0] = DummyPlottable()
        with self.assertRaises(ValueError):
            fig[1::2] = DummyPlottable()

    def test_2d_setitem(self):
        fig = SmartFigure(num_rows=2, num_cols=3)
        with self.assertRaises(ValueError):
            fig[1] = DummyPlottable()
        with self.assertRaises(ValueError):
            fig[1, 1, 1] = DummyPlottable()
        with self.assertRaises(IndexError):
            fig[2, 0] = DummyPlottable()
        with self.assertRaises(IndexError):
            fig[0, 3] = DummyPlottable()
        with self.assertRaises(IndexError):
            fig[0, -1] = DummyPlottable()
        with self.assertRaises(TypeError):
            fig[1.5, 0] = DummyPlottable()
        with self.assertRaises(ValueError):
            fig[0:2] = DummyPlottable()
        with self.assertRaises(ValueError):
            fig[1:2, :, :] = DummyPlottable()
        with self.assertRaises(IndexError):
            fig[1:4, 0] = DummyPlottable()
        with self.assertRaises(IndexError):
            fig[:, 3:4] = DummyPlottable()
        with self.assertRaises(IndexError):
            fig[-1:, 2] = DummyPlottable()
        with self.assertRaises(IndexError):
            fig[1:0, 2] = DummyPlottable()
        with self.assertRaises(ValueError):
            fig[1::2, :] = DummyPlottable()

    def test_elements_in_init(self):
        # Invalid formats
        with self.assertRaises(TypeError):
            SmartFigure(elements=[1])
        with self.assertRaises(TypeError):
            SmartFigure(elements=(DummyPlottable()))
        with self.assertRaises(TypeError):
            SmartFigure(elements=DummyPlottable())
        with self.assertRaises(TypeError):
            SmartFigure(elements=[[[DummyPlottable()]]])
        with self.assertRaises(TypeError):
            SmartFigure(elements="invalid")
        with self.assertRaises(TypeError):
            SmartFigure(elements=["invalid"])
        with self.assertRaises(TypeError):
            SmartFigure(elements=[["invalid"]])

        # Valid
        SmartFigure(elements=(DummyPlottable(),))
        SmartFigure(2, elements=[(DummyPlottable(),), [None]])
        SmartFigure(2, elements=array([(DummyPlottable(),), [None]]))
        SmartFigure(2, elements=(DummyPlottable(), DummyPlottable()))
        SmartFigure(elements=((DummyPlottable(), DummyPlottable())))
        SmartFigure(elements=(DummyPlottable(), DummyPlottable()))

    def test_add_elements(self):
        dummy = DummyPlottable()
        self.fig.add_elements(dummy)
        self.assertIsInstance(self.fig[0, 0][0], DummyPlottable)
        with self.assertRaises(GraphingException):
            fig2 = SmartFigure(num_rows=2, num_cols=2)
            fig2.add_elements(dummy)

    def test_add_all_elements(self):
        self.fig[0] = SmartFigure()
        self.fig._initialize_parent_smart_figure()
        plt.close()
        self.fig[0] = Curve([0, 1], [0, 1])
        self.fig._initialize_parent_smart_figure()
        plt.close()
        self.fig[0] = Scatter([0, 1], [0, 1])
        self.fig._initialize_parent_smart_figure()
        plt.close()
        self.fig[0] = Histogram([0, 1, 2, 3], 4)
        self.fig._initialize_parent_smart_figure()
        plt.close()
        self.fig[0] = Heatmap([[0, 1], [2, 3]])
        self.fig._initialize_parent_smart_figure()
        plt.close()
        self.fig[0] = VectorField([0, 1], [0, 1], [1, 0], [0, 1])
        self.fig._initialize_parent_smart_figure()
        plt.close()
        self.fig[0] = Contour([[0, 1], [0, 1]], [[0, 1], [0, 1]], [[0, 1], [2, 3]])
        self.fig._initialize_parent_smart_figure()
        plt.close()
        self.fig[0] = Stream([0, 1], [0, 1], [[1, 0], [1, 0]], [[0, 1], [0, 1]])
        self.fig._initialize_parent_smart_figure()
        plt.close()
        self.fig[0] = FitFromFunction(lambda x, a: a*x, Curve([0, 1, 2], [0, 1, 2]))
        self.fig._initialize_parent_smart_figure()
        plt.close()
        self.fig[0] = Hlines([0, 1, 2])
        self.fig._initialize_parent_smart_figure()
        plt.close()
        self.fig[0] = Vlines([0, 1, 2])
        self.fig._initialize_parent_smart_figure()
        plt.close()
        self.fig[0] = Point(1, 1)
        self.fig._initialize_parent_smart_figure()
        plt.close()
        self.fig[0] = Text(1, 1, "Test")
        self.fig._initialize_parent_smart_figure()
        plt.close()
        self.fig[0] = Text(1, 1, "Test", relative_to="figure")
        self.fig._initialize_parent_smart_figure()
        plt.close()
        self.fig[0] = Table(["Header1", "Header2"])
        self.fig._initialize_parent_smart_figure()
        plt.close()
        self.fig[0] = Arrow((0, 1), (1, 0))
        self.fig._initialize_parent_smart_figure()
        plt.close()
        self.fig[0] = Line((0, 1), (1, 0))
        self.fig._initialize_parent_smart_figure()
        plt.close()
        self.fig[0] = Polygon([(0, 0), (1, 1), (1, 0)])
        self.fig._initialize_parent_smart_figure()
        plt.close()
        self.fig[0] = Circle(0, 0, 1)
        self.fig._initialize_parent_smart_figure()
        plt.close()
        self.fig[0] = Rectangle(0, 0, 1, 2)
        self.fig._initialize_parent_smart_figure()
        plt.close()

    def test_copy_and_copy_with(self):
        self.fig.x_label = "X"
        fig2 = self.fig.copy()
        self.assertEqual(fig2.x_label, "X")
        fig3 = self.fig.copy_with(x_label="Y")
        self.assertEqual(fig3.x_label, "Y")
        with self.assertRaises(AttributeError):
            self.fig.copy_with(not_a_property=1)

    def test_show_and_save(self):
        dummy = DummyPlottable()
        self.fig.add_elements(dummy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.fig.show(fullscreen=False)
        plt.close(self.fig._figure)
        self.fig.save("test_smart_figure_output.png")
        self.assertTrue(os.path.exists("test_smart_figure_output.png"))
        os.remove("test_smart_figure_output.png")

    def test_auto_assign_default_params(self):
        x = linspace(0, 3 * pi, 200)
        a_curve = Curve(x, sin(x), label="Test Curve")
        self.fig.add_elements(a_curve)
        self.fig._default_params = self.plainDefaults
        self.fig._fill_in_missing_params(a_curve)
        self.assertEqual(a_curve._line_width, 2)

    def test_auto_assign_default_params_horrible(self):
        x = linspace(0, 3 * pi, 200)
        a_curve = Curve(x, sin(x), label="Test Curve")
        a_figure = SmartFigure(figure_style="horrible")
        a_figure.add_elements(a_curve)
        a_figure._default_params = self.horribleDefaults
        a_figure._fill_in_missing_params(a_curve)
        self.assertEqual(a_curve._line_width, 10)

    def test_auto_assign_default_params_skip_predefined(self):
        x = linspace(0, 3 * pi, 200)
        a_curve = Curve(x, sin(x), label="Test Curve", line_width=3)
        self.fig.add_elements(a_curve)
        self.fig._default_params = self.plainDefaults
        self.fig._fill_in_missing_params(a_curve)
        self.assertEqual(a_curve._line_width, 3)

    def test_assign_figure_params_horrible(self):
        a_figure = SmartFigure(figure_style="horrible")
        a_figure.add_elements(self.testCurve)
        a_figure._default_params = self.horribleDefaults
        a_figure._fill_in_missing_params(a_figure)
        self.assertListEqual(list(a_figure._size), [10, 7])

    def test_element_defaults_are_reset(self):
        self.testCurve._line_width = "default"
        self.fig.add_elements(self.testCurve)
        self.fig._initialize_parent_smart_figure()
        self.assertEqual(self.testCurve._line_width, "default")
        self.fig._default_params = self.plainDefaults
        self.fig._fill_in_missing_params(self.testCurve)
        self.assertEqual(self.testCurve._line_width, 2)
        plt.close("all")

    def test_handles_and_labels_cleared(self):
        self.fig.add_elements(self.testCurve)
        self.fig._initialize_parent_smart_figure()
        handles, labels = self.fig._figure.get_axes()[0].get_legend_handles_labels()
        self.assertEqual(len(handles), 1)
        self.assertEqual(len(labels), 1)
        plt.close("all")

    def test_handles_and_labels_added(self):
        self.fig.add_elements(self.testCurve)
        other_curve = Curve(
            self.testCurve._x_data, self.testCurve._y_data, "Other Curve"
        )
        self.fig.add_elements(other_curve)
        self.fig._initialize_parent_smart_figure()
        self.fig._figure.get_axes()[0].get_legend_handles_labels()
        handles, labels = self.fig._figure.get_axes()[0].get_legend_handles_labels()
        self.assertEqual(len(handles), 2)
        self.assertListEqual(labels, ["Test Curve", "Other Curve"])
        # test if still ok when replotting
        self.fig._figure_style = "horrible"
        self.fig._initialize_parent_smart_figure()
        handles, labels = self.fig._figure.get_axes()[0].get_legend_handles_labels()
        self.assertEqual(len(handles), 2)
        self.assertListEqual(labels, ["Test Curve", "Other Curve"])
        plt.close("all")

    def test_fill_in_rc_params_gl(self):
        self.fig.set_visual_params(legend_edge_color="red")
        # Get default params for dim style
        self.fig._default_params = {
            "rc_params": {
                "axes.grid": False,
                "axes.facecolor": "dimgrey",
            }
        }
        # Fill in rc params
        self.fig._fill_in_rc_params()
        # Check axes fill color is updated
        self.assertEqual(plt.rcParams["axes.grid"], False)
        self.assertEqual(plt.rcParams["axes.facecolor"], "dimgrey")
        self.assertEqual(self.fig._user_rc_dict["legend.edgecolor"], "red")

    def test_update_rc_params(self):
        params = {
            "lines.linewidth": 2,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
        }
        self.fig.set_rc_params(params)
        self.assertDictEqual(self.fig._user_rc_dict, params)
        more_params = {
            "lines.linewidth": 3,
            "axes.grid": True,
        }
        self.fig.set_rc_params(more_params)
        resulting_params = {
            "lines.linewidth": 3,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "axes.grid": True,
        }
        self.assertDictEqual(self.fig._user_rc_dict, resulting_params)

    def test_update_rc_params_reset(self):
        params = {
            "lines.linewidth": 2,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
        }
        self.fig.set_rc_params(params)
        self.fig.set_rc_params({"lines.linewidth": 3}, reset=True)
        self.assertDictEqual(self.fig._user_rc_dict, {"lines.linewidth": 3})

    def test_customize_visual_style(self):
        self.fig.set_visual_params(figure_face_color="red", axes_face_color="blue")
        self.assertDictEqual(
            self.fig._user_rc_dict,
            {
                "figure.facecolor": "red",
                "axes.facecolor": "blue",
            },
        )
        self.fig.set_visual_params(axes_face_color="yellow", font_size=19)
        self.assertDictEqual(
            self.fig._user_rc_dict,
            {
                "figure.facecolor": "red",
                "axes.facecolor": "yellow",
                "font.size": 19,
            },
        )

    def test_customize_visual_style_reset(self):
        self.fig.set_visual_params(figure_face_color="red", axes_face_color="blue")
        self.fig.set_visual_params(axes_face_color="yellow", legend_handle_length=1)
        self.fig.set_visual_params(reset=True)
        self.assertDictEqual(self.fig._user_rc_dict, {})

    def test_matplotlib_style_functional(self):
        a_figure = SmartFigure(figure_style="_mpl-gallery")
        a_figure.add_elements(self.testCurve)
        a_figure._initialize_parent_smart_figure()
        plt.close("all")

    def test_set_ticks_and_tick_params(self):
        self.fig.set_ticks(x_ticks=[0, 1], x_tick_labels=["a", "b"])
        self.fig.set_ticks(y_ticks=[0, 1], y_tick_labels=["a", "b"])
        self.fig.set_ticks(minor_x_ticks=[0.1, 0.2], minor_y_tick_spacing=0.5)
        self.fig.set_ticks(x_tick_spacing=None, y_tick_spacing=None)
        self.fig.set_tick_params(axis="x", which="major", direction="in", length=5)
        self.fig.set_tick_params(axis="y", which="minor", color="red", reset=True)
        with self.assertRaises(GraphingException):
            self.fig.set_ticks(x_tick_labels=["a"], x_ticks=None)
        with self.assertRaises(GraphingException):
            self.fig.set_ticks(x_ticks=[1], x_tick_spacing=1)
        with self.assertRaises(GraphingException):
            self.fig.set_ticks(minor_x_ticks=[1], minor_x_tick_spacing=1)

    def test_set_grid(self):
        self.fig.set_grid(visible_x=True, visible_y=False, color="blue", alpha=0.5, line_style="--", line_width=2)
        self.assertTrue(self.fig.show_grid)
        self.fig.show_grid = False
        self.fig.set_grid(show_on_top=True)
        self.assertTrue(self.fig.show_grid)

    def test_set_custom_legend(self):
        self.fig.set_custom_legend(handles=[], labels=[], reset=True)
        self.fig.set_custom_legend(handles=[], labels=[])
        with self.assertRaises(GraphingException):
            self.fig.set_custom_legend(labels=["a"], handles=[])
        line, = plt.plot([0, 1], [0, 1], label="line")
        self.fig.set_custom_legend(handles=[line], labels=["line"], reset=True)
        with self.assertRaises(GraphingException):
            self.fig.set_custom_legend(handles=[line], labels=["a", "b"])

    def test_methods_return_self(self):
        # Test that methods return self for method chaining
        self.assertIs(self.fig.add_elements(), self.fig)
        self.assertIs(self.fig.set_ticks(x_ticks=[0, 1]), self.fig)
        self.assertIs(self.fig.set_tick_params(label_color="green"), self.fig)
        self.assertIs(self.fig.set_grid(visible_x=True), self.fig)
        self.assertIs(self.fig.set_custom_legend(handles=[]), self.fig)
        self.assertIs(self.fig.set_visual_params(figure_face_color="red"), self.fig)
        self.assertIs(self.fig.set_rc_params({"lines.linewidth": 2}), self.fig)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertIs(self.fig.show(), self.fig)
        self.assertIs(self.fig.save("test_smart_figure_output.png"), self.fig)
        os.remove("test_smart_figure_output.png")


if __name__ == "__main__":
    unittest.main()
