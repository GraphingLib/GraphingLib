import os
import unittest
import warnings

from matplotlib import use as matplotlib_use

matplotlib_use("Agg")  # Use non-GUI backend for tests

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy import array, linspace, pi, sin

try:
    import astropy.units as u
    from astropy.wcs import WCS

    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False
    WCS = None
    u = None

from graphinglib.data_plotting_1d import Curve, Histogram, Scatter
from graphinglib.data_plotting_2d import Contour, Heatmap, Stream, VectorField
from graphinglib.file_manager import FileLoader
from graphinglib.fits import FitFromFunction
from graphinglib.graph_elements import (
    GraphingException,
    Hlines,
    Plottable,
    PlottableAxMethod,
    Point,
    Table,
    Text,
    Vlines,
)
from graphinglib.legend_artists import LegendLine
from graphinglib.shapes import Arrow, Circle, Line, Polygon, Rectangle
from graphinglib.smart_figure import SmartFigure, SmartFigureWCS, SmartTwinAxis


class DummyPlottable(Plottable):
    def __init__(self, label="dummy"):
        self.label = label
        self.handle = None
        self._x_data = [0, 1]
        self._y_data = [0, 1]

    def _plot_element(self, ax, z_order, cycle_color=None):
        (self.handle,) = ax.plot(
            self._x_data, self._y_data, label=self.label, color=cycle_color
        )


class TestSmartFigure(unittest.TestCase):
    def setUp(self):
        self.cls = SmartFigure
        self.fig = SmartFigure()
        self.twin_axis = SmartTwinAxis()
        # Set up figures for testing indexing and slicing
        self.fig_2x3 = SmartFigure(num_rows=2, num_cols=3)  # 2 rows, 3 columns
        self.fig_1x4 = SmartFigure(num_rows=1, num_cols=4)  # 1 row, 4 columns (1D)
        self.fig_3x1 = SmartFigure(num_rows=3, num_cols=1)  # 3 rows, 1 column (1D)
        self.fig_2x2 = SmartFigure(num_rows=2, num_cols=2)  # 2 rows, 2 columns
        x = linspace(0, 3 * pi, 200)
        self.testCurve = Curve(x, sin(x), "Test Curve", color="k")
        self.plainDefaults = FileLoader("plain").load()
        self.horribleDefaults = FileLoader("horrible").load()

    def test_init_defaults(self):
        """Test SmartFigure initialization with default parameters."""
        self.assertEqual(self.fig.num_rows, 1)
        self.assertEqual(self.fig.num_cols, 1)
        self.assertEqual(self.fig.figure_style, "default")
        self.assertIsInstance(self.fig, self.cls)

    def test_init_custom_args(self):
        """Test SmartFigure initialization with custom arguments."""
        elements = [
            [DummyPlottable(), DummyPlottable()],
            None,
            DummyPlottable(),
            [None],
            [DummyPlottable(), DummyPlottable(), DummyPlottable()],
            [],
        ]
        annotations = (Text(0.5, 0.5, "Center"), Text(0.1, 0.1, "Corner"))
        fig = self.cls(
            num_rows=2,
            num_cols=3,
            x_label="X",
            y_label="Y",
            size=(8, 6),
            title="Test Figure",
            x_lim=(0, 10),
            y_lim=(-5, 5),
            sub_x_labels=["X1", "X2", "X3"],
            sub_y_labels=["Y1", "Y2"],
            subtitles=[
                "Title 1",
                "Title 2",
                "Title 3",
                "Title 4",
                "Title 5",
                "Title 6",
            ],
            log_scale_x=True,
            log_scale_y=True,
            remove_axes=True,
            aspect_ratio=1.5,
            box_aspect_ratio=0.7,
            remove_x_ticks=True,
            remove_y_ticks=True,
            invert_x_axis=True,
            invert_y_axis=True,
            reference_labels=False,
            global_reference_label=True,
            reference_labels_loc="inside",
            width_padding=0.5,
            height_padding=0,
            width_ratios=[1, 2, 3],
            height_ratios=[1, 2],
            share_x=True,
            share_y=True,
            projection="polar",
            general_legend=True,
            legend_loc="upper right",
            legend_cols=2,
            show_legend=False,
            twin_x_axis=None,
            twin_y_axis=None,
            figure_style="dark",
            elements=elements,
            annotations=annotations,
        )
        self.assertEqual(fig.num_rows, 2)
        self.assertEqual(fig.num_cols, 3)
        self.assertEqual(fig.x_label, "X")
        self.assertEqual(fig.y_label, "Y")
        self.assertEqual(fig.size, (8, 6))
        self.assertEqual(fig.title, "Test Figure")
        self.assertEqual(fig.x_lim, (0, 10))
        self.assertEqual(fig.y_lim, (-5, 5))
        self.assertEqual(fig.sub_x_labels, ["X1", "X2", "X3"])
        self.assertEqual(fig.sub_y_labels, ["Y1", "Y2"])
        self.assertEqual(
            fig.subtitles,
            ["Title 1", "Title 2", "Title 3", "Title 4", "Title 5", "Title 6"],
        )
        self.assertTrue(fig.log_scale_x)
        self.assertTrue(fig.log_scale_y)
        self.assertTrue(fig.remove_axes)
        self.assertEqual(fig.aspect_ratio, 1.5)
        self.assertEqual(fig.box_aspect_ratio, 0.7)
        self.assertTrue(fig.remove_x_ticks)
        self.assertTrue(fig.remove_y_ticks)
        self.assertTrue(fig.invert_x_axis)
        self.assertTrue(fig.invert_y_axis)
        self.assertFalse(fig.reference_labels)
        self.assertTrue(fig.global_reference_label)
        self.assertEqual(fig.reference_labels_loc, "inside")
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
        self.assertIsNone(fig.twin_x_axis)
        self.assertIsNone(fig.twin_y_axis)
        self.assertEqual(fig.figure_style, "dark")
        self.assertEqual(fig[0, 0], elements[0])
        self.assertEqual(fig[0, 1], [])
        self.assertEqual(fig[0, 2], [elements[2]])
        self.assertEqual(fig[1, 0], elements[3])
        self.assertEqual(fig[1, 1], elements[4])
        self.assertEqual(fig[1, 2], elements[5])
        self.assertEqual(fig.annotations, annotations)

    def test_elements_in_init(self):
        """Test elements initialization in constructor."""
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

    def test_num_rows_and_num_cols(self):
        """Test num_rows and num_cols property validation and assignment."""
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

    def test_shape(self):
        """Test shape property."""
        self.assertEqual(self.fig.shape, (1, 1))
        self.assertEqual(self.fig_2x3.shape, (2, 3))
        self.assertEqual(self.fig_1x4.shape, (1, 4))
        self.assertEqual(self.fig_3x1.shape, (3, 1))
        self.assertEqual(self.fig_2x2.shape, (2, 2))
        # Invalid types
        with self.assertRaises(TypeError):
            self.fig.shape = "a", 2
        with self.assertRaises(TypeError):
            self.fig.shape = 2, "a"
        with self.assertRaises(TypeError):
            self.fig.shape = 0.5, 2
        with self.assertRaises(TypeError):
            self.fig.shape = 2, 0.5
        # Invalid values
        with self.assertRaises(ValueError):
            self.fig.shape = 0, 1
        with self.assertRaises(ValueError):
            self.fig.shape = 1, 0
        # Valid set/get
        self.fig.shape = 2, 2
        self.assertEqual(self.fig.num_rows, 2)
        self.assertEqual(self.fig.num_cols, 2)
        # Changing shape with elements present
        new_fig = self.fig.copy_with(num_rows=2, num_cols=2)
        new_fig[:, :] = DummyPlottable()
        with self.assertRaises(GraphingException):
            new_fig.shape = 1, 3
        with self.assertRaises(GraphingException):
            new_fig.shape = 3, 1
        new_fig.shape = 3, 3
        self.assertEqual(new_fig.shape, (3, 3))

    def test_size(self):
        """Test size property validation and assignment."""
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
        """Test x_lim property validation and assignment."""
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.x_lim = 1
        with self.assertRaises(ValueError):
            self.fig.x_lim = (1, 2, 3)
        # Valid
        self.fig.x_lim = (0, 10)
        self.assertEqual(self.fig.x_lim, (0, 10))

    def test_y_lim(self):
        """Test y_lim property validation and assignment."""
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.y_lim = 1
        with self.assertRaises(ValueError):
            self.fig.y_lim = (1, 2, 3)
        # Valid
        self.fig.y_lim = (-5, 5)
        self.assertEqual(self.fig.y_lim, (-5, 5))

    def test_sub_x_labels(self):
        """Test sub_x_labels property validation and assignment."""
        # Test with None (default)
        self.assertIsNone(self.fig.sub_x_labels)

        # Test with valid list of strings
        valid_labels = ("X1", "X2", "X3")
        self.fig.sub_x_labels = valid_labels
        self.assertEqual(self.fig.sub_x_labels, valid_labels)
        self.fig.sub_x_labels = [None, "valid"]
        with self.assertRaises(
            GraphingException
        ):  # different number of sub labels and subplots
            self.fig._initialize_parent_smart_figure()

        # Test invalid types
        with self.assertRaises(TypeError):
            self.fig.sub_x_labels = 123

        self.fig_2x2.sub_x_labels = ["X1", "X2", "X3", "X4"]
        self.fig_2x2.add_elements(*[DummyPlottable() for _ in range(4)])
        self.fig_2x2._initialize_parent_smart_figure()

    def test_sub_y_labels(self):
        """Test sub_y_labels property validation and assignment."""
        # Test with None (default)
        self.assertIsNone(self.fig.sub_y_labels)

        # Test with valid list of strings
        valid_labels = ("Y1", "Y2", "Y3")
        self.fig.sub_y_labels = valid_labels
        self.assertEqual(self.fig.sub_y_labels, valid_labels)
        self.fig.sub_y_labels = [None, "valid"]
        with self.assertRaises(
            GraphingException
        ):  # different number of sub labels and subplots
            self.fig._initialize_parent_smart_figure()

        # Test invalid types
        with self.assertRaises(TypeError):
            self.fig.sub_y_labels = 123

        self.fig_2x2.sub_y_labels = ["Y1", "Y2", "Y3", "Y4"]
        self.fig_2x2.add_elements(*[DummyPlottable() for _ in range(4)])
        self.fig_2x2._initialize_parent_smart_figure()

    def test_subtitles(self):
        """Test subtitles property validation and assignment."""
        # Test with None (default)
        self.assertIsNone(self.fig.subtitles)

        # Test with valid list of strings
        valid_titles = ("Title 1", "Title 2", "Title 3")
        self.fig.subtitles = valid_titles
        self.assertEqual(self.fig.subtitles, valid_titles)
        self.fig.subtitles = [None, "valid"]
        with self.assertRaises(
            GraphingException
        ):  # different number of subtitles and subplots
            self.fig._initialize_parent_smart_figure()

        # Test invalid types
        with self.assertRaises(TypeError):
            self.fig.subtitles = 123

        self.fig_2x2.subtitles = ["Sub 1", "Sub 2", "Sub 3", "Sub 4"]
        self.fig_2x2.add_elements(*[DummyPlottable() for _ in range(4)])
        self.fig_2x2._initialize_parent_smart_figure()

    def test_log_scale_x(self):
        """Test log_scale_x property validation and assignment."""
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.log_scale_x = "yes"
        with self.assertRaises(TypeError):
            self.fig.log_scale_x = 1
        # Valid
        self.fig.log_scale_x = True
        self.assertTrue(self.fig.log_scale_x)

    def test_log_scale_y(self):
        """Test log_scale_y property validation and assignment."""
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.log_scale_y = "yes"
        with self.assertRaises(TypeError):
            self.fig.log_scale_y = 1
        # Valid
        self.fig.log_scale_y = True
        self.assertTrue(self.fig.log_scale_y)

    def test_remove_axes(self):
        """Test remove_axes property validation and assignment."""
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.remove_axes = "yes"
        with self.assertRaises(TypeError):
            self.fig.remove_axes = 1
        # Valid
        self.fig.remove_axes = True
        self.assertTrue(self.fig.remove_axes)

    def test_aspect_ratio(self):
        """Test aspect_ratio property validation and assignment."""
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.aspect_ratio = "yes"
        with self.assertRaises(TypeError):
            self.fig.aspect_ratio = array([1])
        with self.assertRaises(ValueError):
            self.fig.aspect_ratio = -1
        # Valid
        self.fig.aspect_ratio = "auto"
        self.assertEqual(self.fig.aspect_ratio, "auto")
        self.fig.aspect_ratio = "equal"
        self.assertEqual(self.fig.aspect_ratio, "equal")
        self.fig.aspect_ratio = 1.5
        self.assertEqual(self.fig.aspect_ratio, 1.5)

    def test_box_aspect_ratio(self):
        """Test box_aspect_ratio property validation and assignment."""
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.box_aspect_ratio = "yes"
        with self.assertRaises(TypeError):
            self.fig.box_aspect_ratio = array([1])
        with self.assertRaises(ValueError):
            self.fig.box_aspect_ratio = -1
        # Valid
        self.fig.box_aspect_ratio = 0.5
        self.assertEqual(self.fig.box_aspect_ratio, 0.5)
        self.fig.box_aspect_ratio = 3
        self.assertEqual(self.fig.box_aspect_ratio, 3)

    def test_remove_x_ticks(self):
        """Test remove_x_ticks property validation and assignment."""
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.remove_x_ticks = "yes"
        with self.assertRaises(TypeError):
            self.fig.remove_x_ticks = 1
        # Valid
        self.fig.remove_x_ticks = True
        self.assertTrue(self.fig.remove_x_ticks)

    def test_remove_y_ticks(self):
        """Test remove_y_ticks property validation and assignment."""
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.remove_y_ticks = "yes"
        with self.assertRaises(TypeError):
            self.fig.remove_y_ticks = 1
        # Valid
        self.fig.remove_y_ticks = True
        self.assertTrue(self.fig.remove_y_ticks)

    def test_invert_x_axis(self):
        """Test invert_x_axis property validation and assignment."""
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.invert_x_axis = "yes"
        with self.assertRaises(TypeError):
            self.fig.invert_x_axis = 1
        # Valid
        self.fig.invert_x_axis = True
        self.assertTrue(self.fig.invert_x_axis)
        self.fig.invert_x_axis = False
        self.assertFalse(self.fig.invert_x_axis)

    def test_invert_y_axis(self):
        """Test invert_y_axis property validation and assignment."""
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.invert_y_axis = "yes"
        with self.assertRaises(TypeError):
            self.fig.invert_y_axis = 1
        # Valid
        self.fig.invert_y_axis = True
        self.assertTrue(self.fig.invert_y_axis)
        self.fig.invert_y_axis = False
        self.assertFalse(self.fig.invert_y_axis)

    def test_reference_labels(self):
        """Test reference_labels property validation and assignment."""
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.reference_labels = "yes"
        with self.assertRaises(TypeError):
            self.fig.reference_labels = 1
        # Valid
        self.fig.reference_labels = True
        self.assertTrue(self.fig.reference_labels)

    def test_global_reference_label(self):
        """Test global_reference_label property validation and assignment."""
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.global_reference_label = "yes"
        with self.assertRaises(TypeError):
            self.fig.global_reference_label = 1
        # Valid
        self.fig.global_reference_label = True
        self.assertTrue(self.fig.global_reference_label)

    def test_reference_labels_loc(self):
        """Test reference_labels_loc property validation and assignment."""
        # Invalid
        with self.assertRaises(ValueError):
            self.fig.reference_labels_loc = "top"
        with self.assertRaises(ValueError):
            self.fig.reference_labels_loc = True
        with self.assertRaises(ValueError):
            self.fig.reference_labels_loc = 1, 3, 4
        # Valid
        self.fig.reference_labels_loc = "inside"
        self.assertEqual(self.fig.reference_labels_loc, "inside")
        self.fig.reference_labels_loc = "outside"
        self.assertEqual(self.fig.reference_labels_loc, "outside")
        self.fig.reference_labels_loc = 0.5, -0.5
        self.assertEqual(self.fig.reference_labels_loc, (0.5, -0.5))

    def test_width_padding(self):
        """Test width_padding property validation and assignment."""
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.width_padding = "a"
        with self.assertRaises(ValueError):
            self.fig.width_padding = -1
        # Valid
        self.fig.width_padding = 0.1
        self.assertEqual(self.fig.width_padding, 0.1)

    def test_height_padding(self):
        """Test height_padding property validation and assignment."""
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.height_padding = "b"
        with self.assertRaises(ValueError):
            self.fig.height_padding = -2
        # Valid
        self.fig.height_padding = 0.2
        self.assertEqual(self.fig.height_padding, 0.2)

    def test_width_and_height_ratios(self):
        """Test width_ratios and height_ratios property validation and assignment."""
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
        """Test share_x property validation and assignment."""
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.share_x = "yes"
        with self.assertRaises(TypeError):
            self.fig.share_x = 1
        # Valid
        self.fig.share_x = True
        self.assertTrue(self.fig.share_x)

    def test_share_y(self):
        """Test share_y property validation and assignment."""
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.share_y = "yes"
        with self.assertRaises(TypeError):
            self.fig.share_y = 1
        # Valid
        self.fig.share_y = True
        self.assertTrue(self.fig.share_y)

    def test_projection(self):
        """Test projection property validation and assignment."""
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
        """Test general_legend property validation and assignment."""
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.general_legend = "yes"
        with self.assertRaises(TypeError):
            self.fig.general_legend = 1
        # Valid
        self.fig.general_legend = True
        self.assertTrue(self.fig.general_legend)

    def test_legend_loc(self):
        """Test legend_loc property validation and assignment."""
        # Invalid
        with self.assertRaises(ValueError):
            self.fig.legend_loc = "not_a_loc"
        with self.assertRaises(ValueError):
            self.fig.legend_loc = (1,)
        with self.assertRaises(TypeError):
            self.fig.legend_loc = 1
        # Valid
        for pos in [
            "upper right",
            "upper left",
            "lower left",
            "lower right",
            "right",
            "center left",
            "center right",
            "lower center",
            "upper center",
            "center",
            "outside upper center",
            "outside center right",
            "outside lower center",
            "outside center left",
            (0.5, 0.5),
        ]:
            self.fig.legend_loc = pos
            self.assertEqual(self.fig.legend_loc, pos)

    def test_legend_cols(self):
        """Test legend_cols property validation and assignment."""
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
        """Test show_legend property validation and assignment."""
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.show_legend = "yes"
        with self.assertRaises(TypeError):
            self.fig.show_legend = 1
        # Valid
        self.fig.show_legend = True
        self.assertTrue(self.fig.show_legend)

    def test_figure_style(self):
        """Test figure_style property validation and assignment."""
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
        self.fig.figure_style = "matplotlib"
        self.assertEqual(self.fig.figure_style, "matplotlib")

    def test_elements(self):
        """Test elements property validation and assignment."""
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.elements = "not_a_list"
        with self.assertRaises(TypeError):
            self.fig.elements = [1, 2, 3]
        # Valid
        self.fig_1x4.elements = [DummyPlottable(), DummyPlottable()]
        self.assertEqual(len(self.fig_1x4.elements), 2)

        # Setting new elements will clear existing ones
        self.fig_3x1.elements = [
            [DummyPlottable(), DummyPlottable()],
            [None],
            DummyPlottable(),
        ]
        self.assertEqual(len(self.fig_3x1.elements), 3)

    def test_annotations(self):
        """Test annotations property validation and assignment."""
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.annotations = "not_a_list"
        with self.assertRaises(TypeError):
            self.fig.annotations = [1, 2, 3]
        with self.assertRaises(TypeError):
            self.fig.annotations = [Text(1, 1, "s"), Arrow(0, 0, 1, 1)]
        # Valid
        annotations = [
            Text(0.5, 0.5, "Center"),
            Text(0.1, 0.9, "Top Left", h_align="left", v_align="top"),
        ]
        self.fig.annotations = annotations
        self.assertEqual(self.fig.annotations, annotations)

    def test_show_grid(self):
        """Test show_grid property validation and assignment."""
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.show_grid = "yes"
        with self.assertRaises(TypeError):
            self.fig.show_grid = 1
        # Valid
        self.fig.show_grid = True
        self.assertTrue(self.fig.show_grid)

    def test_hide_custom_legend_elements(self):
        """Test hide_custom_legend_elements property validation and assignment."""
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.hide_custom_legend_elements = "yes"
        with self.assertRaises(TypeError):
            self.fig.hide_custom_legend_elements = 1
        # Valid
        self.fig.hide_custom_legend_elements = True
        self.assertTrue(self.fig.hide_custom_legend_elements)

    def test_hide_default_legend_elements(self):
        """Test hide_default_legend_elements property validation and assignment."""
        # Invalid
        with self.assertRaises(TypeError):
            self.fig.hide_default_legend_elements = "yes"
        with self.assertRaises(TypeError):
            self.fig.hide_default_legend_elements = 1
        # Valid
        self.fig.hide_default_legend_elements = True
        self.assertTrue(self.fig.hide_default_legend_elements)

    def test_list_or_item_properties(self):
        """Test all properties that can be either lists or single items except the projection parameter."""
        self.fig_2x3[:, 0] = DummyPlottable()
        self.fig_2x3[0, 1] = DummyPlottable()
        self.fig_2x3[0, 2] = DummyPlottable()
        self.fig_2x3[1, 2] = DummyPlottable()
        # leave the [1,1] subplot empty

        bool_tests = [
            [True, True, False],
            [False, False, True, False],
            [True, False, False, True, False],
        ]
        params = {
            "x_lim": [
                [(0, 1), None, (1, 2), (None, 4)],
                [None, None, (None, 0), (0, None)],
                [(0, 1), (1, 2), None, None, (5, 6)],
            ],
            "y_lim": [
                [(0, 1), None, (1, 2), (None, 4)],
                [None, None, (None, 0), (0, None)],
                [(0, 1), (1, 2), None, None, (5, 6)],
            ],
            "log_scale_x": bool_tests,
            "log_scale_y": bool_tests,
            "remove_axes": bool_tests,
            "aspect_ratio": [
                [1.0, 1.0, 2.0],
                [1.5, 1.0, "auto", "equal"],
                [0.75, 1.0, 1.0, "auto", "equal"],
            ],
            "box_aspect_ratio": [
                [None, 0.5, 2.0],
                [1.5, None, 0.75, 3.0],
                [0.33, None, None, 2.0, 4.0],
            ],
            "remove_x_ticks": bool_tests,
            "remove_y_ticks": bool_tests,
            "invert_x_axis": bool_tests,
            "invert_y_axis": bool_tests,
            "reference_labels": bool_tests,
            "reference_labels_loc": [
                ["inside", (0.5, 0.5), "outside"],
                ["outside", "inside", (0, 0), (1, 1)],
                ["inside", (0.25, 0.75), (0.75, 0.25), "inside", "outside"],
            ],
            "legend_loc": [
                ["upper right", (0.5, 0.5), "lower left"],
                ["upper left", (0, 1), (1, 0), "center"],
                [
                    "center",
                    "lower left",
                    (0.2, 0.7),
                    (0.7, 0.2),
                    "outside lower center",
                ],
            ],
            "legend_cols": [[1, 2, 3], [2, 1, 3, 1], [1, 2, 3, 5, 1]],
            "show_legend": bool_tests,
            "show_grid": bool_tests,
            "hide_default_legend_elements": bool_tests,
            "sub_x_labels": [
                ["X1", "X2", None],
                [None, "X_valid", "X3", None],
                ["X1", None, "X2", None, "X3"],
            ],
            "sub_y_labels": [
                ["Y1", "Y2", None],
                [None, "Y_valid", "Y3", None],
                ["Y1", None, "Y2", None, "Y3"],
            ],
            "subtitles": [
                ["T 1", None, "T 3"],
                [None, "Valid T", "T 2", None],
                ["T 1", None, "T 2", None, "T 3"],
            ],
        }
        self.fig_2x3.set_grid()
        for param, values in params.items():
            fig = self.fig_2x3.copy_with(**{param: values[0]})  # underfilled list
            fig._fill_per_subplot_params()
            fig._figure = plt.figure()
            fig._reference_label_i = 0
            fig._prepare_figure()
            plt.close()

            fig = self.fig_2x3.copy_with(**{param: values[1]})  # equally filled list
            fig._fill_per_subplot_params()
            fig._figure = plt.figure()
            fig._reference_label_i = 0
            fig._prepare_figure()
            plt.close()
            with self.assertRaises(GraphingException):
                fig = self.fig_2x3.copy_with(**{param: values[2]})  # overfilled list
                fig._fill_per_subplot_params()

    def test_projection_list_or_item(self):
        """Test projection property with list or single item assignment."""
        self.fig_2x3[:, 0] = DummyPlottable()
        self.fig_2x3[0, 1] = DummyPlottable()
        self.fig_2x3[0, 2] = DummyPlottable()
        self.fig_2x3[1, 2] = DummyPlottable()
        # leave the [1,1] subplot empty

        self.fig_2x3.projection = [None, "polar", "polar"]  # underfilled list
        self.fig_2x3._fill_per_subplot_params()
        self.fig_2x3.projection = ["polar", None, "polar", None]  # equally filled list
        self.fig_2x3._fill_per_subplot_params()
        with self.assertRaises(GraphingException):
            self.fig_2x3.projection = [
                None,
                "polar",
                None,
                "polar",
                None,
            ]  # overfilled list
            self.fig_2x3._fill_per_subplot_params()

    def test_general_legend_list_or_item_properties(self):
        """Test properties that are invalid as lists when general_legend is True."""
        self.fig_2x2.elements = [DummyPlottable() for _ in range(4)]
        params = {
            "show_legend": [True, False, True, False],
            "legend_cols": [1, 2, 1],
            "legend_loc": ["upper right", "lower left"],
            "hide_default_legend_elements": [True],
        }
        self.fig_2x2._default_params = FileLoader("plain").load()
        self.fig_2x2._reference_label_i = 0
        self.fig_2x2.general_legend = True

        for param, values in params.items():
            fig = self.fig_2x2.copy_with(**{param: values})
            fig._fill_per_subplot_params()
            fig._figure = plt.figure()
            fig._reference_label_i = 0
            with self.assertRaises(GraphingException):
                fig._prepare_figure()
            plt.close(self.fig._figure)

    def test_len_and_setitem_getitem(self):
        """Test length calculation and item assignment/retrieval."""
        dummy = DummyPlottable()
        self.fig_2x2[0, 0] = dummy
        self.assertEqual(len(self.fig_2x2), 1)
        self.assertIsInstance(self.fig_2x2[0, 0][0], DummyPlottable)
        self.fig_2x2[0, 0] = None
        self.assertEqual(len(self.fig_2x2), 0)
        self.assertEqual(self.fig_2x2[0, 0], [])
        with self.assertRaises(TypeError):
            self.fig_2x2[0, 0] = 123  # not Plottable or SmartFigure
        self.fig_2x2[0, :] = dummy
        self.assertIsInstance(self.fig_2x2[0, :][0], DummyPlottable)
        self.fig_2x2[0, :] = None
        self.assertEqual(len(self.fig_2x2), 0)
        dummy1 = DummyPlottable("a")
        dummy2 = DummyPlottable("b")
        self.fig_2x2[0, 1] = [dummy1, dummy2]
        self.assertEqual(len(self.fig_2x2[0, 1]), 2)
        self.assertEqual(self.fig_2x2[0, 1][0].label, "a")
        self.assertEqual(self.fig_2x2[0, 1][1].label, "b")

    def test_1d_setitem(self):
        """Test 1D item assignment for single row/column figures."""
        self.fig_1x4[1] = DummyPlottable()
        self.fig_1x4[0, 1] = DummyPlottable()
        self.fig_1x4[:] = DummyPlottable()
        self.fig_3x1 = self.fig_3x1.copy()
        self.fig_3x1[1] = DummyPlottable()
        self.fig_3x1[1, 0] = DummyPlottable()
        self.fig_3x1[:] = DummyPlottable()
        self.fig_1x4[1] = [DummyPlottable(), DummyPlottable()]
        self.fig_1x4[1] = self.fig_3x1

        # Test negative indexing
        self.fig_1x4[-1] = DummyPlottable()  # Should work
        self.fig_3x1[-1] = DummyPlottable()  # Should work

        with self.assertRaises(TypeError):
            self.fig_1x4[1] = "not a plottable"
        with self.assertRaises(IndexError):
            self.fig_1x4[1, 0] = DummyPlottable()
        with self.assertRaises(ValueError):
            self.fig_1x4[1, 1, 1] = DummyPlottable()
        with self.assertRaises(IndexError):
            self.fig_1x4[4] = DummyPlottable()
        with self.assertRaises(IndexError):
            self.fig_1x4[-5] = DummyPlottable()  # Out of bounds negative
        with self.assertRaises(TypeError):
            self.fig_1x4[1.5] = DummyPlottable()
        with self.assertRaises(IndexError):
            self.fig_1x4[0:5] = DummyPlottable()
        with self.assertRaises(IndexError):
            self.fig_1x4[1:0] = DummyPlottable()
        with self.assertRaises(ValueError):
            self.fig_1x4[1::2] = DummyPlottable()

    def test_2d_setitem(self):
        """Test 2D item assignment for multi-row/column figures."""
        # Test valid negative indexing
        self.fig_2x3[-1, -1] = DummyPlottable()  # Should work
        self.fig_2x3[-2, -3] = DummyPlottable()  # Should work

        with self.assertRaises(ValueError):
            self.fig_2x3[1] = DummyPlottable()
        with self.assertRaises(ValueError):
            self.fig_2x3[1, 1, 1] = DummyPlottable()
        with self.assertRaises(IndexError):
            self.fig_2x3[2, 0] = DummyPlottable()
        with self.assertRaises(IndexError):
            self.fig_2x3[0, 3] = DummyPlottable()
        with self.assertRaises(IndexError):
            self.fig_2x3[-3, 0] = DummyPlottable()  # Out of bounds negative
        with self.assertRaises(IndexError):
            self.fig_2x3[0, -4] = DummyPlottable()  # Out of bounds negative
        with self.assertRaises(TypeError):
            self.fig_2x3[1.5, 0] = DummyPlottable()
        with self.assertRaises(ValueError):
            self.fig_2x3[0:2] = DummyPlottable()
        with self.assertRaises(ValueError):
            self.fig_2x3[1:2, :, :] = DummyPlottable()
        with self.assertRaises(IndexError):
            self.fig_2x3[1:4, 0] = DummyPlottable()
        with self.assertRaises(IndexError):
            self.fig_2x3[:, 3:4] = DummyPlottable()
        with self.assertRaises(IndexError):
            self.fig_2x3[1:0, 2] = DummyPlottable()
        with self.assertRaises(ValueError):
            self.fig_2x3[1::2, :] = DummyPlottable()

    def test_iter(self):
        """Test iteration over subplots."""
        count = 0
        self.fig_2x3.elements = [[None]] * 6
        for subplot in self.fig_2x3:
            self.assertIsInstance(subplot, list)
            count += 1
        self.assertEqual(count, 6)  # 2 rows * 3 columns = 6 subplots

    def test_negative_integer_indexing(self):
        """Test negative integer indexing for element access."""
        mock_element = DummyPlottable("test_element")

        # Test 2D negative integer indexing
        self.fig_2x3[-1, -1] = mock_element  # Last row, last column (1, 2)
        self.assertEqual(len(self.fig_2x3[1, 2]), 1)

        self.fig_2x3[-2, -3] = mock_element  # First row, first column (0, 0)
        self.assertEqual(len(self.fig_2x3[0, 0]), 1)

        self.fig_2x3[-1, -2] = mock_element  # Last row, second-to-last column (1, 1)
        self.assertEqual(len(self.fig_2x3[1, 1]), 1)

        # Test 1D negative indexing (single row)
        self.fig_1x4[-1] = mock_element  # Last column (0, 3)
        self.assertEqual(len(self.fig_1x4[0, 3]), 1)

        self.fig_1x4[-4] = mock_element  # First column (0, 0)
        self.assertEqual(len(self.fig_1x4[0, 0]), 1)

        # Test 1D negative indexing (single column)
        self.fig_3x1[-1] = mock_element  # Last row (2, 0)
        self.assertEqual(len(self.fig_3x1[2, 0]), 1)

        self.fig_3x1[-3] = mock_element  # First row (0, 0)
        self.assertEqual(len(self.fig_3x1[0, 0]), 1)

    def test_negative_slice_indexing(self):
        """Test negative slice indexing for element access."""
        mock_element = DummyPlottable("test_element")

        # Test negative slice indexing
        self.fig_2x3[-2:, -3:-1] = mock_element  # Should span (0:2, 0:2)
        key = (slice(0, 2), slice(0, 2))
        self.assertIn(key, self.fig_2x3._elements)

        # Test slice with only negative start
        self.fig_2x3[-1:, :] = DummyPlottable("test2")  # Last row, all columns
        key = (slice(1, 2), slice(0, 3))
        self.assertIn(key, self.fig_2x3._elements)

    def test_negative_indexing_errors(self):
        """Test error conditions for negative indexing."""
        mock_element = DummyPlottable("test")

        # Test negative index out of bounds
        with self.assertRaises(IndexError):
            self.fig_2x3[-3, 0] = mock_element  # -3 is out of bounds for 2 rows

        with self.assertRaises(IndexError):
            self.fig_2x3[0, -4] = mock_element  # -4 is out of bounds for 3 columns

        # Test negative slice out of bounds
        with self.assertRaises(IndexError):
            self.fig_2x3[-3:, :] = mock_element  # -3 is out of bounds for 2 rows

        # Test negative slice out of bounds
        with self.assertRaises(IndexError):
            self.fig_2x3[-3:, -2:1] = mock_element  # start = stop

    def test_validate_and_normalize_key_negative_indices(self):
        """Test the _validate_and_normalize_key method directly with negative indices."""
        # Test negative integer normalization
        result = self.fig_2x3._validate_and_normalize_key((-1, -1))
        expected = (slice(1, 2), slice(2, 3))  # Should convert to slice objects
        self.assertEqual(result, expected)

        result = self.fig_2x3._validate_and_normalize_key((-2, -3))
        expected = (slice(0, 1), slice(0, 1))
        self.assertEqual(result, expected)

        # Test negative slice normalization
        result = self.fig_2x3._validate_and_normalize_key(
            (slice(-2, None), slice(-3, -1))
        )
        expected = (slice(0, 2), slice(0, 2))
        self.assertEqual(result, expected)

    def test_add_elements(self):
        """Test adding elements to the figure."""
        dummy = DummyPlottable()
        self.fig.add_elements(dummy)
        self.assertIsInstance(self.fig[0, 0][0], DummyPlottable)
        self.fig.add_elements([dummy, dummy], [dummy, dummy])
        self.fig.add_elements()
        with self.assertRaises(TypeError):
            self.fig.add_elements([dummy, [dummy]], dummy)
        with self.assertRaises(TypeError):
            self.fig.add_elements([dummy, self.fig], dummy)
        self.fig_2x2.add_elements(dummy, dummy, dummy, dummy)
        self.fig_2x2.add_elements(dummy, dummy, dummy, self.fig)
        self.fig_2x2.add_elements([None], [dummy, dummy], None, self.fig)
        with self.assertRaises(ValueError):
            self.fig_2x2.add_elements(dummy, dummy, dummy, dummy, dummy)
        with self.assertRaises(ValueError):
            self.fig_2x2.add_elements(
                None, None, [dummy, dummy, None, dummy, dummy], None, None
            )

    def test_add_all_elements(self):
        """Test adding all types of elements to the figure."""
        self.fig[0] = self.fig.copy()
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
        self.fig[0] = Heatmap([[0, 1], [2, 3]], origin_position="lower")
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
        self.fig[0] = FitFromFunction(lambda x, a: a * x, Curve([0, 1, 2], [0, 1, 2]))
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
        self.fig[0] = Text(1, 1, "Test")
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
        self.fig[0] = PlottableAxMethod("bar", [1, 2, 3, 4], [1, 2, 3, 4])
        self.fig._initialize_parent_smart_figure()
        plt.close()

    def test_copy_and_copy_with(self):
        """Test copying and copying with modifications."""
        self.fig.x_label = "X"
        fig2 = self.fig.copy()
        self.assertEqual(fig2.x_label, "X")
        fig3 = self.fig.copy_with(x_label="Y")
        self.assertEqual(fig2.x_label, "X")
        self.assertEqual(fig3.x_label, "Y")
        with self.assertRaises(AttributeError):
            self.fig.copy_with(not_a_property=1)

    def test_show_and_save(self):
        """Test showing and saving the figure."""
        dummy = DummyPlottable()
        self.fig.add_elements(dummy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.fig.show(fullscreen=False)
        plt.close(self.fig._figure)
        self.fig.save("test_smart_figure_output.png")
        self.assertTrue(os.path.exists("test_smart_figure_output.png"))
        os.remove("test_smart_figure_output.png")

        # Test saving with PdfPages
        with PdfPages("test_smart_figure_output.pdf") as pdf:
            self.fig.save(pdf)
            self.fig.save(pdf, split_pdf=True)
        self.assertTrue(os.path.exists("test_smart_figure_output.pdf"))
        os.remove("test_smart_figure_output.pdf")

        # Test saving with split_pdf
        self.fig.save("test_smart_figure_output.pdf", split_pdf=True)
        self.assertTrue(os.path.exists("test_smart_figure_output.pdf"))
        os.remove("test_smart_figure_output.pdf")

        # Test saving with split_pdf and a non-PDF extension
        with self.assertLogs(level="WARNING") as log:
            self.fig.save("test_smart_figure_output.png", split_pdf=True)
            self.assertTrue(any("File extension" in record for record in log.output))
        self.assertTrue(os.path.exists("test_smart_figure_output.pdf"))
        os.remove("test_smart_figure_output.pdf")

        # Test saving with split_pdf and no extension
        with self.assertLogs(level="WARNING") as log:
            self.fig.save("figure", split_pdf=True)
            self.assertTrue(any("File extension" in record for record in log.output))
        self.assertTrue(os.path.exists("figure.pdf"))
        os.remove("figure.pdf")

    def test_auto_assign_default_params(self):
        """Test automatic assignment of default parameters."""
        x = linspace(0, 3 * pi, 200)
        a_curve = Curve(x, sin(x), label="Test Curve")
        self.fig.add_elements(a_curve)
        self.fig._default_params = self.plainDefaults
        self.fig._fill_in_missing_params(a_curve)
        self.assertEqual(a_curve._line_width, 2)

    def test_auto_assign_default_params_horrible(self):
        """Test automatic assignment of default parameters with horrible style."""
        x = linspace(0, 3 * pi, 200)
        a_curve = Curve(x, sin(x), label="Test Curve")
        a_figure = self.fig.copy_with(figure_style="horrible")
        a_figure.add_elements(a_curve)
        a_figure._default_params = self.horribleDefaults
        a_figure._fill_in_missing_params(a_curve)
        self.assertEqual(a_curve._line_width, 10)

    def test_auto_assign_default_params_skip_predefined(self):
        """Test automatic assignment of default parameters, skipping predefined ones."""
        x = linspace(0, 3 * pi, 200)
        a_curve = Curve(x, sin(x), label="Test Curve", line_width=3)
        self.fig.add_elements(a_curve)
        self.fig._default_params = self.plainDefaults
        self.fig._fill_in_missing_params(a_curve)
        self.assertEqual(a_curve._line_width, 3)

    def test_assign_figure_params_horrible(self):
        """Test figure parameter assignment with horrible style."""
        a_figure = self.fig.copy_with(figure_style="horrible")
        a_figure.add_elements(self.testCurve)
        a_figure._default_params = self.horribleDefaults
        a_figure._fill_in_missing_params(a_figure)
        self.assertListEqual(list(a_figure._size), [10, 7])

    def test_element_defaults_are_reset(self):
        """Test that element defaults are reset to defaults."""
        self.testCurve._line_width = "default"
        self.fig.add_elements(self.testCurve)
        self.fig._initialize_parent_smart_figure()
        self.assertEqual(self.testCurve._line_width, "default")
        self.fig._default_params = self.plainDefaults
        self.fig._fill_in_missing_params(self.testCurve)
        self.assertEqual(self.testCurve._line_width, 2)
        plt.close("all")

    def test_handles_and_labels_cleared(self):
        """Test that handles and labels are cleared when adding elements."""
        self.fig.add_elements(self.testCurve)
        self.fig._initialize_parent_smart_figure()
        handles, labels = self.fig._figure.get_axes()[0].get_legend_handles_labels()
        self.assertEqual(len(handles), 1)
        self.assertEqual(len(labels), 1)
        plt.close("all")

    def test_handles_and_labels_added(self):
        """Test that handles and labels are added correctly."""
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
        """Test filling in RC parameters for a GraphingLib style."""
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
        """Test updating RC parameters."""
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
        """Test resetting RC parameters."""
        params = {
            "lines.linewidth": 2,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
        }
        self.fig.set_rc_params(params)
        self.fig.set_rc_params({"lines.linewidth": 3}, reset=True)
        self.assertDictEqual(self.fig._user_rc_dict, {"lines.linewidth": 3})

    def test_customize_visual_style(self):
        """Test customizing visual style parameters."""
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
        """Test resetting visual style parameters."""
        self.fig.set_visual_params(figure_face_color="red", axes_face_color="blue")
        self.fig.set_visual_params(axes_face_color="yellow", legend_handle_length=1)
        self.fig.set_visual_params(reset=True)
        self.assertDictEqual(self.fig._user_rc_dict, {})

    def test_matplotlib_style_functional(self):
        """Test matplotlib style functionality."""
        a_figure = self.fig.copy_with(figure_style="_mpl-gallery")
        a_figure.add_elements(self.testCurve)
        a_figure._initialize_parent_smart_figure()
        plt.close("all")

    def test_matplotlib_keyword_style_functional(self):
        """Test that the 'matplotlib' keyword uses Matplotlib's default style."""
        with plt.style.context("default"):
            expected_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]

        curve = Curve([0, 1], [0, 1])
        a_figure = self.fig.copy_with(figure_style="matplotlib")
        a_figure.add_elements(curve)
        a_figure._initialize_parent_smart_figure()

        self.assertEqual(curve.handle[0].get_color(), expected_color)
        plt.close("all")

    def test_matplotlib_style_preserved_for_nested_smartfigure(self):
        """Test that nested SmartFigures keep matplotlib style defaults."""
        x_data = [0, 1, 2]
        y_data = [0, 1, 0]
        style = "Solarize_Light2"

        try:
            standalone_curve = Curve(x_data, y_data)
            standalone = SmartFigure(figure_style=style, elements=[standalone_curve])
            standalone._initialize_parent_smart_figure()
            standalone_ax = next(
                ax for ax in standalone._figure.get_axes() if ax.get_navigate()
            )
            standalone_facecolor = standalone_ax.get_facecolor()
            standalone_grid_visible = any(
                line.get_visible() for line in standalone_ax.get_xgridlines()
            )
            standalone_tick_color = (
                standalone_ax.get_xticklabels()[0].get_color()
                if standalone_ax.get_xticklabels()
                else None
            )
            standalone_line_color = standalone_curve.handle[0].get_color()

            nested_curve = Curve(x_data, y_data)
            child = SmartFigure(figure_style=style, elements=[nested_curve])
            parent = SmartFigure(
                num_cols=2,
                elements=[child, None],
                figure_style=style,
            )
            parent._initialize_parent_smart_figure()
            nested_ax = next(
                ax
                for ax in parent._get_all_axes_recursive(parent._figure)
                if ax.get_navigate()
            )
            nested_facecolor = nested_ax.get_facecolor()
            nested_grid_visible = any(
                line.get_visible() for line in nested_ax.get_xgridlines()
            )
            nested_tick_color = (
                nested_ax.get_xticklabels()[0].get_color()
                if nested_ax.get_xticklabels()
                else None
            )
            nested_line_color = nested_curve.handle[0].get_color()
        finally:
            plt.close("all")

        self.assertTrue(standalone_grid_visible)
        self.assertEqual(nested_grid_visible, standalone_grid_visible)
        self.assertEqual(nested_facecolor, standalone_facecolor)
        self.assertEqual(nested_tick_color, standalone_tick_color)
        self.assertEqual(nested_line_color, standalone_line_color)

    def test_set_ticks_and_tick_params(self):
        """Test setting ticks and tick parameters."""

        def tick_func(x):
            return f"Label {x}"

        self.fig.set_ticks(
            x_ticks=[0, 1],
            x_tick_labels=["a", "b"],
            y_ticks=[0, 1],
            y_tick_labels=["a", "b"],
        )
        self.fig.set_ticks(
            x_ticks=[0, 1],
            x_tick_labels=tick_func,
            y_ticks=[0, 1],
            y_tick_labels=tick_func,
        )
        self.fig.set_ticks(
            x_tick_spacing=0.5,
            x_tick_labels=tick_func,
            y_tick_spacing=0.5,
            y_tick_labels=tick_func,
        )
        self.fig.set_ticks(minor_x_ticks=[0.1, 0.2], minor_y_tick_spacing=0.5)
        self.fig.set_ticks(x_tick_spacing=None, y_tick_spacing=None)
        self.fig.set_tick_params(axis="x", which="major", direction="in", length=5)
        self.fig.set_tick_params(
            axis="y", which="minor", color="red", length=3, reset=True
        )
        self.assertDictEqual(
            self.fig._tick_params,
            {
                "x major": {"direction": "in", "length": 5},
                "x minor": {},
                "y major": {},
                "y minor": {"color": "red", "length": 3},
            },
        )
        self.fig.set_tick_params(axis="both", which="both", reset=True)
        self.assertDictEqual(
            self.fig._tick_params,
            {"x major": {}, "x minor": {}, "y major": {}, "y minor": {}},
        )
        with self.assertRaises(GraphingException):
            self.fig.set_ticks(x_tick_labels=["a"], x_ticks=None)
        with self.assertRaises(GraphingException):
            self.fig.set_ticks(x_ticks=[1], x_tick_spacing=1)
        with self.assertRaises(GraphingException):
            self.fig.set_ticks(minor_x_ticks=[1], minor_x_tick_spacing=1)
        with self.assertRaises(GraphingException):
            self.fig.set_ticks(
                x_ticks=[0, 1, 2, 3], x_tick_labels=["only one label but 4 ticks"]
            )

    def test_set_grid(self):
        """Test setting grid parameters."""
        self.fig.set_grid(
            visible_x=True,
            visible_y=False,
            color="blue",
            alpha=0.5,
            line_style="--",
            line_width=2,
        )
        self.assertTrue(self.fig.show_grid)
        self.fig.show_grid = False
        self.fig.set_grid()
        self.assertTrue(self.fig.show_grid)

    def test_set_custom_legend(self):
        """Test setting custom legend elements."""
        with self.assertRaises(TypeError):
            self.fig.set_custom_legend(elements=["Not a LegendElement"])
        with self.assertRaises(TypeError):
            self.fig.set_custom_legend(elements="Not a good iterable")
        self.fig.set_custom_legend()
        line = LegendLine(label="Test line", color="green")
        self.fig.set_custom_legend(elements=[line])
        self.fig.set_custom_legend(elements=(line,))
        self.assertEqual(len(self.fig._custom_legend_handles), 2)
        self.assertEqual(self.fig._custom_legend_labels, ["Test line", "Test line"])
        self.fig.set_custom_legend(reset=True)
        self.assertEqual(self.fig._custom_legend_handles, [])
        self.assertEqual(self.fig._custom_legend_labels, [])
        with self.assertRaises(TypeError):
            self.fig.set_custom_legend(elements=[line, "Still not a LegendElement..."])

    def test_set_text_padding_params(self):
        """Test setting text padding parameters."""
        self.fig_2x2.elements = [DummyPlottable() for _ in range(4)]
        self.fig_2x2._default_params = FileLoader("plain").load()
        self.fig_2x2._reference_label_i = 0
        self.fig_2x2._figure = plt.figure()

        # Test initial state (empty _pad_params)
        self.assertEqual(self.fig_2x2._pad_params, {})

        # Test setting individual padding parameters
        self.fig_2x2.set_text_padding_params(x_label_pad=10.0)
        self.assertEqual(self.fig_2x2._pad_params["x_label_pad"], 10.0)

        self.fig_2x2.set_text_padding_params(y_label_pad=15.5, title_pad=1.0)
        self.assertEqual(self.fig_2x2._pad_params["y_label_pad"], 15.5)
        self.assertEqual(
            self.fig_2x2._pad_params["x_label_pad"], 10.0
        )  # Should preserve previous value
        self.assertEqual(self.fig_2x2._pad_params["title_pad"], 1.0)

        # Test setting multiple parameters at once
        self.fig_2x2.set_text_padding_params(
            x_label_pad=5.0, y_label_pad=7.5, title_pad=12.0
        )
        self.assertEqual(self.fig_2x2._pad_params["x_label_pad"], 5.0)
        self.assertEqual(self.fig_2x2._pad_params["y_label_pad"], 7.5)
        self.assertEqual(self.fig_2x2._pad_params["title_pad"], 12.0)

        # Test setting iterable padding parameters
        self.fig_2x2.set_text_padding_params(
            sub_x_labels_pad=[1.0, 2.0, 3.0, 4.0],
            sub_y_labels_pad=(4.5, 5.5, None, None),
            subtitles_pad=array([6.0, 7.0, 8.0, 9.0]),
        )
        self.assertEqual(
            self.fig_2x2._pad_params["sub_x_labels_pad"], [1.0, 2.0, 3.0, 4.0]
        )
        self.assertEqual(
            self.fig_2x2._pad_params["sub_y_labels_pad"], (4.5, 5.5, None, None)
        )
        self.assertEqual(list(self.fig_2x2._pad_params["subtitles_pad"]), [6, 7, 8, 9])

        # Test reset functionality
        self.fig_2x2.set_text_padding_params(reset=True, x_label_pad=25.0)
        self.assertEqual(self.fig_2x2._pad_params["x_label_pad"], 25.0)
        self.fig_2x2.set_text_padding_params(reset=True)
        self.assertEqual(self.fig_2x2._pad_params, {})

        # Test invalid types for single padding parameters
        with self.assertRaises(TypeError):
            self.fig_2x2.set_text_padding_params(x_label_pad="invalid")
        with self.assertRaises(TypeError):
            self.fig_2x2.set_text_padding_params(
                y_label_pad=[1, 2]
            )  # Should be float/int, not list

        # Test invalid types for iterable padding parameters
        with self.assertRaises(TypeError):
            self.fig_2x2.set_text_padding_params(
                sub_x_labels_pad="invalid"
            )  # String not iterable of numbers
        with self.assertRaises(TypeError):
            self.fig_2x2.set_text_padding_params(
                sub_y_labels_pad=[1, 2, "invalid"]
            )  # Invalid element in iterable
        with self.assertRaises(TypeError):
            self.fig_2x2.set_text_padding_params(sub_x_labels_pad=123)  # Not iterable

        # Test empty iterables are valid
        self.fig_2x2.set_text_padding_params(sub_x_labels_pad=[])
        self.assertEqual(self.fig_2x2._pad_params["sub_x_labels_pad"], [])

        # Test list lengths
        self.fig_2x2.set_text_padding_params(
            sub_x_labels_pad=[1, 2, None]
        )  # Underfilled list
        self.fig_2x2._prepare_figure()
        with self.assertRaises(GraphingException):
            self.fig_2x2.set_text_padding_params(
                sub_y_labels_pad=[1.0, 2.0, None, None, 3.0]
            )  # Overfilled list
            self.fig_2x2._prepare_figure()

    def test_set_reference_labels_params(self):
        """Test setting reference labels parameters."""
        # Test initial state (empty _reference_labels_params)
        self.assertEqual(self.fig_2x2._reference_labels_params, {})

        # Test setting individual reference label parameters
        self.fig_2x2.set_reference_labels_params(font_size=12.0)
        self.assertEqual(self.fig_2x2._reference_labels_params["font_size"], 12.0)

        self.fig_2x2.set_reference_labels_params(
            color="red", font_weight="bold", start_index=2
        )
        self.assertEqual(self.fig_2x2._reference_labels_params["color"], "red")
        self.assertEqual(self.fig_2x2._reference_labels_params["font_weight"], "bold")
        self.assertEqual(self.fig_2x2._reference_labels_params["start_index"], 2)
        self.assertEqual(
            self.fig_2x2._reference_labels_params["font_size"], 12.0
        )  # Should preserve previous value

        # Test reset functionality
        self.fig_2x2.set_reference_labels_params(reset=True, font_size=20.0)
        self.assertEqual(self.fig_2x2._reference_labels_params["font_size"], 20.0)
        self.fig_2x2.set_reference_labels_params(reset=True)
        self.assertEqual(self.fig_2x2._reference_labels_params, {})

        # Test invalid types for start index
        with self.assertRaises(TypeError):
            self.fig_2x2.set_reference_labels_params(start_index="invalid")
        with self.assertRaises(ValueError):
            self.fig_2x2.set_reference_labels_params(start_index=-1)

    def test_methods_return_self(self):
        """Test that methods return self for method chaining."""
        self.assertIs(self.fig.add_elements(), self.fig)
        self.assertIs(self.fig.set_ticks(x_ticks=[0, 1]), self.fig)
        self.assertIs(self.fig.set_tick_params(label_color="green"), self.fig)
        self.assertIs(self.fig.set_grid(visible_x=True), self.fig)
        self.assertIs(self.fig.set_custom_legend(elements=[]), self.fig)
        self.assertIs(self.fig.set_text_padding_params(x_label_pad=5.0), self.fig)
        self.assertIs(self.fig.set_reference_labels_params(font_size=5.0), self.fig)
        self.assertIs(self.fig.set_visual_params(figure_face_color="red"), self.fig)
        self.assertIs(self.fig.set_rc_params({"lines.linewidth": 2}), self.fig)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertIs(self.fig.show(), self.fig)
        self.assertIs(self.fig.save("test_smart_figure_output.png"), self.fig)
        os.remove("test_smart_figure_output.png")

    def test_hide_spines(self):
        """Test hiding spines."""
        self.assertIsNone(self.fig._hidden_spines)

        # Invalid
        with self.assertRaises(TypeError):
            self.fig.set_visual_params(hidden_spines=123)
        with self.assertRaises(ValueError):
            self.fig.set_visual_params(hidden_spines="right")
        with self.assertRaises(ValueError):
            self.fig.set_visual_params(hidden_spines=["I am not a direction"])
        with self.assertRaises(ValueError):
            self.fig.set_visual_params(
                hidden_spines=["left", "I am still not a direction", "right"]
            )

        # Valid
        self.fig.set_visual_params(hidden_spines=["left", "right"])
        self.assertListEqual(self.fig._hidden_spines, ["left", "right"])
        self.fig.set_visual_params()
        self.assertListEqual(self.fig._hidden_spines, ["left", "right"])
        self.fig.set_visual_params(hidden_spines=["top"])
        self.assertListEqual(self.fig._hidden_spines, ["top"])

    def test_create_twin_axis_from_smart_figure(self):
        """Test creating twin axis from SmartFigure."""
        twin_y = self.fig.create_twin_axis(is_y=True, label="Twin Y")
        self.assertIsInstance(twin_y, SmartTwinAxis)
        self.assertEqual(twin_y.label, "Twin Y")
        self.assertIs(self.fig._twin_y_axis, twin_y)

        # Test creating x-axis twin
        twin_x = self.fig.create_twin_axis(is_y=False, label="Twin X")
        self.assertIsInstance(twin_x, SmartTwinAxis)
        self.assertEqual(twin_x.label, "Twin X")
        self.assertIs(self.fig._twin_x_axis, twin_x)

        # Test creating in the __init__
        new_fig = self.fig.copy_with(twin_x_axis=twin_x)
        self.assertIsInstance(new_fig.twin_x_axis, SmartTwinAxis)
        self.assertEqual(new_fig.twin_x_axis.label, "Twin X")

    def test_create_twin_axis_validation(self):
        """Test twin axis creation validation."""
        # Test with multi-subplot figure
        with self.assertRaises(GraphingException):
            self.fig_2x2.create_twin_axis()

        with self.assertRaises(GraphingException):
            self.fig_2x2.twin_x_axis = self.twin_axis

        with self.assertRaises(GraphingException):
            self.fig_2x2.twin_y_axis = self.twin_axis

        # Test duplicate twin axis creation
        self.fig.create_twin_axis(is_y=True)
        with self.assertRaises(GraphingException):
            self.fig.create_twin_axis(is_y=True)

        # Test duplicate x-axis twin
        self.fig.create_twin_axis(is_y=False)
        with self.assertRaises(GraphingException):
            self.fig.create_twin_axis(is_y=False)

    def test_twin_axis_integration(self):
        """Test twin axis integration with SmartFigure plotting."""
        # Add main curve
        self.fig.add_elements(self.testCurve)

        # Create and populate twin axis
        twin_y = self.fig.create_twin_axis(is_y=True, label="Twin Y", log_scale=True)
        twin_y.add_elements(Curve([0], [0], label="Twin Curve"))

        # Test that the figure can be prepared
        self.fig._default_params = FileLoader("plain").load()
        self.fig._figure = plt.figure()
        legend_info = self.fig._prepare_figure(make_legend=False)

        # Check that both curves are in the legend
        all_labels = legend_info["labels"]["default"]
        self.assertIn("Test Curve", all_labels)
        self.assertIn("Twin Curve", all_labels)

        plt.close(self.fig._figure)

    def test_twin_axis_both_x_and_y(self):
        """Test creating both x and y twin axes."""
        twin_x = self.fig.create_twin_axis(is_y=False, label="Twin X")
        twin_y = self.fig.create_twin_axis(is_y=True, label="Twin Y")

        # Add elements to both twins
        twin_x.add_elements(Curve([0], [0], label="X Twin"))
        twin_y.add_elements(Curve([0], [0], label="Y Twin"))
        self.fig.add_elements(Curve([0], [0], label="Main"))

        # Test integration
        self.fig._default_params = FileLoader("plain").load()
        self.fig._figure = plt.figure()
        legend_info = self.fig._prepare_figure(make_legend=False)

        # Check all elements are in legend
        all_labels = legend_info["labels"]["default"]
        self.assertIn("Main", all_labels)
        self.assertIn("X Twin", all_labels)
        self.assertIn("Y Twin", all_labels)

        plt.close(self.fig._figure)


@unittest.skipUnless(
    HAS_ASTROPY,
    "Install the optional extra with `pip install graphinglib[astro]` to run WCS tests.",
)
# Test suite for SmartFigureWCS, inheriting from TestSmartFigure to avoid code duplication
class TestSmartFigureWCS(TestSmartFigure):
    """
    Test class for SmartFigureWCS that inherits from TestSmartFigure to avoid code duplication.
    Only tests the specific differences and additions of SmartFigureWCS.
    """

    def setUp(self):
        # Create a simple WCS object for testing
        self.wcs = WCS(naxis=2)
        self.wcs.wcs.crpix = [1, 1]
        self.wcs.wcs.cdelt = [1, 1]
        self.wcs.wcs.crval = [0, 0]
        self.wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        # Create SmartFigureWCS instances
        self.cls = SmartFigureWCS
        self.fig = SmartFigureWCS(projection=self.wcs)
        self.fig_2x3 = SmartFigureWCS(projection=self.wcs, num_rows=2, num_cols=3)
        self.fig_1x4 = SmartFigureWCS(projection=self.wcs, num_rows=1, num_cols=4)
        self.fig_3x1 = SmartFigureWCS(projection=self.wcs, num_rows=3, num_cols=1)
        self.fig_2x2 = SmartFigureWCS(projection=self.wcs, num_rows=2, num_cols=2)
        self.twin_axis = SmartTwinAxis()

        # Set up test data
        x = linspace(0, 3 * pi, 200)
        self.testCurve = Curve(x, sin(x), "Test Curve", color="k")
        self.plainDefaults = FileLoader("plain").load()
        self.horribleDefaults = FileLoader("horrible").load()

    def test_init_requires_wcs_projection(self):
        """Test that SmartFigureWCS requires a WCS projection object."""
        # Valid WCS projection
        self.assertIsInstance(self.fig, self.cls)
        self.assertEqual(self.fig.projection, self.wcs)

        # Invalid projection types should raise exception
        with self.assertRaises(GraphingException):
            self.cls(projection="polar")

        with self.assertRaises(GraphingException):
            self.cls(projection=None)

        with self.assertRaises(GraphingException):
            self.cls(projection=123)

    def test_elements_in_init(self):
        """Test elements initialization in constructor."""
        # Invalid formats
        with self.assertRaises(TypeError):
            SmartFigureWCS(self.wcs, elements=[1])
        with self.assertRaises(TypeError):
            SmartFigureWCS(self.wcs, elements=(DummyPlottable()))
        with self.assertRaises(TypeError):
            SmartFigureWCS(self.wcs, elements=DummyPlottable())
        with self.assertRaises(TypeError):
            SmartFigureWCS(self.wcs, elements=[[[DummyPlottable()]]])
        with self.assertRaises(TypeError):
            SmartFigureWCS(self.wcs, elements="invalid")
        with self.assertRaises(TypeError):
            SmartFigureWCS(self.wcs, elements=["invalid"])
        with self.assertRaises(TypeError):
            SmartFigureWCS(self.wcs, elements=[["invalid"]])

        # Valid
        SmartFigureWCS(self.wcs, elements=(DummyPlottable(),))
        SmartFigureWCS(self.wcs, 2, elements=[(DummyPlottable(),), [None]])
        SmartFigureWCS(self.wcs, 2, elements=array([(DummyPlottable(),), [None]]))
        SmartFigureWCS(self.wcs, 2, elements=(DummyPlottable(), DummyPlottable()))
        SmartFigureWCS(self.wcs, elements=((DummyPlottable(), DummyPlottable())))
        SmartFigureWCS(self.wcs, elements=(DummyPlottable(), DummyPlottable()))

    def test_projection(self):
        """Test the projection property getter and setter."""
        # Valid WCS projection
        new_wcs = WCS(naxis=2)
        new_wcs.wcs.crpix = [2, 2]
        new_wcs.wcs.cdelt = [0.5, 0.5]
        new_wcs.wcs.crval = [10, 10]
        new_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        self.fig.projection = new_wcs
        self.assertEqual(self.fig.projection, new_wcs)

        # Invalid projection types should raise exception
        with self.assertRaises(GraphingException):
            self.fig.projection = "polar"

        with self.assertRaises(GraphingException):
            self.fig.projection = None

    def test_projection_list_or_item(self):
        """Test projection property with list or single item assignment."""
        self.fig_2x3[:, 0] = DummyPlottable()
        self.fig_2x3[0, 1] = DummyPlottable()
        self.fig_2x3[0, 2] = DummyPlottable()
        self.fig_2x3[1, 2] = DummyPlottable()
        # leave the [1,1] subplot empty

        self.fig_2x3._default_params = FileLoader("plain").load()
        self.fig_2x3._reference_label_i = 0
        self.fig_2x3._figure = plt.figure()

        self.fig_2x3.projection = [
            self.wcs,
            self.wcs,
            self.wcs,
            self.wcs,
        ]  # equally filled list
        self.fig_2x3._prepare_figure()
        plt.close(self.fig._figure)
        with self.assertRaises(GraphingException):
            self.fig_2x3.projection = [self.wcs, self.wcs, self.wcs]  # underfilled list
            self.fig_2x3._prepare_figure()
        with self.assertRaises(GraphingException):
            self.fig_2x3.projection = [
                self.wcs,
                self.wcs,
                self.wcs,
                self.wcs,
                self.wcs,
            ]  # overfilled list
            self.fig_2x3._prepare_figure()

    def test_add_all_elements(self):
        """Test adding all types of elements to the figure."""
        self.fig[0] = self.fig.copy()
        self.fig._initialize_parent_smart_figure()
        plt.close()
        self.fig[0] = Curve([0, 1], [0, 1])
        self.fig._initialize_parent_smart_figure()
        plt.close()
        self.fig[0] = Scatter([0, 1], [0, 1])
        self.fig._initialize_parent_smart_figure()
        plt.close()
        self.fig[0] = Heatmap([[0, 1], [2, 3]], origin_position="lower")
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
        self.fig[0] = FitFromFunction(lambda x, a: a * x, Curve([0, 1, 2], [0, 1, 2]))
        self.fig._initialize_parent_smart_figure()
        plt.close()
        self.fig[0] = Hlines([0, 1, 2])
        self.fig._initialize_parent_smart_figure()
        plt.close()
        self.fig[0] = Vlines([0, 1, 2])
        self.fig._initialize_parent_smart_figure()
        plt.close()
        self.fig[0] = Text(1, 1, "Test")
        self.fig._initialize_parent_smart_figure()
        plt.close()
        self.fig[0] = Text(1, 1, "Test")
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

    def test_wcs_specific_attributes(self):
        """Test that SmartFigureWCS has WCS-specific attributes."""
        self.assertIsNone(self.fig._ticks.get("number_of_x_ticks"))
        self.assertIsNone(self.fig._ticks.get("number_of_y_ticks"))
        self.assertIsNone(self.fig._ticks.get("x_tick_formatter"))
        self.assertIsNone(self.fig._ticks.get("y_tick_formatter"))
        self.assertIsNone(self.fig._ticks.get("minor_x_tick_frequency"))
        self.assertIsNone(self.fig._ticks.get("minor_y_tick_frequency"))

        # Check default tick params
        expected_default_tick_params = {
            "x major": {"bottom": True, "top": True, "labelbottom": True},
            "y major": {"left": True, "right": True, "labelleft": True},
            "x minor": {},
            "y minor": {},
        }
        self.assertEqual(self.fig._default_tick_params, expected_default_tick_params)
        self.assertEqual(self.fig._tick_params, expected_default_tick_params)

    def test_set_ticks_wcs_specific(self):
        """Test the WCS-specific set_ticks method."""
        # Test with Quantity objects
        x_ticks = [0, 1, 2] * u.deg
        y_ticks = [0, 1, 2] * u.deg
        x_spacing = 0.5 * u.deg
        y_spacing = 0.5 * u.deg

        # Valid usage
        result = self.fig.set_ticks(
            number_of_x_ticks=5,
            number_of_y_ticks=5,
            x_tick_formatter="hh:mm:ss",
            y_tick_formatter=lambda x: f"{x:.2f}",
            minor_x_tick_frequency=2,
            minor_y_tick_frequency=3,
        )

        # Should return self
        self.assertIs(result, self.fig)

        # Check that attributes are set
        self.assertEqual(self.fig._ticks.get("number_of_x_ticks"), 5)
        self.assertEqual(self.fig._ticks.get("number_of_y_ticks"), 5)
        self.assertEqual(self.fig._ticks.get("x_tick_formatter"), "hh:mm:ss")
        self.assertTrue(callable(self.fig._ticks.get("y_tick_formatter")))
        self.assertEqual(self.fig._ticks.get("minor_x_tick_frequency"), 2)
        self.assertEqual(self.fig._ticks.get("minor_y_tick_frequency"), 3)

        # Test invalid combinations - ticks and number of ticks
        with self.assertRaises(GraphingException):
            self.fig.set_ticks(x_ticks=x_ticks, number_of_x_ticks=5)

        with self.assertRaises(GraphingException):
            self.fig.set_ticks(y_ticks=y_ticks, number_of_y_ticks=5)

        # Test invalid combinations - spacing and number of ticks
        with self.assertRaises(GraphingException):
            self.fig.set_ticks(x_tick_spacing=x_spacing, number_of_x_ticks=5)

        with self.assertRaises(GraphingException):
            self.fig.set_ticks(y_tick_spacing=y_spacing, number_of_y_ticks=5)

    def test_set_ticks_and_tick_params(self):
        """Test the WCS-specific set_tick_params method."""
        # Test setting parameters for both axes
        result = self.fig.set_tick_params(
            axis="both",
            direction="in",
            length=10,
            minor_length=5,
            width=2,
            color="red",
            pad=5,
            label_size=12,
            label_color="blue",
            label_rotation=45,
            draw_bottom_ticks=False,
            draw_top_ticks=True,
            draw_left_ticks=False,
            draw_right_ticks=True,
            draw_bottom_labels=False,
            draw_top_labels=True,
            draw_left_labels=False,
            draw_right_labels=True,
        )

        # Should return self
        self.assertIs(result, self.fig)

        # Check that parameters are set for both axes
        expected_major_params = {
            "bottom": False,
            "top": True,
            "labelbottom": False,
            "labeltop": True,
            "left": False,
            "right": True,
            "labelleft": False,
            "labelright": True,
            "direction": "in",
            "length": 10,
            "width": 2,
            "color": "red",
            "pad": 5,
            "labelsize": 12,
            "labelcolor": "blue",
            "labelrotation": 45,
        }

        self.assertEqual(self.fig._tick_params["x major"], expected_major_params)
        self.assertEqual(self.fig._tick_params["y major"], expected_major_params)
        self.assertEqual(self.fig._tick_params["x minor"], {"length": 5})
        self.assertEqual(self.fig._tick_params["y minor"], {"length": 5})

        # Test setting parameters for individual axes
        self.fig.set_tick_params(axis="x", length=15)
        self.assertEqual(self.fig._tick_params["x major"]["length"], 15)
        self.assertEqual(
            self.fig._tick_params["y major"]["length"], 10
        )  # Should remain unchanged

        # Test reset functionality
        self.fig.set_tick_params(axis="x", reset=True, length=20)
        expected_reset_params = {
            "bottom": True,
            "top": True,
            "labelbottom": True,
            "length": 20,
        }
        self.assertEqual(self.fig._tick_params["x major"], expected_reset_params)
        self.assertEqual(self.fig._tick_params["x minor"], {})

    def test_set_grid_wcs_specific(self):
        """Test the WCS-specific set_grid method."""
        # Test that it calls the parent method with major ticks only
        result = self.fig.set_grid(
            visible_x=True,
            visible_y=False,
            color="red",
            alpha=0.5,
            line_style="--",
            line_width=2,
        )

        # Should return self
        self.assertIs(result, self.fig)

        # Check that grid parameters are set (inherited behavior)
        self.assertTrue(self.fig._show_grid)

    def test_inheritance_behavior(self):
        """Test that SmartFigureWCS inherits all SmartFigure functionality."""
        # Test that all inherited properties work
        self.fig.x_label = "RA"
        self.fig.y_label = "Dec"
        self.fig.title = "WCS Test Figure"

        self.assertEqual(self.fig.x_label, "RA")
        self.assertEqual(self.fig.y_label, "Dec")
        self.assertEqual(self.fig.title, "WCS Test Figure")

        # Test that plotting still works
        dummy = DummyPlottable()
        self.fig.add_elements(dummy)
        self.assertEqual(len(self.fig), 1)

    def test_init_custom_args(self):
        """Test SmartFigureWCS initialization with all parameters."""
        elements = [DummyPlottable(), DummyPlottable()]
        fig = self.cls(
            projection=self.wcs,
            num_rows=2,
            num_cols=2,
            x_label="RA",
            y_label="Dec",
            size=(10, 8),
            title="Test WCS Figure",
            x_lim=(0, 10),
            y_lim=(-5, 5),
            sub_x_labels=["RA1", "RA2"],
            sub_y_labels=["Dec1", "Dec2"],
            subtitles=["Title 1", "Title 2", "Title 3", "Title 4"],
            log_scale_x=False,
            log_scale_y=False,
            remove_axes=False,
            aspect_ratio="equal",
            box_aspect_ratio=0.7,
            remove_x_ticks=False,
            remove_y_ticks=False,
            invert_x_axis=True,
            invert_y_axis=True,
            reference_labels=True,
            global_reference_label=False,
            reference_labels_loc="outside",
            width_padding=0.1,
            height_padding=0.1,
            width_ratios=[1, 2],
            height_ratios=[1, 2],
            share_x=False,
            share_y=False,
            general_legend=False,
            legend_loc="best",
            legend_cols=1,
            show_legend=True,
            figure_style="default",
            elements=elements,
        )

        # Test that WCS-specific attributes are properly initialized
        self.assertEqual(fig.projection, self.wcs)
        self.assertIsNone(fig._ticks.get("number_of_x_ticks"))
        self.assertIsNone(fig._ticks.get("number_of_y_ticks"))

        # Test that inherited attributes work
        self.assertEqual(fig.num_rows, 2)
        self.assertEqual(fig.num_cols, 2)
        self.assertEqual(fig.x_label, "RA")
        self.assertEqual(fig.y_label, "Dec")

    def test_methods_return_self_wcs(self):
        """Test that WCS-specific methods return self for method chaining."""
        result = (
            self.fig.set_ticks(number_of_x_ticks=5)
            .set_tick_params(direction="in")
            .set_grid(visible_x=True)
            .set_text_padding_params(x_label_pad=5.0)
            .set_reference_labels_params(color="green")
        )

        self.assertIs(result, self.fig)

    def test_twin_axis_with_wcs_figure(self):
        """Test twin axis with SmartFigureWCS."""
        # Create a simple WCS object
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [0, 0]
        wcs.wcs.crpix = [1, 1]
        wcs.wcs.cdelt = [0.1, 0.1]
        wcs.wcs.cunit = ["deg", "deg"]

        # Test creating twin axis
        twin_y = self.fig.create_twin_axis(is_y=True, label="WCS Twin")
        twin_y.add_elements(self.testCurve)
        self.fig.add_elements(self.testCurve)

        # Test integration
        self.fig._default_params = FileLoader("plain").load()
        self.fig._figure = plt.figure()
        self.fig._prepare_figure()

        plt.close(self.fig._figure)


class TestSmartTwinAxis(unittest.TestCase):
    """Test class for SmartTwinAxis functionality."""

    def setUp(self):
        self.fig = SmartFigure()
        self.twin_axis = SmartTwinAxis()
        self.x = linspace(0, 3 * pi, 200)
        self.curve1 = Curve(self.x, sin(self.x), "Main Curve", color="blue")
        self.curve2 = Curve(self.x, self.x**2, "Twin Curve", color="red")
        self.plainDefaults = FileLoader("plain").load()
        self.horribleDefaults = FileLoader("horrible").load()

    def test_init_defaults(self):
        """Test SmartTwinAxis initialization with default parameters."""
        self.assertIsNone(self.twin_axis.label)
        self.assertIsNone(self.twin_axis.axis_lim)
        self.assertFalse(self.twin_axis.log_scale)
        self.assertFalse(self.twin_axis.remove_axes)
        self.assertFalse(self.twin_axis.remove_ticks)
        self.assertFalse(self.twin_axis.invert_axis)
        self.assertEqual(len(self.twin_axis._elements), 0)
        self.assertIsNone(self.twin_axis._ticks.get("ticks"))
        self.assertIsNone(self.twin_axis._ticks.get("tick_labels"))
        self.assertIsNone(self.twin_axis._ticks.get("tick_spacing"))
        self.assertIsNone(self.twin_axis._ticks.get("minor_ticks"))
        self.assertIsNone(self.twin_axis._ticks.get("minor_tick_spacing"))
        self.assertEqual(self.twin_axis._tick_params, {"major": {}, "minor": {}})
        self.assertIsNone(self.twin_axis._edge_color)
        self.assertIsNone(self.twin_axis._line_width)
        self.assertEqual(self.twin_axis._user_rc_dict, {})
        self.assertEqual(self.twin_axis._default_params, {})

    def test_init_custom_parameters(self):
        """Test SmartTwinAxis initialization with custom parameters."""
        twin_axis = SmartTwinAxis(
            label="Custom Twin Axis",
            axis_lim=(0, 10),
            log_scale=True,
            remove_axes=True,
            remove_ticks=True,
            invert_axis=True,
            elements=[self.curve1, self.curve2],
        )

        self.assertEqual(twin_axis.label, "Custom Twin Axis")
        self.assertEqual(twin_axis.axis_lim, (0, 10))
        self.assertTrue(twin_axis.log_scale)
        self.assertTrue(twin_axis._remove_axes)
        self.assertTrue(twin_axis._remove_ticks)
        self.assertTrue(twin_axis.invert_axis)
        self.assertEqual(len(twin_axis._elements), 2)

    def test_axis_lim(self):
        """Test axis_lim property validation and assignment."""
        # Invalid
        with self.assertRaises(TypeError):
            self.twin_axis.axis_lim = 1
        with self.assertRaises(ValueError):
            self.twin_axis.axis_lim = (1, 2, 3)
        # Valid
        self.twin_axis.axis_lim = (0, 10)
        self.assertEqual(self.twin_axis.axis_lim, (0, 10))

    def test_log_scale(self):
        """Test log_scale property validation and assignment."""
        # Invalid
        with self.assertRaises(TypeError):
            self.twin_axis.log_scale = "yes"
        with self.assertRaises(TypeError):
            self.twin_axis.log_scale = 1
        # Valid
        self.twin_axis.log_scale = True
        self.assertTrue(self.twin_axis.log_scale)

    def test_remove_axes(self):
        """Test remove_axes property validation and assignment."""
        # Invalid
        with self.assertRaises(TypeError):
            self.twin_axis.remove_axes = "yes"
        with self.assertRaises(TypeError):
            self.twin_axis.remove_axes = 1
        # Valid
        self.twin_axis.remove_axes = True
        self.assertTrue(self.twin_axis.remove_axes)

    def test_remove_ticks(self):
        """Test remove_ticks property validation and assignment."""
        # Invalid
        with self.assertRaises(TypeError):
            self.twin_axis.remove_ticks = "yes"
        with self.assertRaises(TypeError):
            self.twin_axis.remove_ticks = 1
        # Valid
        self.twin_axis.remove_ticks = True
        self.assertTrue(self.twin_axis.remove_ticks)

    def test_invert_axis(self):
        """Test invert_axis property validation and assignment."""
        # Invalid
        with self.assertRaises(TypeError):
            self.twin_axis.invert_axis = "yes"
        with self.assertRaises(TypeError):
            self.twin_axis.invert_axis = 1
        # Valid
        self.twin_axis.invert_axis = True
        self.assertTrue(self.twin_axis.invert_axis)

    def test_elements(self):
        """Test elements property validation and assignment."""
        # Invalid
        with self.assertRaises(TypeError):
            self.twin_axis.elements = "not_a_list"
        with self.assertRaises(TypeError):
            self.twin_axis.elements = [1, 2, 3]
        # Valid
        self.twin_axis.elements = [DummyPlottable(), DummyPlottable()]
        self.assertEqual(len(self.twin_axis.elements), 2)

        # Setting new elements will clear existing ones
        self.twin_axis.elements = [
            DummyPlottable(),
            DummyPlottable(),
            None,
            DummyPlottable(),
        ]
        self.assertEqual(len(self.twin_axis.elements), 3)

    def test_len(self):
        """Test length calculation for twin axis elements."""
        self.twin_axis.elements = [self.curve1, self.curve2]
        self.assertEqual(len(self.twin_axis), 2)

        self.twin_axis.elements = [None, self.curve1, self.curve2, None]
        self.assertEqual(len(self.twin_axis), 2)

        self.twin_axis.elements = []
        self.assertEqual(len(self.twin_axis), 0)

        self.twin_axis.elements = [None, None]
        self.assertEqual(len(self.twin_axis), 0)

    def test_getitem(self):
        """Test item retrieval from twin axis elements."""
        self.twin_axis.elements = [self.curve1, self.curve2]
        self.assertEqual(self.twin_axis[0].label, self.curve1.label)
        self.assertEqual(self.twin_axis[1].label, self.curve2.label)
        self.assertEqual(self.twin_axis[-1].label, self.curve2.label)
        self.assertEqual(self.twin_axis[-2].label, self.curve1.label)

        with self.assertRaises(TypeError):
            _ = self.twin_axis["not_an_index"]
        with self.assertRaises(IndexError):
            _ = self.twin_axis[2]
        with self.assertRaises(IndexError):
            _ = self.twin_axis[-3]

    def test_iter(self):
        """Test iteration over twin axis elements."""
        self.twin_axis.elements = [self.curve1, self.curve2]
        elements = list(iter(self.twin_axis))
        self.assertEqual(len(elements), 2)
        self.assertIs(elements[0], self.curve1)
        self.assertIs(elements[1], self.curve2)

    def test_copy_and_copy_with(self):
        """Test copying and copying with modifications for twin axis."""
        self.twin_axis.label = "X"
        twin_axis_2 = self.twin_axis.copy()
        self.assertEqual(twin_axis_2.label, "X")
        twin_axis_3 = self.twin_axis.copy_with(label="Y")
        self.assertEqual(twin_axis_2.label, "X")
        self.assertEqual(twin_axis_3.label, "Y")
        with self.assertRaises(AttributeError):
            self.twin_axis.copy_with(not_a_property=1)

    def test_add_elements(self):
        """Test adding elements to twin axis."""
        # Add single element
        self.twin_axis.add_elements(self.curve1)
        self.assertEqual(len(self.twin_axis._elements), 1)
        self.assertIs(self.twin_axis._elements[0], self.curve1)

        # Add multiple elements
        self.twin_axis.add_elements(self.curve2, DummyPlottable())
        self.assertEqual(len(self.twin_axis._elements), 3)

        # Test invalid element type
        with self.assertRaises(TypeError):
            self.twin_axis.add_elements("not_plottable")

    def test_auto_assign_default_params(self):
        """Test automatic assignment of default parameters for twin axis."""
        x = linspace(0, 3 * pi, 200)
        a_curve = Curve(x, sin(x), label="Test Curve")
        self.twin_axis.add_elements(a_curve)
        self.twin_axis._default_params = self.plainDefaults
        self.twin_axis._fill_in_missing_params(a_curve, "plain")
        self.assertEqual(a_curve._line_width, 2)

    def test_auto_assign_default_params_horrible(self):
        """Test automatic assignment of default parameters with horrible style for twin axis."""
        x = linspace(0, 3 * pi, 200)
        a_curve = Curve(x, sin(x), label="Test Curve")
        self.twin_axis.add_elements(a_curve)
        self.twin_axis._default_params = self.horribleDefaults
        self.twin_axis._fill_in_missing_params(a_curve, "horrible")
        self.assertEqual(a_curve._line_width, 10)

    def test_auto_assign_default_params_skip_predefined(self):
        """Test automatic assignment of default parameters with predefined values for twin axis."""
        x = linspace(0, 3 * pi, 200)
        a_curve = Curve(x, sin(x), label="Test Curve", line_width=3)
        self.twin_axis.add_elements(a_curve)
        self.twin_axis._default_params = self.plainDefaults
        self.twin_axis._fill_in_missing_params(a_curve, "plain")
        self.assertEqual(a_curve._line_width, 3)

    def test_element_defaults_are_reset(self):
        """Test that element defaults are reset when using default parameters."""
        self.curve1._line_width = "default"
        self.twin_axis.add_elements(self.curve1)
        self.twin_axis._default_params = FileLoader("plain").load()
        self.twin_axis._prepare_twin_axis(
            plt.axes(),
            False,
            plt.rcParams["axes.prop_cycle"].by_key()["color"],
            True,
            0,
            "default",
        )
        self.assertEqual(self.curve1._line_width, "default")
        self.twin_axis._default_params = self.plainDefaults
        self.twin_axis._fill_in_missing_params(self.curve1, "default")
        self.assertEqual(self.curve1._line_width, 2)
        plt.close("all")

    def test_update_rc_params(self):
        """Test updating RC parameters for twin axis."""
        params = {
            "lines.linewidth": 2,
            "axes.labelsize": 10,
            "axes.labelcolor": "blue",
        }
        self.twin_axis.set_rc_params(params)
        self.assertDictEqual(self.twin_axis._user_rc_dict, params)
        more_params = {
            "lines.linewidth": 3,
            "axes.labelpad": 20,
        }
        self.twin_axis.set_rc_params(more_params)
        resulting_params = {
            "lines.linewidth": 3,
            "axes.labelsize": 10,
            "axes.labelcolor": "blue",
            "axes.labelpad": 20,
        }
        self.assertDictEqual(self.twin_axis._user_rc_dict, resulting_params)

    def test_update_rc_params_reset(self):
        """Test resetting RC parameters for twin axis."""
        params = {
            "lines.linewidth": 2,
            "axes.labelsize": 10,
            "axes.labelcolor": "blue",
        }
        self.twin_axis.set_rc_params(params)
        self.twin_axis.set_rc_params({"lines.linewidth": 3}, reset=True)
        self.assertDictEqual(self.twin_axis._user_rc_dict, {"lines.linewidth": 3})

    def test_customize_visual_style(self):
        """Test customizing visual style parameters for twin axis."""
        self.twin_axis.set_visual_params(
            edge_color="blue", font_size=12, font_weight="bold"
        )
        self.twin_axis.set_visual_params()
        self.assertDictEqual(
            self.twin_axis._user_rc_dict,
            {
                "font.size": 12,
                "font.weight": "bold",
            },
        )
        self.assertEqual(self.twin_axis._edge_color, "blue")
        self.twin_axis.set_visual_params(edge_color="yellow", font_size=19)
        self.assertDictEqual(
            self.twin_axis._user_rc_dict,
            {
                "font.size": 19,
                "font.weight": "bold",
            },
        )
        self.assertEqual(self.twin_axis._edge_color, "yellow")

    def test_customize_visual_style_reset(self):
        """Test resetting visual style parameters for twin axis."""
        self.twin_axis.set_visual_params(
            label_color="red", line_width=3, edge_color="orange"
        )
        self.twin_axis.set_visual_params(label_color="yellow", font_size=1)
        self.twin_axis.set_visual_params(reset=True, edge_color="black")
        self.assertDictEqual(self.twin_axis._user_rc_dict, {})
        self.assertEqual(self.twin_axis._edge_color, "black")
        self.assertIsNone(self.twin_axis._line_width)

    def test_matplotlib_style_functional(self):
        """Test matplotlib style functionality for twin axis."""
        self.twin_axis.add_elements(self.curve1)
        self.twin_axis._default_params = FileLoader("plain").load()
        self.twin_axis._prepare_twin_axis(
            plt.axes(),
            True,
            plt.rcParams["axes.prop_cycle"].by_key()["color"],
            True,
            0,
            "mpl-gallery",
        )
        plt.close("all")

    def test_set_ticks_and_tick_params(self):
        """Test setting ticks and tick parameters for twin axis."""

        def tick_func(x):
            return f"Label {x}"

        self.twin_axis.set_ticks(ticks=[0, 1], tick_labels=["a", "b"])
        self.twin_axis.set_ticks(ticks=[0, 1], tick_labels=tick_func)
        self.twin_axis.set_ticks(tick_spacing=0.5, tick_labels=tick_func)
        self.twin_axis.set_ticks(minor_ticks=[0.1, 0.2])
        self.twin_axis.set_ticks(tick_spacing=None)

        self.twin_axis.set_tick_params(which="major", direction="in", length=5)
        self.twin_axis.set_tick_params(which="minor", color="red", reset=True)
        self.assertDictEqual(
            self.twin_axis._tick_params,
            {
                "major": {"direction": "in", "length": 5},
                "minor": {"color": "red"},
            },
        )
        self.twin_axis.set_tick_params(which="both", reset=True)
        self.assertDictEqual(self.twin_axis._tick_params, {"major": {}, "minor": {}})
        with self.assertRaises(GraphingException):
            self.twin_axis.set_ticks(tick_labels=["a"], ticks=None)
        with self.assertRaises(GraphingException):
            self.twin_axis.set_ticks(ticks=[1], tick_spacing=1)
        with self.assertRaises(GraphingException):
            self.twin_axis.set_ticks(minor_ticks=[1], minor_tick_spacing=1)
        with self.assertRaises(GraphingException):
            self.twin_axis.set_ticks(
                ticks=[0, 1, 2, 3], tick_labels=["only one label but 4 ticks"]
            )

    def test_methods_return_self(self):
        """Test that twin axis methods return self for method chaining."""
        self.assertIs(self.twin_axis.add_elements(), self.twin_axis)
        self.assertIs(self.twin_axis.set_ticks(ticks=[0, 1]), self.twin_axis)
        self.assertIs(
            self.twin_axis.set_tick_params(label_color="green"), self.twin_axis
        )
        self.assertIs(
            self.twin_axis.set_visual_params(edge_color="red"), self.twin_axis
        )
        self.assertIs(
            self.twin_axis.set_rc_params({"lines.linewidth": 2}), self.twin_axis
        )

    def test_hide_spines(self):
        """Test hiding the twin axis's spine."""
        self.assertIsNone(self.twin_axis._hide_spine)

        # Invalid
        with self.assertRaises(TypeError):
            self.twin_axis.set_visual_params(hide_spine=["right"])

        # Valid
        self.twin_axis.set_visual_params(hide_spine=True)
        self.assertTrue(self.twin_axis._hide_spine)
        self.twin_axis.set_visual_params(hide_spine=False)
        self.assertFalse(self.twin_axis._hide_spine)


if __name__ == "__main__":
    unittest.main()
