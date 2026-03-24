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


class SmartFigurePropertyMixin:
    def assert_basic_scalar_property_validation(self):
        with self.assertRaises(TypeError):
            self.fig.size = "big"
        with self.assertRaises(ValueError):
            self.fig.size = (1,)
        with self.assertRaises(ValueError):
            self.fig.size = (-1, 2)
        self.fig.size = (5, 4)
        self.assertEqual(self.fig.size, (5, 4))

        with self.assertRaises(TypeError):
            self.fig.x_lim = 1
        with self.assertRaises(ValueError):
            self.fig.x_lim = (1, 2, 3)
        self.fig.x_lim = (0, 10)
        self.assertEqual(self.fig.x_lim, (0, 10))

        with self.assertRaises(TypeError):
            self.fig.y_lim = 1
        with self.assertRaises(ValueError):
            self.fig.y_lim = (1, 2, 3)
        self.fig.y_lim = (-5, 5)
        self.assertEqual(self.fig.y_lim, (-5, 5))

        with self.assertRaises(TypeError):
            self.fig.width_padding = "x"
        with self.assertRaises(ValueError):
            self.fig.width_padding = -1
        self.fig.width_padding = 0.2
        self.assertEqual(self.fig.width_padding, 0.2)

        with self.assertRaises(TypeError):
            self.fig.height_padding = "x"
        with self.assertRaises(ValueError):
            self.fig.height_padding = -1
        self.fig.height_padding = 0.2
        self.assertEqual(self.fig.height_padding, 0.2)

        with self.assertRaises(TypeError):
            self.fig.width_ratios = 1
        with self.assertRaises(TypeError):
            self.fig.height_ratios = 1

        with self.assertRaises(TypeError):
            self.fig.figure_style = 123
        self.fig.figure_style = "default"
        self.assertEqual(self.fig.figure_style, "default")

        with self.assertRaises(TypeError):
            self.fig.annotations = "not_a_list"
        with self.assertRaises(TypeError):
            self.fig.annotations = [1, 2, 3]
        annotations = [Text(0.5, 0.5, "Center"), Text(0.1, 0.1, "Corner")]
        self.fig.annotations = annotations
        self.assertEqual(self.fig.annotations, annotations)

    def assert_visual_and_rc_customization(self):
        self.fig.set_visual_params(figure_face_color="red", axes_face_color="blue")
        self.assertEqual(self.fig._user_rc_dict["figure.facecolor"], "red")
        self.assertEqual(self.fig._user_rc_dict["axes.facecolor"], "blue")

        self.fig.set_visual_params(reset=True)
        self.assertDictEqual(self.fig._user_rc_dict, {})

        params = {
            "lines.linewidth": 2,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
        }
        self.fig.set_rc_params(params)
        self.assertDictEqual(self.fig._user_rc_dict, params)
        self.fig.set_rc_params({"lines.linewidth": 3}, reset=True)
        self.assertDictEqual(self.fig._user_rc_dict, {"lines.linewidth": 3})

    def assert_tick_grid_and_text_methods(self):
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
        self.fig.set_tick_params(axis="x", which="major", direction="in", length=5)
        self.fig.set_grid(
            visible_x=True,
            visible_y=False,
            color="blue",
            alpha=0.5,
            line_style="--",
            line_width=2,
        )
        self.assertTrue(self.fig.show_grid)

        self.fig.set_text_padding_params(x_label_pad=10.0)
        self.assertEqual(self.fig._pad_params["x_label_pad"], 10.0)
        self.fig.set_reference_labels_params(font_size=12.0)
        self.assertEqual(self.fig._reference_labels_params["font_size"], 12.0)

        with self.assertRaises(GraphingException):
            self.fig.set_ticks(x_tick_labels=["a"], x_ticks=None)
        with self.assertRaises(GraphingException):
            self.fig.set_ticks(x_ticks=[1], x_tick_spacing=1)

    def assert_misc_methods_return_self(self):
        self.assertIs(self.fig.set_custom_legend(elements=[]), self.fig)
        self.assertIs(self.fig.set_visual_params(figure_face_color="red"), self.fig)
        self.assertIs(self.fig.set_rc_params({"lines.linewidth": 2}), self.fig)
        self.assertIs(self.fig.set_grid(visible_x=True), self.fig)
        self.assertIs(self.fig.set_text_padding_params(x_label_pad=5.0), self.fig)
        self.assertIs(self.fig.set_reference_labels_params(font_size=5.0), self.fig)


class TestSmartFigureLeaf(unittest.TestCase, SmartFigurePropertyMixin):
    def setUp(self):
        self.fig = SmartFigure()
        self.twin_axis = SmartTwinAxis()
        self.x = linspace(0, 3 * pi, 200)
        self.test_curve = Curve(self.x, sin(self.x), "Test Curve", color="k")
        self.plain_defaults = FileLoader("plain").load()
        self.horrible_defaults = FileLoader("horrible").load()

    def test_init_defaults(self):
        self.assertEqual(self.fig.num_rows, 1)
        self.assertEqual(self.fig.num_cols, 1)
        self.assertEqual(self.fig.figure_style, "default")
        self.assertEqual(len(self.fig), 0)

    def test_init_custom_args(self):
        annotations = (Text(0.5, 0.5, "Center"), Text(0.1, 0.1, "Corner"))
        fig = SmartFigure(
            x_label="X",
            y_label="Y",
            size=(8, 6),
            title="Test Figure",
            x_lim=(0, 10),
            y_lim=(-5, 5),
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
            global_reference_label=False,
            reference_labels_loc="inside",
            general_legend=True,
            legend_loc="upper right",
            legend_cols=2,
            show_legend=False,
            figure_style="dark",
            elements=[DummyPlottable(), DummyPlottable()],
            annotations=annotations,
        )
        self.assertEqual(fig.x_label, "X")
        self.assertEqual(fig.y_label, "Y")
        self.assertEqual(fig.title, "Test Figure")
        self.assertEqual(len(fig.elements), 2)
        self.assertEqual(fig.annotations, annotations)

    def test_elements_in_init(self):
        with self.assertRaises(TypeError):
            SmartFigure(elements=[1])
        with self.assertRaises(TypeError):
            SmartFigure(elements=[[[DummyPlottable()]]])
        single = DummyPlottable("single")
        fig = SmartFigure(elements=single)
        self.assertEqual(len(fig.elements), 1)
        self.assertIs(fig.elements[0], single)
        SmartFigure(elements=(DummyPlottable(),))
        SmartFigure(elements=[DummyPlottable(), DummyPlottable()])

    def test_basic_scalar_property_validation(self):
        self.assert_basic_scalar_property_validation()

    def test_visual_and_rc_customization(self):
        self.assert_visual_and_rc_customization()

    def test_tick_grid_and_text_methods(self):
        self.assert_tick_grid_and_text_methods()

    def test_misc_methods_return_self(self):
        self.assert_misc_methods_return_self()

    def test_mode_semantics(self):
        self.fig.elements = [DummyPlottable("a"), DummyPlottable("b")]
        self.assertEqual(len(self.fig), 2)
        self.assertIsInstance(self.fig.elements[0], DummyPlottable)
        self.assertListEqual([el.label for el in self.fig], ["a", "b"])

    def test_leaf_getitem_raises(self):
        self.fig.elements = [DummyPlottable("a")]
        with self.assertRaises(Exception):
            _ = self.fig[0]
        with self.assertRaises(Exception):
            _ = self.fig[0, 0]
        with self.assertRaises(GraphingException):
            self.fig[0, 0] = DummyPlottable("b")

    def test_leaf_elements_getter_and_setter(self):
        curve_a = DummyPlottable("a")
        curve_b = DummyPlottable("b")
        self.fig.elements = [curve_a]
        self.assertEqual(len(self.fig.elements), 1)
        self.fig.elements += [curve_b]
        self.assertEqual(len(self.fig.elements), 2)

        self.fig.elements = DummyPlottable("c")
        self.assertEqual(len(self.fig.elements), 1)
        self.assertEqual(self.fig.elements[0].label, "c")
        with self.assertRaises(TypeError):
            self.fig.elements = SmartFigure(elements=[DummyPlottable("nested")])

    def test_leaf_add_elements(self):
        dummy = DummyPlottable()
        self.fig.add_elements(dummy)
        self.fig.add_elements(DummyPlottable("b"))
        self.assertEqual(len(self.fig), 2)
        self.assertListEqual([el.label for el in self.fig], ["dummy", "b"])

    def test_leaf_add_and_iadd(self):
        self.fig.elements = [DummyPlottable("a")]
        result = self.fig + DummyPlottable("b")
        self.assertEqual(len(result), 2)
        self.assertEqual(len(self.fig), 1)

        self.fig += [DummyPlottable("c")]
        self.assertEqual(len(self.fig), 2)
        self.assertListEqual([el.label for el in self.fig], ["a", "c"])

    def test_copy_and_copy_with_leaf_mode(self):
        self.fig.elements = [DummyPlottable("a"), DummyPlottable("b")]
        fig2 = self.fig.copy()
        self.assertEqual(len(fig2), 2)
        fig3 = self.fig.copy_with(x_label="Y")
        self.assertEqual(fig3.x_label, "Y")
        self.assertEqual(len(fig3), 2)

    def test_resizing_leaf_promotes_existing_plot_to_first_child(self):
        self.fig.title = "Original"
        self.fig.elements = [DummyPlottable("a")]
        twin = self.fig.create_twin_axis(is_y=True, label="Twin")
        twin.add_elements(DummyPlottable("twin"))

        self.fig.num_cols = 2

        self.assertFalse(self.fig.is_single_subplot)
        self.assertEqual(self.fig[0, 0].elements[0].label, "a")
        self.assertEqual(self.fig[0, 0].title, "Original")
        self.assertIsNotNone(self.fig[0, 0].twin_y_axis)
        self.assertIsNone(self.fig.twin_y_axis)

    def test_create_twin_axis_from_leaf(self):
        twin_y = self.fig.create_twin_axis(is_y=True, label="Twin Y")
        self.assertIsInstance(twin_y, SmartTwinAxis)
        twin_x = self.fig.create_twin_axis(is_y=False, label="Twin X")
        self.assertIsInstance(twin_x, SmartTwinAxis)

    def test_create_twin_axis_validation(self):
        multi_fig = SmartFigure(2, 2)
        with self.assertRaises(GraphingException):
            multi_fig.create_twin_axis()

    def test_twin_axis_integration(self):
        self.fig.add_elements(self.test_curve)
        twin_y = self.fig.create_twin_axis(is_y=True, label="Twin Y", log_scale=True)
        twin_y.add_elements(Curve([0], [0], label="Twin Curve"))

        self.fig._default_params = FileLoader("plain").load()
        self.fig._figure = plt.figure()
        legend_info = self.fig._prepare_figure(make_legend=False)
        self.assertIn("Test Curve", legend_info["labels"]["default"])
        self.assertIn("Twin Curve", legend_info["labels"]["default"])
        plt.close(self.fig._figure)

    def test_show_and_save(self):
        self.fig.add_elements(DummyPlottable())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.fig.show(fullscreen=False)
        plt.close(self.fig._figure)

        self.fig.save("test_smart_figure_output.png")
        self.assertTrue(os.path.exists("test_smart_figure_output.png"))
        os.remove("test_smart_figure_output.png")

        with PdfPages("test_smart_figure_output.pdf") as pdf:
            self.fig.save(pdf)
            self.fig.save(pdf, split_pdf=True)
        self.assertTrue(os.path.exists("test_smart_figure_output.pdf"))
        os.remove("test_smart_figure_output.pdf")

    def test_default_style_propagation(self):
        a_curve = Curve(self.x, sin(self.x), label="Test Curve")
        self.fig.add_elements(a_curve)
        self.fig._default_params = self.plain_defaults
        self.fig._fill_in_missing_params(a_curve)
        self.assertEqual(a_curve._line_width, 2)

        bad_style_fig = self.fig.copy_with(figure_style="horrible")
        bad_style_fig.add_elements(Curve(self.x, sin(self.x), label="Test Curve"))
        bad_style_fig._default_params = self.horrible_defaults
        bad_style_fig._fill_in_missing_params(bad_style_fig.elements[0])
        self.assertEqual(bad_style_fig.elements[0]._line_width, 10)


class TestSmartFigureContainer(unittest.TestCase, SmartFigurePropertyMixin):
    def setUp(self):
        self.fig = SmartFigure(2, 3)
        self.child_curve = DummyPlottable("child")
        self.other_curve = DummyPlottable("other")
        self.plain_defaults = FileLoader("plain").load()

    def test_basic_scalar_property_validation(self):
        self.assert_basic_scalar_property_validation()

    def test_visual_and_rc_customization(self):
        self.assert_visual_and_rc_customization()

    def test_tick_grid_and_text_methods(self):
        self.assert_tick_grid_and_text_methods()

    def test_misc_methods_return_self(self):
        self.assert_misc_methods_return_self()

    def test_container_mode_semantics(self):
        self.fig[0, 0] = DummyPlottable("a")
        self.fig[0, 1] = DummyPlottable("b")
        self.assertEqual(len(self.fig), 2)
        children = list(self.fig)
        self.assertEqual(len(children), 2)
        self.assertTrue(all(isinstance(child, SmartFigure) for child in children))

    def test_one_by_one_can_be_leaf_or_container(self):
        leaf = SmartFigure()
        leaf.elements = [DummyPlottable("a")]
        self.assertEqual(len(leaf), 1)
        self.assertIsInstance(leaf.elements[0], DummyPlottable)

        container = SmartFigure()
        container.num_cols = 2
        container[0, 0] = DummyPlottable("child")
        self.assertEqual(len(container), 1)
        self.assertIsInstance(container[0, 0], SmartFigure)

    def test_container_getitem_returns_child_figures(self):
        self.fig[0, 0] = DummyPlottable("a")
        self.fig[0, 1] = [DummyPlottable("b"), DummyPlottable("c")]
        self.assertIsInstance(self.fig[0, 0], SmartFigure)
        self.assertIsInstance(self.fig[0, 1], SmartFigure)
        self.assertEqual(len(self.fig[0, 0].elements), 1)
        self.assertEqual(len(self.fig[0, 1].elements), 2)

    def test_container_getitem_empty_selection_raises(self):
        with self.assertRaises(Exception):
            _ = self.fig[0, 0]

    def test_container_selection_overlap_raises(self):
        self.fig[0, 0] = DummyPlottable("left")
        self.fig[0, 1] = DummyPlottable("right")
        with self.assertRaises(GraphingException):
            _ = self.fig[0, :]

    def test_setitem_creates_auto_created_leaf_child(self):
        self.fig[0, 0] = self.child_curve
        child = self.fig[0, 0]
        self.assertIsInstance(child, SmartFigure)
        self.assertEqual(len(child.elements), 1)
        self.assertEqual(child.elements[0].label, "child")

        self.fig[0, 1] = [DummyPlottable("a"), DummyPlottable("b")]
        self.assertEqual(len(self.fig[0, 1].elements), 2)

    def test_setitem_inserts_explicit_child(self):
        child = SmartFigure(elements=[DummyPlottable("nested")])
        self.fig[0, 0] = child
        self.assertIs(self.fig[0, 0], child)

    def test_setitem_overwrites_selected_child_and_preserves_span(self):
        self.fig[1, :] = DummyPlottable("span")
        self.assertEqual(self.fig[1, 2].elements[0].label, "span")
        self.fig[1, 1] = DummyPlottable("replaced")
        self.assertEqual(self.fig[1, 0].elements[0].label, "replaced")
        self.assertEqual(self.fig[1, 2].elements[0].label, "replaced")

    def test_setitem_none_deletes_selected_child(self):
        self.fig[0, 0] = DummyPlottable("delete_me")
        self.fig[0, 0] = None
        with self.assertRaises(Exception):
            _ = self.fig[0, 0]

    def test_setitem_invalid_rhs_raises(self):
        with self.assertRaises(TypeError):
            self.fig[0, 0] = 123

    def test_1d_and_2d_indexing_validation(self):
        row_fig = SmartFigure(1, 4)
        row_fig[1] = DummyPlottable("row")
        self.assertIsInstance(row_fig[1], SmartFigure)
        row_fig[-1] = DummyPlottable("last")
        self.assertIsInstance(row_fig[-1], SmartFigure)

        col_fig = SmartFigure(3, 1)
        col_fig[1] = DummyPlottable("col")
        self.assertIsInstance(col_fig[1], SmartFigure)

        with self.assertRaises(IndexError):
            row_fig[4] = DummyPlottable()
        with self.assertRaises(ValueError):
            self.fig[1] = DummyPlottable()

    def test_container_elements_getter_dense_anchor_based(self):
        self.fig[0, 0] = DummyPlottable("a")
        self.fig[1, 1:] = DummyPlottable("span")
        dense = self.fig.elements
        self.assertEqual(len(dense), 6)
        self.assertIsInstance(dense[0], SmartFigure)
        self.assertIsNone(dense[1])
        self.assertIsNone(dense[3])
        self.assertIsInstance(dense[4], SmartFigure)
        self.assertIsNone(dense[5])

    def test_container_elements_setter(self):
        explicit_child = SmartFigure(elements=[DummyPlottable("nested")])
        self.fig.elements = [
            DummyPlottable("a"),
            [DummyPlottable("b"), DummyPlottable("c")],
            explicit_child,
            None,
        ]
        dense = self.fig.elements
        self.assertEqual(len(dense), 6)
        self.assertIsInstance(dense[0], SmartFigure)
        self.assertEqual(len(self.fig[0, 0].elements), 1)
        self.assertEqual(len(self.fig[0, 1].elements), 2)
        self.assertIs(dense[2], explicit_child)
        self.assertIsNone(dense[3])
        self.assertIsNone(dense[4])
        self.assertIsNone(dense[5])

    def test_container_elements_setter_treats_child_shape_as_internal(self):
        child = SmartFigure(1, 2, elements=[DummyPlottable("a"), DummyPlottable("b")])
        self.fig.elements = [child, DummyPlottable("neighbor")]
        dense = self.fig.elements
        self.assertIs(dense[0], child)
        self.assertIsInstance(dense[1], SmartFigure)
        self.assertIsNone(dense[2])
        self.assertEqual(child.num_rows, 1)
        self.assertEqual(child.num_cols, 2)

    def test_container_elements_round_trip(self):
        explicit_child = SmartFigure(elements=[DummyPlottable("nested")])
        self.fig[0, 0] = DummyPlottable("a")
        self.fig[1, 1:] = DummyPlottable("span")
        self.fig[0, 2] = explicit_child
        dense = self.fig.elements

        other = SmartFigure(2, 3)
        other.elements = dense
        other_dense = other.elements
        self.assertEqual(len(other_dense), 6)
        self.assertIsInstance(other_dense[0], SmartFigure)
        self.assertIsInstance(other_dense[2], SmartFigure)
        self.assertIsInstance(other_dense[4], SmartFigure)
        self.assertIsNone(other_dense[5])

    def test_add_elements_creates_children_in_empty_layout(self):
        self.fig.add_elements(DummyPlottable("a"), DummyPlottable("b"))
        self.assertEqual(len(self.fig), 2)
        self.assertEqual(self.fig[0, 0].elements[0].label, "a")
        self.assertEqual(self.fig[0, 1].elements[0].label, "b")

    def test_add_elements_grouped_iterable_creates_single_child_plot(self):
        self.fig.add_elements([DummyPlottable("a"), DummyPlottable("b")])
        self.assertEqual(len(self.fig), 1)
        self.assertEqual(len(self.fig[0, 0].elements), 2)
        self.assertListEqual(
            [el.label for el in self.fig[0, 0].elements], ["a", "b"]
        )

    def test_add_elements_targets_existing_children_then_creates_new_ones(self):
        self.fig[0, 0] = DummyPlottable("a")
        self.fig.add_elements(DummyPlottable("a2"), DummyPlottable("b"))
        self.assertEqual(len(self.fig[0, 0].elements), 2)
        self.assertListEqual(
            [el.label for el in self.fig[0, 0].elements], ["a", "a2"]
        )
        self.assertEqual(self.fig[0, 1].elements[0].label, "b")

    def test_add_elements_targets_children_in_grid_order(self):
        self.fig[0, 0] = DummyPlottable("a")
        self.fig[0, 1] = DummyPlottable("b")
        self.fig.add_elements(DummyPlottable("a2"), DummyPlottable("b2"))
        self.assertEqual(len(self.fig[0, 0].elements), 2)
        self.assertEqual(len(self.fig[0, 1].elements), 2)

    def test_add_elements_can_append_through_spanning_child_position(self):
        self.fig[0, :] = DummyPlottable("span")
        self.fig.add_elements(None, DummyPlottable("added"))
        self.assertEqual(len(self.fig[0, 0].elements), 2)
        self.assertListEqual(
            [el.label for el in self.fig[0, 0].elements], ["span", "added"]
        )

    def test_add_elements_rejects_nested_layout_child_target(self):
        nested_layout = SmartFigure(1, 2)
        nested_layout[0, 0] = DummyPlottable("nested")
        self.fig[0, 0] = nested_layout
        with self.assertRaises(GraphingException):
            self.fig.add_elements(DummyPlottable("bad"))

    def test_add_elements_overflow_raises(self):
        with self.assertRaises(ValueError):
            self.fig.add_elements(
                DummyPlottable("0"),
                DummyPlottable("1"),
                DummyPlottable("2"),
                DummyPlottable("3"),
                DummyPlottable("4"),
                DummyPlottable("5"),
                DummyPlottable("6"),
            )

    def test_parent_index_child_iadd(self):
        self.fig[0, 0] = DummyPlottable("a")
        self.fig[0, 0] += DummyPlottable("b")
        self.assertEqual(len(self.fig[0, 0].elements), 2)
        self.assertListEqual([el.label for el in self.fig[0, 0].elements], ["a", "b"])

    def test_container_add_and_iadd(self):
        self.fig[0, 0] = DummyPlottable("a")
        self.fig[0, 1] = DummyPlottable("b")
        result = self.fig + [DummyPlottable("a2"), DummyPlottable("b2")]
        self.assertEqual(len(result[0, 0].elements), 2)
        self.assertEqual(len(result[0, 1].elements), 2)
        self.assertEqual(len(self.fig[0, 0].elements), 1)

        self.fig += [DummyPlottable("a3"), DummyPlottable("b3")]
        self.assertEqual(len(self.fig[0, 0].elements), 2)
        self.assertEqual(len(self.fig[0, 1].elements), 2)

    def test_container_add_rejects_smartfigure_rhs(self):
        self.fig[0, 0] = DummyPlottable("a")
        with self.assertRaises(Exception):
            _ = self.fig + [SmartFigure(elements=[DummyPlottable("bad")])]

    def test_multi_cell_behavior(self):
        self.fig[0, :] = DummyPlottable("span")
        child = self.fig[0, 1]
        self.assertIs(child, self.fig[0, 2])
        self.assertEqual(child.elements[0].label, "span")

        self.fig[0, 2] = DummyPlottable("replaced")
        self.assertEqual(self.fig[0, 0].elements[0].label, "replaced")

        dense = self.fig.elements
        self.assertIsInstance(dense[0], SmartFigure)
        self.assertIsNone(dense[1])
        self.assertIsNone(dense[2])

        self.fig[0, 1] = None
        with self.assertRaises(Exception):
            _ = self.fig[0, 0]

    def test_multi_cell_overlap_errors(self):
        self.fig[0, :] = DummyPlottable("top")
        self.fig[1, 0] = DummyPlottable("bottom")
        with self.assertRaises(GraphingException):
            _ = self.fig[:, :]

    def test_copy_and_copy_with_container_mode(self):
        self.fig[0, 0] = DummyPlottable("a")
        self.fig[1, 1:] = DummyPlottable("span")
        fig2 = self.fig.copy()
        self.assertEqual(len(fig2), 2)
        self.assertIsInstance(fig2[0, 0], SmartFigure)
        self.assertEqual(fig2[1, 2].elements[0].label, "span")

        fig3 = self.fig.copy_with(title="Container")
        self.assertEqual(fig3.title, "Container")
        self.assertEqual(len(fig3), 2)


class TestSmartFigureContainerRendering(unittest.TestCase):
    def setUp(self):
        self.curve_a = Curve([0, 1], [0, 1], label="A")
        self.curve_b = Curve([0, 1], [1, 0], label="B")
        self.curve_c = Curve([0, 1], [0.5, 0.5], label="C")
        self.plain_defaults = FileLoader("plain").load()

    def _get_plot_axes(self, fig):
        return [ax for ax in fig._figure.get_axes() if ax.get_navigate()]

    def test_auto_created_children_participate_in_parent_shared_axes(self):
        fig = SmartFigure(1, 2, share_x=True)
        fig[0, 0] = self.curve_a
        fig[0, 1] = self.curve_b
        fig._default_params = self.plain_defaults
        fig._reference_label_i = 0
        fig._figure = plt.figure()
        fig._prepare_figure()
        axes = self._get_plot_axes(fig)
        self.assertTrue(axes[0].get_shared_x_axes().joined(axes[0], axes[1]))
        plt.close(fig._figure)

    def test_auto_created_children_consume_parent_projection_lists(self):
        fig = SmartFigure(1, 2, projection=["polar", None])
        fig[0, 0] = self.curve_a
        fig[0, 1] = self.curve_b
        fig._default_params = self.plain_defaults
        fig._reference_label_i = 0
        fig._figure = plt.figure()
        fig._prepare_figure()
        axes = self._get_plot_axes(fig)
        self.assertEqual(axes[0].name, "polar")
        self.assertEqual(axes[1].name, "rectilinear")
        plt.close(fig._figure)

    def test_auto_created_children_respond_to_parent_list_params(self):
        fig = SmartFigure(1, 2, remove_x_ticks=[True, False])
        fig[0, 0] = self.curve_a
        fig[0, 1] = self.curve_b
        fig._default_params = self.plain_defaults
        fig._reference_label_i = 0
        fig._figure = plt.figure()
        fig._prepare_figure()
        axes = self._get_plot_axes(fig)
        self.assertFalse(any(label.get_visible() for label in axes[0].get_xticklabels()))
        plt.close(fig._figure)

    def test_explicit_nested_children_remain_true_nested(self):
        child = SmartFigure(elements=[self.curve_a])
        parent = SmartFigure(1, 2)
        parent[0, 0] = child
        parent[0, 1] = self.curve_b
        parent._default_params = self.plain_defaults
        parent._reference_label_i = 0
        parent._figure = plt.figure()
        parent._prepare_figure()
        self.assertGreaterEqual(len(parent._figure.subfigs), 1)
        plt.close(parent._figure)

    def test_general_legend_aggregation_mixed_children(self):
        child = SmartFigure(elements=[self.curve_a])
        parent = SmartFigure(1, 2, general_legend=True)
        parent[0, 0] = child
        parent[0, 1] = self.curve_b
        parent._default_params = self.plain_defaults
        parent._reference_label_i = 0
        parent._figure = plt.figure()
        parent._prepare_figure()
        self.assertTrue(parent._figure.legends)
        plt.close(parent._figure)

    def test_reference_labels_count_mixed_children(self):
        child = SmartFigure(elements=[self.curve_a], reference_labels=True)
        parent = SmartFigure(1, 2, reference_labels=True)
        parent[0, 0] = child
        parent[0, 1] = self.curve_b
        parent._default_params = self.plain_defaults
        parent._reference_label_i = 0
        parent._figure = plt.figure()
        parent._prepare_figure()
        texts = []
        for ax in self._get_plot_axes(parent):
            texts.extend(text.get_text() for text in ax.texts)
        self.assertTrue(any(text.endswith(")") for text in texts))
        plt.close(parent._figure)

    def test_parent_grid_settings_apply_to_auto_created_children(self):
        fig = SmartFigure(1, 2)
        fig.set_grid()
        fig[0, 0] = self.curve_a
        fig[0, 1] = self.curve_b
        fig._default_params = self.plain_defaults
        fig._reference_label_i = 0
        fig._figure = plt.figure()
        fig._prepare_figure()
        axes = self._get_plot_axes(fig)
        self.assertTrue(any(line.get_visible() for line in axes[0].get_xgridlines()))
        plt.close(fig._figure)

    def test_parent_list_values_map_over_immediate_children_not_cells(self):
        fig = SmartFigure(2, 2, remove_x_ticks=[True, False])
        fig[0, :] = self.curve_a
        fig[1, 0] = self.curve_b
        fig._default_params = self.plain_defaults
        fig._reference_label_i = 0
        fig._figure = plt.figure()
        fig._prepare_figure()
        axes = self._get_plot_axes(fig)
        self.assertEqual(len(axes), 2)
        plt.close(fig._figure)

    def test_parent_list_values_shorter_than_children_pad_with_defaults(self):
        fig = SmartFigure(1, 2, remove_x_ticks=[True])
        fig[0, 0] = self.curve_a
        fig[0, 1] = self.curve_b
        fig._default_params = self.plain_defaults
        fig._reference_label_i = 0
        fig._figure = plt.figure()
        fig._prepare_figure()
        axes = self._get_plot_axes(fig)
        self.assertFalse(
            any(label.get_visible() for label in axes[0].get_xticklabels())
        )
        self.assertTrue(
            any(label.get_visible() for label in axes[1].get_xticklabels())
        )
        plt.close(fig._figure)

    def test_parent_list_values_longer_than_children_raise(self):
        fig = SmartFigure(2, 2, remove_x_ticks=[True, False, True])
        fig[0, 0] = self.curve_a
        fig[1, 1] = self.curve_b
        with self.assertRaises(GraphingException):
            fig._fill_per_subplot_params()

    def test_parent_property_change_after_child_creation_affects_inherited_child(self):
        fig = SmartFigure(1, 2)
        fig[0, 0] = self.curve_a
        fig[0, 1] = self.curve_b
        fig.remove_x_ticks = [True, False]
        fig._default_params = self.plain_defaults
        fig._reference_label_i = 0
        fig._figure = plt.figure()
        fig._prepare_figure()
        axes = self._get_plot_axes(fig)
        self.assertFalse(
            any(label.get_visible() for label in axes[0].get_xticklabels())
        )
        self.assertTrue(
            any(label.get_visible() for label in axes[1].get_xticklabels())
        )
        plt.close(fig._figure)

    def test_explicit_child_override_blocks_parent_inheritance(self):
        child = SmartFigure(elements=[self.curve_a], remove_x_ticks=False)
        parent = SmartFigure(1, 2, remove_x_ticks=[True, True])
        parent[0, 0] = child
        parent[0, 1] = self.curve_b
        parent._default_params = self.plain_defaults
        parent._reference_label_i = 0
        parent._figure = plt.figure()
        parent._prepare_figure()
        axes = self._get_plot_axes(parent)
        self.assertTrue(
            any(label.get_visible() for label in axes[0].get_xticklabels())
        )
        self.assertFalse(
            any(label.get_visible() for label in axes[1].get_xticklabels())
        )
        plt.close(parent._figure)


@unittest.skipUnless(
    HAS_ASTROPY,
    "Install the optional extra with `pip install graphinglib[astro]` to run WCS tests.",
)
class TestSmartFigureWCSLeaf(unittest.TestCase):
    def setUp(self):
        self.wcs = WCS(naxis=2)
        self.wcs.wcs.crpix = [1, 1]
        self.wcs.wcs.cdelt = [1, 1]
        self.wcs.wcs.crval = [0, 0]
        self.wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        self.fig = SmartFigureWCS(projection=self.wcs)
        self.test_curve = Curve([0, 1], [0, 1], label="Curve")

    def test_init_requires_wcs_projection(self):
        self.assertEqual(self.fig.projection, self.wcs)
        with self.assertRaises(GraphingException):
            SmartFigureWCS(projection="polar")
        with self.assertRaises(GraphingException):
            SmartFigureWCS(projection=None)

    def test_leaf_mode_semantics(self):
        self.fig.elements = [self.test_curve]
        self.assertEqual(len(self.fig), 1)
        self.assertIs(self.fig.elements[0], self.test_curve)
        with self.assertRaises(Exception):
            _ = self.fig[0]

    def test_wcs_specific_ticks(self):
        result = self.fig.set_ticks(
            number_of_x_ticks=5,
            number_of_y_ticks=5,
            x_tick_formatter="hh:mm:ss",
            y_tick_formatter=lambda x: f"{x:.2f}",
            minor_x_tick_frequency=2,
            minor_y_tick_frequency=3,
        )
        self.assertIs(result, self.fig)
        self.assertEqual(self.fig._ticks.get("number_of_x_ticks"), 5)

    def test_wcs_twin_axis_smoke(self):
        twin_y = self.fig.create_twin_axis(is_y=True, label="Twin")
        twin_y.add_elements(self.test_curve)
        self.fig.elements = [self.test_curve]
        self.fig._default_params = FileLoader("plain").load()
        self.fig._reference_label_i = 0
        self.fig._figure = plt.figure()
        self.fig._prepare_figure()
        plt.close(self.fig._figure)


@unittest.skipUnless(
    HAS_ASTROPY,
    "Install the optional extra with `pip install graphinglib[astro]` to run WCS tests.",
)
class TestSmartFigureWCSContainer(unittest.TestCase):
    def setUp(self):
        self.wcs = WCS(naxis=2)
        self.wcs.wcs.crpix = [1, 1]
        self.wcs.wcs.cdelt = [1, 1]
        self.wcs.wcs.crval = [0, 0]
        self.wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        self.fig = SmartFigureWCS(projection=self.wcs, num_rows=1, num_cols=2)
        self.curve_a = Curve([0, 1], [0, 1], label="A")
        self.curve_b = Curve([0, 1], [1, 0], label="B")
        self.plain_defaults = FileLoader("plain").load()

    def _get_plot_axes(self, fig):
        return [ax for ax in fig._figure.get_axes() if ax.get_navigate()]

    def test_container_indexing_returns_child_figures(self):
        self.fig[0, 0] = self.curve_a
        self.fig[0, 1] = self.curve_b
        self.assertIsInstance(self.fig[0, 0], SmartFigure)
        self.assertIsInstance(self.fig[0, 1], SmartFigure)

    def test_projection_lists_map_over_immediate_children_not_cells(self):
        self.fig[0, 0] = self.curve_a
        self.fig[0, 1] = self.curve_b
        self.fig.projection = [self.wcs, self.wcs]
        self.fig._default_params = self.plain_defaults
        self.fig._reference_label_i = 0
        self.fig._figure = plt.figure()
        self.fig._prepare_figure()
        axes = self._get_plot_axes(self.fig)
        self.assertEqual(len(axes), 2)
        plt.close(self.fig._figure)

    def test_explicit_non_wcs_child_is_explicit_child(self):
        child = SmartFigure(elements=[self.curve_a])
        self.fig[0, 0] = child
        self.fig[0, 1] = self.curve_b
        self.assertIs(self.fig[0, 0], child)

    def test_wcs_container_with_auto_created_children_preserves_parent_projection(self):
        self.fig[0, 0] = self.curve_a
        self.fig[0, 1] = self.curve_b
        self.fig._default_params = self.plain_defaults
        self.fig._reference_label_i = 0
        self.fig._figure = plt.figure()
        self.fig._prepare_figure()
        axes = self._get_plot_axes(self.fig)
        self.assertEqual(len(axes), 2)
        plt.close(self.fig._figure)

    def test_selected_wcs_child_can_render_standalone(self):
        self.fig.projection = [self.wcs, self.wcs]
        self.fig[0, 0] = self.curve_a
        child = self.fig[0, 0]
        self.assertFalse(isinstance(child.projection, list))
        child._default_params = self.plain_defaults
        child._reference_label_i = 0
        child._figure = plt.figure()
        child._prepare_figure()
        axes = self._get_plot_axes(child)
        self.assertEqual(len(axes), 1)
        plt.close(child._figure)

    def test_twin_axis_on_selected_leaf_child(self):
        self.fig[0, 0] = self.curve_a
        child = self.fig[0, 0]
        twin = child.create_twin_axis(is_y=True, label="child twin")
        self.assertIsInstance(twin, SmartTwinAxis)


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
        self.assertIsNone(self.twin_axis.label)
        self.assertIsNone(self.twin_axis.axis_lim)
        self.assertFalse(self.twin_axis.log_scale)
        self.assertFalse(self.twin_axis.remove_axes)
        self.assertFalse(self.twin_axis.remove_ticks)
        self.assertFalse(self.twin_axis.invert_axis)
        self.assertEqual(len(self.twin_axis._elements), 0)

    def test_init_custom_parameters(self):
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
        self.assertEqual(len(twin_axis._elements), 2)

    def test_init_accepts_single_element(self):
        twin_axis = SmartTwinAxis(elements=self.curve1)
        self.assertEqual(len(twin_axis._elements), 1)
        self.assertIs(twin_axis._elements[0], self.curve1)

    def test_axis_lim(self):
        with self.assertRaises(TypeError):
            self.twin_axis.axis_lim = 1
        with self.assertRaises(ValueError):
            self.twin_axis.axis_lim = (1, 2, 3)
        self.twin_axis.axis_lim = (0, 10)
        self.assertEqual(self.twin_axis.axis_lim, (0, 10))

    def test_log_scale(self):
        with self.assertRaises(TypeError):
            self.twin_axis.log_scale = "yes"
        with self.assertRaises(TypeError):
            self.twin_axis.log_scale = 1
        self.twin_axis.log_scale = True
        self.assertTrue(self.twin_axis.log_scale)

    def test_remove_axes(self):
        with self.assertRaises(TypeError):
            self.twin_axis.remove_axes = "yes"
        with self.assertRaises(TypeError):
            self.twin_axis.remove_axes = 1
        self.twin_axis.remove_axes = True
        self.assertTrue(self.twin_axis.remove_axes)

    def test_remove_ticks(self):
        with self.assertRaises(TypeError):
            self.twin_axis.remove_ticks = "yes"
        with self.assertRaises(TypeError):
            self.twin_axis.remove_ticks = 1
        self.twin_axis.remove_ticks = True
        self.assertTrue(self.twin_axis.remove_ticks)

    def test_invert_axis(self):
        with self.assertRaises(TypeError):
            self.twin_axis.invert_axis = "yes"
        with self.assertRaises(TypeError):
            self.twin_axis.invert_axis = 1
        self.twin_axis.invert_axis = True
        self.assertTrue(self.twin_axis.invert_axis)

    def test_elements(self):
        with self.assertRaises(TypeError):
            self.twin_axis.elements = "not_a_list"
        with self.assertRaises(TypeError):
            self.twin_axis.elements = [1, 2, 3]
        self.twin_axis.elements = DummyPlottable("single")
        self.assertEqual(len(self.twin_axis.elements), 1)
        self.assertEqual(self.twin_axis.elements[0].label, "single")
        self.twin_axis.elements = [DummyPlottable(), DummyPlottable()]
        self.assertEqual(len(self.twin_axis.elements), 2)
        self.twin_axis.elements = [
            DummyPlottable(),
            DummyPlottable(),
            None,
            DummyPlottable(),
        ]
        self.assertEqual(len(self.twin_axis.elements), 3)

    def test_len(self):
        self.twin_axis.elements = [self.curve1, self.curve2]
        self.assertEqual(len(self.twin_axis), 2)
        self.twin_axis.elements = [None, self.curve1, self.curve2, None]
        self.assertEqual(len(self.twin_axis), 2)
        self.twin_axis.elements = []
        self.assertEqual(len(self.twin_axis), 0)
        self.twin_axis.elements = [None, None]
        self.assertEqual(len(self.twin_axis), 0)

    def test_getitem(self):
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
        self.twin_axis.elements = [self.curve1, self.curve2]
        elements = list(iter(self.twin_axis))
        self.assertEqual(len(elements), 2)
        self.assertIs(elements[0], self.curve1)
        self.assertIs(elements[1], self.curve2)

    def test_copy_and_copy_with(self):
        self.twin_axis.label = "X"
        twin_axis_2 = self.twin_axis.copy()
        self.assertEqual(twin_axis_2.label, "X")
        twin_axis_3 = self.twin_axis.copy_with(label="Y")
        self.assertEqual(twin_axis_2.label, "X")
        self.assertEqual(twin_axis_3.label, "Y")
        with self.assertRaises(AttributeError):
            self.twin_axis.copy_with(not_a_property=1)

    def test_add_elements(self):
        self.twin_axis.add_elements(self.curve1)
        self.assertEqual(len(self.twin_axis._elements), 1)
        self.assertIs(self.twin_axis._elements[0], self.curve1)
        self.twin_axis.add_elements(self.curve2, DummyPlottable())
        self.assertEqual(len(self.twin_axis._elements), 3)
        with self.assertRaises(TypeError):
            self.twin_axis.add_elements("not_plottable")

    def test_auto_assign_default_params(self):
        a_curve = Curve(self.x, sin(self.x), label="Test Curve")
        self.twin_axis.add_elements(a_curve)
        self.twin_axis._default_params = self.plainDefaults
        self.twin_axis._fill_in_missing_params(a_curve, "plain")
        self.assertEqual(a_curve._line_width, 2)

    def test_auto_assign_default_params_horrible(self):
        a_curve = Curve(self.x, sin(self.x), label="Test Curve")
        self.twin_axis.add_elements(a_curve)
        self.twin_axis._default_params = self.horribleDefaults
        self.twin_axis._fill_in_missing_params(a_curve, "horrible")
        self.assertEqual(a_curve._line_width, 10)

    def test_auto_assign_default_params_skip_predefined(self):
        a_curve = Curve(self.x, sin(self.x), label="Test Curve", line_width=3)
        self.twin_axis.add_elements(a_curve)
        self.twin_axis._default_params = self.plainDefaults
        self.twin_axis._fill_in_missing_params(a_curve, "plain")
        self.assertEqual(a_curve._line_width, 3)

    def test_element_defaults_are_reset(self):
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
        self.assertDictEqual(
            self.twin_axis._user_rc_dict,
            {
                "lines.linewidth": 3,
                "axes.labelsize": 10,
                "axes.labelcolor": "blue",
                "axes.labelpad": 20,
            },
        )

    def test_update_rc_params_reset(self):
        params = {
            "lines.linewidth": 2,
            "axes.labelsize": 10,
            "axes.labelcolor": "blue",
        }
        self.twin_axis.set_rc_params(params)
        self.twin_axis.set_rc_params({"lines.linewidth": 3}, reset=True)
        self.assertDictEqual(self.twin_axis._user_rc_dict, {"lines.linewidth": 3})

    def test_customize_visual_style(self):
        self.twin_axis.set_visual_params(
            edge_color="blue", font_size=12, font_weight="bold"
        )
        self.twin_axis.set_visual_params()
        self.assertEqual(self.twin_axis._edge_color, "blue")
        self.assertDictEqual(
            self.twin_axis._user_rc_dict,
            {
                "font.size": 12,
                "font.weight": "bold",
            },
        )
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
        self.twin_axis.set_visual_params(
            label_color="red", line_width=3, edge_color="orange"
        )
        self.twin_axis.set_visual_params(label_color="yellow", font_size=1)
        self.twin_axis.set_visual_params(reset=True, edge_color="black")
        self.assertDictEqual(self.twin_axis._user_rc_dict, {})
        self.assertEqual(self.twin_axis._edge_color, "black")
        self.assertIsNone(self.twin_axis._line_width)

    def test_matplotlib_style_functional(self):
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
        self.assertIsNone(self.twin_axis._hide_spine)
        with self.assertRaises(TypeError):
            self.twin_axis.set_visual_params(hide_spine=["right"])
        self.twin_axis.set_visual_params(hide_spine=True)
        self.assertTrue(self.twin_axis._hide_spine)
        self.twin_axis.set_visual_params(hide_spine=False)
        self.assertFalse(self.twin_axis._hide_spine)


if __name__ == "__main__":
    unittest.main()
