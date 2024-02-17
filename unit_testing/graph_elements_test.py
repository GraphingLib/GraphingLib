import unittest
from cgi import test

from matplotlib import pyplot as plt
from numpy import ndarray

from graphinglib.graph_elements import Hlines, Point, Text, Vlines


class TestHlines(unittest.TestCase):
    def setUp(self):
        self.testHlines = Hlines(1, 0, 1, "Test Hlines")

    def test_y_is_list_ndarray_float_int(self):
        self.assertIsInstance(self.testHlines.y, list | ndarray | float | int)

    def test_xmin_is_list_ndarray_float_int(self):
        self.assertIsInstance(self.testHlines.x_min, list | ndarray | float | int)

    def test_ymin_is_list_ndarray_float_int(self):
        self.assertIsInstance(self.testHlines.x_min, list | ndarray | float | int)

    def test_colors_is_str_list_or_none(self):
        self.assertIsInstance(self.testHlines.colors, list | str | None)

    def test_linestyles_is_str_list_or_none(self):
        self.assertIsInstance(self.testHlines.line_styles, list | str | None)

    def test_label_is_str(self):
        self.assertIsInstance(self.testHlines.label, str)


class TestVlines(unittest.TestCase):
    def setUp(self):
        self.testVlines = Vlines(x=[4, 5, 6, 7], y_min=0, y_max=1, label="Test Vlines")

    def test_x_is_list_ndarray_float_int(self):
        self.assertListEqual(list(self.testVlines.x), [4, 5, 6, 7])

    def test_ymin_is_list_ndarray_float_int(self):
        self.assertEqual(self.testVlines.y_min, 0)

    def test_ymax_is_list_ndarray_float_int(self):
        self.assertEqual(self.testVlines.y_max, 1)

    def test_colors_is_default(self):
        self.assertEqual(self.testVlines.colors, "default")

    def test_linestyles_is_str_list_or_none(self):
        self.assertIsInstance(self.testVlines.line_styles, list | str | None)

    def test_label_is_str(self):
        self.assertEqual(self.testVlines.label, "Test Vlines")


class TestPoint(unittest.TestCase):
    def setUp(self):
        self.testPoint = Point(x=0.0, y=0.0, label="Test Point")

    def test_x_is_float(self):
        self.assertEqual(self.testPoint.x, 0.0)

    def test_y_is_float(self):
        self.assertEqual(self.testPoint.y, 0.0)

    def test_label_is_str(self):
        self.assertEqual(self.testPoint.label, "Test Point")

    def test_colors_is_default(self):
        self.assertEqual(self.testPoint.color, "default")

    def test_edge_color_is_default(self):
        self.assertEqual(self.testPoint.edge_color, "default")

    def test_marker_size_is_default(self):
        self.assertEqual(self.testPoint.marker_size, "default")

    def test_marker_style_is_default(self):
        self.assertEqual(self.testPoint.marker_style, "default")

    def test_edge_width_is_default(self):
        self.assertEqual(self.testPoint.edge_width, "default")

    def test_font_size_is_same_as_figure(self):
        self.assertEqual(self.testPoint.font_size, "same as figure")

    def test_text_color_is_k(self):
        self.assertEqual(self.testPoint.text_color, "k")

    def test_h_align_is_left(self):
        self.assertEqual(self.testPoint.h_align, "left")

    def test_v_align_is_bottom(self):
        self.assertEqual(self.testPoint.v_align, "bottom")

    def test_coordinates_property(self):
        self.assertEqual(self.testPoint.get_coordinates(), (0.0, 0.0))


class TestText(unittest.TestCase):
    def test_init(self):
        testText = Text(
            x=0.0,
            y=0.0,
            text="Test Text",
            color="red",
            font_size=12,
            h_align="center",
            v_align="center",
        )
        self.assertEqual(testText.x, 0.0)
        self.assertEqual(testText.y, 0.0)
        self.assertEqual(testText.text, "Test Text")
        self.assertEqual(testText.color, "red")
        self.assertEqual(testText.font_size, 12)
        self.assertEqual(testText.h_align, "center")
        self.assertEqual(testText.v_align, "center")

    def test_add_arrow(self):
        testText = Text(
            x=0.0,
            y=0.0,
            text="Test Text",
            color="red",
            font_size=12,
            h_align="center",
            v_align="center",
        )
        testText.add_arrow(
            points_to=(1, 1), width=0.1, head_width=0.3, head_length=0.2, shrink=0.05
        )
        self.assertEqual(testText._arrow_pointing_to, (1, 1))
        self.assertDictEqual(
            testText.arrow_properties,
            {"width": 0.1, "headwidth": 0.3, "headlength": 0.2, "shrink": 0.05},
        )

    def test_plotting(self):
        testText = Text(
            x=0.0,
            y=0.0,
            text="Test Text",
            color="red",
            font_size=12,
            h_align="center",
            v_align="center",
        )
        testText.add_arrow(
            points_to=(1, 1), width=0.1, head_width=0.3, head_length=0.2, shrink=0.05
        )
        fig, ax = plt.subplots()
        testText._plot_element(ax, 0)
        # Check if the text is plotted correctly
        self.assertEqual(ax.texts[0].get_text(), "Test Text")
        self.assertEqual(ax.texts[0].get_color(), "red")
        self.assertEqual(ax.texts[0].get_fontsize(), 12)
        self.assertEqual(ax.texts[0].get_horizontalalignment(), "center")
        self.assertEqual(ax.texts[0].get_verticalalignment(), "center")
        # Check if the arrow is plotted correctly
        for child in ax.get_children():
            if isinstance(child, plt.Arrow):
                self.assertEqual(child.get_xy(), (0, 0))
                self.assertEqual(child.get_dx(), 1)
                self.assertEqual(child.get_dy(), 1)
                self.assertEqual(child.get_width(), 0.1)
                self.assertEqual(child.get_head_width(), 0.3)
                self.assertEqual(child.get_head_length(), 0.2)
                self.assertEqual(child.get_shrink(), 0.05)

        plt.close(fig)
