import unittest
from numpy import ndarray

from graphinglib.graph_elements import Hlines, Point, Vlines, Text


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


class TestText(unittest.TestCase):
    # TODO: Write tests for Text
    ...
