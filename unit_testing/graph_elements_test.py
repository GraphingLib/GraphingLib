import unittest

from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from numpy import ndarray

from graphinglib.graph_elements import Hlines, Point, Table, Text, Vlines


class TestHlines(unittest.TestCase):
    def setUp(self):
        self.testHlines = Hlines(1, 0, 1, "Test Hlines")

    def test_y_is_list_ndarray_float_int(self):
        self.assertIsInstance(self.testHlines._y, list | ndarray | float | int)

    def test_xmin_is_list_ndarray_float_int(self):
        self.assertIsInstance(self.testHlines._x_min, list | ndarray | float | int)

    def test_ymin_is_list_ndarray_float_int(self):
        self.assertIsInstance(self.testHlines._x_min, list | ndarray | float | int)

    def test_colors_is_str_list_or_none(self):
        self.assertIsInstance(self.testHlines._colors, list | str | None)

    def test_linestyles_is_str_list_or_none(self):
        self.assertIsInstance(self.testHlines._line_styles, list | str | None)

    def test_label_is_str(self):
        self.assertIsInstance(self.testHlines._label, str)

    def test_copy(self):
        testHlinesCopy = self.testHlines.copy()
        self.assertEqual(testHlinesCopy._y, self.testHlines._y)
        self.assertEqual(testHlinesCopy._x_min, self.testHlines._x_min)
        self.assertEqual(testHlinesCopy._x_max, self.testHlines._x_max)
        self.assertEqual(testHlinesCopy._colors, self.testHlines._colors)
        self.assertEqual(testHlinesCopy._line_styles, self.testHlines._line_styles)
        self.assertEqual(testHlinesCopy._label, self.testHlines._label)


class TestVlines(unittest.TestCase):
    def setUp(self):
        self.testVlines = Vlines(x=[4, 5, 6, 7], y_min=0, y_max=1, label="Test Vlines")

    def test_x_is_list_ndarray_float_int(self):
        self.assertListEqual(list(self.testVlines._x), [4, 5, 6, 7])

    def test_ymin_is_list_ndarray_float_int(self):
        self.assertEqual(self.testVlines._y_min, 0)

    def test_ymax_is_list_ndarray_float_int(self):
        self.assertEqual(self.testVlines._y_max, 1)

    def test_colors_is_default(self):
        self.assertEqual(self.testVlines._colors, "default")

    def test_linestyles_is_str_list_or_none(self):
        self.assertIsInstance(self.testVlines._line_styles, list | str | None)

    def test_label_is_str(self):
        self.assertEqual(self.testVlines._label, "Test Vlines")

    def test_copy(self):
        testVlinesCopy = self.testVlines.copy()
        self.assertEqual(list(testVlinesCopy._x), list(self.testVlines._x))
        self.assertEqual(testVlinesCopy._y_min, self.testVlines._y_min)
        self.assertEqual(testVlinesCopy._y_max, self.testVlines._y_max)
        self.assertEqual(testVlinesCopy._colors, self.testVlines._colors)
        self.assertEqual(testVlinesCopy._line_styles, self.testVlines._line_styles)
        self.assertEqual(testVlinesCopy._label, self.testVlines._label)


class TestPoint(unittest.TestCase):
    def setUp(self):
        self.testPoint = Point(x=0.0, y=0.0, label="Test Point")

    def test_x_is_float(self):
        self.assertEqual(self.testPoint._x, 0.0)

    def test_y_is_float(self):
        self.assertEqual(self.testPoint._y, 0.0)

    def test_label_is_str(self):
        self.assertEqual(self.testPoint._label, "Test Point")

    def test_colors_is_default(self):
        self.assertEqual(self.testPoint._color, "default")

    def test_edge_color_is_default(self):
        self.assertEqual(self.testPoint._edge_color, "default")

    def test_marker_size_is_default(self):
        self.assertEqual(self.testPoint._marker_size, "default")

    def test_marker_style_is_default(self):
        self.assertEqual(self.testPoint._marker_style, "default")

    def test_edge_width_is_default(self):
        self.assertEqual(self.testPoint._edge_width, "default")

    def test_font_size_is_same_as_figure(self):
        self.assertEqual(self.testPoint._font_size, "same as figure")

    def test_text_color_is_k(self):
        self.assertEqual(self.testPoint._text_color, "k")

    def test_h_align_is_left(self):
        self.assertEqual(self.testPoint._h_align, "left")

    def test_v_align_is_bottom(self):
        self.assertEqual(self.testPoint._v_align, "bottom")

    def test_coordinates_property(self):
        self.assertEqual(self.testPoint.coordinates, (0.0, 0.0))

    def test_copy(self):
        testPointCopy = self.testPoint.copy()
        self.assertEqual(testPointCopy._x, self.testPoint._x)
        self.assertEqual(testPointCopy._y, self.testPoint._y)
        self.assertEqual(testPointCopy._label, self.testPoint._label)
        self.assertEqual(testPointCopy._color, self.testPoint._color)
        self.assertEqual(testPointCopy._edge_color, self.testPoint._edge_color)
        self.assertEqual(testPointCopy._marker_size, self.testPoint._marker_size)
        self.assertEqual(testPointCopy._marker_style, self.testPoint._marker_style)
        self.assertEqual(testPointCopy._edge_width, self.testPoint._edge_width)
        self.assertEqual(testPointCopy._font_size, self.testPoint._font_size)
        self.assertEqual(testPointCopy._text_color, self.testPoint._text_color)
        self.assertEqual(testPointCopy._h_align, self.testPoint._h_align)
        self.assertEqual(testPointCopy._v_align, self.testPoint._v_align)


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
            if isinstance(child, plt.Annotation):
                self.assertEqual(child.xy, (1, 1))
                self.assertEqual(child.xyann, (0, 0))
                self.assertEqual(child.arrowprops["width"], 0.1)
                self.assertEqual(child.arrowprops["headwidth"], 0.3)
                self.assertEqual(child.arrowprops["headlength"], 0.2)
                self.assertEqual(child.arrowprops["shrink"], 0.05)
        plt.close(fig)

    def test_copy(self):
        testText = Text(
            x=0.0,
            y=0.0,
            text="Test Text",
            color="red",
            font_size=12,
            h_align="center",
            v_align="center",
        )
        testTextCopy = testText.copy()
        self.assertEqual(testTextCopy.x, testText.x)
        self.assertEqual(testTextCopy.y, testText.y)
        self.assertEqual(testTextCopy.text, testText.text)
        self.assertEqual(testTextCopy.color, testText.color)
        self.assertEqual(testTextCopy.font_size, testText.font_size)
        self.assertEqual(testTextCopy.h_align, testText.h_align)
        self.assertEqual(testTextCopy.v_align, testText.v_align)
        self.assertEqual(testTextCopy._arrow_pointing_to, testText._arrow_pointing_to)


class TestTable(unittest.TestCase):
    def test_init(self):
        data = [
            [5, 223.9369, 0.0323, 0.0532, 0.1764],
            [10, 223.9367, 0.0324, 0.0533, 0.1765],
            [15, 223.9367, 0.0325, 0.0534, 0.1764],
            [20, 223.9387, 0.0326, 0.0535, 0.1763],
            [25, 223.9385, 0.0327, 0.0536, 0.1761],
        ]
        columns = [
            "Time (s)",
            "Voltage (V)",
            "Current 1 (A)",
            "Current 2 (A)",
            "Current 3 (A)",
        ]
        rows = ["Series 1", "Series 2", "Series 3", "Series 4", "Series 5"]
        colors = [["#bfbfbf"] * 5] * 5

        table = Table(
            cell_text=data,
            cell_colors=colors,
            cell_align="center",
            col_labels=columns,
            col_widths=[0.1, 0.1, 0.1, 0.1, 0.1],
            col_align="center",
            col_colors=["#bfbfbf"] * 5,
            row_labels=rows,
            row_align="center",
            row_colors=["#bfbfbf"] * 5,
            scaling=(1.1, 1.1),
            location="bottom",
        )

        self.assertListEqual(table.cell_text, data)
        self.assertListEqual(table.cell_colors, colors)
        self.assertEqual(table.cell_align, "center")
        self.assertListEqual(table.col_labels, columns)
        self.assertListEqual(table.col_widths, [0.1, 0.1, 0.1, 0.1, 0.1])
        self.assertEqual(table.col_align, "center")
        self.assertListEqual(table.col_colors, ["#bfbfbf"] * 5)
        self.assertListEqual(table.row_labels, rows)
        self.assertEqual(table.row_align, "center")
        self.assertListEqual(table.row_colors, ["#bfbfbf"] * 5)
        self.assertEqual(table.scaling, (1.1, 1.1))
        self.assertEqual(table.location, "bottom")

    def test_plotting(self):
        data = [
            [5, 223.9369, 0.0323, 0.0532, 0.1764],
            [10, 223.9367, 0.0324, 0.0533, 0.1765],
            [15, 223.9367, 0.0325, 0.0534, 0.1764],
            [20, 223.9387, 0.0326, 0.0535, 0.1763],
            [25, 223.9385, 0.0327, 0.0536, 0.1761],
        ]
        columns = [
            "Time (s)",
            "Voltage (V)",
            "Current 1 (A)",
            "Current 2 (A)",
            "Current 3 (A)",
        ]
        rows = ["Series 1", "Series 2", "Series 3", "Series 4", "Series 5"]
        colors = [["#bfbfbf"] * 5] * 5

        table = Table(
            cell_text=data,
            cell_colors=colors,
            cell_align="center",
            col_labels=columns,
            col_widths=[0.1, 0.1, 0.1, 0.1, 0.1],
            col_align="center",
            col_colors=["#bfbfbf"] * 5,
            row_labels=rows,
            row_align="center",
            row_colors=["#bfbfbf"] * 5,
            scaling=(1.1, 1.1),
            location="bottom",
        )

        fig, ax = plt.subplots()
        table._plot_element(ax, 0)
        #
        # Check text in the table
        self.assertEqual(
            ax.tables[0].get_celld()[(0, 0)].get_text().get_text(), "Time (s)"
        )
        # Check text color
        self.assertEqual(
            ax.tables[0].get_celld()[(0, 0)].get_text().get_color(), "black"
        )
        # Check text alignment
        self.assertEqual(
            ax.tables[0].get_celld()[(0, 0)].get_text().get_horizontalalignment(),
            "center",
        )
        self.assertEqual(
            ax.tables[0].get_celld()[(0, 0)].get_text().get_verticalalignment(),
            "center",
        )
        # Check cell color
        self.assertEqual(
            ax.tables[0].get_celld()[(0, 0)].get_facecolor(),
            to_rgba("#bfbfbf"),
        )
        # Check cell width
        self.assertAlmostEqual(ax.tables[0].get_celld()[(0, 0)].get_width(), 0.1 * 1.1)
        # Now same as above for a row label cell
        self.assertEqual(
            ax.tables[0].get_celld()[(1, -1)].get_text().get_text(), "Series 1"
        )
        self.assertEqual(
            ax.tables[0].get_celld()[(1, -1)].get_text().get_color(), "black"
        )
        self.assertEqual(
            ax.tables[0].get_celld()[(1, -1)].get_text().get_horizontalalignment(),
            "center",
        )
        self.assertEqual(
            ax.tables[0].get_celld()[(1, -1)].get_text().get_verticalalignment(),
            "center",
        )
        self.assertEqual(
            ax.tables[0].get_celld()[(1, -1)].get_facecolor(),
            to_rgba("#bfbfbf"),
        )
        # Now same as above for a cell in the table
        self.assertEqual(
            ax.tables[0].get_celld()[(1, 1)].get_text().get_text(), "223.9369"
        )
        self.assertEqual(
            ax.tables[0].get_celld()[(1, 1)].get_text().get_color(), "black"
        )
        self.assertEqual(
            ax.tables[0].get_celld()[(1, 1)].get_text().get_horizontalalignment(),
            "center",
        )
        self.assertEqual(
            ax.tables[0].get_celld()[(1, 1)].get_text().get_verticalalignment(),
            "center",
        )
        self.assertEqual(
            ax.tables[0].get_celld()[(1, 1)].get_facecolor(),
            to_rgba("#bfbfbf"),
        )
        self.assertEqual(ax.tables[0].get_celld()[(1, 1)].get_width(), 0.1 * 1.1)

    def test_copy(self):
        data = [
            [5, 223.9369, 0.0323, 0.0532, 0.1764],
            [10, 223.9367, 0.0324, 0.0533, 0.1765],
            [15, 223.9367, 0.0325, 0.0534, 0.1764],
            [20, 223.9387, 0.0326, 0.0535, 0.1763],
            [25, 223.9385, 0.0327, 0.0536, 0.1761],
        ]
        columns = [
            "Time (s)",
            "Voltage (V)",
            "Current 1 (A)",
            "Current 2 (A)",
            "Current 3 (A)",
        ]
        rows = ["Series 1", "Series 2", "Series 3", "Series 4", "Series 5"]
        colors = [["#bfbfbf"] * 5] * 5

        table = Table(
            cell_text=data,
            cell_colors=colors,
            cell_align="center",
            col_labels=columns,
            col_widths=[0.1, 0.1, 0.1, 0.1, 0.1],
            col_align="center",
            col_colors=["#bfbfbf"] * 5,
            row_labels=rows,
            row_align="center",
            row_colors=["#bfbfbf"] * 5,
            scaling=(1.1, 1.1),
            location="bottom",
        )
        tableCopy = table.copy()
        self.assertListEqual(tableCopy.cell_text, table.cell_text)
        self.assertListEqual(tableCopy.cell_colors, table.cell_colors)
        self.assertEqual(tableCopy.cell_align, table.cell_align)
        self.assertListEqual(tableCopy.col_labels, table.col_labels)
        self.assertListEqual(tableCopy.col_widths, table.col_widths)
        self.assertEqual(tableCopy.col_align, table.col_align)
        self.assertListEqual(tableCopy.col_colors, table.col_colors)
        self.assertListEqual(tableCopy.row_labels, table.row_labels)
        self.assertEqual(tableCopy.row_align, table.row_align)
        self.assertListEqual(tableCopy.row_colors, table.row_colors)


if __name__ == "__main__":
    unittest.main()
