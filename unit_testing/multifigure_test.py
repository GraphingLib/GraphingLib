import unittest

from graphinglib.data_plotting_1d import Curve
from graphinglib.file_manager import FileLoader
from graphinglib.multifigure import MultiFigure, SubFigure


class TestMultiFigure(unittest.TestCase):
    def setUp(self):
        self.test_curve = Curve.from_function(lambda x: x**2, 0, 10, "Test Curve")
        self.small_multifigure = MultiFigure(1, 1)
        self.big_multifigure = MultiFigure(4, 4)

    def test_create_multifigure(self):
        a_multifig = MultiFigure(num_rows=2, num_cols=2)
        self.assertEqual(a_multifig.num_rows, 2)
        self.assertEqual(a_multifig.num_cols, 2)

        # Test that raises error if num_rows or num_cols is not an integer or is less than 1
        with self.assertRaises(TypeError):
            MultiFigure(num_rows=2.5, num_cols=2)
        with self.assertRaises(ValueError):
            MultiFigure(num_rows=0, num_cols=2)
        with self.assertRaises(ValueError):
            MultiFigure(num_rows=2, num_cols=0)

    def test_create_subfigure(self):
        self.small_multifigure.add_SubFigure(
            row_start=0, col_start=0, row_span=1, col_span=1
        )

    def test_create_subfigure_wrong_span(self):
        # Test that raises error if row_start or col_start is not an integer or is less than 0
        with self.assertRaises(TypeError):
            self.small_multifigure.add_SubFigure(
                row_start=0.5, col_start=0, row_span=1, col_span=1
            )
        with self.assertRaises(ValueError):
            self.small_multifigure.add_SubFigure(
                row_start=-1, col_start=0, row_span=1, col_span=1
            )
        with self.assertRaises(TypeError):
            self.small_multifigure.add_SubFigure(
                row_start=0, col_start=0.5, row_span=1, col_span=1
            )

    def test_create_subfigure_span_too_high(self):
        # Test that raises error if row_span or col_span is greater than or equal to num_rows or num_cols
        with self.assertRaises(ValueError):
            self.small_multifigure.add_SubFigure(
                row_start=0, col_start=0, row_span=2, col_span=1
            )
        with self.assertRaises(ValueError):
            self.small_multifigure.add_SubFigure(
                row_start=0, col_start=0, row_span=1, col_span=2
            )

    def test_create_subfigure_start_too_high(self):
        # Test that raises error if row_start or col_start is greater than or equal to num_rows or num_cols
        with self.assertRaises(ValueError):
            self.small_multifigure.add_SubFigure(
                row_start=1, col_start=0, row_span=1, col_span=1
            )
        with self.assertRaises(ValueError):
            self.small_multifigure.add_SubFigure(
                row_start=0, col_start=1, row_span=1, col_span=1
            )

    def test_fill_in_missing_params_plain(self):
        # Fill in values for plain style
        self.small_multifigure.add_SubFigure(
            row_start=0, col_start=0, row_span=1, col_span=1
        )
        self.small_multifigure.default_params = FileLoader("plain").load()
        self.small_multifigure._fill_in_missing_params(self.small_multifigure)
        self.assertListEqual(self.small_multifigure.size, [6.4, 4.8])

    def test_fill_in_missing_params_horrible(self):
        # Fill in values for horrible style
        self.small_multifigure.add_SubFigure(
            row_start=0, col_start=0, row_span=1, col_span=1
        )
        self.small_multifigure.default_params = FileLoader("horrible").load()
        self.small_multifigure._fill_in_missing_params(self.small_multifigure)
        self.assertEqual(self.small_multifigure.size, [10, 7])

    def test_reset_params_to_default(self):
        self.small_multifigure.add_SubFigure(
            row_start=0, col_start=0, row_span=1, col_span=1
        )
        self.small_multifigure.default_params = FileLoader("plain").load()
        params = self.small_multifigure._fill_in_missing_params(self.small_multifigure)
        self.small_multifigure._reset_params_to_default(self.small_multifigure, params)
        self.assertEqual(self.small_multifigure.size, "default")


class TestSubFigure(unittest.TestCase):
    def setUp(self) -> None:
        self.test_curve = Curve.from_function(lambda x: x**2, 0, 10, "Test Curve")
        self.other_curve = Curve.from_function(lambda x: x**3, 0, 10, "Other Curve")
        self.subfigure = SubFigure(0, 0, 1, 1)

    def test_add_element(self):
        self.subfigure.add_element(self.test_curve)
        self.assertEqual(self.subfigure._elements[0], self.test_curve)

        # Add multiple elements
        self.subfigure.add_element(self.other_curve, self.test_curve)
        self.assertListEqual(
            self.subfigure._elements,
            [self.test_curve, self.other_curve, self.test_curve],
        )

    def test_fill_in_missing_params_plain(self):
        self.subfigure.default_params = FileLoader("plain").load()
        self.subfigure._fill_in_missing_params(self.subfigure)
        self.assertEqual(self.subfigure.log_scale_x, False)

    def test_fill_in_missing_params_curve(self):
        self.subfigure.add_element(self.test_curve)
        self.subfigure.default_params = FileLoader("plain").load()
        self.subfigure._fill_in_missing_params(self.test_curve)
        self.assertEqual(self.test_curve.line_style, "-")
        self.assertEqual(self.test_curve.line_width, 2)

    def test_fill_in_missing_params_horrible(self):
        self.subfigure.default_params = FileLoader("horrible").load()
        self.subfigure._fill_in_missing_params(self.subfigure)
        self.assertEqual(self.subfigure.log_scale_x, False)

    def test_dont_overwrite_specified_params(self):
        self.subfigure.show_grid = False
        self.subfigure.default_params = FileLoader("horrible").load()
        self.subfigure._fill_in_missing_params(self.subfigure)
        self.assertEqual(self.subfigure.show_grid, False)

    def test_dont_overwrite_specified_params_curve(self):
        self.test_curve.line_style = "--"
        self.test_curve.line_width = 3
        self.subfigure.add_element(self.test_curve)
        self.subfigure.default_params = FileLoader("plain").load()
        self.subfigure._fill_in_missing_params(self.test_curve)
        self.assertEqual(self.test_curve.line_style, "--")
        self.assertEqual(self.test_curve.line_width, 3)

    def test_reset_params_to_default(self):
        self.subfigure.default_params = FileLoader("plain").load()
        params = self.subfigure._fill_in_missing_params(self.subfigure)
        self.subfigure._reset_params_to_default(self.subfigure, params)
        self.assertEqual(self.subfigure.log_scale_x, "default")

    def test_raise_exception_if_no_element_added(self):
        multifig = MultiFigure(1, 1)
        multifig.add_SubFigure(0, 0, 1, 1)
        with self.assertRaises(Exception):
            multifig._prepare_MultiFigure()

    def test_all_curves_plotted(self):
        multifig = MultiFigure(1, 1)
        sub1 = multifig.add_SubFigure(0, 0, 1, 1)
        sub1.add_element(self.test_curve)
        multifig._prepare_MultiFigure()
        self.assertEqual(len(sub1._axes.get_lines()), len(sub1._elements))

    def test_handles_and_labels_cleared(self):
        multifig = MultiFigure(1, 1)
        sub1 = multifig.add_SubFigure(row_start=0, col_start=0, row_span=1, col_span=1)
        sub1.add_element(self.test_curve)
        multifig._prepare_MultiFigure()
        self.assertEqual(len(sub1._handles), 0)
        self.assertEqual(len(sub1._labels), 0)

    def test_handles_and_labels_added(self):
        multifig = MultiFigure(1, 1)
        sub1 = multifig.add_SubFigure(row_start=0, col_start=0, row_span=1, col_span=1)
        sub1.add_element(self.test_curve)
        other_curve = Curve(
            self.test_curve.x_data, self.test_curve.y_data, "Other Curve"
        )
        sub1.add_element(other_curve)
        multifig._prepare_MultiFigure()
        handles, labels = sub1._axes.get_legend_handles_labels()
        self.assertEqual(len(handles), 2)
        self.assertListEqual(labels, ["Test Curve", "Other Curve"])
        # test if still ok when replotting
        multifig.figure_style = "horrible"
        multifig._prepare_MultiFigure()
        handles, labels = sub1._axes.get_legend_handles_labels()
        self.assertEqual(len(handles), 2)
        self.assertListEqual(labels, ["Test Curve", "Other Curve"])
