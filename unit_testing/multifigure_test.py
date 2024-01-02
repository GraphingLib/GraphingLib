import unittest

from graphinglib.data_plotting_1d import Curve
from graphinglib.figure import Figure
from graphinglib.file_manager import FileLoader
from graphinglib.multifigure import MultiFigure


class TestMultiFigure(unittest.TestCase):
    def test_create_multifigure(self):
        a_multifig = MultiFigure(num_rows=2, num_cols=2)
        self.assertEqual(a_multifig.num_rows, 2)
        self.assertEqual(a_multifig.num_cols, 2)

        another_multifig = MultiFigure(
            num_rows=2,
            num_cols=2,
            size=(5, 5),
            title="Test Title",
            reference_labels=False,
            reflabel_loc="inside",
            figure_style="plain",
        )
        self.assertEqual(another_multifig.size, (5, 5))
        self.assertEqual(another_multifig.title, "Test Title")
        self.assertEqual(another_multifig.reference_labels, False)
        self.assertEqual(another_multifig.reflabel_loc, "inside")
        self.assertEqual(another_multifig.figure_style, "plain")
        self.assertListEqual(another_multifig._sub_figures, [])
        self.assertDictEqual(another_multifig._rc_dict, {})
        self.assertDictEqual(another_multifig._user_rc_dict, {})

    def test_create_multifigure_raises(self):
        # Test that raises error if num_rows or num_cols is not an integer or is less than 1
        with self.assertRaises(TypeError):
            MultiFigure(num_rows=2.5, num_cols=2)
        with self.assertRaises(ValueError):
            MultiFigure(num_rows=0, num_cols=2)
        with self.assertRaises(ValueError):
            MultiFigure(num_rows=2, num_cols=0)

    def test_add_sub_figure(self):
        a_figure = Figure()
        a_multifig = MultiFigure(num_rows=2, num_cols=2)
        a_multifig.add_sub_figure(a_figure, 0, 0, 1, 1)
        self.assertEqual(a_multifig._sub_figures[0], a_figure)
        self.assertEqual(a_figure.row_start, 0)
        self.assertEqual(a_figure.row_span, 1)
        self.assertEqual(a_figure.col_start, 0)
        self.assertEqual(a_figure.col_span, 1)

        another_figure = Figure()
        a_multifig.add_sub_figure(another_figure, 1, 1, 1, 1)
        self.assertEqual(a_multifig._sub_figures[1], another_figure)
        self.assertEqual(another_figure.row_start, 1)
        self.assertEqual(another_figure.row_span, 1)
        self.assertEqual(another_figure.col_start, 1)
        self.assertEqual(another_figure.col_span, 1)

    def test_add_sub_figure_raises(self):
        a_figure = Figure()
        a_multifig = MultiFigure(num_rows=2, num_cols=2)
        with self.assertRaises(TypeError):
            a_multifig.add_sub_figure(a_figure, 0.5, 0, 1, 1)

        with self.assertRaises(ValueError):
            a_multifig.add_sub_figure(a_figure, 0, 2, 1, 1)

        with self.assertRaises(ValueError):
            a_multifig.add_sub_figure(a_figure, 0, 0, 0, 1)

        with self.assertRaises(ValueError):
            a_multifig.add_sub_figure(a_figure, 0, 0, 1, 0)

        with self.assertRaises(ValueError):
            a_multifig.add_sub_figure(a_figure, 0, 0, 3, 1)

        with self.assertRaises(ValueError):
            a_multifig.add_sub_figure(a_figure, 0, 0, 1, 3)

    def test_stack(self):
        a_figure = Figure()
        another_figure = Figure()
        a_multifig = MultiFigure.stack([a_figure, another_figure])
        self.assertEqual(a_multifig.num_rows, 2)
        self.assertEqual(a_multifig.num_cols, 1)
        self.assertEqual(a_multifig._sub_figures[0], a_figure)
        self.assertEqual(a_multifig._sub_figures[1], another_figure)

        self.assertEqual(a_figure.row_start, 0)
        self.assertEqual(a_figure.row_span, 1)
        self.assertEqual(a_figure.col_start, 0)
        self.assertEqual(a_figure.col_span, 1)

        self.assertEqual(another_figure.row_start, 1)
        self.assertEqual(another_figure.row_span, 1)
        self.assertEqual(another_figure.col_start, 0)
        self.assertEqual(another_figure.col_span, 1)

    def test_row(self):
        a_figure = Figure()
        another_figure = Figure()
        a_multifig = MultiFigure.row([a_figure, another_figure])
        self.assertEqual(a_multifig.num_rows, 1)
        self.assertEqual(a_multifig.num_cols, 2)
        self.assertEqual(a_multifig._sub_figures[0], a_figure)
        self.assertEqual(a_multifig._sub_figures[1], another_figure)

        self.assertEqual(a_figure.row_start, 0)
        self.assertEqual(a_figure.row_span, 1)
        self.assertEqual(a_figure.col_start, 0)
        self.assertEqual(a_figure.col_span, 1)

        self.assertEqual(another_figure.row_start, 0)
        self.assertEqual(another_figure.row_span, 1)
        self.assertEqual(another_figure.col_start, 1)
        self.assertEqual(another_figure.col_span, 1)

    def test_prepare_multifigure(self):
        # TODO: Test this
        pass

    def test_prepare_sub_figure(self):
        # TODO: Test this
        pass

    def test_fill_in_missing_params(self):
        # TODO: Test this
        pass

    def test_reset_params_to_default(self):
        # TODO: Test this
        pass

    def test_fill_in_rc_params(self):
        # TODO: Test this
        pass

    def test_update_rc_params(self):
        a_multifigure = MultiFigure(num_rows=2, num_cols=2)
        params = {
            "lines.linewidth": 2,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
        }
        a_multifigure.update_rc_params(params)
        self.assertDictEqual(a_multifigure._user_rc_dict, params)
        more_params = {
            "lines.linewidth": 3,
            "axes.grid": True,
        }
        a_multifigure.update_rc_params(more_params)
        resulting_params = {
            "lines.linewidth": 3,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "axes.grid": True,
        }
        self.assertDictEqual(a_multifigure._user_rc_dict, resulting_params)

    def test_update_rc_params_reset(self):
        a_multifigure = MultiFigure(num_rows=2, num_cols=2)
        params = {
            "lines.linewidth": 2,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
        }
        a_multifigure.update_rc_params(params)
        a_multifigure.update_rc_params({"lines.linewidth": 3}, reset=True)
        self.assertDictEqual(a_multifigure._user_rc_dict, {"lines.linewidth": 3})

    def test_customize_visual_style(self):
        a_multifigure = MultiFigure(num_rows=2, num_cols=2)
        a_multifigure.customize_visual_style(
            figure_face_color="red", axes_face_color="blue", grid_line_style="dashed"
        )
        self.assertDictEqual(
            a_multifigure._user_rc_dict,
            {
                "figure.facecolor": "red",
                "axes.facecolor": "blue",
                "grid.linestyle": "dashed",
            },
        )
        a_multifigure.customize_visual_style(
            axes_face_color="yellow", grid_line_style="dotted", x_tick_color="green"
        )
        self.assertDictEqual(
            a_multifigure._user_rc_dict,
            {
                "figure.facecolor": "red",
                "axes.facecolor": "yellow",
                "grid.linestyle": "dotted",
                "xtick.color": "green",
            },
        )

    def test_customize_visual_style_reset(self):
        a_multifigure = MultiFigure(num_rows=2, num_cols=2)
        a_multifigure.customize_visual_style(
            figure_face_color="red", axes_face_color="blue", grid_line_style="dashed"
        )
        a_multifigure.customize_visual_style(
            axes_face_color="yellow", grid_line_style="dotted", x_tick_color="green"
        )
        a_multifigure.customize_visual_style(reset=True)
        self.assertDictEqual(a_multifigure._user_rc_dict, {})


if __name__ == "__main__":
    unittest.main()
