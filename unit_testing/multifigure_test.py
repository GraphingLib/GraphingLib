import unittest

from matplotlib import pyplot as plt

from graphinglib.data_plotting_1d import Curve
from graphinglib.figure import Figure
from graphinglib.graph_elements import GraphingException
from graphinglib.multifigure import MultiFigure


class TestMultiFigure(unittest.TestCase):
    def test_create_multifigure(self):
        a_multifig = MultiFigure(num_rows=2, num_cols=2)
        self.assertEqual(a_multifig._num_rows, 2)
        self.assertEqual(a_multifig._num_cols, 2)

        another_multifig = MultiFigure(
            num_rows=2,
            num_cols=2,
            size=(5, 5),
            title="Test Title",
            reference_labels=False,
            reflabel_loc="inside",
            figure_style="plain",
        )
        self.assertEqual(another_multifig._size, (5, 5))
        self.assertEqual(another_multifig._title, "Test Title")
        self.assertEqual(another_multifig._reference_labels, False)
        self.assertEqual(another_multifig._reflabel_loc, "inside")
        self.assertEqual(another_multifig._figure_style, "plain")
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
        a_multifig.add_figure(a_figure, 0, 0, 1, 1)
        self.assertEqual(a_multifig._sub_figures[0], a_figure)
        self.assertEqual(a_figure._row_start, 0)
        self.assertEqual(a_figure._row_span, 1)
        self.assertEqual(a_figure._col_start, 0)
        self.assertEqual(a_figure._col_span, 1)

        another_figure = Figure()
        a_multifig.add_figure(another_figure, 1, 1, 1, 1)
        self.assertEqual(a_multifig._sub_figures[1], another_figure)
        self.assertEqual(another_figure._row_start, 1)
        self.assertEqual(another_figure._row_span, 1)
        self.assertEqual(another_figure._col_start, 1)
        self.assertEqual(another_figure._col_span, 1)

    def test_add_sub_figure_raises(self):
        a_figure = Figure()
        a_multifig = MultiFigure(num_rows=2, num_cols=2)
        with self.assertRaises(TypeError):
            a_multifig.add_figure(a_figure, 0.5, 0, 1, 1)

        with self.assertRaises(ValueError):
            a_multifig.add_figure(a_figure, 0, 2, 1, 1)

        with self.assertRaises(ValueError):
            a_multifig.add_figure(a_figure, 0, 0, 0, 1)

        with self.assertRaises(ValueError):
            a_multifig.add_figure(a_figure, 0, 0, 1, 0)

        with self.assertRaises(ValueError):
            a_multifig.add_figure(a_figure, 0, 0, 3, 1)

        with self.assertRaises(ValueError):
            a_multifig.add_figure(a_figure, 0, 0, 1, 3)

    def test_stack(self):
        a_figure = Figure()
        another_figure = Figure()
        a_multifig = MultiFigure.from_stack([a_figure, another_figure])
        self.assertEqual(a_multifig._num_rows, 2)
        self.assertEqual(a_multifig._num_cols, 1)
        self.assertEqual(a_multifig._sub_figures[0], a_figure)
        self.assertEqual(a_multifig._sub_figures[1], another_figure)

        self.assertEqual(a_figure._row_start, 0)
        self.assertEqual(a_figure._row_span, 1)
        self.assertEqual(a_figure._col_start, 0)
        self.assertEqual(a_figure._col_span, 1)

        self.assertEqual(another_figure._row_start, 1)
        self.assertEqual(another_figure._row_span, 1)
        self.assertEqual(another_figure._col_start, 0)
        self.assertEqual(another_figure._col_span, 1)

    def test_row(self):
        a_figure = Figure()
        another_figure = Figure()
        a_multifig = MultiFigure.from_row([a_figure, another_figure])
        self.assertEqual(a_multifig._num_rows, 1)
        self.assertEqual(a_multifig._num_cols, 2)
        self.assertEqual(a_multifig._sub_figures[0], a_figure)
        self.assertEqual(a_multifig._sub_figures[1], another_figure)

        self.assertEqual(a_figure._row_start, 0)
        self.assertEqual(a_figure._row_span, 1)
        self.assertEqual(a_figure._col_start, 0)
        self.assertEqual(a_figure._col_span, 1)

        self.assertEqual(another_figure._row_start, 0)
        self.assertEqual(another_figure._row_span, 1)
        self.assertEqual(another_figure._col_start, 1)
        self.assertEqual(another_figure._col_span, 1)

    def test_grid(self):
        fig1 = Figure()
        fig2 = Figure()
        fig3 = Figure()
        fig4 = Figure()
        fig5 = Figure()
        fig6 = Figure()
        a_multifig = MultiFigure.from_grid(
            figures=[fig1, fig2, fig3, fig4, fig5, fig6],
            dimensions=(2, 3),
            size=(12, 8),
            title="Complex Example",
        )
        self.assertEqual(a_multifig._num_rows, 2)
        self.assertEqual(a_multifig._num_cols, 3)
        self.assertEqual(a_multifig._sub_figures[0], fig1)
        self.assertEqual(a_multifig._sub_figures[1], fig2)
        self.assertEqual(a_multifig._sub_figures[2], fig3)
        self.assertEqual(a_multifig._sub_figures[3], fig4)
        self.assertEqual(a_multifig._sub_figures[4], fig5)
        self.assertEqual(a_multifig._sub_figures[5], fig6)

        self.assertEqual(fig1._row_start, 0)
        self.assertEqual(fig1._row_span, 1)
        self.assertEqual(fig1._col_start, 0)
        self.assertEqual(fig1._col_span, 1)

        self.assertEqual(fig2._row_start, 0)
        self.assertEqual(fig2._row_span, 1)
        self.assertEqual(fig2._col_start, 1)
        self.assertEqual(fig2._col_span, 1)

        self.assertEqual(fig3._row_start, 0)
        self.assertEqual(fig3._row_span, 1)
        self.assertEqual(fig3._col_start, 2)
        self.assertEqual(fig3._col_span, 1)

        self.assertEqual(fig4._row_start, 1)
        self.assertEqual(fig4._row_span, 1)
        self.assertEqual(fig4._col_start, 0)
        self.assertEqual(fig4._col_span, 1)

        self.assertEqual(fig5._row_start, 1)
        self.assertEqual(fig5._row_span, 1)
        self.assertEqual(fig5._col_start, 1)
        self.assertEqual(fig5._col_span, 1)

        self.assertEqual(fig6._row_start, 1)
        self.assertEqual(fig6._row_span, 1)
        self.assertEqual(fig6._col_start, 2)
        self.assertEqual(fig6._col_span, 1)

    def test_grid_raises(self):
        # Grid too small
        fig1 = Figure()
        fig2 = Figure()
        fig3 = Figure()
        fig4 = Figure()
        fig5 = Figure()
        fig6 = Figure()
        with self.assertRaises(ValueError):
            MultiFigure.from_grid(
                figures=[fig1, fig2, fig3, fig4, fig5, fig6],
                dimensions=(2, 2),
                size=(12, 8),
                title="Complex Example",
            )
        with self.assertRaises(ValueError):
            MultiFigure.from_grid(
                figures=[fig1, fig2, fig3, fig4, fig5],
                dimensions=(2, 1),
                size=(12, 8),
                title="Complex Example",
            )
        # Not integer dimensions
        with self.assertRaises(TypeError):
            MultiFigure.from_grid(
                figures=[fig1, fig2, fig3, fig4, fig5],
                dimensions=(2.5, 10),
                size=(12, 8),
                title="Complex Example",
            )

    def test_prepare_multifigure_raises(self):
        a_multifig = MultiFigure(num_rows=2, num_cols=2, figure_style="Burrrd")
        with self.assertRaises(GraphingException):
            a_multifig._prepare_multi_figure()
        plt.close("all")

    def test_prepare_multifigure_resets_rc(self):
        a_multifig = MultiFigure(num_rows=2, num_cols=2, figure_style="plain")
        a_multifig._prepare_multi_figure()
        self.assertDictEqual(a_multifig._rc_dict, {})
        plt.close("all")

    def test_prepare_multifigure_prepares_figures(self):
        a_figure = Figure(figure_style="plain")
        curve = Curve([1, 2, 3], [1, 2, 3])
        curve2 = Curve([1, 2, 3], [1, 2, 3], line_width=5)
        a_figure.add_elements(curve)
        another_figure = Figure(figure_style="plain")
        another_figure.add_elements(curve2)
        a_multifig = MultiFigure.from_stack(
            [a_figure, another_figure], figure_style="plain"
        )
        a_multifig._prepare_multi_figure()
        self.assertEqual(len(a_figure._axes.get_lines()), 1)
        self.assertEqual(len(another_figure._axes.get_lines()), 1)
        self.assertEqual(a_figure._axes.get_lines()[0].get_linewidth(), 2)
        self.assertEqual(another_figure._axes.get_lines()[0].get_linewidth(), 5)
        self.assertFalse(a_figure._axes.get_legend())
        self.assertFalse(another_figure._axes.get_legend())
        plt.close("all")

    def test_fill_in_missing_params(self):
        a_multifig = MultiFigure(num_rows=2, num_cols=2, figure_style="plain")
        a_multifig._default_params = {"MultiFigure": {"_size": (8, 8)}}
        a_multifig._fill_in_missing_params(a_multifig)
        self.assertEqual(a_multifig._size, (8, 8))

    def test_reset_params_to_default(self):
        a_multifig = MultiFigure(num_rows=2, num_cols=2, size=(8, 8))
        params_to_reset = ["size"]
        a_multifig._reset_params_to_default(a_multifig, params_to_reset)

    def test_fill_in_rc_params_matplotlib(self):
        a_multifig = MultiFigure(num_rows=2, num_cols=2, figure_style="bmh")
        a_multifig._fill_in_rc_params(is_matplotlib_style=True)
        # Check plt.rcParams is updated (show grid is True)
        self.assertTrue(plt.rcParams["axes.grid"])
        # Check axes fill color is updated
        self.assertEqual(plt.rcParams["axes.facecolor"], "#eeeeee")

        a = MultiFigure(num_rows=2, num_cols=2, figure_style="Solarize_Light2")
        a._fill_in_rc_params(is_matplotlib_style=True)
        self.assertEqual(plt.rcParams["axes.grid"], True)
        self.assertEqual(plt.rcParams["axes.facecolor"], "#eee8d5")

    def test_fill_in_rc_params_gl(self):
        a_multifig = MultiFigure(num_rows=2, num_cols=2)
        a_multifig.set_visual_params(legend_edge_color="red")
        # Get default params for dim style
        a_multifig._default_params = {
            "rc_params": {
                "axes.grid": False,
                "axes.facecolor": "dimgrey",
                "legend.edgecolor": "blue",
            }
        }
        # Fill in rc params
        a_multifig._fill_in_rc_params(is_matplotlib_style=False)
        # Check axes fill color is updated
        self.assertEqual(plt.rcParams["axes.grid"], False)
        self.assertEqual(plt.rcParams["axes.facecolor"], "dimgrey")
        self.assertEqual(plt.rcParams["legend.edgecolor"], "red")  # Overridden by user

    def test_update_rc_params(self):
        a_multifigure = MultiFigure(num_rows=2, num_cols=2)
        params = {
            "lines.linewidth": 2,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
        }
        a_multifigure.set_rc_params(params)
        self.assertDictEqual(a_multifigure._user_rc_dict, params)
        more_params = {
            "lines.linewidth": 3,
            "axes.grid": True,
        }
        a_multifigure.set_rc_params(more_params)
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
        a_multifigure.set_rc_params(params)
        a_multifigure.set_rc_params({"lines.linewidth": 3}, reset=True)
        self.assertDictEqual(a_multifigure._user_rc_dict, {"lines.linewidth": 3})

    def test_customize_visual_style(self):
        a_multifigure = MultiFigure(num_rows=2, num_cols=2)
        a_multifigure.set_visual_params(
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
        a_multifigure.set_visual_params(
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
        a_multifigure.set_visual_params(
            figure_face_color="red", axes_face_color="blue", grid_line_style="dashed"
        )
        a_multifigure.set_visual_params(
            axes_face_color="yellow", grid_line_style="dotted", x_tick_color="green"
        )
        a_multifigure.set_visual_params(reset=True)
        self.assertDictEqual(a_multifigure._user_rc_dict, {})

    def test_matplotlib_style_functional(self):
        a_figure = Figure(figure_style="matplotlib")
        curve = Curve([1, 2, 3], [1, 2, 3])
        curve2 = Curve([1, 2, 3], [1, 2, 3], line_width=5)
        a_figure.add_elements(curve)
        another_figure = Figure(figure_style="matplotlib")
        another_figure.add_elements(curve2)
        a_multifig = MultiFigure.from_stack(
            [a_figure, another_figure], figure_style="matplotlib"
        )
        a_multifig._prepare_multi_figure()
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
