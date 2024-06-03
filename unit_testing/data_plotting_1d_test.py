import unittest
from calendar import c
from random import random

from matplotlib.colors import to_hex
from matplotlib.pyplot import close, subplots
from numpy import linspace, ndarray, pi, sin

from graphinglib.data_plotting_1d import Curve, Histogram, Scatter
from graphinglib.figure import Figure
from graphinglib.fits import FitFromPolynomial
from graphinglib.graph_elements import Point


class TestCurve(unittest.TestCase):
    def setUp(self):
        x = linspace(0, 3 * pi, 1000)
        self.testCurve = Curve(x, sin(x), "Test Curve", color="k")

    def test_xdata_is_list_or_ndarray(self):
        self.assertIsInstance(self.testCurve._x_data, list | ndarray)

    def test_ydata_is_list_or_ndarray(self):
        self.assertIsInstance(self.testCurve._y_data, list | ndarray)

    def test_default_value(self):
        self.assertEqual(self.testCurve._line_width, "default")

    def test_color_is_str(self):
        self.assertIsInstance(self.testCurve._color, str)

    def test_label_is_str(self):
        self.assertIsInstance(self.testCurve._label, str)

    def test_from_function(self):
        self.assertIsInstance(
            Curve.from_function(
                lambda x: x**2,
                0,
                1,
                "Test",
                color="k",
                line_width=1,
                line_style="-",
                number_of_points=400,
            ),
            Curve,
        )

    def test_add_errorbars(self):
        self.testCurve.add_errorbars(0.1, 0.1)
        self.testCurve.add_errorbars(0.1, [0.2, 0.3] * 100)
        self.testCurve.add_errorbars([0.2, 0.3] * 100, 0.1)
        self.testCurve.add_errorbars(0.3, None)
        self.testCurve.add_errorbars(None, 0.3)
        self.testCurve.add_errorbars(
            0.1,
            0.1,
            errorbars_color="red",
            errorbars_line_width=10,
            cap_thickness=10,
            cap_width=10,
        )
        self.assertEqual(self.testCurve._errorbars_color, "red")
        self.assertEqual(self.testCurve._errorbars_line_width, 10)
        self.assertEqual(self.testCurve._cap_thickness, 10)
        self.assertEqual(self.testCurve._cap_width, 10)

    def test_create_point_at_x(self):
        point = self.testCurve.create_point_at_x(0.5)
        self.assertEqual(point.x, 0.5)
        self.assertAlmostEqual(point.y, sin(0.5), places=3)

    def test_get_coordinates_at_x(self):
        point = self.testCurve.get_coordinates_at_x(0.5)
        self.assertEqual(point[0], 0.5)
        self.assertAlmostEqual(point[1], sin(0.5), places=3)

    def test_get_points_at_y(self):
        points = self.testCurve.get_coordinates_at_y(0)
        for i, point in enumerate(points):
            self.assertEqual(point[1], 0)
            self.assertAlmostEqual(point[0], i * pi, places=3)

    def test_create_points_at_y(self):
        points = self.testCurve.create_points_at_y(0)
        for i, point in enumerate(points):
            self.assertEqual(point.y, 0)
            self.assertAlmostEqual(point.x, i * pi, places=3)

    def test_curve_is_plotted(self):
        x = linspace(0, 3 * pi, 200)
        self.testCurve = Curve(
            x, sin(x), "Test Curve", color="k", line_width=3, line_style="--"
        )
        _, self.testAxes = subplots()
        self.testCurve._plot_element(self.testAxes, 0)
        self.assertEqual(len(self.testAxes.get_lines()), 1)

    def test_get_derivative_curve(self):
        self.assertIsInstance(self.testCurve.create_derivative_curve(), Curve)

    def test_get_integral_curve(self):
        self.assertIsInstance(self.testCurve.create_integral_curve(), Curve)

    def test_get_tangent_curve(self):
        self.assertIsInstance(self.testCurve.create_tangent_curve(0), Curve)

    def test_get_normal_curve(self):
        self.assertIsInstance(self.testCurve.create_normal_curve(0), Curve)

    def test_area_between(self):
        self.assertAlmostEqual(self.testCurve.get_area_between(0, pi), 2, places=3)

    def test_area_between_fill_under(self):
        curve = Curve.from_function(lambda x: x**2, 0, 1)
        curve2 = Curve.from_function(lambda x: x**3, 0, 1)
        curve2.get_area_between(0, 1, fill_between=True)
        fig = Figure(figure_style="plain")
        fig.add_elements(curve, curve2)
        fig._prepare_figure()
        fill_under_color = to_hex(fig._axes.collections[0].get_facecolor()[0])
        curve2_color = fig._axes.get_lines()[1].get_color()
        self.assertEqual(fill_under_color, curve2_color)
        close("all")

    def test_area_between_fill_under_two_curves(self):
        curve = Curve.from_function(lambda x: x**2, 0, 1)
        curve2 = Curve.from_function(lambda x: x**3, 0, 1)
        area = curve.get_area_between(0, 1, fill_between=True, other_curve=curve2)
        self.assertAlmostEqual(area, 1 / 12, places=3)
        fig = Figure(figure_style="plain")
        fig.add_elements(curve, curve2)
        fig._prepare_figure()
        fill_under_color = to_hex(fig._axes.collections[0].get_facecolor()[0])
        curve_color = fig._axes.get_lines()[0].get_color()
        self.assertEqual(fill_under_color, curve_color)
        close("all")

    def test_slope_at(self):
        self.assertAlmostEqual(self.testCurve.get_slope_at(pi / 2), 0, places=5)

    def test_arc_length_between(self):
        self.assertAlmostEqual(
            self.testCurve.get_arc_length_between(0, pi), 3.820, places=3
        )

    def test_get_intersection_coordinates(self):
        x = linspace(0, 3 * pi, 1000)
        other_curve = Curve(x, 0.005 * x**2 + 0.1, "Other Curve", color="k")
        points = self.testCurve.get_intersection_coordinates(other_curve)
        points_x = [0.1, 2.9962, 6.6072, 8.9052]
        points_y = [0.1, 0.14489, 0.3183, 0.4965]
        self.assertEqual(len(points), 4)
        for i, point in enumerate(points):
            self.assertAlmostEqual(point[0], points_x[i], places=3)
            self.assertAlmostEqual(point[1], points_y[i], places=3)

    def test_intersection(self):
        x = linspace(0, 3 * pi, 1000)
        other_curve = Curve(x, 0.005 * x**2 + 0.1, "Other Curve", color="k")
        points = self.testCurve.create_intersection_points(other_curve)
        points_x = [0.1, 2.9962, 6.6072, 8.9052]
        points_y = [0.1, 0.14489, 0.3183, 0.4965]
        self.assertEqual(len(points), 4)
        for i, point in enumerate(points):
            self.assertIsInstance(point, Point)
            self.assertAlmostEqual(point.x, points_x[i], places=3)
            self.assertAlmostEqual(point.y, points_y[i], places=3)

    def test_add_curves(self):
        x = linspace(0, 3 * pi, 200)
        other_curve = Curve(x, 0.005 * x**2 + 0.1, "Other Curve", color="k")
        curve_sum = self.testCurve + other_curve
        self.assertIsInstance(curve_sum, Curve)
        self.assertAlmostEqual(curve_sum.get_coordinates_at_x(0)[1], 0.1, places=3)
        self.assertAlmostEqual(curve_sum.get_coordinates_at_x(2)[1], 1.029, places=3)

    def test_subtract_curves(self):
        x = linspace(0, 3 * pi, 200)
        other_curve = Curve(x, 0.005 * x**2 + 0.1, "Other Curve", color="k")
        curve_sub = self.testCurve - other_curve
        self.assertIsInstance(curve_sub, Curve)
        self.assertAlmostEqual(curve_sub.get_coordinates_at_x(0)[1], -0.1, places=3)
        self.assertAlmostEqual(curve_sub.get_coordinates_at_x(2)[1], 0.789, places=3)

    def test_multiply_curves(self):
        x = linspace(0, 3 * pi, 200)
        other_curve = Curve(x, 0.005 * x**2 + 0.1, "Other Curve", color="k")
        curve_mult = self.testCurve * other_curve
        self.assertIsInstance(curve_mult, Curve)
        self.assertAlmostEqual(curve_mult.get_coordinates_at_x(0)[1], 0, places=3)
        self.assertAlmostEqual(curve_mult.get_coordinates_at_x(2)[1], 0.109, places=3)

    def test_divide_curves(self):
        x = linspace(0, 3 * pi, 200)
        other_curve = Curve(x, 0.005 * x**2 + 0.1, "Other Curve", color="k")
        curve_div = self.testCurve / other_curve
        self.assertIsInstance(curve_div, Curve)
        self.assertAlmostEqual(curve_div.get_coordinates_at_x(0)[1], 0, places=3)
        self.assertAlmostEqual(curve_div.get_coordinates_at_x(2)[1], 7.57748, places=2)

    def test_add_with_int(self):
        curve_sum = self.testCurve + 2
        self.assertIsInstance(curve_sum, Curve)
        self.assertAlmostEqual(curve_sum.get_coordinates_at_x(0)[1], 2, places=3)
        self.assertAlmostEqual(curve_sum.get_coordinates_at_x(2)[1], 2.909118, places=3)
        # Test with int on the left
        curve_sum = 2 + self.testCurve
        self.assertIsInstance(curve_sum, Curve)
        self.assertAlmostEqual(curve_sum.get_coordinates_at_x(0)[1], 2, places=3)
        self.assertAlmostEqual(curve_sum.get_coordinates_at_x(2)[1], 2.909118, places=3)
        # in place
        self.testCurve += 2
        self.assertAlmostEqual(self.testCurve.get_coordinates_at_x(0)[1], 2, places=3)
        self.assertAlmostEqual(
            self.testCurve.get_coordinates_at_x(2)[1], 2.909118, places=3
        )

    def test_subtract_with_int(self):
        curve_sub = self.testCurve - 2
        self.assertIsInstance(curve_sub, Curve)
        self.assertAlmostEqual(curve_sub.get_coordinates_at_x(0)[1], -2, places=3)
        self.assertAlmostEqual(curve_sub.get_coordinates_at_x(2)[1], -1.09088, places=3)
        # Test with int on the left
        curve_sub = 2 - self.testCurve
        self.assertIsInstance(curve_sub, Curve)
        self.assertAlmostEqual(
            curve_sub.get_coordinates_at_x(0)[1],
            2 - self.testCurve.get_coordinates_at_x(0)[1],
            places=3,
        )
        self.assertAlmostEqual(
            curve_sub.get_coordinates_at_x(2)[1],
            2 - self.testCurve.get_coordinates_at_x(2)[1],
            places=3,
        )
        # in place
        self.testCurve -= 2
        self.assertAlmostEqual(self.testCurve.get_coordinates_at_x(0)[1], -2, places=3)
        self.assertAlmostEqual(
            self.testCurve.get_coordinates_at_x(2)[1], -1.09088, places=3
        )

    def test_multiply_with_int(self):
        curve_mult = self.testCurve * 2
        self.assertIsInstance(curve_mult, Curve)
        self.assertAlmostEqual(curve_mult.get_coordinates_at_x(0)[1], 0, places=3)
        self.assertAlmostEqual(curve_mult.get_coordinates_at_x(2)[1], 1.8182, places=3)
        # Test with int on the left
        curve_mult = 2 * self.testCurve
        self.assertIsInstance(curve_mult, Curve)
        self.assertAlmostEqual(curve_mult.get_coordinates_at_x(0)[1], 0, places=3)
        self.assertAlmostEqual(curve_mult.get_coordinates_at_x(2)[1], 1.8182, places=3)
        # in place
        self.testCurve *= 2
        self.assertAlmostEqual(self.testCurve.get_coordinates_at_x(0)[1], 0, places=3)
        self.assertAlmostEqual(
            self.testCurve.get_coordinates_at_x(2)[1], 1.8182, places=3
        )

    def test_divide_with_int(self):
        curve_div = self.testCurve / 2
        self.assertIsInstance(curve_div, Curve)
        self.assertAlmostEqual(curve_div.get_coordinates_at_x(0)[1], 0, places=3)
        self.assertAlmostEqual(curve_div.get_coordinates_at_x(2)[1], 0.45455, places=3)
        # Test with int on the left
        new_curve = 5 + self.testCurve
        curve_div = 2 / new_curve
        self.assertIsInstance(curve_div, Curve)
        self.assertAlmostEqual(
            curve_div.get_coordinates_at_x(0)[1],
            2 / new_curve.get_coordinates_at_x(0)[1],
            places=3,
        )
        self.assertAlmostEqual(
            curve_div.get_coordinates_at_x(2)[1],
            2 / new_curve.get_coordinates_at_x(2)[1],
            places=3,
        )
        # in place
        self.testCurve /= 2
        self.assertAlmostEqual(self.testCurve.get_coordinates_at_x(0)[1], 0, places=3)
        self.assertAlmostEqual(
            self.testCurve.get_coordinates_at_x(2)[1], 0.45455, places=3
        )

    def test_power_with_int(self):
        curve_pow = self.testCurve**2
        self.assertIsInstance(curve_pow, Curve)
        self.assertAlmostEqual(curve_pow.get_coordinates_at_x(0)[1], 0, places=3)
        self.assertAlmostEqual(
            curve_pow.get_coordinates_at_x(2)[1],
            self.testCurve.get_coordinates_at_x(2)[1] ** 2,
            places=3,
        )
        # in place
        reference_val = self.testCurve.get_coordinates_at_x(2)[1]
        self.testCurve **= 2
        self.assertAlmostEqual(self.testCurve.get_coordinates_at_x(0)[1], 0, places=3)
        self.assertAlmostEqual(
            self.testCurve.get_coordinates_at_x(2)[1], reference_val**2, places=3
        )

    def test_max_curves(self):
        curve_max = max(self.testCurve)
        self.assertAlmostEqual(curve_max, 1, places=3)

    def test_min_curves(self):
        curve_min = min(self.testCurve)
        self.assertAlmostEqual(curve_min, -1, places=3)

    def test_add_different_sizes(self):
        x = linspace(0, 3 * pi, 100)
        other_curve = Curve(x, 0.005 * x**2 + 0.1, "Other Curve", color="k")
        curve_sum = self.testCurve + other_curve
        self.assertIsInstance(curve_sum, Curve)
        self.assertAlmostEqual(curve_sum.get_coordinates_at_x(0)[1], 0.1, places=3)

    def test_absolute_value(self):
        curve = abs(self.testCurve)
        self.assertIsInstance(curve, Curve)
        self.assertListEqual(
            list(curve._y_data), [abs(y) for y in self.testCurve._y_data]
        )

    def test_copy(self):
        curve_copy = self.testCurve.copy()
        self.assertIsInstance(curve_copy, Curve)
        self.assertEqual(curve_copy._label, self.testCurve._label)
        self.assertEqual(curve_copy._color, self.testCurve._color)
        self.assertEqual(curve_copy._line_width, self.testCurve._line_width)
        self.assertEqual(curve_copy._line_style, self.testCurve._line_style)
        self.assertListEqual(list(curve_copy._x_data), list(self.testCurve._x_data))
        self.assertListEqual(list(curve_copy._y_data), list(self.testCurve._y_data))

    def test_create_slice_x(self):
        curve = Curve.from_function(lambda x: x**2, -10, 10, number_of_points=100)
        curve_slice = curve.create_slice_x(-5, 5)
        self.assertIsInstance(curve_slice, Curve)
        correct_x_data = linspace(-10, 10, 100)
        correct_x_data = correct_x_data[correct_x_data >= -5]
        correct_x_data = correct_x_data[correct_x_data <= 5]
        correct_y_data = correct_x_data**2
        self.assertListEqual(list(curve_slice._x_data), list(correct_x_data))
        self.assertListEqual(list(curve_slice._y_data), list(correct_y_data))

    def test_create_slice_y(self):
        curve = Curve.from_function(lambda x: x**2, -10, 10, number_of_points=100)
        curve_slice = curve.create_slice_y(0, 25)
        self.assertIsInstance(curve_slice, Curve)
        correct_x_data = linspace(-10, 10, 100)
        correct_x_data = correct_x_data[correct_x_data**2 <= 25]
        correct_y_data = correct_x_data**2
        self.assertListEqual(list(curve_slice._x_data), list(correct_x_data))
        self.assertListEqual(list(curve_slice._y_data), list(correct_y_data))


class TestScatter(unittest.TestCase):
    def setUp(self):
        x = linspace(0, 3 * pi, 200)
        self.testScatter = Scatter(x, sin(x), "Test Curve")

    def test_scatter_is_scatter(self):
        self.assertIsInstance(self.testScatter, Scatter)

    def test_xdata_is_list_or_ndarray(self):
        self.assertIsInstance(self.testScatter._x_data, list | ndarray)

    def test_color_is_str(self):
        self.assertIsInstance(self.testScatter._face_color, str)
        self.assertIsInstance(self.testScatter._edge_color, str)

    def test_label_is_str(self):
        self.assertIsInstance(self.testScatter._label, str)

    def test_from_function(self):
        self.assertIsInstance(
            Scatter.from_function(
                lambda x: x**2,
                0,
                1,
                "Test",
                face_color="k",
                edge_color="k",
                marker_size=1,
                marker_style="o",
                number_of_points=400,
            ),
            Scatter,
        )

    def test_add_errorbars(self):
        self.testScatter.add_errorbars(0.1, 0.1)
        self.testScatter.add_errorbars(0.1, [0.2, 0.3] * 100)
        self.testScatter.add_errorbars([0.2, 0.3] * 100, 0.1)
        self.testScatter.add_errorbars(0.3, None)
        self.testScatter.add_errorbars(None, 0.3)
        self.testScatter.add_errorbars(
            0.1,
            0.1,
            errorbars_color="red",
            cap_width=10,
            cap_thickness=10,
            errorbars_line_width=10,
        )
        self.assertEqual(self.testScatter._errorbars_color, "red")
        self.assertEqual(self.testScatter._errorbars_line_width, 10)
        self.assertEqual(self.testScatter._cap_thickness, 10)
        self.assertEqual(self.testScatter._cap_width, 10)

    def test_create_point_at_x(self):
        point = self.testScatter.create_point_at_x(0.5)
        self.assertEqual(point.x, 0.5)
        self.assertAlmostEqual(point.y, sin(0.5), places=3)

    def test_get_coordinates_at_x(self):
        point = self.testScatter.get_coordinates_at_x(0.5)
        self.assertEqual(point[0], 0.5)
        self.assertAlmostEqual(point[1], sin(0.5), places=3)

    def test_get_coordinates_at_y(self):
        points = self.testScatter.get_coordinates_at_y(0)
        for i, point in enumerate(points):
            self.assertEqual(point[1], 0)
            self.assertAlmostEqual(point[0], i * pi, places=3)

    def test_create_points_at_y(self):
        points = self.testScatter.create_points_at_y(0)
        for i, point in enumerate(points):
            self.assertEqual(point.y, 0)
            self.assertAlmostEqual(point.x, i * pi, places=3)

    def test_curve_is_plotted(self):
        x = linspace(0, 3 * pi, 200)
        self.testScatter = Scatter(
            x,
            sin(x),
            "test scatter",
            face_color="k",
            edge_color="k",
            marker_size=30,
            marker_style="o",
        )
        _, self.testAxes = subplots()
        self.testScatter._plot_element(self.testAxes, 0)
        self.assertEqual(len(self.testAxes.get_lines()), 1)

    def test_add_scatter(self):
        x = linspace(0, 3 * pi, 200)
        other_scatter = Scatter(x, 0.005 * x**2 + 0.1, "Other Scatter")
        scatter_sum = self.testScatter + other_scatter
        self.assertIsInstance(scatter_sum, Scatter)
        self.assertAlmostEqual(scatter_sum.get_coordinates_at_x(0)[1], 0.1, places=3)
        self.assertAlmostEqual(scatter_sum.get_coordinates_at_x(2)[1], 1.029, places=3)

    def test_add_scatter_with_int(self):
        scatter_sum = self.testScatter + 2
        self.assertIsInstance(scatter_sum, Scatter)
        self.assertAlmostEqual(scatter_sum.get_coordinates_at_x(0)[1], 2, places=3)
        self.assertAlmostEqual(
            scatter_sum.get_coordinates_at_x(2)[1], 2.909118, places=3
        )
        # Test with int on the left
        scatter_sum = 2 + self.testScatter
        self.assertIsInstance(scatter_sum, Scatter)
        self.assertAlmostEqual(scatter_sum.get_coordinates_at_x(0)[1], 2, places=3)
        self.assertAlmostEqual(
            scatter_sum.get_coordinates_at_x(2)[1], 2.909118, places=3
        )
        # in place
        self.testScatter += 2
        self.assertAlmostEqual(self.testScatter.get_coordinates_at_x(0)[1], 2, places=3)
        self.assertAlmostEqual(
            self.testScatter.get_coordinates_at_x(2)[1], 2.909118, places=3
        )

    def test_subtract_scatter(self):
        x = linspace(0, 3 * pi, 200)
        other_scatter = Scatter(x, 0.005 * x**2 + 0.1, "Other Scatter")
        scatter_sub = self.testScatter - other_scatter
        self.assertIsInstance(scatter_sub, Scatter)
        self.assertAlmostEqual(scatter_sub.get_coordinates_at_x(0)[1], -0.1, places=3)
        self.assertAlmostEqual(scatter_sub.get_coordinates_at_x(2)[1], 0.789, places=3)

    def test_subtract_scatter_with_int(self):
        scatter_sub = self.testScatter - 2
        self.assertIsInstance(scatter_sub, Scatter)
        self.assertAlmostEqual(scatter_sub.get_coordinates_at_x(0)[1], -2, places=3)
        self.assertAlmostEqual(
            scatter_sub.get_coordinates_at_x(2)[1], -1.09088, places=3
        )
        # Test with int on the left
        scatter_sub = 2 - self.testScatter
        self.assertIsInstance(scatter_sub, Scatter)
        self.assertAlmostEqual(
            scatter_sub.get_coordinates_at_x(0)[1],
            2 - self.testScatter.get_coordinates_at_x(0)[1],
            places=3,
        )
        self.assertAlmostEqual(
            scatter_sub.get_coordinates_at_x(2)[1],
            2 - self.testScatter.get_coordinates_at_x(2)[1],
            places=3,
        )
        # in place
        self.testScatter -= 2
        self.assertAlmostEqual(
            self.testScatter.get_coordinates_at_x(0)[1], -2, places=3
        )
        self.assertAlmostEqual(
            self.testScatter.get_coordinates_at_x(2)[1], -1.09088, places=3
        )

    def test_multiply_scatter(self):
        x = linspace(0, 3 * pi, 200)
        other_scatter = Scatter(x, 0.005 * x**2 + 0.1, "Other Scatter")
        scatter_mult = self.testScatter * other_scatter
        self.assertIsInstance(scatter_mult, Scatter)
        self.assertAlmostEqual(scatter_mult.get_coordinates_at_x(0)[1], 0, places=3)
        self.assertAlmostEqual(scatter_mult.get_coordinates_at_x(2)[1], 0.109, places=3)

    def test_multiply_scatter_with_int(self):
        scatter_mult = self.testScatter * 2
        self.assertIsInstance(scatter_mult, Scatter)
        self.assertAlmostEqual(scatter_mult.get_coordinates_at_x(0)[1], 0, places=3)
        self.assertAlmostEqual(
            scatter_mult.get_coordinates_at_x(2)[1], 1.8182, places=3
        )
        # Test with int on the left
        scatter_mult = 2 * self.testScatter
        self.assertIsInstance(scatter_mult, Scatter)
        self.assertAlmostEqual(scatter_mult.get_coordinates_at_x(0)[1], 0, places=3)
        self.assertAlmostEqual(
            scatter_mult.get_coordinates_at_x(2)[1], 1.8182, places=3
        )
        # in place
        self.testScatter *= 2
        self.assertAlmostEqual(self.testScatter.get_coordinates_at_x(0)[1], 0, places=3)
        self.assertAlmostEqual(
            self.testScatter.get_coordinates_at_x(2)[1], 1.8182, places=3
        )

    def test_divide_scatter(self):
        x = linspace(0, 3 * pi, 200)
        other_scatter = Scatter(x, 0.005 * x**2 + 0.1, "Other Scatter")
        scatter_div = self.testScatter / other_scatter
        self.assertIsInstance(scatter_div, Scatter)
        self.assertAlmostEqual(scatter_div.get_coordinates_at_x(0)[1], 0, places=3)
        self.assertAlmostEqual(
            scatter_div.get_coordinates_at_x(2)[1], 7.57748, places=2
        )

    def test_divide_scatter_with_int(self):
        scatter_div = self.testScatter / 2
        self.assertIsInstance(scatter_div, Scatter)
        self.assertAlmostEqual(scatter_div.get_coordinates_at_x(0)[1], 0, places=3)
        self.assertAlmostEqual(
            scatter_div.get_coordinates_at_x(2)[1], 0.45455, places=3
        )
        # Test with int on the left
        new_scatter = 5 + self.testScatter
        scatter_div = 2 / new_scatter
        self.assertIsInstance(scatter_div, Scatter)
        self.assertAlmostEqual(
            scatter_div.get_coordinates_at_x(0)[1],
            2 / new_scatter.get_coordinates_at_x(0)[1],
            places=3,
        )
        # in place
        self.testScatter /= 2
        self.assertAlmostEqual(self.testScatter.get_coordinates_at_x(0)[1], 0, places=3)
        self.assertAlmostEqual(
            self.testScatter.get_coordinates_at_x(2)[1], 0.45455, places=3
        )

    def test_power_with_int(self):
        scatter_pow = self.testScatter**2
        self.assertIsInstance(scatter_pow, Scatter)
        self.assertAlmostEqual(scatter_pow.get_coordinates_at_x(0)[1], 0, places=3)
        self.assertAlmostEqual(
            scatter_pow.get_coordinates_at_x(2)[1],
            self.testScatter.get_coordinates_at_x(2)[1] ** 2,
            places=3,
        )
        # in place
        reference_val = self.testScatter.get_coordinates_at_x(2)[1]
        self.testScatter **= 2
        self.assertAlmostEqual(self.testScatter.get_coordinates_at_x(0)[1], 0, places=3)
        self.assertAlmostEqual(
            self.testScatter.get_coordinates_at_x(2)[1], reference_val**2, places=3
        )

    def test_max_scatter(self):
        scatter_max = max(self.testScatter)
        self.assertAlmostEqual(scatter_max, 1, places=3)

    def test_min_scatter(self):
        scatter_min = min(self.testScatter)
        self.assertAlmostEqual(scatter_min, -1, places=3)

    def test_add_different_sizes(self):
        x = linspace(0, 3 * pi, 100)
        other_scatter = Scatter(x, 0.005 * x**2 + 0.1, "Other Scatter")
        with self.assertRaises(ValueError):
            new_scatter = self.testScatter + other_scatter

    def test_absolute_value(self):
        scatter = abs(self.testScatter)
        self.assertIsInstance(scatter, Scatter)
        self.assertListEqual(
            list(scatter._y_data), [abs(y) for y in self.testScatter._y_data]
        )

    def test_copy(self):
        scatter_copy = self.testScatter.copy()
        self.assertIsInstance(scatter_copy, Scatter)
        self.assertEqual(scatter_copy._label, self.testScatter._label)
        self.assertEqual(scatter_copy._face_color, self.testScatter._face_color)
        self.assertEqual(scatter_copy._edge_color, self.testScatter._edge_color)
        self.assertEqual(scatter_copy._marker_size, self.testScatter._marker_size)
        self.assertEqual(scatter_copy._marker_style, self.testScatter._marker_style)
        self.assertListEqual(list(scatter_copy._x_data), list(self.testScatter._x_data))
        self.assertListEqual(list(scatter_copy._y_data), list(self.testScatter._y_data))

    def test_create_slice_x(self):
        scatter = Scatter.from_function(lambda x: x**2, -10, 10, number_of_points=100)
        scatter_slice = scatter.create_slice_x(-5, 5)
        self.assertIsInstance(scatter_slice, Scatter)
        correct_x_data = linspace(-10, 10, 100)
        correct_x_data = correct_x_data[correct_x_data >= -5]
        correct_x_data = correct_x_data[correct_x_data <= 5]
        correct_y_data = correct_x_data**2
        self.assertListEqual(list(scatter_slice._x_data), list(correct_x_data))
        self.assertListEqual(list(scatter_slice._y_data), list(correct_y_data))

    def test_create_slice_y(self):
        scatter = Scatter.from_function(lambda x: x**2, -10, 10, number_of_points=100)
        scatter_slice = scatter.create_slice_y(0, 25)
        self.assertIsInstance(scatter_slice, Scatter)
        correct_x_data = linspace(-10, 10, 100)
        correct_x_data = correct_x_data[correct_x_data**2 <= 25]
        correct_y_data = correct_x_data**2
        self.assertListEqual(list(scatter_slice._x_data), list(correct_x_data))
        self.assertListEqual(list(scatter_slice._y_data), list(correct_y_data))


class TestHistogram(unittest.TestCase):
    def setUp(self):
        self.testHist = Histogram(
            [random() for _ in range(100)],
            20,
            label="Random Distribution",
            face_color="silver",
            edge_color="k",
        )

    def test_label_is_str(self):
        self.assertEqual(self.testHist._label[:19], "Random Distribution")

    def test_xdata_is_list_or_ndarray(self):
        self.assertIsInstance(self.testHist._data, list | ndarray)

    def test_face_color_is_str(self):
        self.assertEqual(self.testHist._face_color, "silver")

    def test_edge_color_is_str(self):
        self.assertEqual(self.testHist._edge_color, "k")

    def test_bins_is_int(self):
        self.assertEqual(self.testHist._number_of_bins, 20)

    def test_alpha_is_default(self):
        self.assertEqual(self.testHist._alpha, "default")

    def test_hist_type_is_str(self):
        self.assertEqual(self.testHist._hist_type, "default")

    def test_plot_residuals_from_fit(self):
        curve = Curve.from_function(lambda x: x**2, 0, 1)
        fit = FitFromPolynomial(curve, degree=2)
        histo = self.testHist.from_fit_residuals(fit, 30)
        self.assertIsInstance(histo, Histogram)

    def test_copy(self):
        hist_copy = self.testHist.copy()
        self.assertIsInstance(hist_copy, Histogram)
        self.assertEqual(hist_copy._label, self.testHist._label)
        self.assertEqual(hist_copy._face_color, self.testHist._face_color)
        self.assertEqual(hist_copy._edge_color, self.testHist._edge_color)
        self.assertEqual(hist_copy._number_of_bins, self.testHist._number_of_bins)
        self.assertEqual(hist_copy._alpha, self.testHist._alpha)
        self.assertEqual(hist_copy._hist_type, self.testHist._hist_type)
        self.assertListEqual(list(hist_copy._data), list(self.testHist._data))


if __name__ == "__main__":
    unittest.main()
