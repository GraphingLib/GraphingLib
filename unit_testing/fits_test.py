import unittest

from numpy import exp, linspace, log, pi, sin

from graphinglib.data_plotting_1d import *
from graphinglib.fits import (FitFromExponential, FitFromFunction,
                              FitFromGaussian, FitFromLog, FitFromPolynomial,
                              FitFromSine, FitFromSquareRoot)


class TestFitFromPolynomial(unittest.TestCase):
    def setUp(self):
        x = linspace(-3, 3 * pi, 1000)
        self.scatter_first_degree = Scatter(x, 3 * x + 2, "k", "Test Curve")
        self.scatter_second_degree = Scatter(x, 4 * x**2 - 3 * x - 2, "k", "Test Curve")
        self.fit_first_degree = FitFromPolynomial(
            self.scatter_first_degree, 1, "k", "First degree fit"
        )
        self.fit_second_degree = FitFromPolynomial(
            self.scatter_second_degree, 2, "k", "Second degree fit"
        )

    def test_first_degree_coeffs(self):
        self.assertListEqual(
            [round(i, 5) for i in list(self.fit_first_degree._coeffs)], [2, 3]
        )

    def test_first_degree_cov(self):
        self.assertIsInstance(self.fit_first_degree._cov_matrix, np.ndarray)
        self.assertEqual(self.fit_first_degree._cov_matrix.shape, (2, 2))

    def test_first_degree_std_dev(self):
        self.assertIsInstance(self.fit_first_degree._standard_deviation, np.ndarray)
        self.assertEqual(self.fit_first_degree._standard_deviation.shape, (2,))

    def test_first_degree_string(self):
        self.assertEqual(str(self.fit_first_degree), "$3.0x^1 + 2.0$")

    def test_first_degree_functions(self):
        self.assertAlmostEqual(self.fit_first_degree._function(5), 17)

    def test_second_degree_coeffs(self):
        self.assertListEqual(
            [round(i, 5) for i in list(self.fit_second_degree._coeffs)], [-2, -3, 4]
        )

    def test_second_degree_cov(self):
        self.assertIsInstance(self.fit_second_degree._cov_matrix, np.ndarray)
        self.assertEqual(self.fit_second_degree._cov_matrix.shape, (3, 3))

    def test_second_degree_std_dev(self):
        self.assertIsInstance(self.fit_second_degree._standard_deviation, np.ndarray)
        self.assertEqual(self.fit_second_degree._standard_deviation.shape, (3,))

    def test_second_degree_string(self):
        self.assertEqual(str(self.fit_second_degree), "$4.0x^2 - 3.0x^1 - 2.0$")

    def test_second_degree_functions(self):
        self.assertAlmostEqual(round(self.fit_second_degree._function(5), 5), 83)

    def test_get_coordinates_at_x(self):
        self.assertAlmostEqual(
            self.fit_first_degree.get_coordinates_at_x(5)[1], 17, places=5
        )
        self.assertAlmostEqual(
            self.fit_second_degree.get_coordinates_at_x(5)[1], 83, places=5
        )

    def test_get_coordinates_at_y(self):
        points = self.fit_first_degree.get_coordinates_at_y(17)
        self.assertAlmostEqual(points[0][0], 5, places=3)
        points = self.fit_second_degree.get_coordinates_at_y(83)
        self.assertAlmostEqual(points[0][0], 5, places=3)

    def test_create_point_at_x(self):
        self.assertAlmostEqual(
            self.fit_first_degree.create_point_at_x(5)._y, 17, places=5
        )
        self.assertAlmostEqual(
            self.fit_second_degree.create_point_at_x(5)._y, 83, places=5
        )

    def test_create_points_at_y(self):
        points = self.fit_first_degree.create_points_at_y(17)
        self.assertAlmostEqual(points[0]._x, 5, places=3)
        points = self.fit_second_degree.create_points_at_y(83)
        self.assertAlmostEqual(points[0]._x, 5, places=3)

    def test_get_derivative_curve(self):
        self.assertIsInstance(self.fit_first_degree.create_derivative_curve(), Curve)
        self.assertIsInstance(self.fit_second_degree.create_derivative_curve(), Curve)

    def test_get_integral_curve(self):
        self.assertIsInstance(self.fit_first_degree.create_integral_curve(), Curve)
        self.assertIsInstance(self.fit_second_degree.create_integral_curve(), Curve)

    def test_get_tangent_curve(self):
        self.assertIsInstance(self.fit_first_degree.create_tangent_curve(0), Curve)
        self.assertIsInstance(self.fit_second_degree.create_tangent_curve(0), Curve)

    def test_get_normal_curve(self):
        self.assertIsInstance(self.fit_first_degree.create_normal_curve(0), Curve)
        self.assertIsInstance(self.fit_second_degree.create_normal_curve(0), Curve)

    def test_area_between(self):
        self.assertAlmostEqual(
            self.fit_first_degree.get_area_between(0, 1), 3.5, places=0
        )
        self.assertAlmostEqual(
            self.fit_second_degree.get_area_between(0, 1), -13 / 6, places=1
        )

    def test_slope_at(self):
        self.assertAlmostEqual(self.fit_first_degree.get_slope_at(0), 3, places=5)
        self.assertAlmostEqual(self.fit_second_degree.get_slope_at(0), -3, places=5)

    def test_arc_length_between(self):
        self.assertAlmostEqual(
            self.fit_first_degree.get_arc_length_between(0, 1), np.sqrt(10), places=0
        )
        self.assertAlmostEqual(
            self.fit_second_degree.get_arc_length_between(0, 1), 2.44455, places=0
        )

    def test_copy(self):
        copy = self.fit_first_degree.copy()
        # check that all attributes are the same
        self.assertEqual(list(copy._coeffs), list(self.fit_first_degree._coeffs))
        self.assertEqual(
            copy._cov_matrix.tolist(), self.fit_first_degree._cov_matrix.tolist()
        )
        self.assertEqual(
            copy._standard_deviation.tolist(),
            self.fit_first_degree._standard_deviation.tolist(),
        )
        self.assertEqual(copy._color, self.fit_first_degree._color)
        self.assertEqual(copy._label, self.fit_first_degree._label)


class TestFitFromSine(unittest.TestCase):
    def setUp(self):
        x = linspace(0, 3 * pi, 1000)
        self.data = Scatter(x, 2 * sin(3 * x + 4) + 5, "k", "Data")
        self.fit = FitFromSine(
            self.data, "Sinusoidal fit", guesses=[2.09, 3.1, 4.2, 5.2]
        )

    def test_parameters(self):
        self.assertListEqual(
            [
                self.fit._amplitude,
                self.fit._frequency_rad,
                self.fit._phase_rad,
                self.fit._vertical_shift,
            ],
            [2, 3, 4, 5],
        )

    def test_cov(self):
        self.assertIsInstance(self.fit._cov_matrix, np.ndarray)
        self.assertEqual(self.fit._cov_matrix.shape, (4, 4))

    def test_std_dev(self):
        self.assertIsInstance(self.fit._standard_deviation, np.ndarray)
        self.assertEqual(self.fit._standard_deviation.shape, (4,))

    def test_string(self):
        self.assertEqual(str(self.fit), "$2.000 \\sin(3.000x + 4.000) + 5.000$")

    def test_function(self):
        self.assertAlmostEqual(self.fit._function(17), 3.00048965)

    def test_get_coordinates_at_x(self):
        self.assertAlmostEqual(self.fit.get_coordinates_at_x(17)[1], 3.00048965)

    def test_get_coordinates_at_y(self):
        points = self.fit.get_coordinates_at_y(5)
        self.assertAlmostEqual(points[0][0], 0.7610626313, places=3)
        self.assertAlmostEqual(points[1][0], 1.8082583879180, places=3)
        self.assertAlmostEqual(points[2][0], 2.8554578303, places=3)
        self.assertAlmostEqual(points[3][0], 3.902653817, places=3)

    def test_create_point_at_x(self):
        self.assertAlmostEqual(self.fit.create_point_at_x(17)._y, 3.00048965)

    def test_create_points_at_y(self):
        points = self.fit.create_points_at_y(5)
        self.assertAlmostEqual(points[0]._x, 0.7610626313, places=3)
        self.assertAlmostEqual(points[1]._x, 1.8082583879180, places=3)
        self.assertAlmostEqual(points[2]._x, 2.8554578303, places=3)
        self.assertAlmostEqual(points[3]._x, 3.902653817, places=3)

    def test_get_derivative_curve(self):
        self.assertIsInstance(self.fit.create_derivative_curve(), Curve)

    def test_get_integral_curve(self):
        self.assertIsInstance(self.fit.create_integral_curve(), Curve)

    def test_get_tangent_curve(self):
        self.assertIsInstance(self.fit.create_tangent_curve(0), Curve)

    def test_get_normal_curve(self):
        self.assertIsInstance(self.fit.create_normal_curve(0), Curve)

    def test_area_between(self):
        self.assertAlmostEqual(
            self.fit.get_area_between(0, np.pi / 2), 7.9228, places=1
        )

    def test_slope_at(self):
        self.assertAlmostEqual(self.fit.get_slope_at(np.pi), -6 * np.cos(4), places=2)

    def test_arc_length_between(self):
        self.assertAlmostEqual(
            self.fit.get_arc_length_between(0, np.pi / 2), 5.538, places=1
        )

    def test_copy(self):
        copy = self.fit.copy()
        # check that all attributes are the same
        self.assertEqual(copy._amplitude, self.fit._amplitude)
        self.assertEqual(copy._frequency_rad, self.fit._frequency_rad)
        self.assertEqual(copy._phase_rad, self.fit._phase_rad)
        self.assertEqual(copy._vertical_shift, self.fit._vertical_shift)
        self.assertEqual(copy._cov_matrix.tolist(), self.fit._cov_matrix.tolist())
        self.assertEqual(
            copy._standard_deviation.tolist(),
            self.fit._standard_deviation.tolist(),
        )
        self.assertEqual(copy._color, self.fit._color)
        self.assertEqual(copy._label, self.fit._label)


class TestFitFromExponential(unittest.TestCase):
    def setUp(self):
        x = linspace(0, 10, 1000)
        self.data = Scatter(x, 2 * exp(3 * x + 4), "Data")
        self.fit = FitFromExponential(
            self.data, "Sinusoidal fit", guesses=[2.09, 3.1, 4.01]
        )

    def test_parameters(self):
        self.assertListEqual([round(i) for i in list(self.fit._parameters)[1:]], [3, 4])

    def test_cov(self):
        self.assertIsInstance(self.fit.cov_matrix, np.ndarray)
        self.assertEqual(self.fit.cov_matrix.shape, (3, 3))

    def test_std_dev(self):
        self.assertIsInstance(self.fit.standard_deviation, np.ndarray)
        self.assertEqual(self.fit.standard_deviation.shape, (3,))

    def test_string(self):
        self.assertEqual(str(self.fit)[6:-7], " \exp(3.000x + ")

    def test_function(self):
        self.assertAlmostEqual(self.fit._function(0.001), 109.524, places=3)

    def test_get_coordinates_at_x(self):
        self.assertAlmostEqual(
            self.fit.get_coordinates_at_x(0.001)[1], 109.524, places=3
        )

    def test_get_coordinates_at_y(self):
        points = self.fit.get_coordinates_at_y(109.524)
        self.assertAlmostEqual(points[0][0], 0.001, places=3)

    def test_create_point_at_x(self):
        self.assertAlmostEqual(self.fit.create_point_at_x(0.001)._y, 109.524, places=3)

    def test_create_points_at_y(self):
        points = self.fit.create_points_at_y(109.524)
        self.assertAlmostEqual(points[0]._x, 0.001, places=3)

    def test_get_derivative_curve(self):
        self.assertIsInstance(self.fit.create_derivative_curve(), Curve)

    def test_get_integral_curve(self):
        self.assertIsInstance(self.fit.create_integral_curve(), Curve)

    def test_get_tangent_curve(self):
        self.assertIsInstance(self.fit.create_tangent_curve(0), Curve)

    def test_get_normal_curve(self):
        self.assertIsInstance(self.fit.create_normal_curve(0), Curve)

    def test_area_between(self):
        curve1 = Curve.from_function(
            lambda x: 2 * exp(3 * x + 4), -0.1, 1.1, number_of_points=10000
        )
        fit = FitFromExponential(curve1, "Exponential fit")
        self.assertAlmostEqual(fit.get_area_between(0, 1), 694.69, places=0)

    def test_slope_at(self):
        self.assertAlmostEqual(
            self.fit.get_slope_at(0.5),
            1468.15,
            places=-1,
        )

    def test_arc_length_between(self):
        self.assertAlmostEqual(
            self.fit.get_arc_length_between(0, 1), 2084.07, places=-3
        )

    def test_copy(self):
        copy = self.fit.copy()
        # check that all attributes are the same
        self.assertEqual(list(copy._parameters), list(self.fit._parameters))
        self.assertEqual(copy.cov_matrix.tolist(), self.fit.cov_matrix.tolist())
        self.assertEqual(
            copy.standard_deviation.tolist(),
            self.fit.standard_deviation.tolist(),
        )
        self.assertEqual(copy._color, self.fit._color)
        self.assertEqual(copy._label, self.fit._label)


class TestFitFromGaussian(unittest.TestCase):
    def setUp(self) -> None:
        x = linspace(-4, 6, 1000)
        self.data = Scatter(x, 5 * np.exp(-(((x - 1) / 1) ** 2) / 2), "Data")
        self.fit = FitFromGaussian(self.data, "Gaussian fit")

    def test_cov(self):
        self.assertIsInstance(self.fit._cov_matrix, np.ndarray)
        self.assertEqual(self.fit._cov_matrix.shape, (3, 3))

    def test_parameters(self):
        params = [self.fit._amplitude, self.fit._mean, self.fit._standard_deviation]
        rounded_params = [round(i, 2) for i in params]
        self.assertListEqual(rounded_params, [5, 1, 1])

    def test_std_dev(self):
        self.assertIsInstance(self.fit._standard_deviation_of_fit_params, np.ndarray)
        self.assertEqual(self.fit._standard_deviation_of_fit_params.shape, (3,))

    def test_str(self):
        self.assertEqual(str(self.fit), "$\mu = 1.000, \sigma = 1.000, A = 5.000$")

    def test_function(self):
        self.assertAlmostEqual(self.fit._function(3), 0.676515, places=3)

    def test_get_coordinates_at_x(self):
        self.assertAlmostEqual(self.fit.get_coordinates_at_x(3)[1], 0.676515, places=3)

    def test_get_coordinates_at_y(self):
        points = self.fit.get_coordinates_at_y(0.676515)
        self.assertAlmostEqual(points[0][0], -1, places=3)
        self.assertAlmostEqual(points[1][0], 3, places=3)

    def test_create_point_at_x(self):
        self.assertAlmostEqual(self.fit.create_point_at_x(3)._y, 0.676515, places=3)

    def test_create_points_at_y(self):
        points = self.fit.create_points_at_y(0.676515)
        self.assertAlmostEqual(points[0]._x, -1, places=3)
        self.assertAlmostEqual(points[1]._x, 3, places=3)

    def test_get_derivative_curve(self):
        self.assertIsInstance(self.fit.create_derivative_curve(), Curve)

    def test_get_integral_curve(self):
        self.assertIsInstance(self.fit.create_integral_curve(), Curve)

    def test_get_tangent_curve(self):
        self.assertIsInstance(self.fit.create_tangent_curve(0), Curve)

    def test_get_normal_curve(self):
        self.assertIsInstance(self.fit.create_normal_curve(0), Curve)

    def test_area_between(self):
        self.assertAlmostEqual(self.fit.get_area_between(0, 1), 4.27812, places=1)

    def test_slope_at(self):
        self.assertAlmostEqual(
            self.fit.get_slope_at(0),
            5 / np.sqrt(np.e),
            places=2,
        )

    def test_arc_length_between(self):
        self.assertAlmostEqual(self.fit.get_arc_length_between(0, 1), 2.27338, places=1)

    def test_copy(self):
        copy = self.fit.copy()
        # check that all attributes are the same
        self.assertEqual(copy._amplitude, self.fit._amplitude)
        self.assertEqual(copy._mean, self.fit._mean)
        self.assertEqual(copy._standard_deviation, self.fit._standard_deviation)
        self.assertEqual(copy._cov_matrix.tolist(), self.fit._cov_matrix.tolist())
        self.assertEqual(
            copy._standard_deviation_of_fit_params.tolist(),
            self.fit._standard_deviation_of_fit_params.tolist(),
        )
        self.assertEqual(copy._color, self.fit._color)
        self.assertEqual(copy._label, self.fit._label)


class TestFitFromSquareRoot(unittest.TestCase):
    def setUp(self) -> None:
        x = linspace(-1, 6, 1000)
        self.data = Scatter(x, 3 * np.sqrt(x + 4) + 5, "Data")
        self.fit = FitFromSquareRoot(self.data, "Square root fit")

    def test_parameters(self):
        rounded_params = [round(i, 3) for i in list(self.fit._parameters)]
        self.assertListEqual(rounded_params, [3, 4, 5])

    def test_cov(self):
        self.assertIsInstance(self.fit._cov_matrix, np.ndarray)
        self.assertEqual(self.fit._cov_matrix.shape, (3, 3))

    def test_std_dev(self):
        self.assertIsInstance(self.fit._standard_deviation, np.ndarray)
        self.assertEqual(self.fit._standard_deviation.shape, (3,))

    def test_str(self):
        self.assertEqual(str(self.fit), "$3.000 \\sqrt{x + 4.000} + 5.000$")

    def test_function(self):
        self.assertAlmostEqual(self.fit._function(3), 12.937, places=3)

    def test_get_coordinates_at_x(self):
        self.assertAlmostEqual(self.fit.get_coordinates_at_x(3)[1], 12.937, places=3)

    def test_get_coordinates_at_y(self):
        points = self.fit.get_coordinates_at_y(12.937)
        self.assertAlmostEqual(points[0][0], 3, places=3)

    def test_create_point_at_x(self):
        self.assertAlmostEqual(self.fit.create_point_at_x(3)._y, 12.937, places=3)

    def test_create_points_at_y(self):
        points = self.fit.create_points_at_y(12.937)
        self.assertAlmostEqual(points[0]._x, 3, places=3)

    def test_get_derivative_curve(self):
        self.assertIsInstance(self.fit.create_derivative_curve(), Curve)

    def test_get_integral_curve(self):
        self.assertIsInstance(self.fit.create_integral_curve(), Curve)

    def test_get_tangent_curve(self):
        self.assertIsInstance(self.fit.create_tangent_curve(0), Curve)

    def test_get_normal_curve(self):
        self.assertIsInstance(self.fit.create_normal_curve(0), Curve)

    def test_area_between(self):
        self.assertAlmostEqual(self.fit.get_area_between(0, 1), 11.361, places=0)

    def test_slope_at(self):
        self.assertAlmostEqual(
            self.fit.get_slope_at(0),
            3 / 4,
            places=4,
        )

    def test_arc_length_between(self):
        self.assertAlmostEqual(self.fit.get_arc_length_between(0, 1), 1.2255, places=1)

    def test_copy(self):
        copy = self.fit.copy()
        # check that all attributes are the same
        self.assertEqual(list(copy._parameters), list(self.fit._parameters))
        self.assertEqual(copy._cov_matrix.tolist(), self.fit._cov_matrix.tolist())
        self.assertEqual(
            copy._standard_deviation.tolist(), self.fit._standard_deviation.tolist()
        )
        self.assertEqual(copy._color, self.fit._color)
        self.assertEqual(copy._label, self.fit._label)


class TestFitFromLog(unittest.TestCase):
    def setUp(self):
        x = linspace(0, 10, 1000)
        self.data = Scatter(x, 2 * log(x + 3) + 4, "Data")
        self.fit = FitFromLog(self.data, "Logarithmic fit")

    def test_parameters(self):
        rounded_params = [round(i) for i in list(self.fit._parameters)]
        self.assertListEqual(rounded_params, [2, 3, 4])

    def test_cov(self):
        self.assertIsInstance(self.fit._cov_matrix, np.ndarray)
        self.assertEqual(self.fit._cov_matrix.shape, (3, 3))

    def test_std_dev(self):
        self.assertIsInstance(self.fit._standard_deviation, np.ndarray)
        self.assertEqual(self.fit._standard_deviation.shape, (3,))

    def test_string(self):
        self.assertEqual(str(self.fit), "$2.000 log_e(x + 3.000) + 4.000$")

    def test_function(self):
        self.assertAlmostEqual(self.fit._function(0.001), 6.19789, places=3)

    def test_get_coordinates_at_x(self):
        self.assertAlmostEqual(
            self.fit.get_coordinates_at_x(0.001)[1], 6.19789, places=3
        )

    def test_get_coordinates_at_y(self):
        points = self.fit.get_coordinates_at_y(6.19789)
        self.assertAlmostEqual(points[0][0], 0.001, places=3)

    def test_create_point_at_x(self):
        self.assertAlmostEqual(self.fit.create_point_at_x(0.001)._y, 6.19789, places=3)

    def test_create_points_at_y(self):
        points = self.fit.create_points_at_y(6.19789)
        self.assertAlmostEqual(points[0]._x, 0.001, places=3)

    def test_get_derivative_curve(self):
        self.assertIsInstance(self.fit.create_derivative_curve(), Curve)

    def test_get_integral_curve(self):
        self.assertIsInstance(self.fit.create_integral_curve(), Curve)

    def test_get_tangent_curve(self):
        self.assertIsInstance(self.fit.create_tangent_curve(0), Curve)

    def test_get_normal_curve(self):
        self.assertIsInstance(self.fit.create_normal_curve(0), Curve)

    def test_area_between(self):
        self.assertAlmostEqual(self.fit.get_area_between(1, 2), 7.004, places=0)

    def test_slope_at(self):
        self.assertAlmostEqual(
            self.fit.get_slope_at(1),
            0.5,
            places=4,
        )

    def test_arc_length_between(self):
        self.assertAlmostEqual(self.fit.get_arc_length_between(1, 2), 1.0954, places=1)

    def test_copy(self):
        copy = self.fit.copy()
        # check that all attributes are the same
        self.assertEqual(list(copy._parameters), list(self.fit._parameters))
        self.assertEqual(copy._cov_matrix.tolist(), self.fit._cov_matrix.tolist())
        self.assertEqual(
            copy._standard_deviation.tolist(),
            self.fit._standard_deviation.tolist(),
        )
        self.assertEqual(copy._color, self.fit._color)
        self.assertEqual(copy._label, self.fit._label)


class TestFitFromFunction(unittest.TestCase):
    def setUp(self):
        x = linspace(1, 10, 1000)
        self.data = Scatter(x, 1 / x + 3 * x, "Data")
        self.fit = FitFromFunction(lambda x, a, b: a / x + b * x, self.data)

    def test_parameters(self):
        rounded_params = [round(i) for i in list(self.fit._parameters)]
        self.assertListEqual(rounded_params, [1, 3])

    def test_cov(self):
        self.assertIsInstance(self.fit._cov_matrix, np.ndarray)
        self.assertEqual(self.fit._cov_matrix.shape, (2, 2))

    def test_std_dev(self):
        self.assertIsInstance(self.fit._standard_deviation, np.ndarray)
        self.assertEqual(self.fit._standard_deviation.shape, (2,))

    def test_function(self):
        self.assertAlmostEqual(self.fit._function(1), 4, places=3)

    def test_get_coordinates_at_x(self):
        self.assertAlmostEqual(self.fit.get_coordinates_at_x(1)[1], 4, places=3)

    def test_get_coordinates_at_y(self):
        points = self.fit.get_coordinates_at_y(4)
        self.assertAlmostEqual(points[0][0], 1, places=3)

    def test_get_derivative_curve(self):
        self.assertIsInstance(self.fit.create_derivative_curve(), Curve)

    def test_get_integral_curve(self):
        self.assertIsInstance(self.fit.create_integral_curve(), Curve)

    def test_get_tangent_curve(self):
        self.assertIsInstance(self.fit.create_tangent_curve(2), Curve)

    def test_get_normal_curve(self):
        self.assertIsInstance(self.fit.create_normal_curve(2), Curve)

    def test_area_between(self):
        self.assertAlmostEqual(self.fit.get_area_between(1, 2), 5.1931, places=3)

    def test_slope_at(self):
        self.assertAlmostEqual(
            self.fit.get_slope_at(1),
            2,
            places=1,
        )

    def test_arc_length_between(self):
        self.assertAlmostEqual(self.fit.get_arc_length_between(1, 2), 2.6937, places=3)

    def test_copy(self):
        copy = self.fit.copy()
        # check that all attributes are the same
        self.assertEqual(list(copy._parameters), list(self.fit._parameters))
        self.assertEqual(copy._cov_matrix.tolist(), self.fit._cov_matrix.tolist())
        self.assertEqual(
            copy._standard_deviation.tolist(),
            self.fit._standard_deviation.tolist(),
        )
        self.assertEqual(copy._color, self.fit._color)
        self.assertEqual(copy._label, self.fit._label)


if __name__ == "__main__":
    unittest.main()
