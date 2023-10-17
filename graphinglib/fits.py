from __future__ import annotations

from functools import partial
from typing import Callable, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from .data_plotting_1d import Curve, Scatter
from .graph_elements import Point


class GeneralFit(Curve):
    """
    Dummy class for curve fits. Defines the interface for all curve fits.

    .. attention:: Not to be used directly.

    Parameters
    ----------
    curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
        The object to be fit.
    label : str, optional
        Label to be displayed in the legend.
    color : str
        Color of the curve.
        Default depends on the ``figure_style`` configuration.
    line_width : int
        Line width of the curve.
        Default depends on the ``figure_style`` configuration.
    line_style : str
        Line style of the curve.
        Default depends on the ``figure_style`` configuration.
    """

    def __init__(
        self,
        curve_to_be_fit: Curve | Scatter,
        label: Optional[str] = None,
        color: str = "default",
        line_width: int | Literal["default"] = "default",
        line_style: str = "default",
    ) -> None:
        """
        Parameters
        ----------
        curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
            The object to be fit.
        label : str, optional
            Label to be displayed in the legend.
        color : str
            Color of the curve.
            Default depends on the ``figure_style`` configuration.
        line_width : int
            Line width of the curve.
            Default depends on the ``figure_style`` configuration.
        line_style : str
            Line style of the curve.
            Default depends on the ``figure_style`` configuration.
        """
        self.curve_to_be_fit = curve_to_be_fit
        self.color = color
        self.line_width = line_width
        if label:
            self.label = label + " : " + "$f(x) = $" + str(self)
        else:
            self.label = "$f(x) = $" + str(self)
        self.line_style = line_style
        self._res_curves_to_be_plotted = False
        self.function: Callable[[np.ndarray], np.ndarray]

    def __str__(self) -> str:
        """
        Create a string representation of the fit function.
        """
        raise NotImplementedError()

    def get_point_at_x(
        self,
        x: float,
        label: str | None = None,
        color: str = "default",
        edge_color: str = "default",
        marker_size: float | Literal["default"] = "default",
        marker_style: str = "default",
        line_width: float | Literal["default"] = "default",
    ) -> Point:
        """
        Gets the point on the curve at a given x value.

        Parameters
        ----------
        x : float
            x value of the point.
        label : str, optional
            Label to be displayed in the legend.
        color : str
            Face color of the point.
            Default depends on the ``figure_style`` configuration.
        edge_color : str
            Edge color of the point.
            Default depends on the ``figure_style`` configuration.
        marker_size : float
            Size of the point.
            Default depends on the ``figure_style`` configuration.
        marker_style : str
            Style of the point.
            Default depends on the ``figure_style`` configuration.
        line_width : float
            Width of the edge of the point.
            Default depends on the ``figure_style`` configuration.

        Returns
        -------
        :class:`~graphinglib.graph_elements.Point` object on the curve at the given x value.
        """
        return Point(
            x,
            self.function(x),
            label=label,
            color=color,
            edge_color=edge_color,
            marker_size=marker_size,
            marker_style=marker_style,
            edge_width=line_width,
        )

    def get_points_at_y(
        self,
        y: float,
        interpolation_kind: str = "linear",
        label: str | None = None,
        color: str = "default",
        edge_color: str = "default",
        marker_size: float | Literal["default"] = "default",
        marker_style: str = "default",
        line_width: float | Literal["default"] = "default",
    ) -> list[Point]:
        """
        Gets the points on the curve at a given y value.

        Parameters
        ----------
        y : float
            y value of the point.
        interpolation_kind : str
            Kind of interpolation to be used.
            Default is "linear".
        label : str, optional
            Label to be displayed in the legend.
        color : str
            Face color of the point.
            Default depends on the ``figure_style`` configuration.
        edge_color : str
            Edge color of the point.
            Default depends on the ``figure_style`` configuration.
        marker_size : float
            Size of the point.
            Default depends on the ``figure_style`` configuration.
        marker_style : str
            Style of the point.
            Default depends on the ``figure_style`` configuration.
        line_width : float
            Width of the edge of the point.
            Default depends on the ``figure_style`` configuration.

        Returns
        -------
        points : list[:class:`~graphinglib.graph_elements.Point`]
            List of :class:`~graphinglib.graph_elements.Point` objects on the curve at the given y value.
        """
        xs = self.curve_to_be_fit.x_data
        ys = self.function(xs)
        crossings = np.where(np.diff(np.sign(ys - y)))[0]
        x_vals: list[float] = []
        for cross in crossings:
            x1, x2 = xs[cross], xs[cross + 1]
            y1, y2 = ys[cross], ys[cross + 1]
            f = interp1d([y1, y2], [x1, x2], kind=interpolation_kind)
            x_val = f(y)
            x_vals.append(float(x_val))
        points = [
            Point(
                x_val,
                y,
                label=label,
                color=color,
                edge_color=edge_color,
                marker_size=marker_size,
                marker_style=marker_style,
                edge_width=line_width,
            )
            for x_val in x_vals
        ]
        return points

    def _plot_element(self, axes: plt.Axes, z_order: int) -> None:
        """
        Plots the element in the specified
        Axes
        """
        (self.handle,) = axes.plot(
            self.x_data,
            self.y_data,
            label=self.label,
            color=self.color,
            linewidth=self.line_width,
            linestyle=self.line_style,
            zorder=z_order,
        )
        if self._res_curves_to_be_plotted:
            x_data = self.x_data
            y_fit = self.y_data
            residuals = self.calculate_residuals()
            std = np.std(residuals)
            y_fit_plus_std = y_fit + (self.res_sigma_multiplier * std)
            y_fit_minus_std = y_fit - (self.res_sigma_multiplier * std)
            axes.plot(
                self.x_data,
                y_fit_minus_std,
                color=self.res_color,
                linewidth=self.res_line_width,
                linestyle=self.res_line_style,
                zorder=z_order,
            )
            axes.plot(
                self.x_data,
                y_fit_plus_std,
                color=self.res_color,
                linewidth=self.res_line_width,
                linestyle=self.res_line_style,
                zorder=z_order,
            )
        if self._fill_curve_between:
            axes.fill_between(
                self.x_data,
                self.y_data,
                where=np.logical_and(
                    self.x_data >= self._fill_curve_between[0],
                    self.x_data <= self._fill_curve_between[1],
                ),
                # color=self.fill_color,
                alpha=0.2,
                zorder=z_order - 2,
            )

    def show_residual_curves(
        self,
        sigma_multiplier: float = 1,
        color: str = "default",
        line_width: float | Literal["default"] = "default",
        line_style: str = "default",
    ) -> None:
        """
        Displays two curves ``"sigma_multiplier"`` standard deviations above and below the fit curve.

        Parameters
        ----------
        sigma_multiplier : float
            Distance in standard deviations from the fit curve.
            Default is 1.
        color : str
            Color of the residual curves.
            Default depends on the ``figure_style`` configuration.
        line_width : float
            Line width of the residual curves.
            Default depends on the ``figure_style`` configuration.
        """
        self._res_curves_to_be_plotted = True
        self.res_sigma_multiplier = sigma_multiplier
        self.res_color = color
        self.res_line_width = line_width
        self.res_line_style = line_style

    def calculate_residuals(self) -> np.ndarray:
        """
        Calculates the residuals of the fit curve.

        Returns
        -------
        residuals : np.ndarray
            Array of residuals.
        """
        y_data = self.function(self.curve_to_be_fit.x_data)
        residuals = y_data - self.curve_to_be_fit.y_data
        return residuals


class FitFromPolynomial(GeneralFit):
    """
    Creates a curve fit (continuous :class:`~graphinglib.data_plotting_1d.Curve`) from an existing curve object using a polynomial fit.

    Fits a polynomial of the form :math:`f(x) = a_0 + a_1 x + a_2 x^2 + ... + a_n x^n` to the given curve. All standard Curve attributes
    and methods are available.

    Parameters
    ----------
    curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
        The object to be fit.
    degree : int
        Degree of the polynomial fit.
    label : str, optional
        Label to be displayed in the legend.
    color : str
        Color of the :class:`~graphinglib.data_plotting_1d.Curve`.
        Default depends on the ``figure_style`` configuration.
    line_width : int
        Line width of the :class:`~graphinglib.data_plotting_1d.Curve`.
        Default depends on the ``figure_style`` configuration.
    line_style : str
        Line style of the :class:`~graphinglib.data_plotting_1d.Curve`.
        Default depends on the ``figure_style`` configuration.

    Attributes
    ----------
    coeffs : np.ndarray
        Coefficients of the polynomial fit. The first element is the coefficient of the lowest order term (constant term).
    cov_matrix : np.ndarray
        Covariance matrix of the polynomial fit (using the same order as the coeffs attribute).
    standard_deviation : np.ndarray
        Standard deviation of the coefficients of the polynomial fit (same order as coeffs).
    function : Callable
        Polynomial function with the parameters of the fit.
    """

    def __init__(
        self,
        curve_to_be_fit: Curve | Scatter,
        degree: int,
        label: Optional[str] = None,
        color: str = "default",
        line_width: int | Literal["default"] = "default",
        line_style: int | Literal["default"] = "default",
    ) -> None:
        """
        Creates a curve fit (continuous :class:`~graphinglib.data_plotting_1d.Curve`) from an existing curve object using a polynomial fit.

        Fits a polynomial of the form :math:`f(x) = a_0 + a_1 x + a_2 x^2 + ... + a_n x^n` to the given curve. All standard Curve attributes
        and methods are available.

        Parameters
        ----------
        curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
            The object to be fit.
        degree : int
            Degree of the polynomial fit.
        label : str, optional
            Label to be displayed in the legend.
        color : str
            Color of the :class:`~graphinglib.data_plotting_1d.Curve`.
            Default depends on the ``figure_style`` configuration.
        line_width : int
            Line width of the :class:`~graphinglib.data_plotting_1d.Curve`.
            Default depends on the ``figure_style`` configuration.
        line_style : str
            Line style of the :class:`~graphinglib.data_plotting_1d.Curve`.
            Default depends on the ``figure_style`` configuration.

        Attributes
        ----------
        coeffs : np.ndarray
            Coefficients of the polynomial fit. The first element is the coefficient of the lowest order term (constant term).
        cov_matrix : np.ndarray
            Covariance matrix of the polynomial fit (using the same order as the coeffs attribute).
        standard_deviation : np.ndarray
            Standard deviation of the coefficients of the polynomial fit (same order as coeffs).
        function : Callable
            Polynomial function with the parameters of the fit.s of the fit.
        """
        self.curve_to_be_fit = curve_to_be_fit
        inversed_coeffs, inversed_cov_matrix = np.polyfit(
            self.curve_to_be_fit.x_data, self.curve_to_be_fit.y_data, degree, cov=True
        )
        self.coeffs = inversed_coeffs[::-1]
        self.cov_matrix = np.flip(inversed_cov_matrix)
        self.standard_deviation = np.sqrt(np.diag(self.cov_matrix))
        self.function = self._polynomial_func_with_params()
        self.color = color
        self.line_width = line_width
        if label:
            self.label = label + " : " + "$f(x) = $" + str(self)
        else:
            self.label = "$f(x) = $" + str(self)
        self.line_style = line_style
        self._res_curves_to_be_plotted = False
        number_of_points = 500
        self.x_data = np.linspace(
            self.curve_to_be_fit.x_data[0],
            self.curve_to_be_fit.x_data[-1],
            number_of_points,
        )
        self.y_data = self.function(self.x_data)
        self._fill_curve_between = False

    def __str__(self) -> str:
        """
        Creates a string representation of the polynomial function.
        """
        coeff_chunks = []
        power_chunks = []
        ordered_rounded_coeffs = [round(coeff, 3) for coeff in self.coeffs[::-1]]
        for coeff, power in zip(
            ordered_rounded_coeffs, range(len(ordered_rounded_coeffs) - 1, -1, -1)
        ):
            if coeff == 0:
                continue
            coeff_chunks.append(self._format_coeff(coeff))
            power_chunks.append(self._format_power(power))
        coeff_chunks[0] = coeff_chunks[0].lstrip("+ ")
        return (
            "$"
            + "".join(
                [coeff_chunks[i] + power_chunks[i] for i in range(len(coeff_chunks))]
            )
            + "$"
        )

    @staticmethod
    def _format_coeff(coeff: float) -> str:
        """
        Formats a coefficient to be displayed in the string representation of the polynomial function.
        """
        return " - {0}".format(abs(coeff)) if coeff < 0 else " + {0}".format(coeff)

    @staticmethod
    def _format_power(power: int) -> str:
        """
        Formats a power to be displayed in the string representation of the polynomial function.
        """
        return "x^{0}".format(power) if power != 0 else ""

    def _polynomial_func_with_params(
        self,
    ) -> Callable[[float | np.ndarray], float | np.ndarray]:
        """
        Creates a polynomial function with the parameters of the fit.

        Returns
        -------
        function : Callable
            Polynomial function with the parameters of the fit.
        """
        return lambda x: sum(
            coeff * x**exponent for exponent, coeff in enumerate(self.coeffs)
        )


class FitFromSine(GeneralFit):
    """
    Create a curve fit (continuous :class:`~graphinglib.data_plotting_1d.Curve`) from an existing
    :class:`~graphinglib.data_plotting_1d.Curve` object using a sinusoidal fit.

    Fits a sine function of the form :math:`f(x) = a sin(bx + c) + d` to the given curve. All standard
    :class:`~graphinglib.data_plotting_1d.Curve` attributes and methods are available.

    Parameters
    ----------
    curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
        The object to be fit.
    label : str, optional
        Label to be displayed in the legend.
    guesses : ArrayLike, optional
        Initial guesses for the parameters of the fit (order: amplitude (a), frequency (b), phase (c), vertical shift (d) as written above).
    color : str
        Color of the curve.
        Default depends on the ``figure_style`` configuration.
    line_width : int
        Line width of the curve.
        Default depends on the ``figure_style`` configuration.
    line_style : str
        Line style of the curve.
        Default depends on the ``figure_style`` configuration.

    Attributes
    ----------
    amplitude : float
        Amplitude of the sine function.
    frequency_rad : float
        Frequency of the sine function in radians.
    phase : float
        Phase of the sine function.
    vertical_shift : float
        Vertical shift of the sine function.
    cov_matrix : np.ndarray
        Covariance matrix of the parameters of the fit.
    standard_deviation : np.ndarray
        Standard deviation of the parameters of the fit.
    function : Callable
        Sine function with the parameters of the fit.
    """

    def __init__(
        self,
        curve_to_be_fit: Curve | Scatter,
        label: Optional[str] = None,
        guesses: Optional[ArrayLike] = None,
        color: str = "default",
        line_width: str = "default",
        line_style: str = "default",
    ) -> None:
        """
        Create a curve fit (continuous :class:`~graphinglib.data_plotting_1d.Curve`) from an existing
        :class:`~graphinglib.data_plotting_1d.Curve` object using a sinusoidal fit.

        Fits a sine function of the form :math:`f(x) = a sin(bx + c) + d` to the given curve. All standard
        :class:`~graphinglib.data_plotting_1d.Curve` attributes and methods are available.

        Parameters
        ----------
        curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
            The object to be fit.
        label : str, optional
            Label to be displayed in the legend.
        guesses : ArrayLike, optional
            Initial guesses for the parameters of the fit (order: amplitude (a), frequency (b), phase (c), vertical shift (d) as written above).
        color : str
            Color of the curve.
            Default depends on the ``figure_style`` configuration.
        line_width : int
            Line width of the curve.
            Default depends on the ``figure_style`` configuration.
        line_style : str
            Line style of the curve.
            Default depends on the ``figure_style`` configuration.

        Attributes
        ----------
        amplitude : float
            Amplitude of the sine function.
        frequency_rad : float
            Frequency of the sine function in radians.
        phase : float
            Phase of the sine function.
        vertical_shift : float
            Vertical shift of the sine function.
        cov_matrix : np.ndarray
            Covariance matrix of the parameters of the fit.
        standard_deviation : np.ndarray
            Standard deviation of the parameters of the fit.
        function : Callable
            Sine function with the parameters of the fit.
        """
        self.curve_to_be_fit = curve_to_be_fit
        self.guesses = guesses
        self._calculate_parameters()
        self.function = self._sine_func_with_params()
        self.color = color
        if label:
            self.label = label + " : " + "$f(x) = $" + str(self)
        else:
            self.label = "$f(x) = $" + str(self)
        self.line_width = line_width
        self.line_style = line_style
        self._res_curves_to_be_plotted = False
        number_of_points = 500
        self.x_data = np.linspace(
            self.curve_to_be_fit.x_data[0],
            self.curve_to_be_fit.x_data[-1],
            number_of_points,
        )
        self.y_data = self.function(self.x_data)
        self._fill_curve_between = False

    def __str__(self) -> str:
        """
        Creates a string representation of the sine function.
        """
        part1 = f"{self.amplitude:.3f} \sin({self.frequency_rad:.3f}x"
        part2 = (
            f" + {self.phase:.3f})" if self.phase >= 0 else f" - {abs(self.phase):.3f})"
        )
        part3 = (
            f" + {self.vertical_shift:.3f}"
            if self.vertical_shift >= 0
            else f" - {abs(self.vertical_shift):.3f}"
        )
        return f"${part1 + part2 + part3}$"

    def _calculate_parameters(self) -> None:
        """
        Calculates the parameters of the fit.
        """
        parameters, self.cov_matrix = curve_fit(
            self._sine_func_template,
            self.curve_to_be_fit.x_data,
            self.curve_to_be_fit.y_data,
            p0=self.guesses,
        )
        self.amplitude, self.frequency_rad, self.phase, self.vertical_shift = parameters
        self.standard_deviation = np.sqrt(np.diag(self.cov_matrix))

    @staticmethod
    def _sine_func_template(
        x: np.ndarray, a: float, b: float, c: float, d: float
    ) -> np.ndarray:
        """
        Function to be passed to the ``curve_fit`` function.
        """
        return a * np.sin(b * x + c) + d

    def _sine_func_with_params(
        self,
    ) -> Callable[[float | np.ndarray], float | np.ndarray]:
        """
        Creates a sine function with the parameters of the fit.

        Returns
        -------
        function : Callable
            Sine function with the parameters of the fit.
        """
        return (
            lambda x: self.amplitude * np.sin(self.frequency_rad * x + self.phase)
            + self.vertical_shift
        )


class FitFromExponential(GeneralFit):
    """
    Create a curve fit (continuous :class:`~graphinglib.data_plotting_1d.Curve`) from an existing
    :class:`~graphinglib.data_plotting_1d.Curve` object using an exponential fit.

    Parameters
    ----------
    curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
        The object to be fit.
    label : str, optional
        Label to be displayed in the legend.
    guesses : ArrayLike, optional
        Initial guesses for the parameters of the fit. Order is a, b, c as written above.
    color : str
        Color of the curve.
        Default depends on the ``figure_style`` configuration.
    line_width : int
        Line width of the curve.
        Default depends on the ``figure_style`` configuration.
    line_style : str
        Line style of the curve.
        Default depends on the ``figure_style`` configuration.

    Attributes
    ----------
    parameters : np.ndarray
        Parameters of the fit (same order as guesses).
    cov_matrix : np.ndarray
        Covariance matrix of the parameters of the fit.
    standard_deviation : np.ndarray
        Standard deviation of the parameters of the fit.
    function : Callable
        Exponential function with the parameters of the fit.
    """

    def __init__(
        self,
        curve_to_be_fit: Curve | Scatter,
        label: Optional[str] = None,
        guesses: Optional[ArrayLike] = None,
        color: str = "default",
        line_width: int | Literal["default"] = "default",
        line_style: str = "default",
    ) -> None:
        """
        Create a curve fit (continuous :class:`~graphinglib.data_plotting_1d.Curve`) from an existing
        :class:`~graphinglib.data_plotting_1d.Curve` object using an exponential fit.

        Parameters
        ----------
        curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
            The object to be fit.
        label : str, optional
            Label to be displayed in the legend.
        guesses : ArrayLike, optional
            Initial guesses for the parameters of the fit. Order is a, b, c as written above.
        color : str
            Color of the curve.
            Default depends on the ``figure_style`` configuration.
        line_width : int
            Line width of the curve.
            Default depends on the ``figure_style`` configuration.
        line_style : str
            Line style of the curve.
            Default depends on the ``figure_style`` configuration.

        Attributes
        ----------
        parameters : np.ndarray
            Parameters of the fit (same order as guesses).
        cov_matrix : np.ndarray
            Covariance matrix of the parameters of the fit.
        standard_deviation : np.ndarray
            Standard deviation of the parameters of the fit.
        function : Callable
            Exponential function with the parameters of the fit.
        """
        self.curve_to_be_fit = curve_to_be_fit
        self.guesses = guesses
        self._calculate_parameters()
        self.function = self._exp_func_with_params()
        self.color = color
        if label:
            self.label = label + " : " + "$f(x) = $" + str(self)
        else:
            self.label = "$f(x) = $" + str(self)
        self.line_width = line_width
        self.line_style = line_style
        self._res_curves_to_be_plotted = False
        number_of_points = 500
        self.x_data = np.linspace(
            self.curve_to_be_fit.x_data[0],
            self.curve_to_be_fit.x_data[-1],
            number_of_points,
        )
        self.y_data = self.function(self.x_data)
        self._fill_curve_between = False

    def __str__(self) -> str:
        """
        Creates a string representation of the exponential function.
        """
        part1 = f"{self.parameters[0]:.3f} \exp({self.parameters[1]:.3f}x"
        part2 = (
            f" + {self.parameters[2]:.3f})"
            if self.parameters[2] >= 0
            else f" - {abs(self.parameters[2]):.3f})"
        )
        return f"${part1 + part2}$"

    def _calculate_parameters(self) -> None:
        """
        Calculates the parameters of the fit.
        """
        parameters, self.cov_matrix = curve_fit(
            self._exp_func_template,
            self.curve_to_be_fit.x_data,
            self.curve_to_be_fit.y_data,
            p0=self.guesses,
        )
        self.parameters = parameters
        self.standard_deviation = np.sqrt(np.diag(self.cov_matrix))

    @staticmethod
    def _exp_func_template(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """
        Function to be passed to the ``curve_fit`` function.
        """
        return a * np.exp(b * x + c)

    def _exp_func_with_params(
        self,
    ) -> Callable[[float | np.ndarray], float | np.ndarray]:
        """
        Creates an exponential function with the parameters of the fit.

        Returns
        -------
        function : Callable
            Exponential function with the parameters of the fit.
        """
        return lambda x: self.parameters[0] * np.exp(
            self.parameters[1] * x + self.parameters[2]
        )


class FitFromGaussian(GeneralFit):
    """
    Create a curve fit (continuous :class:`~graphinglib.data_plotting_1d.Curve`) from an existing
    :class:`~graphinglib.data_plotting_1d.Curve` object using a gaussian fit.

    Fits a gaussian function of the form :math:`f(x) = A e^{-\\frac{(x - \mu)^2}{2 \sigma^2}}` to the given curve.
    All standard :class:`~graphinglib.data_plotting_1d.Curve` attributes and methods are available.

    Parameters
    ----------
    curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
        The object to be fit.
    label : str, optional
        Label to be displayed in the legend.
    guesses : ArrayLike, optional
        Initial guesses for the parameters of the fit.
    color : str
        Color of the curve.
        Default depends on the ``figure_style`` configuration.
    line_width : int
        Line width of the curve.
        Default depends on the ``figure_style`` configuration.
    line_style : str
        Line style of the curve.
        Default depends on the ``figure_style`` configuration.

    Attributes
    ----------
    amplitude : float
        Amplitude of the gaussian function.
    mean : float
        Mean of the gaussian function.
    standard_deviation : float
        Standard deviation of the gaussian function.

        .. warning::

            The ``standard_deviation`` attribute doesn't represent the standard deviation of the fit parameters as it does in the other fit classes. Instead, it represents the standard deviation of the gaussian function (it is one of parameters of the fit). The standard deviation of the fit parameters can be found in the ``standard_deviation_of_fit_params`` attribute.

    cov_matrix : np.ndarray
        Covariance matrix of the parameters of the fit.
    standard_deviation_of_fit_params : np.ndarray
        Standard deviation of the parameters of the fit.
    function : Callable
        Gaussian function with the parameters of the fit.
    """

    def __init__(
        self,
        curve_to_be_fit: Curve | Scatter,
        label: Optional[str] = None,
        guesses: Optional[ArrayLike] = None,
        color: str = "default",
        line_width: int | Literal["default"] = "default",
        line_style: str = "default",
    ) -> None:
        """
        Create a curve fit (continuous :class:`~graphinglib.data_plotting_1d.Curve`) from an existing
        :class:`~graphinglib.data_plotting_1d.Curve` object using a gaussian fit.

        Fits a gaussian function of the form :math:`f(x) = A e^{-\\frac{(x - \mu)^2}{2 \sigma^2}}` to the given curve.
        All standard :class:`~graphinglib.data_plotting_1d.Curve` attributes and methods are available.

        Parameters
        ----------
        curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
            The object to be fit.
        label : str, optional
            Label to be displayed in the legend.
        guesses : ArrayLike, optional
            Initial guesses for the parameters of the fit.
        color : str
            Color of the curve.
            Default depends on the ``figure_style`` configuration.
        line_width : int
            Line width of the curve.
            Default depends on the ``figure_style`` configuration.
        line_style : str
            Line style of the curve.
            Default depends on the ``figure_style`` configuration.

        Attributes
        ----------
        amplitude : float
            Amplitude of the gaussian function.
        mean : float
            Mean of the gaussian function.
        standard_deviation : float
            Standard deviation of the gaussian function.
        cov_matrix : np.ndarray
            Covariance matrix of the parameters of the fit.
        standard_deviation_of_fit_params : np.ndarray
            Standard deviation of the parameters of the fit.
        function : Callable
            Gaussian function with the parameters of the fit.

        Warning
        -------
        The ``standard_deviation`` attribute doesn't represent the standard deviation of the fit parameters as it does in the other fit classes. Instead, it represents the standard deviation of the gaussian function (it is one of parameters of the fit). The standard deviation of the fit parameters can be found in the ``standard_deviation_of_fit_params`` attribute.
        """
        self.curve_to_be_fit = curve_to_be_fit
        self.guesses = guesses
        self._calculate_parameters()
        self.function = self._gaussian_func_with_params()
        self.color = color
        if label:
            self.label = label + " : " + str(self)
        else:
            self.label = str(self)
        self.line_width = line_width
        self.line_style = line_style
        self._res_curves_to_be_plotted = False
        number_of_points = 500
        self.x_data = np.linspace(
            self.curve_to_be_fit.x_data[0],
            self.curve_to_be_fit.x_data[-1],
            number_of_points,
        )
        self.y_data = self.function(self.x_data)
        self._fill_curve_between = False

    def __str__(self) -> str:
        """
        Creates a string representation of the gaussian function.
        """
        return f"$\mu = {self.mean:.3f}, \sigma = {self.standard_deviation:.3f}, A = {self.amplitude:.3f}$"

    def _calculate_parameters(self) -> None:
        """
        Calculates the parameters of the fit.
        """
        parameters, self.cov_matrix = curve_fit(
            self._gaussian_func_template,
            self.curve_to_be_fit.x_data,
            self.curve_to_be_fit.y_data,
            p0=self.guesses,
        )
        self.amplitude = parameters[0]
        self.mean = parameters[1]
        self.standard_deviation = parameters[2]
        self.standard_deviation_of_fit_params = np.sqrt(np.diag(self.cov_matrix))

    @staticmethod
    def _gaussian_func_template(
        x: np.ndarray, amplitude: float, mean: float, standard_deviation: float
    ) -> np.ndarray:
        """
        Function to be passed to the ``curve_fit`` function.
        """
        return amplitude * np.exp(-(((x - mean) / standard_deviation) ** 2) / 2)

    def _gaussian_func_with_params(
        self,
    ) -> Callable[[float | np.ndarray], float | np.ndarray]:
        """
        Creates a gaussian function with the parameters of the fit.

        Returns
        -------
        function : Callable
            Gaussian function with the parameters of the fit.
        """
        return lambda x: self.amplitude * np.exp(
            -(((x - self.mean) / self.standard_deviation) ** 2) / 2
        )


class FitFromSquareRoot(GeneralFit):
    """
    Create a curve fit (continuous :class:`~graphinglib.data_plotting_1d.Curve`) from an existing
    :class:`~graphinglib.data_plotting_1d.Curve` object using a square root fit.

    Fits a square root function of the form :math:`f(x) = a \sqrt{x + b} + c` to the given curve. All standard
    :class:`~graphinglib.data_plotting_1d.Curve` attributes and methods are available.

    Parameters
    ----------
    curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
        The object to be fit.
    label : str, optional
        Label to be displayed in the legend.
    guesses : ArrayLike, optional
        Initial guesses for the parameters of the fit. Order is a, b, c as written above.
    color : str
        Color of the curve.
        Default depends on the ``figure_style`` configuration.
    line_width : int
        Line width of the curve.
        Default depends on the ``figure_style`` configuration.
    line_style : str
        Line style of the curve.
        Default depends on the ``figure_style`` configuration.

    Attributes
    ----------
    parameters : np.ndarray
        Parameters of the fit (same order as guesses).
    cov_matrix : np.ndarray
        Covariance matrix of the parameters of the fit.
    standard_deviation : np.ndarray
        Standard deviation of the parameters of the fit.
    function : Callable
        Square root function with the parameters of the fit.
    """

    def __init__(
        self,
        curve_to_be_fit: Curve | Scatter,
        label: Optional[str] = None,
        guesses: Optional[ArrayLike] = None,
        color: str = "default",
        line_width: int | Literal["default"] = "default",
        line_style: str = "default",
    ) -> None:
        """
        Create a curve fit (continuous :class:`~graphinglib.data_plotting_1d.Curve`) from an existing
        :class:`~graphinglib.data_plotting_1d.Curve` object using a square root fit.

        Fits a square root function of the form :math:`f(x) = a \sqrt{x + b} + c` to the given curve. All standard
        :class:`~graphinglib.data_plotting_1d.Curve` attributes and methods are available.

        Parameters
        ----------
        curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
            The object to be fit.
        label : str, optional
            Label to be displayed in the legend.
        guesses : ArrayLike, optional
            Initial guesses for the parameters of the fit. Order is a, b, c as written above.
        color : str
            Color of the curve.
            Default depends on the ``figure_style`` configuration.
        line_width : int
            Line width of the curve.
            Default depends on the ``figure_style`` configuration.
        line_style : str
            Line style of the curve.
            Default depends on the ``figure_style`` configuration.

        Attributes
        ----------
        parameters : np.ndarray
            Parameters of the fit (same order as guesses).
        cov_matrix : np.ndarray
            Covariance matrix of the parameters of the fit.
        standard_deviation : np.ndarray
            Standard deviation of the parameters of the fit.
        function : Callable
            Square root function with the parameters of the fit.
        """
        self.curve_to_be_fit = curve_to_be_fit
        self.guesses = guesses
        self._calculate_parameters()
        self.function = self._square_root_func_with_params()
        self.color = color
        if label:
            self.label = label + " : " + str(self)
        else:
            self.label = str(self)
        self.line_width = line_width
        self.line_style = line_style
        self._res_curves_to_be_plotted = False
        number_of_points = 500
        self.x_data = np.linspace(
            self.curve_to_be_fit.x_data[0],
            self.curve_to_be_fit.x_data[-1],
            number_of_points,
        )
        self.y_data = self.function(self.x_data)
        self._fill_curve_between = False

    def __str__(self) -> str:
        """
        Creates a string representation of the square root function.
        """
        return f"${self.parameters[0]:.3f} \sqrt{{x {'+' if self.parameters[1] > 0 else '-'} {abs(self.parameters[1]):.3f}}} {'+' if self.parameters[2] > 0 else '-'} {abs(self.parameters[2]):.3f}$"

    def _calculate_parameters(self) -> None:
        """
        Calculates the parameters of the fit.
        """
        parameters, self.cov_matrix = curve_fit(
            self._square_root_func_template,
            self.curve_to_be_fit.x_data,
            self.curve_to_be_fit.y_data,
            p0=self.guesses,
        )
        self.parameters = parameters
        self.standard_deviation = np.sqrt(np.diag(self.cov_matrix))

    @staticmethod
    def _square_root_func_template(
        x: np.ndarray, a: float, b: float, c: float
    ) -> np.ndarray:
        """
        Function to be passed to the ``curve_fit`` function.
        """
        return a * np.sqrt(x + b) + c

    def _square_root_func_with_params(
        self,
    ) -> Callable[[float | np.ndarray], float | np.ndarray]:
        """
        Creates a square root function with the parameters of the fit.

        Returns
        -------
        function : Callable
            Square root function with the parameters of the fit.
        """
        return (
            lambda x: self.parameters[0] * np.sqrt(x + self.parameters[1])
            + self.parameters[2]
        )


class FitFromLog(GeneralFit):
    """
    Create a curve fit (continuous :class:`~graphinglib.data_plotting_1d.Curve`) from an existing
    :class:`~graphinglib.data_plotting_1d.Curve` object using a logarithmic fit.

    Fits a logarithmic function of the form :math:`f(x) = a \log_{base}(x + b) + c` to the given curve. All standard
    :class:`~graphinglib.data_plotting_1d.Curve` attributes and methods are available.

    Parameters
    ----------
    curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
        The object to be fit.
    label : str, optional
        Label to be displayed in the legend.
    log_base : float
        Base of the logarithm.
        Default is e.
    guesses : ArrayLike, optional
        Initial guesses for the parameters of the fit. Order is a, b, c as written above.
    color : str
        Color of the curve.
        Default depends on the ``figure_style`` configuration.
    line_width : int
        Line width of the curve.
        Default depends on the ``figure_style`` configuration.
    line_style : str
        Line style of the curve.
        Default depends on the ``figure_style`` configuration.

    Attributes
    ----------
    parameters : np.ndarray
        Parameters of the fit (same order as guesses).
    cov_matrix : np.ndarray
        Covariance matrix of the parameters of the fit.
    standard_deviation : np.ndarray
        Standard deviation of the parameters of the fit.
    function : Callable
        Logarithmic function with the parameters of the fit.
    """

    def __init__(
        self,
        curve_to_be_fit: Curve | Scatter,
        label: Optional[str] = None,
        log_base: float = np.e,
        guesses: Optional[ArrayLike] = None,
        color: str = "default",
        line_width: int | Literal["default"] = "default",
        line_style: str = "default",
    ) -> None:
        """
        Create a curve fit (continuous :class:`~graphinglib.data_plotting_1d.Curve`) from an existing
        :class:`~graphinglib.data_plotting_1d.Curve` object using a logarithmic fit.

        Fits a logarithmic function of the form :math:`f(x) = a \log_{base}(x + b) + c` to the given curve. All standard
        :class:`~graphinglib.data_plotting_1d.Curve` attributes and methods are available.

        Parameters
        ----------
        curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
            The object to be fit.
        label : str, optional
            Label to be displayed in the legend.
        log_base : float
            Base of the logarithm.
            Default is e.
        guesses : ArrayLike, optional
            Initial guesses for the parameters of the fit. Order is a, b, c as written above.
        color : str
            Color of the curve.
            Default depends on the ``figure_style`` configuration.
        line_width : int
            Line width of the curve.
            Default depends on the ``figure_style`` configuration.
        line_style : str
            Line style of the curve.
            Default depends on the ``figure_style`` configuration.

        Attributes
        ----------
        parameters : np.ndarray
            Parameters of the fit (same order as guesses).
        cov_matrix : np.ndarray
            Covariance matrix of the parameters of the fit.
        standard_deviation : np.ndarray
            Standard deviation of the parameters of the fit.
        function : Callable
            Logarithmic function with the parameters of the fit.
        """
        self.curve_to_be_fit = curve_to_be_fit
        self.log_base = log_base
        self.guesses = guesses
        self._calculate_parameters()
        self.function = self._log_func_with_params()
        self.color = color
        if label:
            self.label = label + " : " + str(self)
        else:
            self.label = str(self)
        self.line_width = line_width
        self.line_style = line_style
        self._res_curves_to_be_plotted = False
        number_of_points = 500
        self.x_data = np.linspace(
            self.curve_to_be_fit.x_data[0],
            self.curve_to_be_fit.x_data[-1],
            number_of_points,
        )
        self.y_data = self.function(self.x_data)
        self._fill_curve_between = False

    def __str__(self) -> str:
        """
        Creates a string representation of the logarithmic function.
        """
        return f"${self.parameters[0]:.3f} log_{self.log_base if self.log_base != np.e else 'e'}(x {'-' if self.parameters[1] < 0 else '+'} {abs(self.parameters[1]):.3f}) {'-' if self.parameters[2] < 0 else '+'} {abs(self.parameters[2]):.3f}$"

    def _calculate_parameters(self) -> None:
        """
        Calculates the parameters of the fit.
        """
        self.parameters, self.cov_matrix = curve_fit(
            self._log_func_template(),
            self.curve_to_be_fit.x_data,
            self.curve_to_be_fit.y_data,
            p0=self.guesses,
        )
        self.standard_deviation = np.sqrt(np.diag(self.cov_matrix))

    def _log_func_template(
        self,
    ) -> Callable[[float | np.ndarray, float, float, float], float | np.ndarray]:
        """
        Function to be passed to the ``curve_fit`` function.
        """
        return lambda x, a, b, c: a * (np.log(x + b) / np.log(self.log_base)) + c

    def _log_func_with_params(
        self,
    ) -> Callable[[float | np.ndarray], float | np.ndarray]:
        """
        Creates a logarithmic function with the parameters of the fit.

        Returns
        -------
        function : Callable
            Logarithmic function with the parameters of the fit.
        """
        return (
            lambda x: self.parameters[0]
            * (np.log(x + self.parameters[1]) / np.log(self.log_base))
            + self.parameters[2]
        )


class FitFromFunction(GeneralFit):
    """
    Create a curve fit (continuous :class:`~graphinglib.data_plotting_1d.Curve`) from a
    :class:`~graphinglib.data_plotting_1d.Curve` object using an arbitrary function passed as an argument.

    Fits a function of the form :math:`f(x, a, b, c, ...)` to the given curve. All standard
    :class:`~graphinglib.data_plotting_1d.Curve` attributes and methods are available.

    Parameters
    ----------
    function : Callable
        Function to be passed to the curve_fit function.
    curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
        The object to be fit.
    label : str, optional
        Label to be displayed in the legend.
    guesses : ArrayLike, optional
        Initial guesses for the parameters of the fit. Order is a, b, c, ... as written above.
    color : str
        Color of the curve.
        Default depends on the ``figure_style`` configuration.
    line_width : int
        Line width of the curve.
        Default depends on the ``figure_style`` configuration.
    line_style : str
        Line style of the curve.
        Default depends on the ``figure_style`` configuration.

    Attributes
    ----------
    parameters : np.ndarray
        Parameters of the fit (same order as guesses).
    cov_matrix : np.ndarray
        Covariance matrix of the parameters of the fit.
    standard_deviation : np.ndarray
        Standard deviation of the parameters of the fit.
    function : Callable
        Function with the parameters of the fit.
    """

    def __init__(
        self,
        function: Callable,
        curve_to_be_fit: Curve | Scatter,
        label: Optional[str] = None,
        guesses: Optional[ArrayLike] = None,
        color: str = "default",
        line_width: int | Literal["default"] = "default",
        line_style: str = "default",
    ):
        """
        Create a curve fit (continuous :class:`~graphinglib.data_plotting_1d.Curve`) from a
        :class:`~graphinglib.data_plotting_1d.Curve` object using an arbitrary function passed as an argument.

        Fits a function of the form :math:`f(x, a, b, c, ...)` to the given curve. All standard
        :class:`~graphinglib.data_plotting_1d.Curve` attributes and methods are available.

        Parameters
        ----------
        function : Callable
            Function to be passed to the curve_fit function.
        curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
            The object to be fit.
        label : str, optional
            Label to be displayed in the legend.
        guesses : ArrayLike, optional
            Initial guesses for the parameters of the fit. Order is a, b, c, ... as written above.
        color : str
            Color of the curve.
            Default depends on the ``figure_style`` configuration.
        line_width : int
            Line width of the curve.
            Default depends on the ``figure_style`` configuration.
        line_style : str
            Line style of the curve.
            Default depends on the ``figure_style`` configuration.

        Attributes
        ----------
        parameters : np.ndarray
            Parameters of the fit (same order as guesses).
        cov_matrix : np.ndarray
            Covariance matrix of the parameters of the fit.
        standard_deviation : np.ndarray
            Standard deviation of the parameters of the fit.
        function : Callable
            Function with the parameters of the fit.
        """
        self._function_template = function
        self.curve_to_be_fit = curve_to_be_fit
        self.guesses = guesses
        self.color = color
        self.line_width = line_width
        self.line_style = line_style

        self._calculate_parameters()
        self.function = self._get_function_with_params()
        self.label = label
        self._res_curves_to_be_plotted = False
        number_of_points = 500
        self.x_data = np.linspace(
            self.curve_to_be_fit.x_data[0],
            self.curve_to_be_fit.x_data[-1],
            number_of_points,
        )
        self.y_data = self.function(self.x_data)
        self._fill_curve_between = False

    def _calculate_parameters(self) -> None:
        """
        Calculates the parameters of the fit.
        """
        self.parameters, self.cov_matrix = curve_fit(
            self._function_template,
            self.curve_to_be_fit.x_data,
            self.curve_to_be_fit.y_data,
            p0=self.guesses,
        )
        self.standard_deviation = np.sqrt(np.diag(self.cov_matrix))

    def _get_function_with_params(self) -> Callable:
        """
        Creates a function with the parameters of the fit.

        Returns
        -------
        function : Callable
            Function with the parameters of the fit.
        """
        argument_names = self._function_template.__code__.co_varnames[
            : self._function_template.__code__.co_argcount
        ][1:]
        args_dict = {
            argument_names[i]: self.parameters[i] for i in range(len(argument_names))
        }
        return partial(self._function_template, **args_dict)
