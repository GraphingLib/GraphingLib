from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import Callable, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit

from .data_plotting_1d import Curve, Scatter
from .graph_elements import Point

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


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
        self._curve_to_be_fit = curve_to_be_fit
        self._color = color
        self._line_width = line_width
        if label:
            self._label = label + " : " + "$f(x) = $" + str(self)
        else:
            self._label = "$f(x) = $" + str(self)
        self._line_style = line_style

        self._function: Callable[[np.ndarray], np.ndarray]

        self._setup_attributes()

    def _setup_attributes(self) -> None:
        self._res_curves_to_be_plotted = False
        self._res_sigma_multiplier = None
        self._res_color = None
        self._res_line_width = None
        self._res_line_style = None

        self._show_errorbars: bool = False
        self._errorbars_color = None
        self._errorbars_line_width = None
        self._cap_thickness = None
        self._cap_width = None

        self._show_error_curves: bool = False
        self._error_curves_fill_between: bool = False
        self._error_curves_color = None
        self._error_curves_line_style = None
        self._error_curves_line_width = None

        self._fill_between_bounds: Optional[tuple[float, float]] = None
        self._fill_between_other_curve: Optional[Self] = None
        self._fill_between_color: Optional[str] = None

    @property
    def curve_to_be_fit(self) -> Curve | Scatter:
        return self._curve_to_be_fit

    @curve_to_be_fit.setter
    def curve_to_be_fit(self, curve: Curve | Scatter) -> None:
        self._curve_to_be_fit = curve

    @property
    def function(self) -> Callable[[np.ndarray], np.ndarray]:
        return self._function

    def __str__(self) -> str:
        """
        Create a string representation of the fit function.
        """
        raise NotImplementedError()

    def get_coordinates_at_x(self, x: float) -> tuple[float, float]:
        return (x, self._function(x))

    def create_point_at_x(
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
            self._function(x),
            label=label,
            color=color,
            edge_color=edge_color,
            marker_size=marker_size,
            marker_style=marker_style,
            edge_width=line_width,
        )

    def get_coordinates_at_y(
        self, y: float, interpolation_method: str = "linear"
    ) -> list[tuple[float, float]]:
        return super().get_coordinates_at_y(y, interpolation_method)

    def create_points_at_y(
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
        Creates the Points on the curve at a given y value.

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
        list[:class:`~graphinglib.graph_elements.Point`]
            List of :class:`~graphinglib.graph_elements.Point` objects on the curve at the given y value.
        """
        coord_pairs = self.get_coordinates_at_y(y, interpolation_kind)
        points = [
            Point(
                coord[0],
                coord[1],
                label=label,
                color=color,
                edge_color=edge_color,
                marker_size=marker_size,
                marker_style=marker_style,
                edge_width=line_width,
            )
            for coord in coord_pairs
        ]
        return points

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified
        Axes
        """
        params = {
            "color": self._color,
            "linewidth": self._line_width,
            "linestyle": self._line_style,
        }
        params = {key: value for key, value in params.items() if value != "default"}
        (self.handle,) = axes.plot(
            self._x_data,
            self._y_data,
            label=self._label,
            zorder=z_order,
            **params,
        )
        if self._res_curves_to_be_plotted:
            y_fit = self._y_data
            residuals = self.get_residuals()
            std = np.std(residuals)
            y_fit_plus_std = y_fit + (self._res_sigma_multiplier * std)
            y_fit_minus_std = y_fit - (self._res_sigma_multiplier * std)
            params = {
                "color": self._res_color,
                "linewidth": self._res_line_width,
                "linestyle": self._res_line_style,
            }
            params = {key: value for key, value in params.items() if value != "default"}
            axes.plot(
                self._x_data,
                y_fit_minus_std,
                zorder=z_order,
                **params,
            )
            axes.plot(
                self._x_data,
                y_fit_plus_std,
                zorder=z_order,
                **params,
            )
        if self._fill_between_bounds:
            kwargs = {"alpha": 0.2}
            if self._fill_between_color:
                kwargs["color"] = self._fill_between_color
            else:
                kwargs["color"] = self.handle[0].get_color()
            params = {key: value for key, value in kwargs.items() if value != "default"}
            axes.fill_between(
                self._x_data,
                self._y_data,
                where=np.logical_and(
                    self._x_data >= self._fill_between_bounds[0],
                    self._x_data <= self._fill_between_bounds[1],
                ),
                zorder=z_order - 2,
                **params,
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
        self._res_sigma_multiplier = sigma_multiplier
        self._res_color = color
        self._res_line_width = line_width
        self._res_line_style = line_style

    def get_residuals(self) -> np.ndarray:
        """
        Calculates the residuals of the fit curve.

        Returns
        -------
        residuals : np.ndarray
            Array of residuals.
        """
        y_data = self._function(self._curve_to_be_fit._x_data)
        residuals = y_data - self._curve_to_be_fit._y_data
        return residuals

    def get_Rsquared(self) -> float:
        """
        Calculates the :math:`R^2` value of the fit curve.

        Returns
        -------
        Rsquared : float
            :math:`R^2` value
        """
        Rsquared = 1 - (
            np.sum(self.get_residuals() ** 2)
            / np.sum(
                (self._curve_to_be_fit._y_data - np.mean(self._curve_to_be_fit._y_data))
                ** 2
            )
        )
        return Rsquared

    def copy(self) -> Self:
        return deepcopy(self)


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
            Polynomial function with the parameters of the fit.
        """
        self._curve_to_be_fit = curve_to_be_fit
        inversed_coeffs, inversed_cov_matrix = np.polyfit(
            self._curve_to_be_fit._x_data,
            self._curve_to_be_fit._y_data,
            degree,
            cov=True,
        )
        self._coeffs = inversed_coeffs[::-1]
        self._cov_matrix = np.flip(inversed_cov_matrix)
        self._standard_deviation = np.sqrt(np.diag(self._cov_matrix))
        self._function = self._polynomial_func_with_params()
        self._color = color
        self._line_width = line_width
        if label:
            self._label = label + " : " + "$f(x) = $" + str(self)
        else:
            self._label = "$f(x) = $" + str(self)
        self._line_style = line_style
        self._res_curves_to_be_plotted = False
        number_of_points = (
            len(self._curve_to_be_fit._x_data)
            if len(self._curve_to_be_fit._x_data) > 500
            else 500
        )
        self._x_data = np.linspace(
            self._curve_to_be_fit._x_data[0],
            self._curve_to_be_fit._x_data[-1],
            number_of_points,
        )
        self._y_data = self._function(self._x_data)

        self._setup_attributes()

    @property
    def coeffs(self) -> np.ndarray:
        return self._coeffs

    @property
    def cov_matrix(self) -> np.ndarray:
        return self._cov_matrix

    @property
    def standard_deviation(self) -> np.ndarray:
        return self._standard_deviation

    def __str__(self) -> str:
        """
        Creates a string representation of the polynomial function.
        """
        coeff_chunks = []
        power_chunks = []
        ordered_rounded_coeffs = [round(coeff, 3) for coeff in self._coeffs[::-1]]
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
            coeff * x**exponent for exponent, coeff in enumerate(self._coeffs)
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
        frequency_deg : float
            Frequency of the sine function in degrees.
        phase_rad : float
            Phase of the sine function.
        phase_deg : float
            Phase of the sine function in degrees.
        vertical_shift : float
            Vertical shift of the sine function.
        parameters : np.ndarray
            Parameters of the fit (amplitude, frequency (rad), phase (rad), vertical shift)
        cov_matrix : np.ndarray
            Covariance matrix of the parameters of the fit.
        standard_deviation : np.ndarray
            Standard deviation of the parameters of the fit.
        function : Callable
            Sine function with the parameters of the fit.
        """
        self._curve_to_be_fit = curve_to_be_fit
        self._guesses = guesses
        self._calculate_parameters()
        self._function = self._sine_func_with_params()
        self._color = color
        if label:
            self._label = label + " : " + "$f(x) = $" + str(self)
        else:
            self._label = "$f(x) = $" + str(self)
        self._line_width = line_width
        self._line_style = line_style
        self._res_curves_to_be_plotted = False
        number_of_points = (
            len(self._curve_to_be_fit._x_data)
            if len(self._curve_to_be_fit._x_data) > 500
            else 500
        )
        self._x_data = np.linspace(
            self._curve_to_be_fit._x_data[0],
            self._curve_to_be_fit._x_data[-1],
            number_of_points,
        )
        self._y_data = self._function(self._x_data)

        self._setup_attributes()

    @property
    def amplitude(self) -> float:
        return self._amplitude

    @property
    def frequency_rad(self) -> float:
        return self._frequency_rad

    @property
    def frequency_deg(self) -> float:
        return self._frequency_deg

    @property
    def phase_rad(self) -> float:
        return self._phase_rad

    @property
    def phase_deg(self) -> float:
        return self._phase_deg

    @property
    def vertical_shift(self) -> float:
        return self._vertical_shift

    @property
    def cov_matrix(self) -> np.ndarray:
        return self._cov_matrix

    @property
    def standard_deviation(self) -> np.ndarray:
        return self._standard_deviation

    @property
    def parameters(self) -> np.ndarray:
        return self._parameters

    def __str__(self) -> str:
        """
        Creates a string representation of the sine function.
        """
        part1 = f"{self._amplitude:.3f} \sin({self._frequency_rad:.3f}x"
        part2 = (
            f" + {self._phase_rad:.3f})"
            if self._phase_rad >= 0
            else f" - {abs(self._phase_rad):.3f})"
        )
        part3 = (
            f" + {self._vertical_shift:.3f}"
            if self._vertical_shift >= 0
            else f" - {abs(self._vertical_shift):.3f}"
        )
        return f"${part1 + part2 + part3}$"

    def _calculate_parameters(self) -> None:
        """
        Calculates the parameters of the fit.
        """
        self._parameters, self._cov_matrix = curve_fit(
            self._sine_func_template,
            self._curve_to_be_fit._x_data,
            self._curve_to_be_fit._y_data,
            p0=self._guesses,
        )
        self._amplitude, self._frequency_rad, self._phase_rad, self._vertical_shift = (
            self._parameters
        )
        self._standard_deviation = np.sqrt(np.diag(self._cov_matrix))

        # Calculate the frequency and phase in degrees
        self._frequency_deg = np.degrees(self._frequency_rad)
        self._phase_deg = np.degrees(self._phase_rad)

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
        Callable
            Sine function with the parameters of the fit.
        """
        return (
            lambda x: self._amplitude
            * np.sin(self._frequency_rad * x + self._phase_rad)
            + self._vertical_shift
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
        Create a curve fit (continuous :class:`~graphinglib.data_plotting_1d.Curve`)
        of the form :math:`f(x) = a \exp(bx + c)` from an existing :class:`~graphinglib.data_plotting_1d.Curve`
        object using an exponential fit.

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
        self._curve_to_be_fit = curve_to_be_fit
        self._guesses = guesses
        self._calculate_parameters()
        self._function = self._exp_func_with_params()
        self._color = color
        if label:
            self._label = label + " : " + "$f(x) = $" + str(self)
        else:
            self._label = "$f(x) = $" + str(self)
        self._line_width = line_width
        self._line_style = line_style
        self._res_curves_to_be_plotted = False
        number_of_points = (
            len(self._curve_to_be_fit._x_data)
            if len(self._curve_to_be_fit._x_data) > 500
            else 500
        )
        self._x_data = np.linspace(
            self._curve_to_be_fit._x_data[0],
            self._curve_to_be_fit._x_data[-1],
            number_of_points,
        )
        self._y_data = self._function(self._x_data)

        self._setup_attributes()

    @property
    def parameters(self) -> np.ndarray:
        return self._parameters

    @property
    def cov_matrix(self) -> np.ndarray:
        return self._cov_matrix

    @property
    def standard_deviation(self) -> np.ndarray:
        return self._standard_deviation

    def __str__(self) -> str:
        """
        Creates a string representation of the exponential function.
        """
        part1 = f"{self._parameters[0]:.3f} \exp({self._parameters[1]:.3f}x"
        part2 = (
            f" + {self._parameters[2]:.3f})"
            if self._parameters[2] >= 0
            else f" - {abs(self._parameters[2]):.3f})"
        )
        return f"${part1 + part2}$"

    def _calculate_parameters(self) -> None:
        """
        Calculates the parameters of the fit.
        """
        self._parameters, self._cov_matrix = curve_fit(
            self._exp_func_template,
            self._curve_to_be_fit._x_data,
            self._curve_to_be_fit._y_data,
            p0=self._guesses,
        )
        self._standard_deviation = np.sqrt(np.diag(self._cov_matrix))

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
        return lambda x: self._parameters[0] * np.exp(
            self._parameters[1] * x + self._parameters[2]
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
        Initial guesses for the parameters of the fit. Order is amplitude (A), mean (mu), standard deviation (sigma).
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
        self._curve_to_be_fit = curve_to_be_fit
        self._guesses = guesses
        self._calculate_parameters()
        self._function = self._gaussian_func_with_params()
        self._color = color
        if label:
            self._label = label + " : " + str(self)
        else:
            self._label = str(self)
        self._line_width = line_width
        self._line_style = line_style
        self._res_curves_to_be_plotted = False
        number_of_points = (
            len(self._curve_to_be_fit._x_data)
            if len(self._curve_to_be_fit._x_data) > 500
            else 500
        )
        self._x_data = np.linspace(
            self._curve_to_be_fit._x_data[0],
            self._curve_to_be_fit._x_data[-1],
            number_of_points,
        )
        self._y_data = self._function(self._x_data)

        self._setup_attributes()

    @property
    def amplitude(self) -> float:
        return self._amplitude

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def standard_deviation(self) -> float:
        return self._standard_deviation

    @property
    def cov_matrix(self) -> np.ndarray:
        return self._cov_matrix

    @property
    def standard_deviation_of_fit_params(self) -> np.ndarray:
        return self._standard_deviation_of_fit_params

    @property
    def parameters(self) -> np.ndarray:
        return self._parameters

    def __str__(self) -> str:
        """
        Creates a string representation of the gaussian function.
        """
        return f"$\mu = {self._mean:.3f}, \sigma = {self._standard_deviation:.3f}, A = {self._amplitude:.3f}$"

    def _calculate_parameters(self) -> None:
        """
        Calculates the parameters of the fit.
        """
        self._parameters, self._cov_matrix = curve_fit(
            self._gaussian_func_template,
            self._curve_to_be_fit._x_data,
            self._curve_to_be_fit._y_data,
            p0=self._guesses,
        )
        self._amplitude = self._parameters[0]
        self._mean = self._parameters[1]
        self._standard_deviation = self._parameters[2]
        self._standard_deviation_of_fit_params = np.sqrt(np.diag(self._cov_matrix))

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
        return lambda x: self._amplitude * np.exp(
            -(((x - self._mean) / self._standard_deviation) ** 2) / 2
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
        self._curve_to_be_fit = curve_to_be_fit
        self._guesses = guesses
        self._calculate_parameters()
        self._function = self._square_root_func_with_params()
        self._color = color
        if label:
            self._label = label + " : " + str(self)
        else:
            self._label = str(self)
        self._line_width = line_width
        self._line_style = line_style
        self._res_curves_to_be_plotted = False
        number_of_points = (
            len(self._curve_to_be_fit._x_data)
            if len(self._curve_to_be_fit._x_data) > 500
            else 500
        )
        self._x_data = np.linspace(
            self._curve_to_be_fit._x_data[0],
            self._curve_to_be_fit._x_data[-1],
            number_of_points,
        )
        self._y_data = self._function(self._x_data)

        self._setup_attributes()

    @property
    def parameters(self) -> np.ndarray:
        return self._parameters

    @property
    def cov_matrix(self) -> np.ndarray:
        return self.cov_matrix

    @property
    def standard_deviation(self) -> np.ndarray:
        return self.standard_deviation

    def __str__(self) -> str:
        """
        Creates a string representation of the square root function.
        """
        return f"${self._parameters[0]:.3f} \sqrt{{x {'+' if self._parameters[1] > 0 else '-'} {abs(self._parameters[1]):.3f}}} {'+' if self._parameters[2] > 0 else '-'} {abs(self._parameters[2]):.3f}$"

    def _calculate_parameters(self) -> None:
        """
        Calculates the parameters of the fit.
        """
        self._parameters, self._cov_matrix = curve_fit(
            self._square_root_func_template,
            self._curve_to_be_fit._x_data,
            self._curve_to_be_fit._y_data,
            p0=self._guesses,
        )
        self._standard_deviation = np.sqrt(np.diag(self._cov_matrix))

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
            lambda x: self._parameters[0] * np.sqrt(x + self._parameters[1])
            + self._parameters[2]
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
        self._curve_to_be_fit = curve_to_be_fit
        self._log_base = log_base
        self._guesses = guesses
        self._calculate_parameters()
        self._function = self._log_func_with_params()
        self._color = color
        if label:
            self._label = label + " : " + str(self)
        else:
            self._label = str(self)
        self._line_width = line_width
        self._line_style = line_style
        self._res_curves_to_be_plotted = False
        number_of_points = (
            len(self._curve_to_be_fit._x_data)
            if len(self._curve_to_be_fit._x_data) > 500
            else 500
        )
        self._x_data = np.linspace(
            self._curve_to_be_fit._x_data[0],
            self._curve_to_be_fit._x_data[-1],
            number_of_points,
        )
        self._y_data = self._function(self._x_data)

        self._setup_attributes()

    @property
    def parameters(self) -> np.ndarray:
        return self._parameters

    @property
    def cov_matrix(self) -> np.ndarray:
        return self._cov_matrix

    @property
    def standard_deviation(self) -> np.ndarray:
        return self._standard_deviation

    def __str__(self) -> str:
        """
        Creates a string representation of the logarithmic function.
        """
        return f"${self._parameters[0]:.3f} log_{self._log_base if self._log_base != np.e else 'e'}(x {'-' if self._parameters[1] < 0 else '+'} {abs(self._parameters[1]):.3f}) {'-' if self._parameters[2] < 0 else '+'} {abs(self._parameters[2]):.3f}$"

    def _calculate_parameters(self) -> None:
        """
        Calculates the parameters of the fit.
        """
        self._parameters, self._cov_matrix = curve_fit(
            self._log_func_template(),
            self._curve_to_be_fit._x_data,
            self._curve_to_be_fit._y_data,
            p0=self._guesses,
        )
        self._standard_deviation = np.sqrt(np.diag(self._cov_matrix))

    def _log_func_template(
        self,
    ) -> Callable[[float | np.ndarray, float, float, float], float | np.ndarray]:
        """
        Function to be passed to the ``curve_fit`` function.
        """
        return lambda x, a, b, c: a * (np.log(x + b) / np.log(self._log_base)) + c

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
            lambda x: self._parameters[0]
            * (np.log(x + self._parameters[1]) / np.log(self._log_base))
            + self._parameters[2]
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
        Initial guesses for the parameters of the fit. Order is a, b, c, ...
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
        self._curve_to_be_fit = curve_to_be_fit
        self._guesses = guesses
        self._color = color
        self._line_width = line_width
        self._line_style = line_style

        self._calculate_parameters()
        self._function = self._get_function_with_params()
        self._label = label
        self._res_curves_to_be_plotted = False
        number_of_points = (
            len(self._curve_to_be_fit._x_data)
            if len(self._curve_to_be_fit._x_data) > 500
            else 500
        )
        self._x_data = np.linspace(
            self._curve_to_be_fit._x_data[0],
            self._curve_to_be_fit._x_data[-1],
            number_of_points,
        )
        self._y_data = self._function(self._x_data)

        self._setup_attributes()

    @property
    def parameters(self) -> np.ndarray:
        return self._parameters

    @property
    def cov_matrix(self) -> np.ndarray:
        return self._cov_matrix

    @property
    def standard_deviation(self) -> np.ndarray:
        return self._standard_deviation

    def _calculate_parameters(self) -> None:
        """
        Calculates the parameters of the fit.
        """
        self._parameters, self._cov_matrix = curve_fit(
            self._function_template,
            self._curve_to_be_fit._x_data,
            self._curve_to_be_fit._y_data,
            p0=self._guesses,
        )
        self._standard_deviation = np.sqrt(np.diag(self._cov_matrix))

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
            argument_names[i]: self._parameters[i] for i in range(len(argument_names))
        }
        return partial(self._function_template, **args_dict)


class FitFromFOTF(GeneralFit):
    """
    Create a curve fit (continuous :class:`~graphinglib.data_plotting_1d.Curve`) from an existing :class:`~graphinglib.data_plotting_1d.Curve` object using a first order transfer function (FOTF) fit.

    Fits a first order transfer function of the form :math:`f(x) = K\left(1-e^{-\\frac{t}{\\tau}}\\right)` to the given curve. All standard :class:`~graphinglib.data_plotting_1d.Curve` attributes and methods are available.

    Parameters
    ----------
    curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
        The object to be fit.
    label : str, optional
        Label to be displayed in the legend.
    guesses : ArrayLike, optional
        Initial guesses for the parameters of the fit. Order is K, tau.
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
    gain : float
        Gain of the first order transfer function.
    time_constant : float
        Time constant of the first order transfer function.
    cov_matrix : np.ndarray
        Covariance matrix of the parameters of the fit.
    standard_deviation : np.ndarray
        Standard deviation of the parameters of the fit.
    function : Callable
        First order transfer function with the parameters of the fit.
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
        Create a curve fit (continuous :class:`~graphinglib.data_plotting_1d.Curve`) from an existing :class:`~graphinglib.data_plotting_1d.Curve` object using a first order transfer function (FOTF) fit.

        Fits a first order transfer function of the form :math:`f(x) = K \left(1 - e^{-\\frac{t}{\\tau}}\\right)` to the given curve. All standard :class:`~graphinglib.data_plotting_1d.Curve` attributes and methods are available.

        Parameters
        ----------
        curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
            The object to be fit.
        label : str, optional
            Label to be displayed in the legend.
        guesses : ArrayLike, optional
            Initial guesses for the parameters of the fit. Order is K, tau.
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
        gain : float
            Gain of the first order transfer function.
        time_constant : float
            Time constant of the first order transfer function.
        cov_matrix : np.ndarray
            Covariance matrix of the parameters of the fit.
        standard_deviation : np.ndarray
            Standard deviation of the parameters of the fit.
        function : Callable
            First order transfer function with the parameters of the fit.
        """
        self._curve_to_be_fit = curve_to_be_fit
        self._guesses = guesses
        self._calculate_parameters()
        self._function = self._fotf_func_with_params()
        self._color = color
        if label:
            self._label = label + " : " + str(self)
        else:
            self._label = str(self)
        self._line_width = line_width
        self._line_style = line_style
        self._res_curves_to_be_plotted = False
        number_of_points = (
            len(self._curve_to_be_fit._x_data)
            if len(self._curve_to_be_fit._x_data) > 500
            else 500
        )
        self._x_data = np.linspace(
            self._curve_to_be_fit._x_data[0],
            self._curve_to_be_fit._x_data[-1],
            number_of_points,
        )
        self._y_data = self._function(self._x_data)

        self._setup_attributes()

    @property
    def gain(self) -> float:
        return self._gain

    @property
    def time_constant(self) -> float:
        return self._time_constant

    @property
    def cov_matrix(self) -> np.ndarray:
        return self._cov_matrix

    @property
    def standard_deviation(self) -> np.ndarray:
        return self._standard_deviation

    @property
    def parameters(self) -> np.ndarray:
        return self._parameters

    def __str__(self) -> str:
        """
        Creates a string representation of the first order transfer function.
        """
        return f"$K = {self._gain:.3f}, \\tau = {self._time_constant:.3f}$"

    def _calculate_parameters(self) -> None:
        """
        Calculates the parameters of the fit.
        """
        self._parameters, self._cov_matrix = curve_fit(
            self._fotf_func_template,
            self._curve_to_be_fit._x_data,
            self._curve_to_be_fit._y_data,
            p0=self._guesses,
        )
        self._gain = self._parameters[0]
        self._time_constant = self._parameters[1]
        self._standard_deviation = np.sqrt(np.diag(self._cov_matrix))

    @staticmethod
    def _fotf_func_template(
        x: np.ndarray, gain: float, time_constant: float
    ) -> np.ndarray:
        """
        Function to be passed to the ``curve_fit`` function.
        """
        return gain * (1 - np.exp(-x / time_constant))

    def _fotf_func_with_params(
        self,
    ) -> Callable[[float | np.ndarray], float | np.ndarray]:
        """
        Creates a first order transfer function with the parameters of the fit.

        Returns
        -------
        function : Callable
            First order transfer function with the parameters of the fit.
        """
        return lambda x: self._gain * (1 - np.exp(-x / self._time_constant))
