from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import NoneType
from typing import Callable, Literal, Optional, Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, is_color_like, to_rgba, Colormap, to_rgba_array
from matplotlib.patches import Polygon
from numpy.typing import ArrayLike
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

from .graph_elements import Point

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


class Fit(Protocol):
    """
    Dummy class to allow type hinting of Fit objects.
    """

    def _plot_element(self, axes: plt.Axes, z_order: int) -> None:
        """
        Plots the element in the specified axes.
        """
        pass

    def show_residual_curves(
        self,
        sigma_multiplier: float,
        color: str,
        line_width: float,
        line_style: str,
    ) -> None:
        pass

    def get_residuals(self) -> np.ndarray:
        pass


@dataclass
class Curve:
    """
    This class implements a general continuous curve.

    Parameters
    ----------
    x_data, y_data : ArrayLike
        Arrays of x and y values to be plotted.
    label : str, optional
        Label to be displayed in the legend.
    color : str
        Color of the curve.
        Default depends on the ``figure_style`` configuration.
    line_width : float
        Width of the curve.
        Default depends on the ``figure_style`` configuration.
    line_style : str
        Style of the curve.
        Default depends on the ``figure_style`` configuration.
    """

    def __init__(
        self,
        x_data: ArrayLike,
        y_data: ArrayLike,
        label: Optional[str] = None,
        color: str = "default",
        line_width: float | Literal["default"] = "default",
        line_style: str = "default",
    ) -> None:
        self.handle = None
        self._x_data = np.asarray(x_data)
        self._y_data = np.asarray(y_data)
        self._label = label
        self._color = color
        self._line_width = line_width
        self._line_style = line_style

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

    @classmethod
    def from_function(
        cls,
        func: Callable[[ArrayLike], ArrayLike],
        x_min: float,
        x_max: float,
        label: Optional[str] = None,
        color: str = "default",
        line_width: float | Literal["default"] = "default",
        line_style: str = "default",
        number_of_points: int = 500,
    ) -> Self:
        """
        Creates a :class:`~graphinglib.data_plotting_1d.Curve` from a function and a range of x values.

        Parameters
        ----------
        func : Callable[[ArrayLike], ArrayLike]
            Function to be plotted. Works with regular functions and lambda functions.
        x_min, x_max : float
            The :class:`~graphinglib.data_plotting_1d.Curve` will be plotted between these two values.
        label : str, optional
            Label to be displayed in the legend.
        color : str
            Color of the curve.
            Default depends on the ``figure_style`` configuration.
        line_width : float
            Width of the curve.
            Default depends on the ``figure_style`` configuration.
        number_of_points : int
            Number of points to be used to plot the curve (resolution).
            Defaults to 500.

        Returns
        -------
        A :class:`~graphinglib.data_plotting_1d.Curve` object created from the given function and x range.
        """
        x_data = np.linspace(x_min, x_max, number_of_points)
        y_data = func(x_data)
        return cls(x_data, y_data, label, color, line_width, line_style)

    @property
    def x_data(self) -> np.ndarray:
        return self._x_data

    @x_data.setter
    def x_data(self, x_data: ArrayLike) -> None:
        self._x_data = np.asarray(x_data)

    @property
    def y_data(self) -> np.ndarray:
        return self._y_data

    @y_data.setter
    def y_data(self, y_data: ArrayLike) -> None:
        self._y_data = np.asarray(y_data)

    @property
    def label(self) -> Optional[str]:
        return self._label

    @label.setter
    def label(self, label: Optional[str]) -> None:
        self._label = label

    @property
    def color(self) -> str:
        return self._color

    @color.setter
    def color(self, color: str) -> None:
        self._color = color

    @property
    def line_width(self) -> float | Literal["default"]:
        return self._line_width

    @line_width.setter
    def line_width(self, line_width: float | Literal["default"]) -> None:
        self._line_width = line_width

    @property
    def line_style(self) -> str:
        return self._line_style

    @line_style.setter
    def line_style(self, line_style: str) -> None:
        self._line_style = line_style

    @property
    def show_errorbars(self) -> bool:
        return self._show_errorbars

    @show_errorbars.setter
    def show_errorbars(self, show_errorbars: bool) -> None:
        self._show_errorbars = show_errorbars

    @property
    def errorbars_color(self) -> str:
        return self._errorbars_color

    @errorbars_color.setter
    def errorbars_color(self, errorbars_color: str) -> None:
        self._errorbars_color = errorbars_color

    @property
    def errorbars_line_width(self) -> float | Literal["default"]:
        return self._errorbars_line_width

    @errorbars_line_width.setter
    def errorbars_line_width(
        self, errorbars_line_width: float | Literal["default"]
    ) -> None:
        self._errorbars_line_width = errorbars_line_width

    @property
    def cap_thickness(self) -> float | Literal["default"]:
        return self._cap_thickness

    @cap_thickness.setter
    def cap_thickness(self, cap_thickness: float | Literal["default"]) -> None:
        self._cap_thickness = cap_thickness

    @property
    def cap_width(self) -> float | Literal["default"]:
        return self._cap_width

    @cap_width.setter
    def cap_width(self, cap_width: float | Literal["default"]) -> None:
        self._cap_width = cap_width

    @property
    def show_error_curves(self) -> bool:
        return self._show_error_curves

    @show_error_curves.setter
    def show_error_curves(self, show_error_curves: bool) -> None:
        self._show_error_curves = show_error_curves

    @property
    def error_curves_fill_between(self) -> bool:
        return self._error_curves_fill_between

    @error_curves_fill_between.setter
    def error_curves_fill_between(self, error_curves_fill_between: bool) -> None:
        self._error_curves_fill_between = error_curves_fill_between

    @property
    def error_curves_color(self) -> str:
        return self._error_curves_color

    @error_curves_color.setter
    def error_curves_color(self, error_curves_color: str) -> None:
        self._error_curves_color = error_curves_color

    @property
    def error_curves_line_style(self) -> str:
        return self._error_curves_line_style

    @error_curves_line_style.setter
    def error_curves_line_style(self, error_curves_line_style: str) -> None:
        self._error_curves_line_style = error_curves_line_style

    @property
    def error_curves_line_width(self) -> float | Literal["default"]:
        return self._error_curves_line_width

    @error_curves_line_width.setter
    def error_curves_line_width(
        self, error_curves_line_width: float | Literal["default"]
    ) -> None:
        self._error_curves_line_width = error_curves_line_width

    @property
    def fill_between_bounds(self) -> tuple[float, float]:
        return self._fill_between_bounds

    @fill_between_bounds.setter
    def fill_between_bounds(self, fill_between_bounds: tuple[float, float]) -> None:
        self._fill_between_bounds = fill_between_bounds

    @property
    def fill_between_other_curve(self) -> Self:
        return self._fill_between_other_curve

    @fill_between_other_curve.setter
    def fill_between_other_curve(self, fill_between_other_curve: Self) -> None:
        self._fill_between_other_curve = fill_between_other_curve

    @property
    def fill_between_color(self) -> str:
        return self._fill_between_color

    @fill_between_color.setter
    def fill_between_color(self, fill_between_color: str) -> None:
        self._fill_between_color = fill_between_color

    def __add__(self, other: Self | float) -> Self:
        """
        Defines the addition of two curves or a curve and a number.
        """
        if isinstance(other, Curve):
            if not np.array_equal(self._x_data, other._x_data):
                if len(self._x_data) > len(other._x_data):
                    x_data = other._x_data
                    y_data = interp1d(self._x_data, self._y_data)(x_data)
                    return Curve(x_data, y_data + other._y_data)
                else:
                    x_data = self._x_data
                    y_data = interp1d(other._x_data, other._y_data)(x_data)
                    return Curve(x_data, y_data + self._y_data)

            new_y_data = self._y_data + other._y_data
            return Curve(self._x_data, new_y_data)
        elif isinstance(other, (int, float)):
            new_y_data = self._y_data + other
            return Curve(self._x_data, new_y_data)
        else:
            raise TypeError("Can only add a curve to another curve or a number.")

    def __radd__(self, other: Self | float) -> Self:
        return self.__add__(other)

    def __iadd__(self, other: Self | float) -> Self:
        if isinstance(other, Curve):
            if not np.array_equal(self._x_data, other._x_data):
                if len(self._x_data) > len(other._x_data):
                    x_data = other._x_data
                    y_data = interp1d(self._x_data, self._y_data)(x_data)
                    self._y_data = y_data + other._y_data
                    return self
                else:
                    x_data = self._x_data
                    y_data = interp1d(other._x_data, other._y_data)(x_data)
                    self._y_data = y_data + self._y_data
                    return self
            self._y_data += other._y_data
            return self
        elif isinstance(other, (int, float)):
            self._y_data += other
            return self

    def __sub__(self, other: Self | float) -> Self:
        """
        Defines the subtraction of two curves or a curve and a number.
        """
        if isinstance(other, Curve):
            if not np.array_equal(self._x_data, other._x_data):
                if len(self._x_data) > len(other._x_data):
                    x_data = other._x_data
                    y_data = interp1d(self._x_data, self._y_data)(x_data)
                    return Curve(x_data, y_data - other._y_data)
                else:
                    x_data = self._x_data
                    y_data = interp1d(other._x_data, other._y_data)(x_data)
                    return Curve(x_data, self._y_data - y_data)
            new_y_data = self._y_data - other._y_data
            return Curve(self._x_data, new_y_data)
        elif isinstance(other, (int, float)):
            new_y_data = self._y_data - other
            return Curve(self._x_data, new_y_data)
        else:
            raise TypeError("Can only subtract a curve from another curve or a number.")

    def __rsub__(self, other: Self | float) -> Self:
        return (self * -1) + other

    def __isub__(self, other: Self | float) -> Self:
        if isinstance(other, Curve):
            if not np.array_equal(self._x_data, other._x_data):
                if len(self._x_data) > len(other._x_data):
                    x_data = other._x_data
                    y_data = interp1d(self._x_data, self._y_data)(x_data)
                    self._y_data = y_data - other._y_data
                    return self
                else:
                    x_data = self._x_data
                    y_data = interp1d(other._x_data, other._y_data)(x_data)
                    self._y_data = self._y_data - y_data
                    return self
            self._y_data -= other._y_data
            return self
        elif isinstance(other, (int, float)):
            self._y_data -= other
            return self

    def __mul__(self, other: Self | float) -> Self:
        """
        Defines the multiplication of two curves or a curve and a number.
        """
        if isinstance(other, Curve):
            if not np.array_equal(self._x_data, other._x_data):
                if len(self._x_data) > len(other._x_data):
                    x_data = other._x_data
                    y_data = interp1d(self._x_data, self._y_data)(x_data)
                    return Curve(x_data, y_data * other._y_data)
                else:
                    x_data = self._x_data
                    y_data = interp1d(other._x_data, other._y_data)(x_data)
                    return Curve(x_data, y_data * self._y_data)
            new_y_data = self._y_data * other._y_data
            return Curve(self._x_data, new_y_data)
        elif isinstance(other, (int, float)):
            new_y_data = self._y_data * other
            return Curve(self._x_data, new_y_data)
        else:
            raise TypeError("Can only multiply a curve by another curve or a number.")

    def __rmul__(self, other: Self | float) -> Self:
        return self.__mul__(other)

    def __imul__(self, other: Self | float) -> Self:
        if isinstance(other, Curve):
            if not np.array_equal(self._x_data, other._x_data):
                if len(self._x_data) > len(other._x_data):
                    x_data = other._x_data
                    y_data = interp1d(self._x_data, self._y_data)(x_data)
                    self._y_data = y_data * other._y_data
                    return self
                else:
                    x_data = self._x_data
                    y_data = interp1d(other._x_data, other._y_data)(x_data)
                    self._y_data = self._y_data * y_data
                    return self
            self._y_data *= other._y_data
            return self
        elif isinstance(other, (int, float)):
            self._y_data *= other
            return self

    def __truediv__(self, other: Self | float) -> Self:
        """
        Defines the division of two curves or a curve and a number.
        """
        if isinstance(other, Curve):
            if not np.array_equal(self._x_data, other._x_data):
                if len(self._x_data) > len(other._x_data):
                    x_data = other._x_data
                    y_data = interp1d(self._x_data, self._y_data)(x_data)
                    return Curve(x_data, y_data / other._y_data)
                else:
                    x_data = self._x_data
                    y_data = interp1d(other._x_data, other._y_data)(x_data)
                    return Curve(x_data, self._y_data / y_data)
            new_y_data = self._y_data / other._y_data
            return Curve(self._x_data, new_y_data)
        elif isinstance(other, (int, float)):
            new_y_data = self._y_data / other
            return Curve(self._x_data, new_y_data)
        else:
            raise TypeError("Can only divide a curve by another curve or a number.")

    def __rtruediv__(self, other: Self | float) -> Self:
        try:
            return (self**-1) * other
        except ZeroDivisionError:
            raise ZeroDivisionError("Cannot divide by zero.")

    def __itruediv__(self, other: Self | float) -> Self:
        if isinstance(other, Curve):
            if not np.array_equal(self._x_data, other._x_data):
                if len(self._x_data) > len(other._x_data):
                    x_data = other._x_data
                    y_data = interp1d(self._x_data, self._y_data)(x_data)
                    self._y_data = y_data / other._y_data
                    return self
                else:
                    x_data = self._x_data
                    y_data = interp1d(other._x_data, other._y_data)(x_data)
                    self._y_data = self._y_data / y_data
                    return self
            self._y_data /= other._y_data
            return self
        elif isinstance(other, (int, float)):
            self._y_data /= other
            return self

    def __pow__(self, other: float) -> Self:
        """
        Defines the power of a curve to a number.
        """
        if isinstance(other, (int, float)):
            new_y_data = self._y_data**other
            return Curve(self._x_data, new_y_data)
        else:
            raise TypeError("Can only raise a curve to another curve or a number.")

    def __ipow__(self, other: float) -> Self:
        self._y_data **= other
        return self

    def __iter__(self):
        """
        Defines the iteration of a curve. Returns the y values.
        """
        return iter(self._y_data)

    def __abs__(self) -> Self:
        """
        Returns the absolute value of the curve.
        """
        return Curve(self._x_data, np.abs(self._y_data))

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.data_plotting_1d.Curve`.
        """
        return deepcopy(self)

    def create_slice_x(
        self,
        x1: float,
        x2: float,
        label: Optional[str] = None,
        color: str = "default",
        line_width: float | Literal["default"] = "default",
        line_style: str = "default",
        copy_first: bool = False,
    ) -> Self:
        """
        Creates a slice of the curve between two x values.

        Parameters
        ----------
        x1, x2 : float
            The x values between which the slice is to be created.
        label : str, optional
            Label of the slice to be displayed in the legend.
        color : str
            Color of the slice.
            Default depends on the ``figure_style`` configuration.
        line_width : float
            Width of the slice.
            Default depends on the ``figure_style`` configuration.
        line_style : str
            Style of the slice.
            Default depends on the ``figure_style`` configuration.
        copy_first : bool
            If ``True``, a copy of the curve (with all its parameters) will be returned with the slicing applied. Any other parameters passed to this method will also be applied to the copied curve. If ``False``, a new curve will be created with the slicing applied and the parameters passed to this method.

        Returns
        -------
        A :class:`~graphinglib.data_plotting_1d.Curve` object which is the slice of the original curve between the two x values.
        """
        mask = (self._x_data >= x1) & (self._x_data <= x2)
        x_data = self._x_data[mask]
        y_data = self._y_data[mask]
        if copy_first:
            copy = self.copy()
            copy._x_data = x_data
            copy._y_data = y_data
            if label is not None:
                copy._label = label
            if color != "default":
                copy._color = color
            if line_width != "default":
                copy._line_width = line_width
            if line_style != "default":
                copy._line_style = line_style
            return copy
        else:
            return Curve(x_data, y_data, label, color, line_width, line_style)

    def create_slice_y(
        self,
        y1: float,
        y2: float,
        label: Optional[str] = None,
        color: str = "default",
        line_width: float | Literal["default"] = "default",
        line_style: str = "default",
        copy_first: bool = False,
    ) -> Self:
        """
        Creates a slice of the curve between two y values.

        Parameters
        ----------
        y1, y2 : float
            The y values between which the slice is to be created.
        label : str, optional
            Label of the slice to be displayed in the legend.
        color : str
            Color of the slice.
            Default depends on the ``figure_style`` configuration.
        line_width : float
            Width of the slice.
            Default depends on the ``figure_style`` configuration.
        line_style : str
            Style of the slice.
            Default depends on the ``figure_style`` configuration.
        copy_first : bool
            If ``True``, a copy of the curve (with all its parameters) will be returned with the slicing applied. Any other parameters passed to this method will also be applied to the copied curve. If ``False``, a new curve will be created with the slicing applied and the parameters passed to this method.

        Returns
        -------
        A :class:`~graphinglib.data_plotting_1d.Curve` object which is the slice of the original curve between the two y values.
        """
        mask = (self._y_data >= y1) & (self._y_data <= y2)
        x_data = self._x_data[mask]
        y_data = self._y_data[mask]
        if copy_first:
            copy = self.copy()
            copy._x_data = x_data
            copy._y_data = y_data
            if label is not None:
                copy._label = label
            if color != "default":
                copy._color = color
            if line_width != "default":
                copy._line_width = line_width
            if line_style != "default":
                copy._line_style = line_style
            return copy
        else:
            return Curve(x_data, y_data, label, color, line_width, line_style)

    def add_errorbars(
        self,
        x_error: Optional[ArrayLike] = None,
        y_error: Optional[ArrayLike] = None,
        cap_width: float | Literal["default"] = "default",
        errorbars_color: str = "default",
        errorbars_line_width: float | Literal["default"] = "default",
        cap_thickness: float | Literal["default"] = "default",
    ) -> None:
        """
        Adds errorbars to the :class:`~graphinglib.data_plotting_1d.Curve`.

        Parameters
        ----------
        x_error, y_error : ArrayLike, optional
            Arrays of x and y errors. Use one or both.
        cap_width : float
            Width of the errorbar caps.
            Default depends on the ``figure_style`` configuration.
        errorbars_color : str
            Color of the errorbars.
            Default depends on the ``figure_style`` configuration.
        errorbars_line_width : float
            Width of the errorbars.
            Default depends on the ``figure_style`` configuration.
        cap_thickness : float
            Thickness of the errorbar caps.
            Default depends on the ``figure_style`` configuration.
        """
        self._show_errorbars = True
        self._x_error = np.array(x_error) if x_error is not None else x_error
        self._y_error = np.array(y_error) if y_error is not None else y_error
        self._errorbars_color = errorbars_color
        self._errorbars_line_width = errorbars_line_width
        self._cap_thickness = cap_thickness
        self._cap_width = cap_width

    def add_error_curves(
        self,
        y_error: Optional[ArrayLike] = None,
        error_curves_color: str = "default",
        error_curves_line_style: str = "default",
        error_curves_line_width: float | Literal["default"] = "default",
        error_curves_fill_between: bool | Literal["default"] = "default",
    ) -> None:
        """
        Adds error curves to the :class:`~graphinglib.data_plotting_1d.Curve`.

        Parameters
        ----------
        y_error : ArrayLike, optional
            Array of y errors.
        error_curves_color : str
            Color of the error curves.
            Default depends on the ``figure_style`` configuration.
        error_curves_line_style : str
            Line style of the error curves.
            Default depends on the ``figure_style`` configuration.
        error_curves_line_width : float
            Line width of the error curves.
            Default depends on the ``figure_style`` configuration.
        error_curves_fill_between : bool
            Whether or not to fill the area between the two error curves.
            Default depends on the ``figure_style`` configuration.
        """
        self._show_error_curves = True
        self._y_error = np.array(y_error) if y_error is not None else y_error
        self._error_curves_color = error_curves_color
        self._error_curves_line_style = error_curves_line_style
        self._error_curves_line_width = error_curves_line_width
        self._error_curves_fill_between = error_curves_fill_between

    def get_coordinates_at_x(
        self,
        x: float,
        interpolation_method: str = "linear",
    ) -> tuple[float, float]:
        """
        Gets the coordinates of the curve at a given x value.

        Parameters
        ----------
        x : float
            The x value of the desired coordinates.
        interpolation_method : str,
            The type of interpolation to be used, as defined in ``scipy.interpolate.interp1d``.

            .. seealso:: `scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_

            Defaults to "linear".

        Returns
        -------
        tuple[float, float]
            The coordinates of the curve at the given x value.
        """
        return (
            x,
            float(interp1d(self._x_data, self._y_data, kind=interpolation_method)(x)),
        )

    def create_point_at_x(
        self,
        x: float,
        interpolation_method: str = "linear",
        label: Optional[str] = None,
        color: str = "default",
        edge_color: str = "default",
        marker_size: float | Literal["default"] = "default",
        marker_style: str = "default",
        line_width: float | Literal["default"] = "default",
    ) -> Point:
        """
        Creates a point on the curve at a given x value.

        Parameters
        ----------
        x : float
            The x value of the point.
        interpolation_method : str,
            The type of interpolation to be used, as defined in ``scipy.interpolate.interp1d``.

            .. seealso:: `scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_

            Defaults to "linear".
        label : str, optional
            Point's label to be displayed in the legend.
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
            Width of the point edge.
            Default depends on the ``figure_style`` configuration.

        Returns
        -------
        :class:`~graphinglib.graph_elements.Point`
            The point on the curve at the given x value.
        """
        point = Point(
            x,
            self.get_coordinates_at_x(x, interpolation_method)[1],
            label=label,
            color=color,
            edge_color=edge_color,
            marker_size=marker_size,
            marker_style=marker_style,
            edge_width=line_width,
        )
        return point

    def get_coordinates_at_y(
        self,
        y: float,
        interpolation_method: str = "linear",
    ) -> list[tuple[float, float]]:
        """
        Gets the coordinates of the curve at a given y value. Can return multiple coordinate pairs if the curve crosses the y value multiple times.

        Parameters
        ----------
        y : float
            The y value of the desired coordinates.
        interpolation_method : str,
            The type of interpolation to be used, as defined in ``scipy.interpolate.interp1d``.

            .. seealso:: `scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_

            Defaults to "linear".

        Returns
        -------
        list[tuple[float, float]]
            The coordinates of the points on the curve at the given y value.
        """
        xs = self._x_data
        ys = self._y_data
        crossings = np.where(np.diff(np.sign(ys - y)))[0]
        x_vals: list[float] = []
        for cross in crossings:
            x1, x2 = xs[cross], xs[cross + 1]
            y1, y2 = ys[cross], ys[cross + 1]
            f = interp1d([y1, y2], [x1, x2], kind=interpolation_method)
            x_val = f(y)
            x_vals.append(float(x_val))
        points = [(x_val, y) for x_val in x_vals]
        return points

    def create_points_at_y(
        self,
        y: float,
        interpolation_method: str = "linear",
        label: str | None = None,
        color: str = "default",
        edge_color: str = "default",
        marker_size: float | Literal["default"] = "default",
        marker_style: str = "default",
        line_width: float | Literal["default"] = "default",
    ) -> list[Point]:
        """
        Gets the points on the curve at a given y value. Can return multiple Point objects if the curve crosses the y value multiple times.

        Parameters
        ----------
        y : float
            The y value of the desired points.
        interpolation_method : str
            The type of interpolation to be used, as defined in ``scipy.interpolate.interp1d``.

            .. seealso:: `scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_

            Defaults to "linear".
        label : str, optional
            Point label to be displayed in the legend.
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
            Width of the point edge.
            Default depends on the ``figure_style`` configuration.

        Returns
        -------
        list[:class:`~graphinglib.graph_elements.Point`]
            The Point objects on the curve at the given y value.
        """
        pairs = self.get_coordinates_at_y(y, interpolation_method)
        points = [
            Point(
                pair[0],
                pair[1],
                label=label,
                color=color,
                edge_color=edge_color,
                marker_size=marker_size,
                marker_style=marker_style,
                edge_width=line_width,
            )
            for pair in pairs
        ]
        return points

    def create_derivative_curve(
        self,
        label: Optional[str] = None,
        color: str = "default",
        line_width: float | Literal["default"] = "default",
        line_style: str = "default",
        copy_first: bool = False,
    ) -> Self:
        """
        Creates a new curve which is the derivative of the original curve.

        Parameters
        ----------
        label : str, optional
            Label of the new curve to be displayed in the legend.
        color : str
            Color of the new curve.
            Default depends on the ``figure_style`` configuration.
        line_width : float
            Width of the new curve.
            Default depends on the ``figure_style`` configuration.
        line_style : str
            Style of the new curve.
            Default depends on the ``figure_style`` configuration.
        copy_first : bool
            If ``True``, a copy of the curve (with all its parameters) will be returned with the derivative applied. Any other parameters passed to this method will also be applied to the copied curve. If ``False``, a new curve will be created with the derivative applied and the parameters passed to this method.

        Returns
        -------
        A :class:`~graphinglib.data_plotting_1d.Curve` object which is the derivative of the original curve.
        """
        x_data = self._x_data
        y_data = np.gradient(self._y_data, x_data)
        if copy_first:
            copy = self.copy()
            copy._y_data = y_data
            if label is not None:
                copy._label = label
            if color != "default":
                copy._color = color
            if line_width != "default":
                copy._line_width = line_width
            if line_style != "default":
                copy._line_style = line_style
            return copy
        else:
            return Curve(x_data, y_data, label, color, line_width, line_style)

    def create_integral_curve(
        self,
        initial_value: float = 0,
        label: Optional[str] = None,
        color: str = "default",
        line_width: float | Literal["default"] = "default",
        line_style: str = "default",
        copy_first: bool = False,
    ) -> Self:
        """
        Creates a new curve which is the integral of the original curve.

        Parameters
        ----------
        initial_value : float, optional
            The value of the integral at the first x value (initial condition).
            Defaults to 0.
        label : str, optional
            Label of the new curve to be displayed in the legend.
        color : str
            Color of the new curve.
            Default depends on the ``figure_style`` configuration.
        line_width : float
            Width of the new curve.
            Default depends on the ``figure_style`` configuration.
        line_style : str
            Style of the new curve.
            Default depends on the ``figure_style`` configuration.
        copy_first : bool
            If ``True``, a copy of the curve (with all its parameters) will be returned with the integral applied. Any other parameters passed to this method will also be applied to the copied curve. If ``False``, a new curve will be created with the integral applied and the parameters passed to this method.

        Returns
        -------
        A :class:`~graphinglib.data_plotting_1d.Curve` object which is the integral of the original curve.
        """
        # calculate the integral curve using cumulative trapezoidal integration
        y_data = (
            cumulative_trapezoid(self._y_data, self._x_data, initial=0) + initial_value
        )
        if copy_first:
            copy = self.copy()
            copy._y_data = y_data
            if label is not None:
                copy._label = label
            if color != "default":
                copy._color = color
            if line_width != "default":
                copy._line_width = line_width
            if line_style != "default":
                copy._line_style = line_style
            return copy
        else:
            return Curve(self._x_data, y_data, label, color, line_width, line_style)

    def create_tangent_curve(
        self,
        x: float,
        label: Optional[str] = None,
        color: str = "default",
        line_width: float | Literal["default"] = "default",
        line_style: str = "default",
        copy_first: bool = False,
    ) -> Self:
        """
        Creates a new curve which is the tangent to the original curve at a given x value.

        Parameters
        ----------
        x : float
            The x value at which the tangent is to be calculated.
        label : str, optional
            Label of the new curve to be displayed in the legend.
        color : str
            Color of the new curve.
            Default depends on the ``figure_style`` configuration.
        line_width : float
            Width of the new curve.
            Default depends on the ``figure_style`` configuration.
        line_style : str
            Style of the new curve.
            Default depends on the ``figure_style`` configuration.
        copy_first : bool
            If ``True``, a copy of the curve (with all its parameters) will be returned with the tangent applied. Any other parameters passed to this method will also be applied to the copied curve. If ``False``, a new curve will be created with the tangent applied and the parameters passed to this method.

        Returns
        -------
        :class:`~graphinglib.data_plotting_1d.Curve`
            A :class:`~graphinglib.data_plotting_1d.Curve` object which is the tangent to the original curve at a given x value.
        """
        point = self.get_coordinates_at_x(x)
        gradient = self.create_derivative_curve().get_coordinates_at_x(x)[1]
        y_data = gradient * (self._x_data - x) + point[1]
        if copy_first:
            copy = self.copy()
            copy._y_data = y_data
            if label is not None:
                copy._label = label
            if color != "default":
                copy._color = color
            if line_width != "default":
                copy._line_width = line_width
            if line_style != "default":
                copy._line_style = line_style
            return copy
        else:
            tangent_curve = Curve(
                self._x_data, y_data, label, color, line_width, line_style
            )
            return tangent_curve

    def create_normal_curve(
        self,
        x: float,
        label: Optional[str] = None,
        color: str = "default",
        line_width: float | Literal["default"] = "default",
        line_style: str = "default",
        copy_first: bool = False,
    ) -> Self:
        """
        Creates a new curve which is the normal to the original curve at a given x value.

        Parameters
        ----------
        x : float
            The x value at which the normal is to be calculated.
        label : str, optional
            Label of the new curve to be displayed in the legend.
        color : str
            Color of the new curve.
            Default depends on the ``figure_style`` configuration.
        line_width : float
            Width of the new curve.
            Default depends on the ``figure_style`` configuration.
        line_style : str
            Style of the new curve.
            Default depends on the ``figure_style`` configuration.
        copy_first : bool
            If ``True``, a copy of the curve (with all its parameters) will be returned with the normal applied. Any other parameters passed to this method will also be applied to the copied curve. If ``False``, a new curve will be created with the normal applied and the parameters passed to this method.

        Returns
        -------
        :class:`~graphinglib.data_plotting_1d.Curve`
            A :class:`~graphinglib.data_plotting_1d.Curve` object which is the normal to the original curve at a given x value.
        """
        point = self.get_coordinates_at_x(x)
        gradient = self.create_derivative_curve().get_coordinates_at_x(x)[1]
        y_data = -1 / gradient * (self._x_data - x) + point[1]
        if copy_first:
            copy = self.copy()
            copy._y_data = y_data
            if label is not None:
                copy._label = label
            if color != "default":
                copy._color = color
            if line_width != "default":
                copy._line_width = line_width
            if line_style != "default":
                copy._line_style = line_style
            return copy
        else:
            normal_curve = Curve(
                self._x_data, y_data, label, color, line_width, line_style
            )
            return normal_curve

    def get_slope_at(self, x: float) -> float:
        """
        Calculates the slope of the curve at a given x value.

        Parameters
        ----------
        x : float
            The x value at which the slope is to be calculated.

        Returns
        -------
        The slope of the curve (float) at the given x value.
        """
        return self.create_derivative_curve().get_coordinates_at_x(x)[1]

    def get_arc_length_between(self, x1: float, x2: float) -> float:
        """
        Calculates the arc length of the curve between two x values.

        Parameters
        ----------
        x1, x2 : float
            The x values between which the arc length is to be calculated.

        Returns
        -------
        The arc length of the curve (float) between the two given x values.
        """
        y_data = self._y_data
        x_data = self._x_data
        # f = interp1d(x_data, y_data)
        # x = np.linspace(x1, x2, 1000)
        # y = f(x)
        x = x_data[(x_data >= x1) & (x_data <= x2)]
        y = y_data[(x_data >= x1) & (x_data <= x2)]
        return np.trapz(np.sqrt(1 + np.gradient(y, x) ** 2), x)

    def get_area_between(
        self,
        x1: float,
        x2: float,
        fill_between: bool = False,
        fill_color: str = "default",
        other_curve: Optional[Self] = None,
    ) -> float:
        """
        Calculates the area between the curve and the x axis between two x values.
        This is the definite integral of the curve between the two x values.

        Parameters
        ----------
        x1, x2 : float
            The x values between which the area is to be calculated.
        fill_between : bool
            Whether to fill the specified area between the curve and the x axis when displaying.
            Defaults to ``False``.
        fill_color : str
            Color of the area between the curve and the x axis when ``fill_between`` is set to ``True``.
            Default depends on the ``figure_style`` configuration.
        other_curve : :class:`~graphinglib.data_plotting_1d.Curve`, optional
            If specified, the area between the two curves will be calculated instead of the area between the curve and the x axis.

        Returns
        -------
        The area (float) between the curve and the x axis (or between the two curves) between the two given x values.
        """
        if other_curve is None:
            if fill_between:
                self._fill_between_bounds = (x1, x2)
                self._fill_between_color = fill_color
            y_data = self._y_data
            x_data = self._x_data
            mask = (x_data >= x1) & (x_data <= x2)
            y = y_data[mask]
            x = x_data[mask]
            return np.trapz(y, x)
        else:
            if fill_between:
                self._fill_between_bounds = (x1, x2)
                self._fill_between_color = fill_color
                self._fill_between_other_curve = other_curve
            if np.array_equal(self._x_data, other_curve._x_data):
                # No need to interpolate
                mask = (self._x_data >= x1) & (self._x_data <= x2)
                common_x = self._x_data[mask]
                y1 = self._y_data[mask]
                y2 = other_curve._y_data[mask]
            else:
                # Interpolate to get common x values
                density_x1 = len(self._x_data) / (self._x_data[-1] - self._x_data[0])
                density_x2 = len(other_curve._x_data) / (
                    other_curve._x_data[-1] - other_curve._x_data[0]
                )
                higher_density = max(density_x1, density_x2)
                num_of_values = int(np.ceil(higher_density * (x2 - x1)))
                common_x = np.linspace(x1, x2, num_of_values)
                y1 = np.interp(common_x, self._x_data, self._y_data)
                y2 = np.interp(common_x, other_curve._x_data, other_curve._y_data)

            difference = y1 - y2
            area = np.trapz(difference, common_x)
            return area

    def get_intersection_coordinates(
        self,
        other: Self,
    ) -> list[tuple[float, float]]:
        """
        Calculates the coordinates of the intersection points between two curves.

        Parameters
        ----------
        other : :class:`~graphinglib.data_plotting_1d.Curve`
            The other curve to calculate the intersections with.

        Returns
        -------
        list[tuple[float, float]]
            A list of tuples of coordinates which are the intersection points between the two curves.
        """
        y = self._y_data - other._y_data
        s = np.abs(np.diff(np.sign(y))).astype(bool)
        intersections_x = self._x_data[:-1][s] + np.diff(self._x_data)[s] / (
            np.abs(y[1:][s] / y[:-1][s]) + 1
        )
        intersections_y = np.interp(intersections_x, self._x_data, self._y_data)
        points = []
        for i in range(len(intersections_x)):
            x_val = intersections_x[i]
            y_val = intersections_y[i]
            points.append((x_val, y_val))
        return points

    def create_intersection_points(
        self,
        other: Self,
        labels: Optional[list[str] | str] = None,
        colors: list[str] | str = "default",
        edge_colors: list[str] | str = "default",
        marker_sizes: list[float] | float | Literal["default"] = "default",
        marker_styles: list[str] | str = "default",
        edge_widths: list[float] | float | Literal["default"] = "default",
    ) -> list[Point]:
        """
        Creates the intersection Points between two curves.

        Parameters
        ----------
        other : :class:`~graphinglib.data_plotting_1d.Curve`
            The other curve to calculate the intersections with.
        as_point_objects : bool
            Whether to return a list of :class:`~graphinglib.graph_elements.Point` objects (True) or a list of tuples of coordinates (False).
            Defaults to False.
        labels : list[str] or str, optional
            Labels of the intersection points to be displayed in the legend.
            If a single string is passed, all intersection points will have the same label.
        colors : list[str] or str
            Face colors of the intersection points.
            If a single string is passed, all intersection points will have the same color.
            Default depends on the ``figure_style`` configuration.
        edge_colors : list[str] or str
            Edge colors of the intersection points.
            If a single string is passed, all intersection points will have the same color.
            Default depends on the ``figure_style`` configuration.
        marker_sizes : list[float] or float
            Sizes of the intersection points.
            If a single float is passed, all intersection points will have the same size.
            Default depends on the ``figure_style`` configuration.
        marker_styles : list[str] or str
            Styles of the intersection points.
            If a single string is passed, all intersection points will have the same style.
            Default depends on the ``figure_style`` configuration.
        edge_widths : list[float] or float
            Widths of the intersection points.
            If a single float is passed, all intersection points will have the same width.
            Default depends on the ``figure_style`` configuration.

        Returns
        -------
        list[:class:`~graphinglib.graph_elements.Point`] or list[tuple[float, float]]
            A list of :class:`~graphinglib.graph_elements.Point` objects which are the intersection points between the two curves.
        """
        y = self._y_data - other._y_data
        s = np.abs(np.diff(np.sign(y))).astype(bool)
        intersections_x = self._x_data[:-1][s] + np.diff(self._x_data)[s] / (
            np.abs(y[1:][s] / y[:-1][s]) + 1
        )
        point_coords = self.get_intersection_coordinates(other)
        point_objects = []
        for i in range(len(intersections_x)):
            try:
                assert isinstance(labels, list)
                label = labels[i]
            except (IndexError, TypeError, AssertionError):
                label = labels
            try:
                assert isinstance(colors, list)
                color = colors[i]
            except (IndexError, TypeError, AssertionError):
                color = colors
            try:
                assert isinstance(edge_colors, list)
                edge_color = edge_colors[i]
            except (IndexError, TypeError, AssertionError):
                edge_color = edge_colors
            try:
                assert isinstance(marker_sizes, list)
                marker_size = marker_sizes[i]
            except (IndexError, TypeError, AssertionError):
                marker_size = marker_sizes
            try:
                assert isinstance(marker_styles, list)
                marker_style = marker_styles[i]
            except (IndexError, TypeError, AssertionError):
                marker_style = marker_styles
            try:
                assert isinstance(edge_widths, list)
                edge_width = edge_widths[i]
            except (IndexError, TypeError, AssertionError):
                edge_width = edge_widths
            point = point_coords[i]
            point_objects.append(
                Point(
                    point[0],
                    point[1],
                    label=label,
                    color=color,
                    edge_color=edge_color,
                    marker_size=marker_size,
                    marker_style=marker_style,
                    edge_width=edge_width,
                )
            )
        return point_objects

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified axes.
        """
        if self._show_errorbars:
            params = {
                "color": self._color,
                "linewidth": self._line_width,
                "linestyle": self._line_style,
                "elinewidth": (
                    self._errorbars_line_width
                    if self._errorbars_line_width != "same as curve"
                    else self._line_width
                ),
                "capsize": self._cap_width,
                "capthick": (
                    self._cap_thickness
                    if self._cap_thickness != "same as curve"
                    else self._line_width
                ),
                "ecolor": (
                    self._errorbars_color
                    if self._errorbars_color != "same as curve"
                    else self._color
                ),
            }
            params = {k: v for k, v in params.items() if v != "default"}
            self.handle = axes.errorbar(
                self._x_data,
                self._y_data,
                xerr=self._x_error,
                yerr=self._y_error,
                label=self._label,
                zorder=z_order,
                **params,
            )
        else:
            params = {
                "color": self._color,
                "linewidth": self._line_width,
                "linestyle": self._line_style,
            }
            params = {k: v for k, v in params.items() if v != "default"}
            self.handle = axes.errorbar(
                self._x_data,
                self._y_data,
                label=self._label,
                zorder=z_order,
                **params,
            )
        if self._show_error_curves:
            max_y = (
                self._y_data + self._y_error
                if self._y_error is not None
                else self._y_data
            )
            min_y = (
                self._y_data - self._y_error
                if self._y_error is not None
                else self._y_data
            )

            params = {
                "color": (
                    self._error_curves_color
                    if self._error_curves_color != "same as curve"
                    else self.handle[0].get_color()
                ),
                "linestyle": (
                    self._error_curves_line_style
                    if self._error_curves_line_style != "same as curve"
                    else self._line_style
                ),
                "linewidth": (
                    self._error_curves_line_width
                    if self._error_curves_line_width != "same as curve"
                    else self._line_width
                ),
            }

            params = {k: v for k, v in params.items() if v != "default"}

            axes.plot(
                self._x_data,
                min_y,
                **params,
            )
            axes.plot(
                self._x_data,
                max_y,
                **params,
            )
            if self._error_curves_fill_between:
                axes.fill_between(
                    self._x_data,
                    max_y,
                    min_y,
                    color=self.handle[0].get_color(),
                    alpha=0.2,
                )
        if self._fill_between_bounds:
            params = {"alpha": 0.2}
            params["color"] = (
                self._fill_between_color
                if self._fill_between_color != "same as curve"
                else self.handle[0].get_color()
            )
            params = {k: v for k, v in params.items() if v != "default"}
            if self._fill_between_other_curve:
                self_y_data = self._y_data
                self_x_data = self._x_data
                other_y_data = self._fill_between_other_curve._y_data
                other_x_data = self._fill_between_other_curve._x_data
                x_data = np.linspace(
                    self._fill_between_bounds[0],
                    self._fill_between_bounds[1],
                    max(len(self_x_data), len(other_x_data)),
                )
                self_y_data = interp1d(self_x_data, self_y_data)(x_data)
                other_y_data = interp1d(other_x_data, other_y_data)(x_data)
                params["x"] = x_data
                params["y1"] = self_y_data
                params["y2"] = other_y_data
                where_x_data = x_data
            else:
                params["x"] = self._x_data
                params["y1"] = self._y_data
                where_x_data = self._x_data

            axes.fill_between(
                where=np.logical_and(
                    where_x_data >= self._fill_between_bounds[0],
                    where_x_data <= self._fill_between_bounds[1],
                ),
                zorder=z_order - 2,
                **params,
            )


@dataclass
class Scatter:
    """
    This class implements a general scatter plot.

    Parameters
    ----------
    x_data, y_data : ArrayLike
        Arrays of x and y values to be plotted.
    label : str, optional
        Label to be displayed in the legend.
    face_color : str or ArrayLike or None
        Face color of the points. If an array of intensities is provided, the values are mapped to the specified color
        map. If None, marker faces are transparent.
        Default depends on the ``figure_style`` configuration.
    edge_color : str or ArrayLike or None
        Edge color of the points. If an array of intensities is provided, the values are mapped to the specified color
        map. If None, marker edges are transparent.
        Default depends on the ``figure_style`` configuration.
    color_map : str or Colormap
        Color map of the stream lines, to be used in combination with the color parameter to specify intensity.
        Default depends on the ``figure_style`` configuration.
    show_color_bar : bool
        Whether or not to display the color bar next to the plot.
        Default depends on the ``figure_style`` configuration.
    marker_size : float
        Size of the points.
        Default depends on the ``figure_style`` configuration.
    marker_style : str
        Style of the points.
        Default depends on the ``figure_style`` configuration.
    """

    def __init__(
        self,
        x_data: ArrayLike,
        y_data: ArrayLike,
        label: Optional[str] = None,
        face_color: str | ArrayLike | NoneType = "default",
        edge_color: str | ArrayLike | NoneType = "default",
        color_map: str | Colormap = "default",
        show_color_bar: bool | Literal["default"] = "default",
        marker_size: float | Literal["default"] = "default",
        marker_style: str = "default",
    ) -> None:
        """
        This class implements a general scatter plot.

        Parameters
        ----------
        x_data, y_data : ArrayLike
            Arrays of x and y values to be plotted.
        label : str, optional
            Label to be displayed in the legend.
        face_color : str or ArrayLike or None
            Face color of the points. If an array of intensities is provided, the values are mapped to the specified
            color map. If None, marker faces are transparent.
            Default depends on the ``figure_style`` configuration.
        edge_color : str or ArrayLike or None
            Edge color of the points. If an array of intensities is provided, the values are mapped to the specified
            color map. If None, marker edges are transparent.
            Default depends on the ``figure_style`` configuration.
        color_map : str or Colormap
            Color map of the stream lines, to be used in combination with the color parameter to specify intensity.
            Default depends on the ``figure_style`` configuration.
        show_color_bar : bool
            Whether or not to display the color bar next to the plot.
            Default depends on the ``figure_style`` configuration.
        marker_size : float
            Size of the points.
            Default depends on the ``figure_style`` configuration.
        marker_style : str
            Style of the points.
            Default depends on the ``figure_style`` configuration.
        """
        self.handle = None
        self.errorbars_handle = None
        self._x_data = np.asarray(x_data)
        self._y_data = np.asarray(y_data)
        self._label = label
        self._face_color = face_color
        self._edge_color = edge_color
        self._color_map = color_map
        self._show_color_bar = show_color_bar
        self._marker_size = marker_size
        self._marker_style = marker_style

        self._show_errorbars: bool = False
        self._errorbars_line_width: float = 1.0
        self._cap_width: float = 3.0
        self._cap_thickness: float = 1.0
        self._errorbars_color: Optional[str] = None

    @classmethod
    def from_function(
        cls,
        func: Callable[[ArrayLike], ArrayLike],
        x_min: float,
        x_max: float,
        label: Optional[str] = None,
        face_color: str | ArrayLike | NoneType = "default",
        edge_color: str | ArrayLike | NoneType = "default",
        color_map: str | Colormap = "default",
        show_color_bar: bool | Literal["default"] = "default",
        marker_size: int | Literal["default"] = "default",
        marker_style: str = "default",
        number_of_points: int = 30,
    ) -> Self:
        """
        Creates a scatter plot from a function and a range of x values.

        Parameters
        ----------
        func : Callable[[ArrayLike], ArrayLike]
            The function to be plotted.
        x_min, x_max : float
            The scatter plot will be created for x values between x_min and x_max.
        label : str, optional
            Label to be displayed in the legend.
        face_color : str or ArrayLike or None
            Face color of the points. If an array of intensities is provided, the values are mapped to the specified
            color map. If None, marker faces are transparent.
        edge_color : str or ArrayLike or None
            Edge color of the points. If an array of intensities is provided, the values are mapped to the specified
            color map. If None, marker edges are transparent.
            Default depends on the ``figure_style`` configuration.
        color_map : str or Colormap
            Color map of the stream lines, to be used in combination with the color parameter to specify intensity.
            Default depends on the ``figure_style`` configuration.
        show_color_bar : bool
            Whether or not to display the color bar next to the plot.
            Default depends on the ``figure_style`` configuration.
        marker_size : int
            Size of the points.
            Default depends on the ``figure_style`` configuration.
        marker_style : str
            Style of the points.
            Default depends on the ``figure_style`` configuration.
        number_of_points : int
            Number of points to be plotted.
            Defaults to 30.

        Returns
        -------
        A :class:`~graphinglib.data_plotting_1d.Scatter` object created from a function and a range of x values.
        """
        x_data = np.linspace(x_min, x_max, number_of_points)
        y_data = func(x_data)
        return cls(
            x_data,
            y_data,
            label,
            face_color,
            edge_color,
            color_map,
            show_color_bar,
            marker_size,
            marker_style,
        )

    @property
    def x_data(self) -> np.ndarray:
        return self._x_data

    @x_data.setter
    def x_data(self, x_data: ArrayLike) -> None:
        self._x_data = np.asarray(x_data)

    @property
    def y_data(self) -> np.ndarray:
        return self._y_data

    @y_data.setter
    def y_data(self, y_data: ArrayLike) -> None:
        self._y_data = np.asarray(y_data)

    @property
    def label(self) -> str | None:
        return self._label

    @label.setter
    def label(self, label: str | None) -> None:
        self._label = label

    @property
    def face_color(self) -> str | ArrayLike:
        return self._face_color

    @face_color.setter
    def face_color(self, face_color: str | ArrayLike) -> None:
        self._face_color = face_color

    @property
    def edge_color(self) -> str:
        return self._edge_color

    @edge_color.setter
    def edge_color(self, edge_color: str) -> None:
        self._edge_color = edge_color

    @property
    def color_map(self) -> str | Colormap:
        return self._color_map

    @color_map.setter
    def color_map(self, color_map: str | Colormap) -> None:
        self._color_map = color_map

    @property
    def show_color_bar(self) -> bool:
        return self._show_color_bar

    @show_color_bar.setter
    def show_color_bar(self, show_color_bar: bool) -> None:
        self._show_color_bar = show_color_bar

    @property
    def marker_size(self) -> float | Literal["default"]:
        return self._marker_size

    @marker_size.setter
    def marker_size(self, marker_size: float | Literal["default"]) -> None:
        self._marker_size = marker_size

    @property
    def marker_style(self) -> str:
        return self._marker_style

    @marker_style.setter
    def marker_style(self, marker_style: str) -> None:
        self._marker_style = marker_style

    @property
    def show_errorbars(self) -> bool:
        return self._show_errorbars

    @show_errorbars.setter
    def show_errorbars(self, show_errorbars: bool) -> None:
        self._show_errorbars = show_errorbars

    @property
    def errorbars_line_width(self) -> float:
        return self._errorbars_line_width

    @errorbars_line_width.setter
    def errorbars_line_width(self, errorbars_line_width: float) -> None:
        self._errorbars_line_width = errorbars_line_width

    @property
    def cap_width(self) -> float:
        return self._cap_width

    @cap_width.setter
    def cap_width(self, cap_width: float) -> None:
        self._cap_width = cap_width

    @property
    def cap_thickness(self) -> float:
        return self._cap_thickness

    @cap_thickness.setter
    def cap_thickness(self, cap_thickness: float) -> None:
        self._cap_thickness = cap_thickness

    @property
    def errorbars_color(self) -> str:
        return self._errorbars_color

    @errorbars_color.setter
    def errorbars_color(self, errorbars_color: str) -> None:
        self._errorbars_color = errorbars_color

    def __add__(self, other: Self | float) -> Self:
        """
        Defines the addition of two scatter plots or a scatter plot and a number.
        """
        if isinstance(other, Scatter):
            try:
                assert np.array_equal(self._x_data, other._x_data)
            except AssertionError:
                raise ValueError(
                    "Cannot add two scatter plots with different x values."
                )
            new_y_data = self._y_data + other._y_data
            return Scatter(self._x_data, new_y_data)
        elif isinstance(other, (int, float)):
            new_y_data = self._y_data + other
            return Scatter(self._x_data, new_y_data)
        else:
            raise TypeError(
                "Can only add a scatter plot to another scatter plot or a number."
            )

    def __radd__(self, other: Self | float) -> Self:
        """
        Defines the reverse addition of a scatter plot and a number.
        """
        return self.__add__(other)

    def __iadd__(self, other: Self | float) -> Self:
        if isinstance(other, Scatter):
            try:
                assert np.array_equal(self._x_data, other._x_data)
            except AssertionError:
                raise ValueError(
                    "Cannot add two scatter plots with different x values."
                )
            self._y_data += other._y_data
            return self
        elif isinstance(other, (int, float)):
            self._y_data += other
            return self

    def __sub__(self, other: Self | float) -> Self:
        """
        Defines the subtraction of two scatter plots or a scatter plot and a number.
        """
        if isinstance(other, Scatter):
            try:
                assert np.array_equal(self._x_data, other._x_data)
            except AssertionError:
                raise ValueError(
                    "Cannot subtract two scatter plots with different x values."
                )
            new_y_data = self._y_data - other._y_data
            return Scatter(self._x_data, new_y_data)
        elif isinstance(other, (int, float)):
            new_y_data = self._y_data - other
            return Scatter(self._x_data, new_y_data)
        else:
            raise TypeError(
                "Can only subtract a scatter plot from another scatter plot or a number."
            )

    def __rsub__(self, other: Self | float) -> Self:
        """
        Defines the reverse subtraction of a scatter plot and a number.
        """
        return (self * -1) + other

    def __isub__(self, other: Self | float) -> Self:
        if isinstance(other, Scatter):
            try:
                assert np.array_equal(self._x_data, other._x_data)
            except AssertionError:
                raise ValueError(
                    "Cannot subtract two scatter plots with different x values."
                )
            self._y_data -= other._y_data
            return self
        elif isinstance(other, (int, float)):
            self._y_data -= other
            return self

    def __mul__(self, other: Self | float) -> Self:
        """
        Defines the multiplication of two scatter plots or a scatter plot and a number.
        """
        if isinstance(other, Scatter):
            try:
                assert np.array_equal(self._x_data, other._x_data)
            except AssertionError:
                raise ValueError(
                    "Cannot multiply two scatter plots with different x values."
                )
            new_y_data = self._y_data * other._y_data
            return Scatter(self._x_data, new_y_data)
        elif isinstance(other, (int, float)):
            new_y_data = self._y_data * other
            return Scatter(self._x_data, new_y_data)
        else:
            raise TypeError(
                "Can only multiply a scatter plot by another scatter plot or a number."
            )

    def __rmul__(self, other: Self | float) -> Self:
        """
        Defines the reverse multiplication of a scatter plot and a number.
        """
        return self.__mul__(other)

    def __imul__(self, other: Self | float) -> Self:
        if isinstance(other, Scatter):
            try:
                assert np.array_equal(self._x_data, other._x_data)
            except AssertionError:
                raise ValueError(
                    "Cannot multiply two scatter plots with different x values."
                )
            self._y_data *= other._y_data
            return self
        elif isinstance(other, (int, float)):
            self._y_data *= other
            return self

    def __truediv__(self, other: Self | float) -> Self:
        """
        Defines the division of two scatter plots or a scatter plot and a number.
        """
        if isinstance(other, Scatter):
            try:
                assert np.array_equal(self._x_data, other._x_data)
            except AssertionError:
                raise ValueError(
                    "Cannot divide two scatter plots with different x values."
                )
            new_y_data = self._y_data / other._y_data
            return Scatter(self._x_data, new_y_data)
        elif isinstance(other, (int, float)):
            new_y_data = self._y_data / other
            return Scatter(self._x_data, new_y_data)
        else:
            raise TypeError(
                "Can only divide a scatter plot by another scatter plot or a number."
            )

    def __rtruediv__(self, other: Self | float) -> Self:
        """
        Defines the division of two scatter plots or a scatter plot and a number.
        """
        try:
            return (self**-1) * other
        except ZeroDivisionError:
            raise ZeroDivisionError("Cannot divide by zero.")

    def __itruediv__(self, other: Self | float) -> Self:
        if isinstance(other, Scatter):
            try:
                assert np.array_equal(self._x_data, other._x_data)
            except AssertionError:
                raise ValueError(
                    "Cannot divide two scatter plots with different x values."
                )
            self._y_data /= other._y_data
            return self
        elif isinstance(other, (int, float)):
            self._y_data /= other
            return self

    def __pow__(self, other: float) -> Self:
        """
        Defines the power of a scatter plot to a number.
        """
        if isinstance(other, (int, float)):
            new_y_data = self._y_data**other
            return Scatter(self._x_data, new_y_data)
        else:
            raise TypeError(
                "Can only raise a scatter plot to another scatter plot or a number."
            )

    def __ipow__(self, other: float) -> Self:
        self._y_data **= other
        return self

    def __iter__(self):
        """
        Defines the iteration of a scatter plot. Returns the y values.
        """
        return iter(self._y_data)

    def __abs__(self) -> Self:
        """
        Defines the absolute value of a scatter plot.
        """
        new_y_data = np.abs(self._y_data)
        return Scatter(self._x_data, new_y_data)

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.data_plotting_1d.Scatter` object.
        """
        return deepcopy(self)

    def create_slice_x(
        self,
        x_min: float,
        x_max: float,
        label: Optional[str] = None,
        face_color: str | ArrayLike | NoneType = "default",
        edge_color: str | ArrayLike | NoneType = "default",
        color_map: str | Colormap = "default",
        show_color_bar: bool | Literal["default"] = "default",
        marker_size: float | Literal["default"] = "default",
        marker_style: str = "default",
        copy_first: bool = False,
    ) -> Self:
        """
        Creates a slice of the scatter plot between two x values.

        Parameters
        ----------
        x_min, x_max : float
            The slice will be created between x_min and x_max.
        label : str, optional
            Label to be displayed in the legend.
        face_color : str or ArrayLike or None
            Face color of the points. If an array of intensities is provided, the values are mapped to the specified
            color map. If None, marker faces are transparent.
            Default depends on the ``figure_style`` configuration.
        edge_color : str or ArrayLike or None
            Edge color of the points. If an array of intensities is provided, the values are mapped to the specified
            color map. If None, marker edges are transparent.
            Default depends on the ``figure_style`` configuration.
        color_map : str or Colormap
            Color map of the stream lines, to be used in combination with the color parameter to specify intensity.
            Default depends on the ``figure_style`` configuration.
        show_color_bar : bool
            Whether or not to display the color bar next to the plot.
            Default depends on the ``figure_style`` configuration.
        marker_size : float
            Size of the points.
            Default depends on the ``figure_style`` configuration.
        marker_style : str
            Style of the points.
            Default depends on the ``figure_style`` configuration.
        copy_first : bool
            If ``True``, a copy of the scatter plot (with all its parameters) will be returned with the slice applied. Any other parameters passed to this method will also be applied to the copied scatter plot. If ``False``, a new scatter plot will be created with the slice applied and the parameters passed to this method.

        Returns
        -------
        :class:`~graphinglib.data_plotting_1d.Scatter`
            A new :class:`~graphinglib.data_plotting_1d.Scatter` object which is a slice of the original scatter plot.
        """
        mask = (self._x_data >= x_min) & (self._x_data <= x_max)
        if copy_first:
            copy = self.copy()
            copy._x_data = self._x_data[mask]
            copy._y_data = self._y_data[mask]
            if label is not None:
                copy._label = label
            if face_color != "default":
                copy._face_color = face_color
            if edge_color != "default":
                copy._edge_color = edge_color
            if color_map != "default":
                copy._color_map = color_map
            if show_color_bar != "default":
                copy._show_color_bar = show_color_bar
            if marker_size != "default":
                copy._marker_size = marker_size
            if marker_style != "default":
                copy._marker_style = marker_style
            return copy
        else:
            return Scatter(
                self._x_data[mask],
                self._y_data[mask],
                label,
                face_color,
                edge_color,
                color_map,
                show_color_bar,
                marker_size,
                marker_style,
            )

    def create_slice_y(
        self,
        y_min: float,
        y_max: float,
        label: Optional[str] = None,
        face_color: str | ArrayLike | NoneType = "default",
        edge_color: str | ArrayLike | NoneType = "default",
        color_map: str | Colormap | Literal["default"] = "default",
        show_color_bar: bool | Literal["default"] = "default",
        marker_size: float | Literal["default"] = "default",
        marker_style: str = "default",
        copy_first: bool = False,
    ) -> Self:
        """
        Creates a slice of the scatter plot between two y values.

        Parameters
        ----------
        y_min, y_max : float
            The slice will be created between y_min and y_max.
        label : str, optional
            Label to be displayed in the legend.
        face_color : str or ArrayLike or None
            Face color of the points. If an array of intensities is provided, the values are mapped to the specified
            color map. If None, marker faces are transparent.
            Default depends on the ``figure_style`` configuration.
        edge_color : str or ArrayLike or None
            Edge color of the points. If an array of intensities is provided, the values are mapped to the specified
            color map. If None, marker edges are transparent.
            Default depends on the ``figure_style`` configuration.
        color_map : str or Colormap
            Color map of the stream lines, to be used in combination with the color parameter to specify intensity.
            Default depends on the ``figure_style`` configuration.
        show_color_bar : bool
            Whether or not to display the color bar next to the plot.
            Default depends on the ``figure_style`` configuration.
        marker_size : float
            Size of the points.
            Default depends on the ``figure_style`` configuration.
        marker_style : str
            Style of the points.
            Default depends on the ``figure_style`` configuration.
        copy_first : bool
            If ``True``, a copy of the scatter plot (with all its parameters) will be returned with the slice applied. Any other parameters passed to this method will also be applied to the copied scatter plot. If ``False``, a new scatter plot will be created with the slice applied and the parameters passed to this method.

        Returns
        -------
        :class:`~graphinglib.data_plotting_1d.Scatter`
            A new :class:`~graphinglib.data_plotting_1d.Scatter` object which is a slice of the original scatter plot.
        """
        mask = (self._y_data >= y_min) & (self._y_data <= y_max)
        if copy_first:
            copy = self.copy()
            copy._x_data = self._x_data[mask]
            copy._y_data = self._y_data[mask]
            if label is not None:
                copy._label = label
            if face_color != "default":
                copy._face_color = face_color
            if edge_color != "default":
                copy._edge_color = edge_color
            if color_map != "default":
                copy._color_map = color_map
            if show_color_bar != "default":
                copy._show_color_bar = show_color_bar
            if marker_size != "default":
                copy._marker_size = marker_size
            if marker_style != "default":
                copy._marker_style = marker_style
            return copy
        else:
            return Scatter(
                self._x_data[mask],
                self._y_data[mask],
                label,
                face_color,
                edge_color,
                color_map,
                show_color_bar,
                marker_size,
                marker_style,
            )

    def add_errorbars(
        self,
        x_error: Optional[ArrayLike] = None,
        y_error: Optional[ArrayLike] = None,
        cap_width: float | Literal["default"] = "default",
        errorbars_color: str = "default",
        errorbars_line_width: float | Literal["default"] = "default",
        cap_thickness: float | Literal["default"] = "default",
    ) -> None:
        """
        Adds errorbars to the scatter plot.

        Parameters
        ----------
        x_error, y_error : ArrayLike, optional
            Arrays of x and y errors. Use one or both.
        cap_width : float
            Width of the errorbar caps.
            Default depends on the ``figure_style`` configuration.
        errorbars_color : str
            Color of the errorbars.
            Default depends on the ``figure_style`` configuration.
        errorbars_line_width : float
            Width of the errorbars.
            Default depends on the ``figure_style`` configuration.
        cap_thickness : float
            Thickness of the errorbar caps.
            Default depends on the ``figure_style`` configuration.
        """
        self._show_errorbars = True
        self._x_error = np.array(x_error) if x_error is not None else x_error
        self._y_error = np.array(y_error) if y_error is not None else y_error
        self._errorbars_color = errorbars_color
        self._errorbars_line_width = errorbars_line_width
        self._cap_thickness = cap_thickness
        self._cap_width = cap_width

    def get_coordinates_at_x(
        self,
        x: float,
        interpolation_method: str = "linear",
    ) -> tuple[float, float]:
        """
        Gets the coordinates of the point on the curve at a given x value.

        Parameters
        ----------
        x : float
            The x value of the point.
        interpolation_method : str,
            The type of interpolation to be used, as defined in ``scipy.interpolate.interp1d``.

            .. seealso:: `scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_

            Defaults to "linear".

        Returns
        -------
        tuple[float, float]
            The coordinates of the point on the curve at the given x value.
        """
        return (
            x,
            float(interp1d(self._x_data, self._y_data, kind=interpolation_method)(x)),
        )

    def create_point_at_x(
        self,
        x: float,
        interpolation_method: str = "linear",
        label: Optional[str] = None,
        color: str = "default",
        edge_color: str = "default",
        marker_size: float | Literal["default"] = "default",
        marker_style: str = "default",
        line_width: float | Literal["default"] = "default",
    ) -> Point:
        """
        Creates a Point on the curve at a given x value.

        Parameters
        ----------
        x : float
            The x value of the point.
        interpolation_method : str,
            The type of interpolation to be used, as defined in ``scipy.interpolate.interp1d``.

            .. seealso:: `scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_

            Defaults to "linear".
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
            Width of the point edge.
            Default depends on the ``figure_style`` configuration.

        Returns
        -------
        :class:`~graphinglib.graph_elements.Point`
            The Point on the curve at the given x value.
        """
        point = Point(
            x,
            self.get_coordinates_at_x(x, interpolation_method)[1],
            label=label,
            color=color,
            edge_color=edge_color,
            marker_size=marker_size,
            marker_style=marker_style,
            edge_width=line_width,
        )
        return point

    def get_coordinates_at_y(
        self,
        y: float,
        interpolation_method: str = "linear",
    ) -> list[tuple[float, float]]:
        """
        Gets the coordinates the curve at a given y value. Can return multiple coordinate pairs if the curve crosses the y value multiple times.

        Parameters
        ----------
        y : float
            The y value of the point.
        interpolation_method : str,
            The type of interpolation to be used, as defined in ``scipy.interpolate.interp1d``.

            .. seealso:: `scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_

            Defaults to "linear".

        Returns
        -------
        list[tuple[float, float]]
            The coordinates of the points on the curve at the given y value.
        """
        xs = self._x_data
        ys = self._y_data
        assert isinstance(xs, np.ndarray) and isinstance(ys, np.ndarray)
        crossings = np.where(np.diff(np.sign(ys - y)))[0]
        x_vals: list[float] = []
        for cross in crossings:
            x1, x2 = xs[cross], xs[cross + 1]
            y1, y2 = ys[cross], ys[cross + 1]
            f = interp1d([y1, y2], [x1, x2], kind=interpolation_method)
            x_val = f(y)
            x_vals.append(float(x_val))
        points = [(x_val, y) for x_val in x_vals]
        return points

    def create_points_at_y(
        self,
        y: float,
        interpolation_method: str = "linear",
        label: Optional[str] = None,
        color: str = "default",
        edge_color: str = "default",
        marker_size: float | Literal["default"] = "default",
        marker_style: str = "default",
        line_width: float | Literal["default"] = "default",
    ) -> list[Point]:
        """
        Creates the Points on the curve at a given y value. Can return multiple Points if the curve crosses the y value multiple times.

        Parameters
        ----------
        y : float
            The y value of the desired points.
        interpolation_method : str
            The type of interpolation to be used, as defined in ``scipy.interpolate.interp1d``.

            .. seealso:: `scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_

            Defaults to "linear".
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
            Width of the point edge.
            Default depends on the ``figure_style`` configuration.

        Returns
        -------
        list[:class:`~graphinglib.graph_elements.Point`]
            The Point objects on the curve at the given y value.
        """
        coords = self.get_coordinates_at_y(y, interpolation_method)
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
            for coord in coords
        ]
        return points

    def _get_contrasting_shade(self, color: str | tuple[int, int, int]) -> str:
        """
        Gives the most contrasting shade (black/white) for a given color. The algorithm used comes from this Stack
        Exchange answer : https://ux.stackexchange.com/a/82068.

        Parameters
        ----------
        color : str or tuple[int, int, int]
            Color that needs to be contrasted. This can either be a known matplotlib color string or a RGB code, given
            as a tuple of integers that take 0-255.

        Returns
        -------
        shade : str
            Shade (black/white) that contrasts the most with the given color.
        """
        if isinstance(color, str):
            color = to_rgba_array(color)[0, :3] * 255

        R, G, B = color

        if R <= 10:
            Rg = R / 3294
        else:
            Rg = (R / 269 + 0.0513) ** 2.4

        if G <= 10:
            Gg = G / 3294
        else:
            Gg = (G / 269 + 0.0513) ** 2.4

        if B <= 10:
            Bg = B / 3294
        else:
            Bg = (B / 269 + 0.0513) ** 2.4

        L = 0.2126 * Rg + 0.7152 * Gg + 0.0722 * Bg
        if L < 0.5:
            return "white"
        else:
            return "black"

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified axes.
        """
        # Check that either face color or edge color is not None
        if self._face_color is None and self._edge_color is None:
            raise ValueError(
                "Both face color and edge color cannot be None. Please set at least one of them to a valid color."
            )

        # Check that either face color or edge color is not a list/array/tuple of intensities
        if isinstance(self._face_color, (list, tuple, np.ndarray)) and isinstance(
            self._edge_color, (list, tuple, np.ndarray)
        ):
            # If they're 3 or 4 element arrays, assume they're RGB or RGBA values
            if len(self._face_color) in [3, 4] or len(self._edge_color) in [3, 4]:
                pass
            else:
                raise ValueError(
                    "Both face color and edge color cannot be lists/arrays/tuples of intensities or colors. Please set at least one of them to a valid color or set one of them to None."
                )

        # Convert face color to matplotlib notation
        if self._face_color is None:
            # Set to transparent
            mpl_face_color = "none"
        elif isinstance(self._face_color, str) and self._face_color == "default":
            # Use color cycle (figure uses a matplotlib style)
            mpl_face_color = None
        elif isinstance(self._face_color, str) and self._face_color == "color cycle":
            # Use color cycle
            mpl_face_color = None
        else:
            # Use specified color
            mpl_face_color = self._face_color

        # Convert edge color to matplotlib notation
        if self._edge_color is None:
            # Set to transparent
            mpl_edge_color = "none"
        elif isinstance(self._edge_color, str) and self._edge_color == "default":
            # Use color cycle (figure uses a matplotlib style)
            mpl_edge_color = None
        elif isinstance(self._edge_color, str) and self._edge_color == "color cycle":
            # Use color cycle
            mpl_edge_color = kwargs["cycle_color"]
        else:
            # Use specified color
            mpl_edge_color = self._edge_color

        # Check whether to use color map (one of the colors is an array of intensities)
        if isinstance(self._face_color, (list, tuple, np.ndarray)):
            if all(isinstance(i, (int, float)) for i in self._face_color):
                color_map = plt.get_cmap(self._color_map)
                norm = Normalize(vmin=min(self._face_color), vmax=max(self._face_color))
                mpl_face_color = [color_map(norm(i)) for i in self._face_color]
        elif isinstance(self._edge_color, (list, tuple, np.ndarray)):
            if all(isinstance(i, (int, float)) for i in self._edge_color):
                color_map = plt.get_cmap(self._color_map)
                norm = Normalize(vmin=min(self._edge_color), vmax=max(self._edge_color))
                mpl_edge_color = [color_map(norm(i)) for i in self._edge_color]

        params = {
            "s": self._marker_size,
            "marker": self._marker_style if self._marker_style != "default" else "o",
        }
        params = {k: v for k, v in params.items() if v != "default"}
        params["facecolors"] = mpl_face_color
        params["edgecolors"] = mpl_edge_color
        self.handle = axes.scatter(
            self._x_data,
            self._y_data,
            label=self._label,
            zorder=z_order,
            **params,
        )
        if self._show_errorbars:
            # Convert errorbars color to matplotlib notation
            if self._errorbars_color is None:
                raise ValueError(
                    "Errorbars color cannot be None. Please set the errorbars color to a valid color."
                )
            elif isinstance(self._errorbars_color, str) and (
                self._errorbars_color == "default"
            ):
                # Use color cycle
                mpl_errorbars_color = None
            elif (
                isinstance(self._errorbars_color, str)
                and self._errorbars_color == "same as scatter"
            ):
                marker_edge_color = self.handle.get_edgecolor()
                marker_face_color = self.handle.get_facecolor()
                if is_color_like(marker_edge_color):
                    mpl_errorbars_color = marker_edge_color
                elif is_color_like(marker_face_color):
                    mpl_errorbars_color = marker_face_color
                else:
                    ax_face_color = plt.rcParams["axes.facecolor"]
                    mpl_errorbars_color = self._get_contrasting_shade(ax_face_color)
            elif isinstance(self._errorbars_color, str):
                # Use specified color
                mpl_errorbars_color = self._errorbars_color
            else:
                raise ValueError("Errorbars color must be a string.")

            errorbar_params = {
                "markerfacecolor": None,
                "markeredgecolor": None,
                "elinewidth": self._errorbars_line_width,
                "capsize": self._cap_width,
                "capthick": self._cap_thickness,
                "linestyle": "none",
            }
            errorbar_params = {
                k: v for k, v in errorbar_params.items() if v != "default"
            }
            errorbar_params["ecolor"] = mpl_errorbars_color
            self.errorbars_handle = axes.errorbar(
                self._x_data,
                self._y_data,
                xerr=self._x_error,
                yerr=self._y_error,
                zorder=z_order - 1,
                **errorbar_params,
            )

        if (
            self._show_color_bar
            and self._face_color is not None
            and not isinstance(self.face_color, str)
        ):
            # Create color bar from face color intensities
            color_map = plt.get_cmap(self._color_map)
            norm = Normalize(vmin=min(self._face_color), vmax=max(self._face_color))
            sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=axes)

        if (
            self._show_color_bar
            and self._edge_color is not None
            and not isinstance(self.edge_color, str)
        ):
            # Create color bar from edge color intensities
            color_map = plt.get_cmap(self._color_map)
            norm = Normalize(vmin=min(self._edge_color), vmax=max(self._edge_color))
            sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=axes)


@dataclass
class Histogram:
    """
    This class implements a general histogram.

    Parameters
    ----------
    data : ArrayLike
        Array of values to be plotted.
    number_of_bins : int
        Number of bins to be used in the histogram.
    label : str, optional
        Label to be displayed in the legend.
    face_color : str
        Face color of the histogram.
        Default depends on the ``figure_style`` configuration.
    edge_color : str
        Edge color of the histogram.
        Default depends on the ``figure_style`` configuration.
    hist_type : str
        Type of the histogram. Can be "bar", "barstacked", "step", "stepfilled".
        Default depends on the ``figure_style`` configuration.
    alpha : float
        Transparency of the histogram.
        Default depends on the ``figure_style`` configuration.
    line_width : float
        Width of the histogram edge.
        Default depends on the ``figure_style`` configuration.
    normalize : bool
        Whether or not to normalize the histogram.
        Default depends on the ``figure_style`` configuration.
    show_pdf : str
        Whether or not to show the probability density function.
        Can be "normal" or "gaussian".
        Default depends on the ``figure_style`` configuration.
    show_params : bool
        Whether or not to show the mean and standard deviation of the data.
        Default depends on the ``figure_style`` configuration.
    """

    def __init__(
        self,
        data: ArrayLike,
        number_of_bins: int,
        label: Optional[str] = None,
        face_color: str = "default",
        edge_color: str = "default",
        hist_type: str = "default",
        alpha: float | Literal["default"] = "default",
        line_width: float | Literal["default"] = "default",
        normalize: bool | Literal["default"] = "default",
        show_params: bool | Literal["default"] = "default",
    ) -> None:
        """
        This class implements a general histogram.

        Parameters
        ----------
        data : ArrayLike
            Array of values to be plotted.
        number_of_bins : int
            Number of bins to be used in the histogram.
        label : str, optional
            Label to be displayed in the legend.
        face_color : str
            Face color of the histogram.
            Default depends on the ``figure_style`` configuration.
        edge_color : str
            Edge color of the histogram.
            Default depends on the ``figure_style`` configuration.
        hist_type : str
            Type of the histogram. Can be "bar", "barstacked", "step", "stepfilled".
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Transparency of the histogram.
            Default depends on the ``figure_style`` configuration.
        line_width : float
            Width of the histogram edge.
            Default depends on the ``figure_style`` configuration.
        normalize : bool
            Whether or not to normalize the histogram.
            Default depends on the ``figure_style`` configuration.
        show_pdf : str
            Whether or not to show the probability density function.
            Can be "normal" or "gaussian".
            Default depends on the ``figure_style`` configuration.
        show_params : bool
            Whether or not to show the mean and standard deviation of the data.
            Default depends on the ``figure_style`` configuration.
        """
        self._data = np.asarray(data)
        self._number_of_bins = number_of_bins
        self._label = label
        self._face_color = face_color
        self._edge_color = edge_color
        self._hist_type = hist_type
        self._alpha = alpha
        self._line_width = line_width
        self._normalize = normalize
        self._show_params = show_params

        self._show_pdf = False
        self._pdf_type = None
        self._pdf_show_mean = None
        self._pdf_show_std = None
        self._pdf_curve_color = None
        self._pdf_mean_color = None
        self._pdf_std_color = None

        self._mean = np.mean(self._data)
        self._standard_deviation = np.std(self._data)
        _parameters = np.histogram(
            self._data, bins=self._number_of_bins, density=self._normalize
        )
        self._bin_heights, bin_edges = _parameters[0], _parameters[1]
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = bin_edges[1:] - bin_width / 2
        self._bin_width = bin_width
        self._bin_centers = bin_centers
        self._bin_edges = bin_edges

    @classmethod
    def from_fit_residuals(
        cls,
        fit: Fit,
        number_of_bins: int,
        label: Optional[str] = None,
        face_color: str = "default",
        edge_color: str = "default",
        hist_type: str = "default",
        alpha: int | Literal["default"] = "default",
        line_width: int | Literal["default"] = "default",
        normalize: bool | Literal["default"] = "default",
        show_params: bool | Literal["default"] = "default",
    ) -> Self:
        """
        Calculates the residuals of a fit and plots them as a histogram.

        Parameters
        ----------
        fit : Fit
            The fit from which the residuals are to be calculated.
        number_of_bins : int
            Number of bins to be used in the histogram.
        label : str, optional
            Label to be displayed in the legend.
        face_color : str
            Face color of the histogram.
            Default depends on the ``figure_style`` configuration.
        edge_color : str
            Edge color of the histogram.
            Default depends on the ``figure_style`` configuration.
        hist_type : str
            Type of the histogram. Can be "bar", "barstacked", "step", "stepfilled".
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Transparency of the histogram.
            Default depends on the ``figure_style`` configuration.
        line_width : float
            Width of the histogram edge.
            Default depends on the ``figure_style`` configuration.
        normalize : bool
            Whether or not to normalize the histogram.
            Default depends on the ``figure_style`` configuration.
        show_pdf : str
            Whether or not to show the probability density function.
            Can be "normal" or "gaussian".
            Default depends on the ``figure_style`` configuration.
        show_params : bool
            Whether or not to show the mean and standard deviation of the data.
            Default depends on the ``figure_style`` configuration.

        Returns
        -------
        A :class:`~graphinglib.data_plotting_1d.Histogram` object created from the residuals of a fit.
        """
        residuals = fit.get_residuals()
        return cls(
            residuals,
            number_of_bins,
            label,
            face_color,
            edge_color,
            hist_type,
            alpha,
            line_width,
            normalize,
            show_params,
        )

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, data: ArrayLike) -> None:
        self._data = np.array(data)
        self._mean = np.mean(self._data)
        self._standard_deviation = np.std(self._data)
        _parameters = np.histogram(
            self._data, bins=self._number_of_bins, density=self._normalize
        )
        self._bin_heights, bin_edges = _parameters[0], _parameters[1]
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = bin_edges[1:] - bin_width / 2
        self._bin_width = bin_width
        self._bin_centers = bin_centers
        self._bin_edges = bin_edges

    @property
    def number_of_bins(self) -> int:
        return self._number_of_bins

    @number_of_bins.setter
    def number_of_bins(self, number_of_bins: int) -> None:
        self._number_of_bins = number_of_bins
        _parameters = np.histogram(
            self._data, bins=self._number_of_bins, density=self._normalize
        )
        self._bin_heights, bin_edges = _parameters[0], _parameters[1]
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = bin_edges[1:] - bin_width / 2
        self._bin_width = bin_width
        self._bin_centers = bin_centers
        self._bin_edges = bin_edges

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, label: str) -> None:
        self._label = label

    @property
    def face_color(self) -> str:
        return self._face_color

    @face_color.setter
    def face_color(self, face_color: str) -> None:
        self._face_color = face_color

    @property
    def edge_color(self) -> str:
        return self._edge_color

    @edge_color.setter
    def edge_color(self, edge_color: str) -> None:
        self._edge_color = edge_color

    @property
    def hist_type(self) -> str:
        return self._hist_type

    @hist_type.setter
    def hist_type(self, hist_type: str) -> None:
        self._hist_type = hist_type

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        self._alpha = alpha

    @property
    def line_width(self) -> float:
        return self._line_width

    @line_width.setter
    def line_width(self, line_width: float) -> None:
        self._line_width = line_width

    @property
    def normalize(self) -> bool:
        return self._normalize

    @normalize.setter
    def normalize(self, normalize: bool) -> None:
        self._normalize = normalize

    @property
    def show_params(self) -> bool:
        return self._show_params

    @show_params.setter
    def show_params(self, show_params: bool) -> None:
        self._show_params = show_params

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def standard_deviation(self) -> float:
        return self._standard_deviation

    @property
    def parameters(self) -> tuple[float, float]:
        return self._mean, self._standard_deviation

    @property
    def bin_heights(self) -> np.ndarray:
        return self._bin_heights

    @property
    def bin_centers(self) -> np.ndarray:
        return self._bin_centers

    @property
    def bin_edges(self) -> np.ndarray:
        return self._bin_edges

    def _create_label(self) -> None:
        """
        Creates the label of the histogram (with or without parameters).
        """
        lab = self._label
        if lab and self._show_params:
            lab += (
                " :\n"
                + f"$\mu$ = {0 if abs(self._mean) < 1e-3 else self._mean:.3f}, $\sigma$ = {self._standard_deviation:.3f}"
            )
        elif self._show_params:
            lab = f"$\mu$ = {0 if abs(self._mean) < 1e-3 else self._mean:.3f}, $\sigma$ = {self._standard_deviation:.3f}"
        self._label = lab

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.data_plotting_1d.Histogram` object.
        """
        return deepcopy(self)

    def _normal_normalized(self, x: ArrayLike) -> ArrayLike:
        """
        Calculates the normalized gaussian curve from the mean and standard deviation of the data.

        Parameters
        ----------
        x : ArrayLike
            The x values at which the gaussian curve is to be calculated.

        Returns
        -------
        The corresponding array of y values of the gaussian curve.
        """
        x = np.array(x)
        return (1 / (self._standard_deviation * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * (((x - self._mean) / self._standard_deviation) ** 2)
        )

    def _normal_not_normalized(self, x: ArrayLike) -> ArrayLike:
        """
        Calculates the (not normalized) gaussian curve from the mean and standard deviation of the data.

        Parameters
        ----------
        x : ArrayLike
            The x values at which the gaussian curve is to be calculated.

        Returns
        -------
        The corresponding array of y values of the gaussian curve.
        """
        x = np.array(x)
        return sum(self._bin_heights) * self._bin_width * self._normal_normalized(x)

    def add_pdf(
        self,
        type: str = "normal",
        show_mean: bool | Literal["default"] = "default",
        show_std: bool | Literal["default"] = "default",
        curve_color: str | Literal["default"] = "default",
        mean_color: str | Literal["default"] = "default",
        std_color: str | Literal["default"] = "default",
    ) -> None:
        """
        Shows the probability density function of the histogram.

        Parameters
        ----------
        type : str
            The type of probability density function to be shown.
            Currently only "normal" is supported.
            Defaults to "normal".
        show_mean : bool
            Whether or not to show the mean of the data.
            Default depends on the ``figure_style`` configuration.
        show_std : bool
            Whether or not to show the standard deviation of the data.
            Default depends on the ``figure_style`` configuration.
        curve_color : str
            Color of the probability density function curve.
            Default depends on the ``figure_style`` configuration.
        mean_color : str
            Color of the mean line.
            Default depends on the ``figure_style`` configuration.
        std_color : str
            Color of the standard deviation lines.
            Default depends on the ``figure_style`` configuration.
        """
        if type != "normal":
            raise ValueError("Currently, only 'normal' distribution is supported.")
        self._show_pdf = True
        self._pdf_type = type
        self._pdf_show_mean = show_mean
        self._pdf_show_std = show_std
        self._pdf_curve_color = curve_color
        self._pdf_mean_color = mean_color
        self._pdf_std_color = std_color

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified axes.
        """
        self._create_label()
        params = {
            "facecolor": (
                to_rgba(self._face_color, self._alpha)
                if self._face_color != "default" and self._alpha != "default"
                else "default"
            ),
            "edgecolor": (
                to_rgba(self._edge_color, 1)
                if self._edge_color != "default"
                else self._edge_color
            ),
            "linewidth": self._line_width,
        }
        params = {k: v for k, v in params.items() if v != "default"}
        self.handle = Polygon(
            np.array([[0, 2, 2, 3, 3, 1, 1, 0, 0], [0, 0, 1, 1, 2, 2, 3, 3, 0]]).T,
            **params,
        )
        params = {
            "facecolor": (
                to_rgba(self._face_color, self._alpha)
                if self._face_color != "default" and self._alpha != "default"
                else "default"
            ),
            "edgecolor": (
                to_rgba(self._edge_color, 1)
                if self._edge_color != "default"
                else self._edge_color
            ),
            "histtype": self._hist_type,
            "linewidth": self._line_width,
            "density": self._normalize,
        }
        params = {k: v for k, v in params.items() if v != "default"}
        axes.hist(
            self._data,
            bins=self._number_of_bins,
            label=self._label,
            zorder=z_order - 1,
            **params,
        )
        if self._show_pdf:
            normal = (
                self._normal_normalized
                if self._normalize
                else self._normal_not_normalized
            )
            num_of_points = 500
            x_data = np.linspace(self._bin_edges[0], self._bin_edges[-1], num_of_points)
            y_data = normal(x_data)
            params = {
                "color": self._pdf_curve_color,
            }
            params = {k: v for k, v in params.items() if v != "default"}
            axes.plot(
                x_data,
                y_data,
                zorder=z_order,
                **params,
            )
            curve_max_y = normal(self._mean)
            curve_std_y = normal(self._mean + self._standard_deviation)
            if self._pdf_show_std:
                params = {}
                if self._pdf_std_color != "default":
                    params["colors"] = [self._pdf_std_color, self._pdf_std_color]
                plt.vlines(
                    [
                        self._mean - self._standard_deviation,
                        self._mean + self._standard_deviation,
                    ],
                    [0, 0],
                    [curve_std_y, curve_std_y],
                    linestyles=["dashed"],
                    zorder=z_order - 1,
                    **params,
                )
            if self._pdf_show_mean:
                params = {}
                if self._pdf_mean_color != "default":
                    params["colors"] = [self._pdf_mean_color]
                plt.vlines(
                    self._mean,
                    0,
                    curve_max_y,
                    linestyles=["dashed"],
                    zorder=z_order - 1,
                    **params,
                )
