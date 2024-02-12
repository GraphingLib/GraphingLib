from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Polygon
from numpy.typing import ArrayLike
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

    x_data: ArrayLike
    y_data: ArrayLike
    label: Optional[str] = None
    color: str = "default"
    line_width: float | Literal["default"] = "default"
    line_style: str = "default"
    show_errorbars: bool = field(default=False, init=False)
    _fill_curve_between: Optional[tuple[float, float]] = field(init=False, default=None)

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

    def __post_init__(self):
        self.x_data = np.array(self.x_data)
        self.y_data = np.array(self.y_data)

    def __add__(self, other: Self | float) -> Self:
        """
        Defines the addition of two curves or a curve and a number.
        """
        if isinstance(other, Curve):
            if not np.array_equal(self.x_data, other.x_data):
                if len(self.x_data) > len(other.x_data):
                    x_data = other.x_data
                    y_data = interp1d(self.x_data, self.y_data)(x_data)
                    return Curve(x_data, y_data + other.y_data)
                else:
                    x_data = self.x_data
                    y_data = interp1d(other.x_data, other.y_data)(x_data)
                    return Curve(x_data, y_data + self.y_data)

            new_y_data = self.y_data + other.y_data
            return Curve(self.x_data, new_y_data)
        elif isinstance(other, (int, float)):
            new_y_data = self.y_data + other
            return Curve(self.x_data, new_y_data)
        else:
            raise TypeError("Can only add a curve to another curve or a number.")

    def __radd__(self, other: Self | float) -> Self:
        return self.__add__(other)

    def __sub__(self, other: Self | float) -> Self:
        """
        Defines the subtraction of two curves or a curve and a number.
        """
        if isinstance(other, Curve):
            if not np.array_equal(self.x_data, other.x_data):
                if len(self.x_data) > len(other.x_data):
                    x_data = other.x_data
                    y_data = interp1d(self.x_data, self.y_data)(x_data)
                    return Curve(x_data, y_data - other.y_data)
                else:
                    x_data = self.x_data
                    y_data = interp1d(other.x_data, other.y_data)(x_data)
                    return Curve(x_data, self.y_data - y_data)
            new_y_data = self.y_data - other.y_data
            return Curve(self.x_data, new_y_data)
        elif isinstance(other, (int, float)):
            new_y_data = self.y_data - other
            return Curve(self.x_data, new_y_data)
        else:
            raise TypeError("Can only subtract a curve from another curve or a number.")

    def __rsub__(self, other: Self | float) -> Self:
        # 2 - curve => -curve + 2
        return (self * -1) + other

    def __mul__(self, other: Self | float) -> Self:
        """
        Defines the multiplication of two curves or a curve and a number.
        """
        if isinstance(other, Curve):
            if not np.array_equal(self.x_data, other.x_data):
                if len(self.x_data) > len(other.x_data):
                    x_data = other.x_data
                    y_data = interp1d(self.x_data, self.y_data)(x_data)
                    return Curve(x_data, y_data * other.y_data)
                else:
                    x_data = self.x_data
                    y_data = interp1d(other.x_data, other.y_data)(x_data)
                    return Curve(x_data, y_data * self.y_data)
            new_y_data = self.y_data * other.y_data
            return Curve(self.x_data, new_y_data)
        elif isinstance(other, (int, float)):
            new_y_data = self.y_data * other
            return Curve(self.x_data, new_y_data)
        else:
            raise TypeError("Can only multiply a curve by another curve or a number.")

    def __rmul__(self, other: Self | float) -> Self:
        return self.__mul__(other)

    def __truediv__(self, other: Self | float) -> Self:
        """
        Defines the division of two curves or a curve and a number.
        """
        if isinstance(other, Curve):
            if not np.array_equal(self.x_data, other.x_data):
                if len(self.x_data) > len(other.x_data):
                    x_data = other.x_data
                    y_data = interp1d(self.x_data, self.y_data)(x_data)
                    return Curve(x_data, y_data / other.y_data)
                else:
                    x_data = self.x_data
                    y_data = interp1d(other.x_data, other.y_data)(x_data)
                    return Curve(x_data, self.y_data / y_data)
            new_y_data = self.y_data / other.y_data
            return Curve(self.x_data, new_y_data)
        elif isinstance(other, (int, float)):
            new_y_data = self.y_data / other
            return Curve(self.x_data, new_y_data)
        else:
            raise TypeError("Can only divide a curve by another curve or a number.")

    def __rtruediv__(self, other: Self | float) -> Self:
        try:
            return (self**-1) * other
        except ZeroDivisionError:
            raise ZeroDivisionError("Cannot divide by zero.")

    def __pow__(self, other: float) -> Self:
        """
        Defines the power of a curve to a number.
        """
        if isinstance(other, (int, float)):
            new_y_data = self.y_data**other
            return Curve(self.x_data, new_y_data)
        else:
            raise TypeError("Can only raise a curve to another curve or a number.")

    def __iter__(self):
        """
        Defines the iteration of a curve. Returns the y values.
        """
        return iter(self.y_data)

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
        self.show_errorbars = True
        self.x_error = np.array(x_error) if x_error is not None else x_error
        self.y_error = np.array(y_error) if y_error is not None else y_error
        self.errorbars_color = errorbars_color
        self.errorbars_line_width = errorbars_line_width
        self.cap_thickness = cap_thickness
        self.cap_width = cap_width

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
            float(interp1d(self.x_data, self.y_data, kind=interpolation_method)(x)),
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
    ) -> Point | tuple[float, float]:
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
        xs = self.x_data
        ys = self.y_data
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

        Returns
        -------
        A :class:`~graphinglib.data_plotting_1d.Curve` object which is the derivative of the original curve.
        """
        x_data = self.x_data
        y_data = np.gradient(self.y_data, x_data)
        return Curve(x_data, y_data, label, color, line_width, line_style)

    def create_integral_curve(
        self,
        label: Optional[str] = None,
        color: str = "default",
        line_width: float | Literal["default"] = "default",
        line_style: str = "default",
    ) -> Self:
        """
        Creates a new curve which is the integral of the original curve.

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

        Returns
        -------
        A :class:`~graphinglib.data_plotting_1d.Curve` object which is the integral of the original curve.
        """
        x_data = self.x_data
        y_data = np.cumsum(self.y_data) * np.diff(x_data)[0]
        return Curve(x_data, y_data, label, color, line_width, line_style)

    def create_tangent_curve(
        self,
        x: float,
        label: Optional[str] = None,
        color: str = "default",
        line_width: float | Literal["default"] = "default",
        line_style: str = "default",
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

        Returns
        -------
        :class:`~graphinglib.data_plotting_1d.Curve`
            A :class:`~graphinglib.data_plotting_1d.Curve` object which is the tangent to the original curve at a given x value.
        """
        point = self.get_coordinates_at_x(x)
        gradient = self.create_derivative_curve().get_coordinates_at_x(x)[1]
        y_data = gradient * (self.x_data - x) + point[1]
        tangent_curve = Curve(self.x_data, y_data, label, color, line_width, line_style)
        return tangent_curve

    def create_normal_curve(
        self,
        x: float,
        label: Optional[str] = None,
        color: str = "default",
        line_width: float | Literal["default"] = "default",
        line_style: str = "default",
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

        Returns
        -------
        :class:`~graphinglib.data_plotting_1d.Curve`
            A :class:`~graphinglib.data_plotting_1d.Curve` object which is the normal to the original curve at a given x value.
        """
        point = self.get_coordinates_at_x(x)
        gradient = self.create_derivative_curve().get_coordinates_at_x(x)[1]
        y_data = -1 / gradient * (self.x_data - x) + point[1]
        normal_curve = Curve(self.x_data, y_data, label, color, line_width, line_style)
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
        y_data = self.y_data
        x_data = self.x_data
        f = interp1d(x_data, y_data)
        x = np.linspace(x1, x2, 1000)
        y = f(x)
        return np.trapz(np.sqrt(1 + np.gradient(y, x) ** 2), x)

    def get_area_between(
        self,
        x1: float,
        x2: float,
        fill_under: bool = False,
        fill_color: str = "default",
    ) -> float:
        """
        Calculates the area between the curve and the x axis between two x values.
        This is the definite integral of the curve between the two x values.

        Parameters
        ----------
        x1, x2 : float
            The x values between which the area is to be calculated.
        fill_under : bool
            Whether to fill the specified area between the curve and the x axis when displaying.
            Defaults to ``False``.
        fill_color : str
            Color of the area between the curve and the x axis when ``fill_under`` is set to ``True``.
            Default depends on the ``figure_style`` configuration.

        Returns
        -------
        The area (float) between the curve and the x axis between the two given x values.
        """
        if fill_under:
            self._fill_curve_between = (x1, x2)
            self.fill_under_color = fill_color
        y_data = self.y_data
        x_data = self.x_data
        f = interp1d(x_data, y_data)
        x = np.linspace(x1, x2, 1000)
        y = f(x)
        return np.trapz(y, x)

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
        y = self.y_data - other.y_data
        s = np.abs(np.diff(np.sign(y))).astype(bool)
        intersections_x = self.x_data[:-1][s] + np.diff(self.x_data)[s] / (
            np.abs(y[1:][s] / y[:-1][s]) + 1
        )
        intersections_y = np.interp(intersections_x, self.x_data, self.y_data)
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
        y = self.y_data - other.y_data
        s = np.abs(np.diff(np.sign(y))).astype(bool)
        intersections_x = self.x_data[:-1][s] + np.diff(self.x_data)[s] / (
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

    def _plot_element(self, axes: plt.Axes, z_order: int) -> None:
        """
        Plots the element in the specified axes.
        """
        if self.show_errorbars:
            params = {
                "color": self.color,
                "linewidth": self.line_width,
                "linestyle": self.line_style,
                "elinewidth": self.errorbars_line_width,
                "capsize": self.cap_width,
                "capthick": self.cap_thickness,
                "ecolor": self.errorbars_color,
            }
            params = {k: v for k, v in params.items() if v != "default"}
            self.handle = axes.errorbar(
                self.x_data,
                self.y_data,
                xerr=self.x_error,
                yerr=self.y_error,
                label=self.label,
                zorder=z_order,
                **params,
            )
        else:
            params = {
                "color": self.color,
                "linewidth": self.line_width,
                "linestyle": self.line_style,
            }
            params = {k: v for k, v in params.items() if v != "default"}
            self.handle = axes.errorbar(
                self.x_data,
                self.y_data,
                label=self.label,
                zorder=z_order,
                **params,
            )
        if self._fill_curve_between:
            kwargs = {"alpha": 0.2}
            if self.fill_under_color:
                kwargs["color"] = self.fill_under_color
            else:
                kwargs["color"] = self.handle[0].get_color()
            kwargs = {k: v for k, v in kwargs.items() if v != "default"}
            axes.fill_between(
                self.x_data,
                self.y_data,
                where=np.logical_and(
                    self.x_data >= self._fill_curve_between[0],
                    self.x_data <= self._fill_curve_between[1],
                ),
                zorder=z_order - 2,
                **kwargs,
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
    face_color : str
        Face color of the points.
        Default depends on the ``figure_style`` configuration.
    edge_color : str
        Edge color of the points.
        Default depends on the ``figure_style`` configuration.
    marker_size : float
        Size of the points.
        Default depends on the ``figure_style`` configuration.
    marker_style : str
        Style of the points.
        Default depends on the ``figure_style`` configuration.
    errorbars : bool
        Whether or not to display errorbars.
        Defaults to ``False``.
    """

    x_data: ArrayLike
    y_data: ArrayLike
    label: Optional[str] = None
    face_color: str = "default"
    edge_color: str = "default"
    marker_size: float | Literal["default"] = "default"
    marker_style: str = "default"
    errorbars: bool = field(default=False, init=False)

    @classmethod
    def from_function(
        cls,
        func: Callable[[ArrayLike], ArrayLike],
        x_min: float,
        x_max: float,
        label: Optional[str] = None,
        face_color: str = "default",
        edge_color: str = "default",
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
        face_color : str
            Face color of the points.
            Default depends on the ``figure_style`` configuration.
        edge_color : str
            Edge color of the points.
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
            x_data, y_data, label, face_color, edge_color, marker_size, marker_style
        )

    def __post_init__(self) -> None:
        self.x_data = np.array(self.x_data)
        self.y_data = np.array(self.y_data)

    def __add__(self, other: Self | float) -> Self:
        """
        Defines the addition of two scatter plots or a scatter plot and a number.
        """
        if isinstance(other, Scatter):
            try:
                assert np.array_equal(self.x_data, other.x_data)
            except AssertionError:
                raise ValueError(
                    "Cannot add two scatter plots with different x values."
                )
            new_y_data = self.y_data + other.y_data
            return Scatter(self.x_data, new_y_data)
        elif isinstance(other, (int, float)):
            new_y_data = self.y_data + other
            return Scatter(self.x_data, new_y_data)
        else:
            raise TypeError(
                "Can only add a scatter plot to another scatter plot or a number."
            )

    def __radd__(self, other: Self | float) -> Self:
        """
        Defines the reverse addition of a scatter plot and a number.
        """
        return self.__add__(other)

    def __sub__(self, other: Self | float) -> Self:
        """
        Defines the subtraction of two scatter plots or a scatter plot and a number.
        """
        if isinstance(other, Scatter):
            try:
                assert np.array_equal(self.x_data, other.x_data)
            except AssertionError:
                raise ValueError(
                    "Cannot subtract two scatter plots with different x values."
                )
            new_y_data = self.y_data - other.y_data
            return Scatter(self.x_data, new_y_data)
        elif isinstance(other, (int, float)):
            new_y_data = self.y_data - other
            return Scatter(self.x_data, new_y_data)
        else:
            raise TypeError(
                "Can only subtract a scatter plot from another scatter plot or a number."
            )

    def __rsub__(self, other: Self | float) -> Self:
        """
        Defines the reverse subtraction of a scatter plot and a number.
        """
        return (self * -1) + other

    def __mul__(self, other: Self | float) -> Self:
        """
        Defines the multiplication of two scatter plots or a scatter plot and a number.
        """
        if isinstance(other, Scatter):
            try:
                assert np.array_equal(self.x_data, other.x_data)
            except AssertionError:
                raise ValueError(
                    "Cannot multiply two scatter plots with different x values."
                )
            new_y_data = self.y_data * other.y_data
            return Scatter(self.x_data, new_y_data)
        elif isinstance(other, (int, float)):
            new_y_data = self.y_data * other
            return Scatter(self.x_data, new_y_data)
        else:
            raise TypeError(
                "Can only multiply a scatter plot by another scatter plot or a number."
            )

    def __rmul__(self, other: Self | float) -> Self:
        """
        Defines the reverse multiplication of a scatter plot and a number.
        """
        return self.__mul__(other)

    def __truediv__(self, other: Self | float) -> Self:
        """
        Defines the division of two scatter plots or a scatter plot and a number.
        """
        if isinstance(other, Scatter):
            try:
                assert np.array_equal(self.x_data, other.x_data)
            except AssertionError:
                raise ValueError(
                    "Cannot divide two scatter plots with different x values."
                )
            new_y_data = self.y_data / other.y_data
            return Scatter(self.x_data, new_y_data)
        elif isinstance(other, (int, float)):
            new_y_data = self.y_data / other
            return Scatter(self.x_data, new_y_data)
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

    def __pow__(self, other: float) -> Self:
        """
        Defines the power of a scatter plot to a number.
        """
        if isinstance(other, (int, float)):
            new_y_data = self.y_data**other
            return Scatter(self.x_data, new_y_data)
        else:
            raise TypeError(
                "Can only raise a scatter plot to another scatter plot or a number."
            )

    def __iter__(self):
        """
        Defines the iteration of a scatter plot. Returns the y values.
        """
        return iter(self.y_data)

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
        self.errorbars = True
        self.x_error = np.array(x_error) if x_error is not None else x_error
        self.y_error = np.array(y_error) if y_error is not None else y_error
        self.errorbars_color = errorbars_color
        self.errorbars_line_width = errorbars_line_width
        self.cap_thickness = cap_thickness
        self.cap_width = cap_width

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
            float(interp1d(self.x_data, self.y_data, kind=interpolation_method)(x)),
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
        xs = self.x_data
        ys = self.y_data
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

    def _plot_element(self, axes: plt.Axes, z_order: int) -> None:
        """
        Plots the element in the specified axes.
        """
        if self.errorbars:
            params = {
                "markerfacecolor": self.face_color,
                "markeredgecolor": self.edge_color,
                "markersize": self.marker_size,
                "marker": self.marker_style,
                "elinewidth": self.errorbars_line_width,
                "capsize": self.cap_width,
                "capthick": self.cap_thickness,
                "ecolor": self.errorbars_color,
                "linestyle": "none",
            }
            if params["marker"] == "default":
                params["marker"] = "o"
            params = {k: v for k, v in params.items() if v != "default"}
            self.handle = axes.errorbar(
                self.x_data,
                self.y_data,
                xerr=self.x_error,
                yerr=self.y_error,
                label=self.label,
                zorder=z_order,
                **params,
            )
        else:
            params = {
                "markerfacecolor": self.face_color,
                "markeredgecolor": self.edge_color,
                "markersize": self.marker_size,
                "marker": self.marker_style,
                "linestyle": "none",
            }
            if params["marker"] == "default":
                params["marker"] = "o"
            params = {k: v for k, v in params.items() if v != "default"}
            self.handle = axes.errorbar(
                self.x_data,
                self.y_data,
                label=self.label,
                zorder=z_order,
                **params,
            )


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

    data: ArrayLike
    number_of_bins: int
    label: Optional[str] = None
    face_color: str = "default"
    edge_color: str = "default"
    hist_type: str = "default"
    alpha: float | Literal["default"] = "default"
    line_width: float | Literal["default"] = "default"
    normalize: bool | Literal["default"] = "default"
    show_params: bool | Literal["default"] = "default"

    def __post_init__(self) -> None:
        self.data = np.array(self.data)
        self.mean = np.mean(self.data)
        self.standard_deviation = np.std(self.data)
        parameters = np.histogram(
            self.data, bins=self.number_of_bins, density=self.normalize
        )
        self._bin_heights, bin_edges = parameters[0], parameters[1]
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = bin_edges[1:] - bin_width / 2
        self._bin_width = bin_width
        self._bin_centers = bin_centers
        self._bin_edges = bin_edges
        self._create_label()
        self._show_pdf = False

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

    def _create_label(self) -> None:
        """
        Creates the label of the histogram (with or without parameters).
        """
        lab = self.label
        if lab and self.show_params:
            lab += (
                " :\n"
                + f"$\mu$ = {0 if abs(self.mean) < 1e-3 else self.mean:.3f}, $\sigma$ = {self.standard_deviation:.3f}"
            )
        elif self.show_params:
            lab = f"$\mu$ = {0 if abs(self.mean) < 1e-3 else self.mean:.3f}, $\sigma$ = {self.standard_deviation:.3f}"
        self.label = lab

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
        return (1 / (self.standard_deviation * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * (((x - self.mean) / self.standard_deviation) ** 2)
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

    def _plot_element(self, axes: plt.Axes, z_order: int) -> None:
        """
        Plots the element in the specified axes.
        """
        params = {
            "facecolor": (
                to_rgba(self.face_color, self.alpha)
                if self.face_color != "default" and self.alpha != "default"
                else "default"
            ),
            "edgecolor": (
                to_rgba(self.edge_color, 1)
                if self.edge_color != "default"
                else self.edge_color
            ),
            "linewidth": self.line_width,
        }
        params = {k: v for k, v in params.items() if v != "default"}
        self.handle = Polygon(
            np.array([[0, 2, 2, 3, 3, 1, 1, 0, 0], [0, 0, 1, 1, 2, 2, 3, 3, 0]]).T,
            **params,
        )
        params = {
            "facecolor": (
                to_rgba(self.face_color, self.alpha)
                if self.face_color != "default" and self.alpha != "default"
                else "default"
            ),
            "edgecolor": (
                to_rgba(self.edge_color, 1)
                if self.edge_color != "default"
                else self.edge_color
            ),
            "histtype": self.hist_type,
            "linewidth": self.line_width,
            "density": self.normalize,
        }
        params = {k: v for k, v in params.items() if v != "default"}
        axes.hist(
            self.data,
            bins=self.number_of_bins,
            label=self.label,
            zorder=z_order - 1,
            **params,
        )
        if self._show_pdf:
            normal = (
                self._normal_normalized
                if self.normalize
                else self._normal_not_normalized
            )
            num_of_points = 500
            x_data = np.linspace(self._bin_edges[0], self._bin_edges[-1], num_of_points)
            y_data = normal(x_data)
            params = {
                "color": self.pdf_curve_color,
            }
            params = {k: v for k, v in params.items() if v != "default"}
            axes.plot(
                x_data,
                y_data,
                zorder=z_order,
                **params,
            )
            curve_max_y = normal(self.mean)
            curve_std_y = normal(self.mean + self.standard_deviation)
            if self.pdf_show_std:
                params = {}
                if self.pdf_std_color != "default":
                    params["colors"] = [self.pdf_std_color, self.pdf_std_color]
                plt.vlines(
                    [
                        self.mean - self.standard_deviation,
                        self.mean + self.standard_deviation,
                    ],
                    [0, 0],
                    [curve_std_y, curve_std_y],
                    linestyles=["dashed"],
                    zorder=z_order - 1,
                    **params,
                )
            if self.pdf_show_mean:
                params = {}
                if self.pdf_mean_color != "default":
                    params["colors"] = [self.pdf_mean_color]
                plt.vlines(
                    self.mean,
                    0,
                    curve_max_y,
                    linestyles=["dashed"],
                    zorder=z_order - 1,
                    **params,
                )

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
            raise ValueError("Currently only normal distribution is supported.")
        self._show_pdf = True
        self._pdf_type = type
        self.pdf_show_mean = show_mean
        self.pdf_show_std = show_std
        self.pdf_curve_color = curve_color
        self.pdf_mean_color = mean_color
        self.pdf_std_color = std_color
