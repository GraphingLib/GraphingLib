from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Protocol, Self

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Polygon
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d

from .graph_elements import Point


class Fit(Protocol):
    def _plot_element(self, axes: plt.Axes, z_order: int) -> None:
        pass

    def show_residual_curves(
        self,
        sigma_multiplier: float,
        color: str,
        line_width: float,
        line_style: str,
    ) -> None:
        pass

    def calculate_residuals(self) -> np.ndarray:
        pass


@dataclass
class Curve:
    """
    A general continuous curve.
    """

    x_data: ArrayLike
    y_data: ArrayLike
    label: Optional[str] = None
    color: str = "default"
    line_width: float | Literal["default"] = "default"
    line_style: str = "default"
    errorbars: bool = field(default=False, init=False)
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
        x_data = np.linspace(x_min, x_max, number_of_points)
        y_data = func(x_data)
        return cls(x_data, y_data, label, color, line_width, line_style)

    def __post_init__(self):
        self.x_data = np.array(self.x_data)
        self.y_data = np.array(self.y_data)

    def __add__(self, other: Self | float) -> Self:
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

    def __sub__(self, other: Self | float) -> Self:
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

    def __mul__(self, other: Self | float) -> Self:
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

    def __truediv__(self, other: Self | float) -> Self:
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

    def __iter__(self):
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
        self.errorbars = True
        self.x_error = np.array(x_error)
        self.y_error = np.array(y_error)
        self.errorbars_color = errorbars_color
        self.errorbars_line_width = errorbars_line_width
        self.cap_thickness = cap_thickness
        self.cap_width = cap_width

    def get_point_at_x(
        self,
        x: float,
        interpolation_kind: str = "linear",
        label: Optional[str] = None,
        color: str = "default",
        edge_color: str = "default",
        marker_size: float | Literal["default"] = "default",
        marker_style: str = "default",
        line_width: float | Literal["default"] = "default",
    ) -> Point:
        point = Point(
            x,
            float(interp1d(self.x_data, self.y_data, kind=interpolation_kind)(x)),
            label=label,
            color=color,
            edge_color=edge_color,
            marker_size=marker_size,
            marker_style=marker_style,
            edge_width=line_width,
        )
        return point

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
        xs = self.x_data
        ys = self.y_data
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

    def get_derivative_curve(
        self,
        label: Optional[str] = None,
        color: str = "default",
        line_width: float | Literal["default"] = "default",
        line_style: str = "default",
    ) -> Self:
        x_data = self.x_data
        y_data = np.gradient(self.y_data, x_data)
        return Curve(x_data, y_data, label, color, line_width, line_style)

    def get_integral_curve(
        self,
        label: Optional[str] = None,
        color: str = "default",
        line_width: float | Literal["default"] = "default",
        line_style: str = "default",
    ) -> Self:
        x_data = self.x_data
        y_data = np.cumsum(self.y_data) * np.diff(x_data)[0]
        return Curve(x_data, y_data, label, color, line_width, line_style)

    def get_tangent_curve(
        self,
        x: float,
        label: Optional[str] = None,
        color: str = "default",
        line_width: float | Literal["default"] = "default",
        line_style: str = "default",
    ) -> Self:
        point = self.get_point_at_x(x)
        gradient = self.get_derivative_curve().get_point_at_x(x).y
        y_data = gradient * (self.x_data - x) + point.y
        tangent_curve = Curve(self.x_data, y_data, label, color, line_width, line_style)
        return tangent_curve

    def get_normal_curve(
        self,
        x: float,
        label: Optional[str] = None,
        color: str = "default",
        line_width: float | Literal["default"] = "default",
        line_style: str = "default",
    ) -> Self:
        point = self.get_point_at_x(x)
        gradient = self.get_derivative_curve().get_point_at_x(x).y
        y_data = -1 / gradient * (self.x_data - x) + point.y
        normal_curve = Curve(self.x_data, y_data, label, color, line_width, line_style)
        return normal_curve

    def slope_at(self, x: float) -> float:
        return self.get_derivative_curve().get_point_at_x(x).y

    def arc_length_between(self, x1: float, x2: float) -> float:
        y_data = self.y_data
        x_data = self.x_data
        f = interp1d(x_data, y_data)
        x = np.linspace(x1, x2, 1000)
        y = f(x)
        return np.trapz(np.sqrt(1 + np.gradient(y, x) ** 2), x)

    def area_between(self, x1: float, x2: float, fill_under: bool = False) -> float:
        if fill_under:
            self._fill_curve_between = (x1, x2)
        y_data = self.y_data
        x_data = self.x_data
        f = interp1d(x_data, y_data)
        x = np.linspace(x1, x2, 1000)
        y = f(x)
        return np.trapz(y, x)

    # A method with calculates the intersection point of two functions. Returns a list of Point objects. Uses interp and npwhere
    def intersection(
        self,
        other: Self,
        labels: Optional[list[str] | str] = None,
        colors: list[str] | str = "default",
        edge_colors: list[str] | str = "default",
        marker_sizes: list[float] | float | Literal["default"] = "default",
        marker_styles: list[str] | str = "default",
        edge_widths: list[float] | float | Literal["default"] = "default",
    ) -> list[Point]:
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
            points.append(
                Point(
                    x_val,
                    y_val,
                    label=label,
                    color=color,
                    edge_color=edge_color,
                    marker_size=marker_size,
                    marker_style=marker_style,
                    edge_width=edge_width,
                )
            )
        return points

    def _plot_element(self, axes: plt.Axes, z_order: int) -> None:
        (self.handle,) = axes.plot(
            self.x_data,
            self.y_data,
            color=self.color,
            linewidth=self.line_width,
            linestyle=self.line_style,
            label=self.label,
            zorder=z_order,
        )
        if self.errorbars:
            axes.errorbar(
                self.x_data,
                self.y_data,
                xerr=self.x_error,
                yerr=self.y_error,
                color=self.errorbars_color,
                elinewidth=self.errorbars_line_width,
                capsize=self.cap_width,
                capthick=self.cap_thickness,
                fmt="none",
                zorder=z_order - 1,
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


@dataclass
class Scatter:
    """
    A general scatter plot.
    """

    x_data: ArrayLike
    y_data: ArrayLike
    label: Optional[str] = None
    face_color: str = "default"
    edge_color: str = "default"
    marker_size: float | Literal["default"] = "default"
    marker_style: str = "default"
    errorbars: bool = False

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
        number_of_points: int = 500,
    ) -> Self:
        x_data = np.linspace(x_min, x_max, number_of_points)
        y_data = func(x_data)
        return cls(
            x_data, y_data, label, face_color, edge_color, marker_size, marker_style
        )

    def __post_init__(self) -> None:
        self.x_data = np.array(self.x_data)
        self.y_data = np.array(self.y_data)

    def __add__(self, other: Self | float) -> Self:
        """Check that x arrays are the same length and the same values, then add y arrays."""
        try:
            assert np.array_equal(self.x_data, other.x_data)
        except AssertionError:
            raise ValueError("Cannot add two scatter plots with different x values.")
        if isinstance(other, Scatter):
            new_y_data = self.y_data + other.y_data
            return Scatter(self.x_data, new_y_data)
        elif isinstance(other, (int, float)):
            new_y_data = self.y_data + other
            return Scatter(self.x_data, new_y_data)
        else:
            raise TypeError(
                "Can only add a scatter plot to another scatter plot or a number."
            )

    def __sub__(self, other: Self | float) -> Self:
        """Check that x arrays are the same length and the same values, then subtract y arrays."""
        try:
            assert np.array_equal(self.x_data, other.x_data)
        except AssertionError:
            raise ValueError(
                "Cannot subtract two scatter plots with different x values."
            )
        if isinstance(other, Scatter):
            new_y_data = self.y_data - other.y_data
            return Scatter(self.x_data, new_y_data)
        elif isinstance(other, (int, float)):
            new_y_data = self.y_data - other
            return Scatter(self.x_data, new_y_data)
        else:
            raise TypeError(
                "Can only subtract a scatter plot from another scatter plot or a number."
            )

    def __mul__(self, other: Self | float) -> Self:
        """Check that x arrays are the same length and the same values, then multiply y arrays."""
        try:
            assert np.array_equal(self.x_data, other.x_data)
        except AssertionError:
            raise ValueError(
                "Cannot multiply two scatter plots with different x values."
            )
        if isinstance(other, Scatter):
            new_y_data = self.y_data * other.y_data
            return Scatter(self.x_data, new_y_data)
        elif isinstance(other, (int, float)):
            new_y_data = self.y_data * other
            return Scatter(self.x_data, new_y_data)
        else:
            raise TypeError(
                "Can only multiply a scatter plot by another scatter plot or a number."
            )

    def __truediv__(self, other: Self | float) -> Self:
        """Check that x arrays are the same length and the same values, then divide y arrays."""
        try:
            assert np.array_equal(self.x_data, other.x_data)
        except AssertionError:
            raise ValueError("Cannot divide two scatter plots with different x values.")
        if isinstance(other, Scatter):
            new_y_data = self.y_data / other.y_data
            return Scatter(self.x_data, new_y_data)
        elif isinstance(other, (int, float)):
            new_y_data = self.y_data / other
            return Scatter(self.x_data, new_y_data)
        else:
            raise TypeError(
                "Can only divide a scatter plot by another scatter plot or a number."
            )

    def __iter__(self):
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
        self.errorbars = True
        self.x_error = np.array(x_error)
        self.y_error = np.array(y_error)
        self.errorbars_color = errorbars_color
        self.errorbars_line_width = errorbars_line_width
        self.cap_thickness = cap_thickness
        self.cap_width = cap_width

    def get_point_at_x(
        self,
        x: float,
        interpolation_kind: str = "linear",
        label: Optional[str] = None,
        color: str = "default",
        edge_color: str = "default",
        marker_size: float | Literal["default"] = "default",
        marker_style: str = "default",
        line_width: float | Literal["default"] = "default",
    ) -> Point:
        point = Point(
            x,
            float(interp1d(self.x_data, self.y_data, kind=interpolation_kind)(x)),
            label=label,
            color=color,
            edge_color=edge_color,
            marker_size=marker_size,
            marker_style=marker_style,
            edge_width=line_width,
        )
        return point

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
        xs = self.x_data
        ys = self.y_data
        assert isinstance(xs, np.ndarray) and isinstance(ys, np.ndarray)
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
        self.handle = axes.scatter(
            self.x_data,
            self.y_data,
            color=self.face_color,
            edgecolors=self.edge_color,
            s=self.marker_size,
            marker=self.marker_style,
            label=self.label,
            zorder=z_order,
        )
        if self.errorbars:
            axes.errorbar(
                self.x_data,
                self.y_data,
                xerr=self.x_error,
                yerr=self.y_error,
                color=self.errorbars_color,
                elinewidth=self.errorbars_line_width,
                capsize=self.cap_width,
                capthick=self.cap_thickness,
                fmt="none",
                zorder=z_order - 1,
            )


@dataclass
class Histogram:
    """
    A histogram plot with minor changes to the lable icon.
    """

    x_data: ArrayLike
    number_of_bins: int
    label: Optional[str] = None
    face_color: str = "default"
    edge_color: str = "default"
    hist_type: str = "default"
    alpha: float | Literal["default"] = "default"
    line_width: float | Literal["default"] = "default"
    normalize: bool | Literal["default"] = "default"
    show_pdf: str = "default"
    show_params: bool | Literal["default"] = "default"

    def __post_init__(self) -> None:
        self.x_data = np.array(self.x_data)
        self.mean = np.mean(self.x_data)
        self.standard_deviation = np.std(self.x_data)
        parameters = np.histogram(
            self.x_data, bins=self.number_of_bins, density=self.normalize
        )
        self._bin_heights, bin_edges = parameters[0], parameters[1]
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = bin_edges[1:] - bin_width / 2
        self._bin_width = bin_width
        self._bin_centers = bin_centers
        self._bin_edges = bin_edges
        self._create_label()

    @classmethod
    def plot_residuals_from_fit(
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
        show_pdf: str | Literal["default"] = "default",
        show_params: bool | Literal["default"] = "default",
    ) -> Self:
        residuals = fit.calculate_residuals()
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
            show_pdf,
            show_params,
        )

    def _create_label(self) -> None:
        lab = self.label
        if lab and self.show_params:
            lab += (
                " :\n"
                + f"$\mu$ = {self.mean:.3f}, $\sigma$ = {self.standard_deviation:.3f}"
            )
        elif self.show_params:
            lab = f"$\mu$ = {self.mean:.3f}, $\sigma$ = {self.standard_deviation:.3f}"
        self.label = lab

    def _normal_normalized(self, x: ArrayLike) -> ArrayLike:
        x = np.array(x)
        return (1 / (self.standard_deviation * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * (((x - self.mean) / self.standard_deviation) ** 2)
        )

    def _normal_not_normalized(self, x: ArrayLike) -> ArrayLike:
        x = np.array(x)
        return sum(self._bin_heights) * self._bin_width * self._normal_normalized(x)

    def _plot_element(self, axes: plt.Axes, z_order: int) -> None:
        self.handle = Polygon(
            np.array([[0, 2, 2, 3, 3, 1, 1, 0, 0], [0, 0, 1, 1, 2, 2, 3, 3, 0]]).T,
            facecolor=to_rgba(self.face_color, self.alpha),
            edgecolor=to_rgba(self.edge_color, 1),
            linewidth=1,
        )
        axes.hist(
            self.x_data,
            bins=self.number_of_bins,
            facecolor=to_rgba(self.face_color, self.alpha),
            edgecolor=to_rgba(self.edge_color, 1),
            label=self.label,
            histtype=self.hist_type,
            linewidth=self.line_width,
            density=self.normalize,
            zorder=z_order - 1,
        )
        if self.show_pdf in ["normal", "gaussian"]:
            normal = (
                self._normal_normalized
                if self.normalize
                else self._normal_not_normalized
            )
            num_of_points = 500
            x_data = np.linspace(self._bin_edges[0], self._bin_edges[-1], num_of_points)
            y_data = normal(x_data)
            axes.plot(
                x_data,
                y_data,
                color=self.edge_color,
                zorder=z_order,
            )
            curve_max_y = normal(self.mean)
            curve_std_y = normal(self.mean + self.standard_deviation)
            plt.vlines(
                [
                    self.mean - self.standard_deviation,
                    self.mean,
                    self.mean + self.standard_deviation,
                ],
                [0, 0, 0],
                [curve_std_y, curve_max_y, curve_std_y],
                linestyles=["dashed"],
                colors=["k", "r", "k"],
                zorder=z_order - 1,
            )


def remove_duplicates(numbers: list[float], relative_tolerance: float):
    result: list[float] = []
    for num in numbers:
        similar_points: list[float] = [num]
        for existing_num in result:
            if abs(num - existing_num) <= relative_tolerance * abs(existing_num):
                similar_points.append(existing_num)
        if len(similar_points) > 1:
            average = sum(similar_points) / len(similar_points)
            result = [x for x in result if x not in similar_points]
            result.append(average)
        else:
            result.append(num)
    return result
