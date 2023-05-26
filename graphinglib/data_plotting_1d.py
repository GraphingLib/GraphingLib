from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Self

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Polygon
from numpy.typing import ArrayLike


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

    def _calculate_residuals(self) -> np.ndarray:
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
    line_width: int = "default"
    line_style: str = "default"
    errorbars: bool = False

    @classmethod
    def from_function(
        cls,
        func: Callable[[ArrayLike], ArrayLike],
        x_min: float,
        x_max: float,
        label: Optional[str] = None,
        color: str = "default",
        line_width: int = "default",
        line_style: str = "default",
        number_of_points: int = 500,
    ) -> Self:
        x_data = np.linspace(x_min, x_max, number_of_points)
        y_data = func(x_data)
        return cls(x_data, y_data, label, color, line_width, line_style)

    def add_errorbars(
        self,
        x_error: Optional[ArrayLike] = None,
        y_error: Optional[ArrayLike] = None,
        cap_width: float = "default",
        errorbars_color: str = "default",
        errorbars_line_width: float = "default",
        cap_thickness: float = "default",
    ) -> None:
        self.errorbars = True
        self.x_error = x_error
        self.y_error = y_error
        self.errorbars_color = errorbars_color
        self.errorbars_line_width = errorbars_line_width
        self.cap_thickness = cap_thickness
        self.cap_width = cap_width

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
    marker_size: float = "default"
    marker_style: str = "default"
    errorbars: bool = False

    @classmethod
    def from_function(
        cls,
        func: Callable,
        x_min: float,
        x_max: float,
        label: Optional[str] = None,
        face_color: str = "default",
        edge_color: str = "default",
        marker_size: int = "default",
        marker_style: str = "default",
        number_of_points: int = 500,
    ) -> Self:
        x_data = np.linspace(x_min, x_max, number_of_points)
        y_data = func(x_data)
        return cls(
            x_data, y_data, label, face_color, edge_color, marker_size, marker_style
        )

    def add_errorbars(
        self,
        x_error: Optional[ArrayLike] = None,
        y_error: Optional[ArrayLike] = None,
        cap_width: float = "default",
        errorbars_color: str = "default",
        errorbars_line_width: float = "default",
        cap_thickness: float = "default",
    ) -> None:
        self.errorbars = True
        self.x_error = x_error
        self.y_error = y_error
        self.errorbars_color = errorbars_color
        self.errorbars_line_width = errorbars_line_width
        self.cap_thickness = cap_thickness
        self.cap_width = cap_width

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
    alpha: float = "default"
    line_width: float = "default"
    normalize: bool = "default"
    show_pdf: str = "default"
    show_params: bool = "default"

    def __post_init__(self) -> None:
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
        alpha: int = "default",
        line_width: int = "default",
        normalize: bool = "default",
        show_pdf: str = "default",
        show_params: bool = "default",
    ) -> Self:
        residuals = fit._calculate_residuals()
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
        if self.label and self.show_params:
            lab += (
                " :\n"
                + f"$\mu$ = {self.mean:.3f}, $\sigma$ = {self.standard_deviation:.3f}"
            )
        elif self.show_params:
            lab = f"$\mu$ = {self.mean:.3f}, $\sigma$ = {self.standard_deviation:.3f}"
        self.label = lab

    def _normal_normalized(self, x: ArrayLike) -> ArrayLike:
        return (1 / (self.standard_deviation * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * (((x - self.mean) / self.standard_deviation) ** 2)
        )

    def _normal_not_normalized(self, x: ArrayLike) -> ArrayLike:
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
