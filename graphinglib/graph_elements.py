from dataclasses import dataclass, field
from typing import Literal, Optional, Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from numpy.typing import ArrayLike

from .legend_artists import VerticalLineCollection


class Plottable(Protocol):
    def _plot_element(self, axes: plt.Axes, z_order: int):
        pass


class GraphingException(Exception):
    """
    General exception raised for the GraphingLib module.
    """

    pass


class Hlines:
    """
    Horizontal lines.
    """

    def __init__(
        self,
        y: ArrayLike,
        x_min: ArrayLike,
        x_max: ArrayLike,
        label: Optional[str] = None,
        colors: list[str] | str = "default",
        line_styles: list[str] | str = "default",
    ) -> None:
        if isinstance(y, (int, float)):
            self.y = y
        elif isinstance(y, (list, np.ndarray)):
            self.y = np.array(y)
        if isinstance(x_min, (int, float)):
            self.x_min = x_min
        elif isinstance(x_min, (list, np.ndarray)):
            self.x_min = np.array(x_min)
        if isinstance(x_max, (int, float)):
            self.x_max = x_max
        elif isinstance(x_max, (list, np.ndarray)):
            self.x_max = np.array(x_max)
        self.label = label
        self.colors = colors
        self.line_styles = line_styles
        if isinstance(self.y, (int, float)) and isinstance(
            self.colors, (list, np.ndarray)
        ):
            raise GraphingException("There can't be multiple colors for a single line!")
        if isinstance(self.y, (int, float)) and isinstance(
            self.line_styles, (list, np.ndarray)
        ):
            raise GraphingException(
                "There can't be multiple linestyles for a single line!"
            )
        if (
            isinstance(self.y, (list, np.ndarray))
            and isinstance(self.colors, list)
            and isinstance(self.line_styles, list)
        ):
            if len(self.y) != len(self.colors) or len(self.y) != len(self.line_styles):
                raise GraphingException(
                    "There must be the same number of colors, "
                    + "linestyles and lines!"
                )

    def _plot_element(self, axes: plt.Axes, z_order: int) -> None:
        if isinstance(self.y, list) and len(self.y) > 1:
            axes.hlines(
                self.y,
                self.x_min,
                self.x_max,
                colors=self.colors,
                linestyles=self.line_styles,
                zorder=z_order,
            )
            self.handle = LineCollection(
                [[(0, 0)]] * (len(self.y) if len(self.y) <= 3 else 3),
                color=self.colors,
                linestyle="solid",
            )
        else:
            self.handle = axes.hlines(
                self.y,
                self.x_min,
                self.x_max,
                colors=self.colors,
                linestyles=self.line_styles,
                zorder=z_order,
            )


class Vlines:
    """
    Vertical lines.
    """

    def __init__(
        self,
        x: ArrayLike,
        y_min: ArrayLike,
        y_max: ArrayLike,
        label: Optional[str] = None,
        colors: list[str] | str = "default",
        line_styles: list[str] | str = "default",
    ) -> None:
        if isinstance(x, (int, float)):
            self.x = x
        elif isinstance(x, (list, np.ndarray)):
            self.x = np.array(x)
        if isinstance(y_min, (int, float)):
            self.y_min = y_min
        elif isinstance(y_min, (list, np.ndarray)):
            self.y_min = np.array(y_min)
        if isinstance(y_max, (int, float)):
            self.y_max = y_max
        elif isinstance(y_max, (list, np.ndarray)):
            self.y_max = np.array(y_max)
        self.label = label
        self.colors = colors
        self.line_styles = line_styles
        if isinstance(self.x, (int, float)) and isinstance(
            self.colors, (list, np.ndarray)
        ):
            raise GraphingException("There can't be multiple colors for a single line!")
        if isinstance(self.x, (int, float)) and isinstance(
            self.line_styles, (list, np.ndarray)
        ):
            raise GraphingException(
                "There can't be multiple linestyles for a single line!"
            )
        if (
            isinstance(self.x, (list, np.ndarray))
            and isinstance(self.colors, list)
            and isinstance(self.line_styles, list)
        ):
            if len(self.x) != len(self.colors) or len(self.x) != len(self.line_styles):
                raise GraphingException(
                    "There must be the same number of colors, "
                    + "linestyles and lines!"
                )

    def _plot_element(self, axes: plt.Axes, z_order: int) -> None:
        if isinstance(self.x, list) and len(self.x) > 1:
            axes.vlines(
                self.x,
                self.y_min,
                self.y_max,
                colors=self.colors,
                linestyles=self.line_styles,
                zorder=z_order,
            )
            self.handle = VerticalLineCollection(
                [[(0, 0)]] * (len(self.x) if len(self.x) <= 4 else 4),
                color=self.colors,
                linestyle="solid",
            )
        else:
            self.handle = axes.vlines(
                self.x,
                self.y_min,
                self.y_max,
                colors=self.colors,
                linestyles=self.line_styles,
                zorder=z_order,
            )


class Point:
    """
    A general point object.
    """

    def __init__(
        self,
        x: float,
        y: float,
        label: Optional[str] = None,
        color: str = "default",
        edge_color: str = "default",
        marker_size: float | Literal["default"] = "default",
        marker_style: str = "default",
        edge_width: float | Literal["default"] = "default",
        font_size: int | Literal["same as figure"] = "same as figure",
        text_color: str = "k",
        h_align: str = "left",
        v_align: str = "bottom",
    ) -> None:
        if not isinstance(x, int | float) or not isinstance(y, int | float):
            raise GraphingException(
                "The x and y coordinates for a point must be a single number each!"
            )
        else:
            self.x = x
            self.y = y
        self.label = label
        self.color = color
        self.edge_color = edge_color
        self.marker_size = marker_size
        self.marker_style = marker_style
        self.edge_width = edge_width
        self.font_size = font_size
        self.text_color = text_color
        self.h_align = h_align
        self.v_align = v_align
        self._show_coordinates: bool = False

    def add_coordinates(self) -> None:
        self._show_coordinates = True

    def _plot_element(self, axes: plt.Axes, z_order: int) -> None:
        size = self.font_size if self.font_size != "same as figure" else None
        prefix = " " if self.h_align == "left" else ""
        postfix = " " if self.h_align == "right" else ""
        if self.label is not None and not self._show_coordinates:
            point_label = prefix + self.label + postfix
        else:
            point_label = None
        axes.scatter(
            self.x,
            self.y,
            c=self.color,
            edgecolors=self.edge_color,
            s=self.marker_size,
            marker=self.marker_style,
            linewidths=self.edge_width,
            zorder=z_order,
        )
        axes.annotate(
            point_label,
            (self.x, self.y),
            color=self.text_color,
            fontsize=size,
            horizontalalignment=self.h_align,
            verticalalignment=self.v_align,
            zorder=z_order,
        )
        if self._show_coordinates:
            prefix = " " if self.h_align == "left" else ""
            postfix = " " if self.h_align == "right" else ""
            if self.label is not None:
                point_label = (
                    prefix
                    + self.label
                    + " : "
                    + f"({self.x:.3f}, {self.y:.3f})"
                    + postfix
                )
            else:
                point_label = prefix + f"({self.x:.3f}, {self.y:.3f})" + postfix
            axes.annotate(
                point_label,
                (self.x, self.y),
                color=self.text_color,
                fontsize=size,
                horizontalalignment=self.h_align,
                verticalalignment=self.v_align,
                zorder=z_order,
            )


@dataclass
class Text:
    """
    A text object to be displayed on a graph.
    """

    x: float
    y: float
    text: str
    color: str = "default"
    font_size: float | Literal["same as figure"] = "same as figure"
    h_align: str = "default"
    v_align: str = "default"
    _arrow_pointing_to: Optional[tuple[float]] = field(default=None, init=False)

    def attach_arrow(
        self,
        points_to: tuple[float],
        width: Optional[float] = None,
        shrink: Optional[float] = None,
        head_width: Optional[float] = None,
        head_length: Optional[float] = None,
    ) -> None:
        self._arrow_pointing_to = points_to
        self.width = width
        self.shrink = shrink
        self.arrow_properties = {"color": self.color}
        if width is not None:
            self.arrow_properties["width"] = width
        if shrink is not None:
            self.arrow_properties["shrink"] = shrink
        if head_width is not None:
            self.arrow_properties["headwidth"] = head_width
        if head_length is not None:
            self.arrow_properties["headlength"] = head_length

    def _plot_element(self, axes: plt.Axes, z_order: int) -> None:
        size = self.font_size if self.font_size != "same as figure" else None
        axes.text(
            self.x,
            self.y,
            self.text,
            horizontalalignment=self.h_align,
            verticalalignment=self.v_align,
            color=self.color,
            fontsize=size,
            zorder=z_order,
        )
        if self._arrow_pointing_to is not None:
            axes.annotate(
                self.text,
                self._arrow_pointing_to,
                xytext=(self.x, self.y),
                color=self.color,
                arrowprops=self.arrow_properties,
                fontsize=size,
                horizontalalignment=self.h_align,
                verticalalignment=self.v_align,
                zorder=z_order,
            )
