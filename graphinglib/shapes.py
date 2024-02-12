from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from .graph_elements import Point

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


@dataclass
class Circle:
    """This class implements a Circle object with a given center and radius.

    Parameters
    ----------
    x_center : float
        The x coordinate of the :class:`~graphinglib.shapes.Circle`.
    y_center : float
        The y coordinate of the :class:`~graphinglib.shapes.Circle`.
    radius : float
        The radius of the :class:`~graphinglib.shapes.Circle`.
    fill : bool, optional
        Whether the circle should be filled or not.
        Default depends on the ``figure_style`` configuration.
    color : str, optional
        The color of the circle (both the line and the fill).
        Default depends on the ``figure_style`` configuration.
    line_width : float, optional
        The width of the line.
        Default depends on the ``figure_style`` configuration.
    line_style : str, optional
        The style of the line.
        Default depends on the ``figure_style`` configuration.
    fill_alpha : float, optional
        The alpha value of the fill.
        Default depends on the ``figure_style`` configuration.
    """

    x_center: float
    y_center: float
    radius: float
    fill: bool = "default"
    color: str = "default"
    line_width: float | Literal["default"] = "default"
    line_style: str = "default"
    fill_alpha: float | Literal["default"] = "default"

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("The radius must be positive")

    def __contains__(self, point: Point) -> bool:
        return (point.x - self.x_center) ** 2 + (
            point.y - self.y_center
        ) ** 2 <= self.radius**2

    def get_area(self) -> float:
        """Returns the area of the circle.

        Returns
        -------
        float
            The area of the circle.
        """
        return np.pi * self.radius**2

    def get_circumference(self) -> float:
        """Returns the circumference of the circle.

        Returns
        -------
        float
            The circumference of the circle.
        """
        return 2 * np.pi * self.radius

    def get_center_coordinates(self) -> tuple[float, float]:
        """Returns the coordinates of the center of the circle.

        Returns
        -------
        tuple of floats
            The coordinates of the center of the circle.
        """
        return (self.x_center, self.y_center)

    def create_center_point(self) -> Point:
        """Returns the center point of the circle.

        Returns
        -------
        tuple of floats or :class:`~graphinglib.graph_elements.Point`
            The center point of the circle.
        """
        return Point(self.x_center, self.y_center)

    def get_coordinates_at_x(self, x: float) -> list[tuple[float, float]]:
        """Returns the coordinates on the circle at the specified x coordinate.

        Parameters
        ----------
        x : float
            The x coordinate of the points.

        Returns
        -------
        list[tuple[float, float]]
            The coordinates on the circle at the specified x coordinate.
        """
        if x < self.x_center - self.radius or x > self.x_center + self.radius:
            raise ValueError(
                f"x must be between {self.x_center - self.radius} and {self.x_center + self.radius}"
            )
        y = np.sqrt(self.radius**2 - (x - self.x_center) ** 2)
        if y == 0:
            return [(x, y)]
        return [(x, y), (x, -y)]

    def create_points_at_x(self, x: float) -> list[Point]:
        """Returns the points on the circle at the specified x coordinate.

        Parameters
        ----------
        x : float
            The x coordinate of the points.

        Returns
        -------
        list[:class:`~graphinglib.graph_elements.Point`]
            The points on the circle at the specified x coordinate.
        """
        if x < self.x_center - self.radius or x > self.x_center + self.radius:
            raise ValueError(
                f"x must be between {self.x_center - self.radius} and {self.x_center + self.radius}"
            )
        y = np.sqrt(self.radius**2 - (x - self.x_center) ** 2)
        if y == 0:
            return [Point(x, y)]
        return [Point(x, y), Point(x, -y)]

    def get_coordinates_at_y(self, y: float) -> list[tuple[float, float]]:
        """Returns the coordinates on the circle at the specified y coordinate.

        Parameters
        ----------
        y : float
            The y coordinate of the points.

        Returns
        -------
        list[tuple[float, float]]
            The coordinates on the circle at the specified y coordinate.
        """
        if y < self.y_center - self.radius or y > self.y_center + self.radius:
            raise ValueError(
                f"y must be between {self.y_center - self.radius} and {self.y_center + self.radius}"
            )
        x = np.sqrt(self.radius**2 - (y - self.y_center) ** 2)
        if x == 0:
            return [(x, y)]
        return [(x, y), (-x, y)]

    def create_points_at_y(self, y: float) -> list[Point]:
        """Returns the points on the circle at the specified y coordinate.

        Parameters
        ----------
        y : float
            The y coordinate of the points.

        Returns
        -------
        list[]:class:`~graphinglib.graph_elements.Point`]
            The points on the circle at the specified y coordinate.
        """
        if y < self.y_center - self.radius or y > self.y_center + self.radius:
            raise ValueError(
                f"y must be between {self.y_center - self.radius} and {self.y_center + self.radius}"
            )
        x = np.sqrt(self.radius**2 - (y - self.y_center) ** 2)
        if x == 0:
            return [Point(x, y)]
        return [Point(x, y), Point(-x, y)]

    def get_coordinates_at_angle(
        self, angle: float, degrees=False
    ) -> tuple[float, float]:
        """Returns the coordinates on the circle at the specified angle.

        Parameters
        ----------
        angle : float
            The angle of the point.
        degrees : bool, optional
            Whether the angle is in degrees or radians.
            Default is ``False`` (angle is in radians).

        Returns
        -------
        tuple of floats
            The coordinates on the circle at the specified angle.
        """
        if degrees:
            angle = np.radians(angle)
        x = self.x_center + self.radius * np.cos(angle)
        y = self.y_center + self.radius * np.sin(angle)
        return (x, y)

    def create_point_at_angle(self, angle: float, degrees=False) -> Point:
        """Returns the point on the circle at the specified angle.

        Parameters
        ----------
        angle : float
            The angle of the point.
        degrees : bool, optional
            Whether the angle is in degrees or radians.
            Default is ``False`` (angle is in radians).

        Returns
        -------
        :class:`~graphinglib.graph_elements.Point`
            The point on the circle at the specified angle.
        """
        if degrees:
            angle = np.radians(angle)
        x = self.x_center + self.radius * np.cos(angle)
        y = self.y_center + self.radius * np.sin(angle)
        return Point(x, y)

    def _plot_element(self, axes: plt.Axes, z_order: int) -> None:
        """
        Plots the element in the specified
        `Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_.
        """

        def xy(r, phi, x0, y0):
            return r * np.cos(phi) + x0, r * np.sin(phi) + y0

        phi = np.linspace(0, 2 * np.pi, 100)
        x, y = xy(self.radius, phi, self.x_center, self.y_center)
        params = {
            "color": self.color,
            "linewidth": self.line_width,
            "linestyle": self.line_style,
        }

        params = {k: v for k, v in params.items() if v != "default"}
        axes.plot(x, y, zorder=z_order, **params)
        if self.fill:
            params = {
                "alpha": self.fill_alpha,
                "facecolor": self.color,
            }
            params = {k: v for k, v in params.items() if v != "default"}
            axes.fill_between(x, y, self.y_center, zorder=z_order, **params)


@dataclass
class Rectangle:
    """This class implements a Rectangle object with a given bottom left corner, width and height.

    Parameters
    ----------
    x_bottom_left : float
        The x coordinate of the bottom left corner of the :class:`~graphinglib.shapes.Rectangle`.
    y_bottom_left : float
        The y coordinate of the bottom left corner of the :class:`~graphinglib.shapes.Rectangle`.
    width : float
        The width of the :class:`~graphinglib.shapes.Rectangle`.
    height : float
        The height of the :class:`~graphinglib.shapes.Rectangle`.
    fill : bool, optional
        Whether the rectangle should be filled or not.
        Default depends on the ``figure_style`` configuration.
    color : str, optional
        The color of the rectangle (both the line and the fill).
        Default depends on the ``figure_style`` configuration.
    line_width : float, optional
        The width of the line.
        Default depends on the ``figure_style`` configuration.
    line_style : str, optional
        The style of the line.
        Default depends on the ``figure_style`` configuration.
    fill_alpha : float, optional
        The alpha value of the fill.
        Default depends on the ``figure_style`` configuration.
    """

    x_bottom_left: float
    y_bottom_left: float
    width: float
    height: float
    fill: bool = "default"
    color: str = "default"
    line_width: float | Literal["default"] = "default"
    line_style: str = "default"
    fill_alpha: float | Literal["default"] = "default"

    def __post_init__(self):
        if self.width <= 0:
            raise ValueError("The width must be positive")
        if self.height <= 0:
            raise ValueError("The height must be positive")

    @classmethod
    def from_center(
        cls,
        x: float,
        y: float,
        width: float,
        height: float,
        fill: bool = "default",
        color: str = "default",
        line_width: float | Literal["default"] = "default",
        line_style: str = "default",
        fill_alpha: float | Literal["default"] = "default",
    ) -> Self:
        """Creates a :class:`~graphinglib.shapes.Rectangle` from its center point, width and height.

        Parameters
        ----------
        x : float
            The x coordinate of the center point.
        y : float
            The y coordinate of the center point.
        width : float
            The width of the :class:`~graphinglib.shapes.Rectangle`.
        height : float
            The height of the :class:`~graphinglib.shapes.Rectangle`.
        fill : bool, optional
            Whether the rectangle should be filled or not.
            Default depends on the ``figure_style`` configuration.
        color : str, optional
            The color of the rectangle (both the line and the fill).
            Default depends on the ``figure_style`` configuration.
        line_width : float, optional
            The width of the line.
            Default depends on the ``figure_style`` configuration.
        line_style : str, optional
            The style of the line.
            Default depends on the ``figure_style`` configuration.
        fill_alpha : float, optional
            The alpha value of the fill.
            Default depends on the ``figure_style`` configuration.
        """
        return cls(
            x - width / 2,
            y - height / 2,
            width,
            height,
            fill,
            color,
            line_width,
            line_style,
            fill_alpha,
        )

    @classmethod
    def from_points(
        cls,
        point1: Point,
        point2: Point,
        fill: bool = "default",
        color: str = "default",
        line_width: float | Literal["default"] = "default",
        line_style: str = "default",
        fill_alpha: float | Literal["default"] = "default",
    ) -> Self:
        """Creates a :class:`~graphinglib.shapes.Rectangle` from two of its opposite corners.

        Parameters
        ----------
        point1 : :class:`~graphinglib.graph_elements.Point`
            The first point.
        point2 : :class:`~graphinglib.graph_elements.Point`
            The second point.
        fill : bool, optional
            Whether the rectangle should be filled or not.
            Default depends on the ``figure_style`` configuration.
        color : str, optional
            The color of the rectangle (both the line and the fill).
            Default depends on the ``figure_style`` configuration.
        line_width : float, optional
            The width of the line.
            Default depends on the ``figure_style`` configuration.
        line_style : str, optional
            The style of the line.
            Default depends on the ``figure_style`` configuration.
        fill_alpha : float, optional
            The alpha value of the fill.
            Default depends on the ``figure_style`` configuration.
        """
        if point1.x == point2.x or point1.y == point2.y:
            raise ValueError("The points must not be on the same line")
        return cls(
            min(point1.x, point2.x),
            min(point1.y, point2.y),
            abs(point1.x - point2.x),
            abs(point1.y - point2.y),
            fill,
            color,
            line_width,
            line_style,
            fill_alpha,
        )

    def __contains__(self, point: Point) -> bool:
        return (self.x_bottom_left <= point.x <= self.x_bottom_left + self.width) and (
            self.y_bottom_left <= point.y <= self.y_bottom_left + self.height
        )

    def get_area(self) -> float:
        """Returns the area of the rectangle.

        Returns
        -------
        float
            The area of the rectangle.
        """
        return self.width * self.height

    def get_perimeter(self) -> float:
        """Returns the perimeter of the rectangle.

        Returns
        -------
        float
            The perimeter of the rectangle.
        """
        return 2 * (self.width + self.height)

    def get_center_coordinates(self) -> tuple[float, float] | Point:
        """Returns the center coordinates of the rectangle.

        Returns
        -------
        tuple[float, float]
            The center point of the rectangle.
        """
        return (
            self.x_bottom_left + self.width / 2,
            self.y_bottom_left + self.height / 2,
        )

    def create_center_point(self) -> Point:
        """Returns the center point of the rectangle.

        Returns
        -------
        :class:`~graphinglib.graph_elements.Point`
            The center point of the rectangle.
        """
        return Point(
            self.x_bottom_left + self.width / 2,
            self.y_bottom_left + self.height / 2,
        )

    def get_coordinates_at_x(self, x: float) -> list[tuple[float, float]]:
        """Returns the coordinates on the rectangle at the specified x coordinate.

        Parameters
        ----------
        x : float
            The x coordinate of the points.

        Returns
        -------
        list[tuple[float, float]]
            The coordinates on the rectangle at the specified x coordinate.
        """
        if x <= self.x_bottom_left or x >= self.x_bottom_left + self.width:
            raise ValueError(
                f"x must be between {self.x_bottom_left} and {self.x_bottom_left + self.width}"
            )
        return [
            (x, self.y_bottom_left),
            (x, self.y_bottom_left + self.height),
        ]

    def create_points_at_x(self, x: float) -> list[Point]:
        """Returns the points on the rectangle at the specified x coordinate.

        Parameters
        ----------
        x : float
            The x coordinate of the points.

        Returns
        -------
        list[:class:`~graphinglib.graph_elements.Point`]
            The points on the rectangle at the specified x coordinate.
        """
        if x <= self.x_bottom_left or x >= self.x_bottom_left + self.width:
            raise ValueError(
                f"x must be between {self.x_bottom_left} and {self.x_bottom_left + self.width}"
            )
        return [
            Point(x, self.y_bottom_left),
            Point(x, self.y_bottom_left + self.height),
        ]

    def get_coordinates_at_y(self, y: float) -> list[tuple[float, float]]:
        """Returns the coordinates on the rectangle at the specified y coordinate.

        Parameters
        ----------
        y : float
            The y coordinate of the points.

        Returns
        -------
        list[tuple[float, float]]
            The coordinates on the rectangle at the specified y coordinate.
        """
        if y <= self.y_bottom_left or y >= self.y_bottom_left + self.height:
            raise ValueError(
                f"y must be between {self.y_bottom_left} and {self.y_bottom_left + self.height}"
            )
        return [
            (self.x_bottom_left, y),
            (self.x_bottom_left + self.width, y),
        ]

    def create_points_at_y(self, y: float) -> list[Point]:
        """Returns the points on the rectangle at the specified y coordinate.

        Parameters
        ----------
        y : float
            The y coordinate of the points.

        Returns
        -------
        list[:class:`~graphinglib.graph_elements.Point`]
            The points on the rectangle at the specified y coordinate.
        """
        if y <= self.y_bottom_left or y >= self.y_bottom_left + self.height:
            raise ValueError(
                f"y must be between {self.y_bottom_left} and {self.y_bottom_left + self.height}"
            )
        return [
            Point(self.x_bottom_left, y),
            Point(self.x_bottom_left + self.width, y),
        ]

    def _plot_element(self, axes: plt.Axes, z_order: int) -> None:
        """
        Plots the element in the specified
        `Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_.
        """
        params = {
            "color": self.color,
            "linewidth": self.line_width,
            "linestyle": self.line_style,
        }

        params = {k: v for k, v in params.items() if v != "default"}
        axes.plot(
            [
                self.x_bottom_left,
                self.x_bottom_left + self.width,
                self.x_bottom_left + self.width,
                self.x_bottom_left,
                self.x_bottom_left,
            ],
            [
                self.y_bottom_left,
                self.y_bottom_left,
                self.y_bottom_left + self.height,
                self.y_bottom_left + self.height,
                self.y_bottom_left,
            ],
            zorder=z_order,
            **params,
        )
        if self.fill:
            params = {
                "alpha": self.fill_alpha,
                "facecolor": self.color,
            }
            params = {k: v for k, v in params.items() if v != None}
            axes.fill_between(
                [
                    self.x_bottom_left,
                    self.x_bottom_left + self.width,
                    self.x_bottom_left + self.width,
                    self.x_bottom_left,
                    self.x_bottom_left,
                ],
                [
                    self.y_bottom_left,
                    self.y_bottom_left,
                    self.y_bottom_left + self.height,
                    self.y_bottom_left + self.height,
                    self.y_bottom_left,
                ],
                zorder=z_order,
                **params,
            )


@dataclass
class Arrow:
    """This class implements an arrow object.

    Parameters
    ----------
    pointA : tuple[float, float]
        Point A of the arrow. If the arrow is single-sided, refers to the tail.
    pointB : tuple[float, float]
        Point B of the arrow. If the arrow is douple-sided, refers to the head.
    color : str
        Color of the arrow. Default depends on the ``figure_style``configuration.
    width : float, optional
        Arrow line width. Default depends on the ``figure_style`` configuration.
    head_size : float, optional
        Scales the size of the arrow head.
        Default depends on the ``figure_style`` configuration.
    shrink : float
        Fraction of the total length of the arrow to shrink from both ends.
        A value of 0.5 means the arrow is no longer visible.
        Defaults to 0.
    two_sided : bool
        If ``True``, the arrow is double-sided. Defaults to ``False``
    """

    pointA: tuple[float, float]
    pointB: tuple[float, float]
    color: str = "default"
    width: float | Literal["default"] = "default"
    head_size: float | Literal["default"] = "default"
    shrink: float = 0
    two_sided: bool = False

    def _shrink_points(self):
        x_length, y_length = (
            self.pointA[0] - self.pointB[0],
            self.pointA[1] - self.pointB[1],
        )
        newA = (
            self.pointB[0] + (1 - self.shrink) * x_length,
            self.pointB[1] + (1 - self.shrink) * y_length,
        )
        newB = (
            self.pointA[0] - (1 - self.shrink) * x_length,
            self.pointA[1] - (1 - self.shrink) * y_length,
        )
        return newA, newB

    def _plot_element(self, axes: plt.Axes, z_order: int):
        if self.two_sided:
            self._style = "<|-|>"
        else:
            self._style = "-|>"
        head_length, head_width = self.head_size * 0.4, self.head_size * 0.2
        props = {
            "arrowstyle": f"{self._style}, head_width={head_width}, head_length={head_length}",
            "color": self.color,
            "linewidth": self.width,
        }
        if self.shrink != 0:
            shrinkPointA, shrinkPointB = self._shrink_points()
            print(shrinkPointA, shrinkPointB)
            axes.annotate(
                "",
                shrinkPointB,
                shrinkPointA,
                zorder=z_order,
                arrowprops=props,
            )
        else:
            axes.annotate(
                "",
                self.pointB,
                self.pointA,
                zorder=z_order,
                arrowprops=props,
            )


@dataclass
class Line:
    """This class implements a line object.

    Parameters
    ----------
    pointA : tuple[float, float]
        Point A of the line.
    pointB : tuple[float, float]
        Point B of the line.
    color : str
        Color of the line. Default depends on the ``figure_style``configuration.
    width : float, optional
        Line width. Default depends on the ``figure_style`` configuration.
    capped_line : bool
        If ``True``, the line is capped on both ends. Defaults to ``False``.
    cap_width : float
        Width of the caps. Default depends on the ``figure_style`` configuration.
    """

    pointA: tuple[float, float]
    pointB: tuple[float, float]
    color: str = "default"
    width: float | Literal["default"] = "default"
    capped_line: bool = False
    cap_width: float | Literal["default"] = "default"

    def _plot_element(self, axes: plt.axes, z_order: int):
        if self.capped_line:
            style = f"|-|, widthA={self.cap_width/2}, widthB={self.cap_width/2}"
        else:
            style = "-"
        props = {
            "arrowstyle": style,
            "color": self.color,
            "linewidth": self.width,
        }
        axes.annotate(
            "",
            self.pointA,
            self.pointB,
            zorder=z_order,
            arrowprops=props,
        )
