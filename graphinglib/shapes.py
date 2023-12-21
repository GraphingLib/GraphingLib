from dataclasses import dataclass
from typing import Any, Literal, Self

import matplotlib.pyplot as plt
import numpy as np

from .data_plotting_1d import Curve
from .graph_elements import Point


@dataclass
class Circle:
    x_center: float
    y_center: float
    radius: float
    fill: bool = "default"
    color: str = "default"
    line_width: float | Literal["default"] = "default"
    line_style: str = "default"
    fill_alpha: float | Literal["default"] = "default"

    def __post_init__(self):
        if self.radius <= 0:
            raise ValueError("The radius must be positive")

    def __contains__(self, point: Point):
        return (point.x - self.x_center) ** 2 + (
            point.y - self.y_center
        ) ** 2 <= self.radius**2

    def area(self):
        return np.pi * self.radius**2

    def circumference(self):
        return 2 * np.pi * self.radius

    def get_center_point(self, as_point_object=False):
        if as_point_object:
            return Point(self.x_center, self.y_center)
        return (self.x_center, self.y_center)

    def get_equation(self):
        return f"(x - {self.x_center})^2 + (y - {self.y_center})^2 = {self.radius}^2"

    def get_points_at_x(self, x: float, as_point_object=False):
        if x < self.x_center - self.radius or x > self.x_center + self.radius:
            raise ValueError(
                f"x must be between {self.x_center - self.radius} and {self.x_center + self.radius}"
            )
        y = np.sqrt(self.radius**2 - (x - self.x_center) ** 2)
        if y == 0:
            if as_point_object:
                return [Point(x, y)]
            return [(x, y)]
        if as_point_object:
            return [Point(x, y), Point(x, -y)]
        return [(x, y), (x, -y)]

    def get_points_at_y(self, y: float, as_point_object=False):
        if y < self.y_center - self.radius or y > self.y_center + self.radius:
            raise ValueError(
                f"y must be between {self.y_center - self.radius} and {self.y_center + self.radius}"
            )
        x = np.sqrt(self.radius**2 - (y - self.y_center) ** 2)
        if x == 0:
            if as_point_object:
                return [Point(x, y)]
            return [(x, y)]
        if as_point_object:
            return [Point(x, y), Point(-x, y)]
        return [(x, y), (-x, y)]

    def get_point_at_angle(self, angle: float, degrees=False, as_point_object=False):
        if degrees:
            angle = np.radians(angle)
        x = self.x_center + self.radius * np.cos(angle)
        y = self.y_center + self.radius * np.sin(angle)
        if as_point_object:
            return Point(x, y)
        return (x, y)

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
            params = {k: v for k, v in params.items() if v != None}
            axes.fill_between(x, y, self.y_center, zorder=z_order, **params)


@dataclass
class Rectangle:
    x_bottom_left: float
    y_bottom_left: float
    width: float
    height: float
    fill: bool = "default"
    color: str = "default"
    line_width: float | Literal["default"] = "default"
    line_style: str = "default"
    fill_alpha: float | Literal["default"] = "default"

    @classmethod
    def from_center(cls, x: float, y: float, width: float, height: float):
        return cls(
            x - width / 2,
            y - height / 2,
            width,
            height,
        )

    @classmethod
    def from_points(cls, point1: Point, point2: Point):
        if point1.x == point2.x or point1.y == point2.y:
            raise ValueError("The points must not be on the same line")
        return cls(
            min(point1.x, point2.x),
            min(point1.y, point2.y),
            abs(point1.x - point2.x),
            abs(point1.y - point2.y),
        )

    def __contains__(self, point: Point):
        return (self.x_bottom_left <= point.x <= self.x_bottom_left + self.width) and (
            self.y_bottom_left <= point.y <= self.y_bottom_left + self.height
        )

    def area(self):
        return self.width * self.height

    def get_center_point(self):
        return Point(
            self.x_bottom_left + self.width / 2, self.y_bottom_left + self.height / 2
        )

    def get_equation(self):
        return f"{self.x_bottom_left} <= x <= {self.x_bottom_left + self.width} and {self.y_bottom_left} <= y <= {self.y_bottom_left + self.height}"

    def get_points_at_x(self, x: float):
        if x < self.x_bottom_left or x > self.x_bottom_left + self.width:
            raise ValueError(
                f"x must be between {self.x_bottom_left} and {self.x_bottom_left + self.width}"
            )
        return [
            Point(x, self.y_bottom_left),
            Point(x, self.y_bottom_left + self.height),
        ]

    def get_points_at_y(self, y: float):
        if y < self.y_bottom_left or y > self.y_bottom_left + self.height:
            raise ValueError(
                f"y must be between {self.y_bottom_left} and {self.y_bottom_left + self.height}"
            )
        return [Point(self.x_bottom_left, y), Point(self.x_bottom_left + self.width, y)]

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
