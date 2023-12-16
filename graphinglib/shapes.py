from dataclasses import dataclass
from typing import Literal, Self

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

    def __contains__(self, point: Point):
        return (point.x - self.x_center) ** 2 + (
            point.y - self.y_center
        ) ** 2 <= self.radius**2

    def area(self):
        return np.pi * self.radius**2

    def circumference(self):
        return 2 * np.pi * self.radius

    def get_center_point(self):
        return Point(self.x_center, self.y_center)

    def get_equation(self):
        return f"(x - {self.x_center})^2 + (y - {self.y_center})^2 = {self.radius}^2"

    def get_points_at_x(self, x: float):
        if x < self.x_center - self.radius or x > self.x_center + self.radius:
            raise ValueError(
                f"x must be between {self.x_center - self.radius} and {self.x_center + self.radius}"
            )
        y = np.sqrt(self.radius**2 - (x - self.x_center) ** 2)
        return [Point(x, y), Point(x, -y)]

    def get_points_at_y(self, y: float):
        if y < self.y_center - self.radius or y > self.y_center + self.radius:
            raise ValueError(
                f"y must be between {self.y_center - self.radius} and {self.y_center + self.radius}"
            )
        x = np.sqrt(self.radius**2 - (y - self.y_center) ** 2)
        return [Point(x, y), Point(-x, y)]

    def get_point_at_angle(self, angle: float, degrees=False):
        if degrees:
            angle = np.radians(angle)
        x = self.x_center + self.radius * np.cos(angle)
        y = self.y_center + self.radius * np.sin(angle)
        return Point(x, y)

    def overlap_area(self, other: Self) -> float:
        """
        Returns the area of overlap between two circles.
        """
        x1, y1, r1 = self.x_center, self.y_center, self.radius
        x2, y2, r2 = other.x_center, other.y_center, other.radius
        d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Check if circles do not intersect
        if d >= r1 + r2:
            return 0
        # Check if one circle is completely inside the other
        elif d <= abs(r2 - r1) and r1 >= r2:
            # Circle 2 is completely inside circle 1
            return np.pi * r2**2
        elif d <= abs(r2 - r1) and r1 < r2:
            # Circle 1 is completely inside circle 2
            return np.pi * r1**2
        else:
            # Calculate the area of intersection
            term1 = r1**2 * np.arccos((d**2 + r1**2 - r2**2) / (2 * d * r1))
            term2 = r2**2 * np.arccos((d**2 + r2**2 - r1**2) / (2 * d * r2))
            term3 = 0.5 * np.sqrt(
                (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)
            )
            return term1 + term2 - term3

    def get_intersection_points(self, other: Self | Curve) -> list[Point]:
        if isinstance(other, Curve):
            pass
        elif isinstance(other, Circle):
            # Check if circles are too far apart
            if (
                np.sqrt(
                    (self.x_center - other.x_center) ** 2
                    + (self.y_center - other.y_center) ** 2
                )
                > self.radius + other.radius
            ):
                return []
            # Check if one circle is inside the other
            elif np.sqrt(
                (self.x_center - other.x_center) ** 2
                + (self.y_center - other.y_center) ** 2
            ) < abs(self.radius - other.radius):
                return []
            # Check if circles are the same
            elif (
                self.x_center == other.x_center
                and self.y_center == other.y_center
                and self.radius == other.radius
            ):
                return []
            # Calculate intersection points
            else:
                d = np.sqrt(
                    (self.x_center - other.x_center) ** 2
                    + (self.y_center - other.y_center) ** 2
                )
                a = (self.radius**2 - other.radius**2 + d**2) / (2 * d)
                h = np.sqrt(self.radius**2 - a**2)
                x2 = self.x_center + a * (other.x_center - self.x_center) / d
                y2 = self.y_center + a * (other.y_center - self.y_center) / d
                x3 = x2 + h * (other.y_center - self.y_center) / d
                y3 = y2 - h * (other.x_center - self.x_center) / d
                x4 = x2 - h * (other.y_center - self.y_center) / d
                y4 = y2 + h * (other.x_center - self.x_center) / d
                if x3 == x4 and y3 == y4:
                    # If the circles are tangent
                    return [Point(x3, y3)]
                else:
                    return [Point(x3, y3), Point(x4, y4)]

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
        print(params)
        params = {k: v for k, v in params.items() if v != "default"}
        axes.plot(x, y, zorder=z_order, **params)
        if self.fill:
            params = {
                "alpha": self.fill_alpha,
                "facecolor": self.color,
            }
            params = {k: v for k, v in params.items() if v != None}
            axes.fill_between(x, y, self.y_center, zorder=z_order, **params)
