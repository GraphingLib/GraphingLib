from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import shapely as sh
import shapely.ops as ops
from matplotlib.patches import Polygon as MPLPolygon
from shapely import LineString
from shapely import Polygon as ShPolygon

from .data_plotting_1d import Curve
from .graph_elements import Point

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


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

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.shapes.Arrow` object.
        """
        return deepcopy(self)

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

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.shapes.Line` object.
        """
        return deepcopy(self)

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


class Polygon:
    """This class implements a Polygon object.

    Parameters
    ----------
    points : list[tuple[float, float]]
        List of points that define the polygon.
    fill : bool, optional
        Whether the polygon should be filled or not.
        Default depends on the ``figure_style`` configuration.
    color : str, optional
        The color of the polygon (both the line and the fill).
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

    def __init__(
        self,
        points: list[tuple[float, float]],
        fill: bool = "default",
        edge_color: str = "default",
        fill_color: str = "default",
        line_width: float | Literal["default"] = "default",
        line_style: str = "default",
        fill_alpha: float | Literal["default"] = "default",
    ):
        self.fill = fill
        self.edge_color = edge_color
        self.fill_color = fill_color
        self.line_width = line_width
        self.line_style = line_style
        self.fill_alpha = fill_alpha
        self.sh_polygon = ShPolygon(points)

    def __contains__(self, point: Point) -> bool:
        return self.sh_polygon.contains(sh.geometry.Point(point.x, point.y))

    @property
    def vertices(self):
        return np.array(self.sh_polygon.exterior.coords)

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.shapes.Polygon` object.
        """
        return deepcopy(self)

    def get_area(self) -> float:
        """Returns the area of the polygon.

        Returns
        -------
        float
            The area of the polygon.
        """
        return self.sh_polygon.area

    def get_perimeter(self) -> float:
        """Returns the perimeter of the polygon.

        Returns
        -------
        float
            The perimeter of the polygon.
        """
        return self.sh_polygon.length

    def get_centroid_coordinates(self) -> tuple[float, float]:
        """Returns the center coordinates of the polygon.

        Returns
        -------
        tuple[float, float]
            The center coordinates of the polygon.
        """
        return self.sh_polygon.centroid.coords[0]

    def create_centroid_point(self) -> Point:
        """Returns the center point of the polygon.

        Returns
        -------
        :class:`~graphinglib.graph_elements.Point`
            The center point of the polygon.
        """
        return Point(*self.get_centroid_coordinates())

    def create_intersection(self, other: Self, copy_style: bool = False) -> Self:
        """
        Returns the intersection of the polygon with another polygon.

        Parameters
        ----------
        other : :class:`~graphinglib.shapes.Polygon`
            The other polygon.
        copy_style : bool, optional
            If ``True``, the current polygon's parameters are copied to the new polygon. If ``False``, the new polygon will have default parameters. Default is ``False``.

        Returns
        -------
        :class:`~graphinglib.shapes.Polygon`
            The intersection of the two polygons.
        """
        if copy_style:
            new_poly = self.copy()
            new_poly.sh_polygon = self.sh_polygon.intersection(other.sh_polygon)
            return new_poly
        else:
            return Polygon(
                list(self.sh_polygon.intersection(other.sh_polygon).exterior.coords)
            )

    def create_union(self, other: Self, copy_style: bool = False) -> Self:
        """
        Returns the union of the polygon with another polygon.

        Parameters
        ----------
        other : :class:`~graphinglib.shapes.Polygon`
            The other polygon.
        copy_style : bool, optional
            If ``True``, the current polygon's parameters are copied to the new polygon. If ``False``, the new polygon will have default parameters. Default is ``False``.

        Returns
        -------
        :class:`~graphinglib.shapes.Polygon`
            The union of the two polygons.
        """
        if copy_style:
            new_poly = self.copy()
            new_poly.sh_polygon = self.sh_polygon.union(other.sh_polygon)
            return new_poly
        else:
            return Polygon(
                list(self.sh_polygon.union(other.sh_polygon).exterior.coords)
            )

    def create_difference(self, other: Self, copy_style: bool = False) -> Self:
        """
        Returns the difference of the polygon with another polygon.

        Parameters
        ----------
        other : :class:`~graphinglib.shapes.Polygon`
            The other polygon to subtract from the current polygon.
        copy_style : bool, optional
            If ``True``, the current polygon's parameters are copied to the new polygon. If ``False``, the new polygon will have default parameters. Default is ``False``.

        Returns
        -------
        :class:`~graphinglib.shapes.Polygon`
            The difference of the two polygons.
        """
        if copy_style:
            new_poly = self.copy()
            new_poly.sh_polygon = self.sh_polygon.difference(other.sh_polygon)
            return new_poly
        else:
            return Polygon(
                list(self.sh_polygon.difference(other.sh_polygon).exterior.coords)
            )

    def translate(self, dx: float, dy: float) -> Self | None:
        """
        Translates the polygon by the specified amount.

        Parameters
        ----------
        dx : float
            The amount to move the polygon in the x direction.
        dy : float
            The amount to move the polygon in the y direction.
        """
        self.sh_polygon = sh.affinity.translate(self.sh_polygon, xoff=dx, yoff=dy)

    def rotate(
        self,
        angle: float,
        center: Optional[tuple[float, float]] = None,
        use_rad: bool = False,
    ) -> Self:
        """
        Rotates the polygon by the specified angle.

        Parameters
        ----------
        angle : float
            The angle by which to rotate the polygon (in degrees by default).
        center : tuple[float, float], optional
            The center of rotation. If not specified, the centroid of the polygon is used.
        use_rad : bool, optional
            Set to ``True`` if the angle is in radians instead of degrees. Default is ``False``.
        """
        if center is None:
            center = self.get_centroid_coordinates()

        # Use shapely.affinity.rotate to rotate the polygon
        self.sh_polygon = sh.affinity.rotate(
            self.sh_polygon, angle, origin=center, use_radians=use_rad
        )

    def scale(
        self,
        x_scale: float,
        y_scale: float,
        center: Optional[tuple[float, float]] = None,
    ) -> Self:
        """
        Scales the polygon by the specified factors.

        Parameters
        ----------
        x_scale : float
            The factor by which to scale the polygon in the x direction.
        y_scale : float
            The factor by which to scale the polygon in the y direction.
        center : tuple[float, float], optional
            The center of the scaling. If not specified, the centroid of the polygon is used.
        """
        if center is None:
            center = self.get_centroid_coordinates()

        # Use shapely.affinity.scale to scale the polygon
        self.sh_polygon = sh.affinity.scale(
            self.sh_polygon, xfact=x_scale, yfact=y_scale, origin=center
        )

    def skew(
        self,
        x_skew: float,
        y_skew: float,
        center: Optional[tuple[float, float]] = None,
        use_rad: bool = False,
    ) -> Self:
        """
        Skews the polygon by the specified factors.

        Parameters
        ----------
        x_skew : float
            The factor by which to skew the polygon in the x direction.
        y_skew : float
            The factor by which to skew the polygon in the y direction.
        center : tuple[float, float], optional
            The center of the skewing. If not specified, the centroid of the polygon is used.
        use_rad : bool, optional
            Set to ``True`` if the skewing factors are in radians instead of degrees. Default is ``False``.
        """
        if center is None:
            center = self.get_centroid_coordinates()

        # Use shapely.affinity.skew to skew the polygon
        self.sh_polygon = sh.affinity.skew(
            self.sh_polygon, xs=x_skew, ys=y_skew, origin=center, use_radians=use_rad
        )

    def split(self, curve: Curve, copy_style: bool = False) -> list[Self]:
        """
        Splits the polygon by a curve.

        Parameters
        ----------
        curve : :class:`~graphinglib.data_plotting_1d.Curve`
            The curve to split the polygon by.
        copy_style : bool, optional
            If ``True``, the current polygon's parameters are copied to the new polygons. If ``False``, the new polygons will have default parameters. Default is ``False``.

        Returns
        -------
        list[:class:`~shapely.geometry.polygon.Polygon`]
            The list of polygons resulting from the split.
        """
        if not isinstance(curve, Curve):
            raise TypeError("The curve must be a Curve object")
        sh_curve = LineString([(x, y) for x, y in zip(curve.x_data, curve.y_data)])
        split_sh_polygons = ops.split(self.sh_polygon, sh_curve)
        split_sh_polygons = [
            p.simplify(0.001 * p.length) for p in list(split_sh_polygons.geoms)
        ]
        polygons = [Polygon(list(p.exterior.coords)) for p in split_sh_polygons]
        if copy_style:
            for polygon in polygons:
                polygon.fill = self.fill
                polygon.fill_color = self.fill_color
                polygon.edge_color = self.edge_color
                polygon.line_width = self.line_width
                polygon.line_style = self.line_style
                polygon.fill_alpha = self.fill_alpha
        return polygons

    def linear_transformation(self, matrix: np.ndarray) -> Self:
        """
        Applies a transformation matrix to the polygon.

        Parameters
        ----------
        transform : numpy.ndarray
            The transformation matrix to apply. The matrix should be a 2x2 matrix for 2D transformations.
        """
        new_points = np.dot(self.vertices, matrix)
        self.sh_polygon = ShPolygon(new_points)

    def get_intersection_coordinates(self, other: Self) -> list[tuple[float, float]]:
        """
        Returns the coordinates of the intersection points of the borders of the two polygons.

        Parameters
        ----------
        other : :class:`~graphinglib.shapes.Polygon`
            The other polygon.

        Returns
        -------
        list[tuple[float, float]]
            The coordinates of the intersection of the two polygons.
        """
        intersection = self.sh_polygon.boundary.intersection(other.sh_polygon.boundary)
        return [(p.x, p.y) for p in intersection.geoms]

    def create_intersection_points(self, other: Self | Curve) -> list[Point]:
        """
        Returns the intersection points of the borders of the two polygons.

        Parameters
        ----------
        other : :class:`~graphinglib.shapes.Polygon` or :class:`~graphinglib.data_plotting_1d.Curve`
            The other polygon.

        Returns
        -------
        list[:class:`~graphinglib.graph_elements.Point`]
            The intersection points of the two polygons.
        """
        if isinstance(other, Curve):
            # create curve points from the x_data and y_data of the curve
            other_points = [(x, y) for x, y in zip(other.x_data, other.y_data)]
            other_boundary = LineString(other_points)

            intersection = self.sh_polygon.boundary.intersection(other_boundary)
            return [Point(p.x, p.y) for p in intersection.geoms]
        elif isinstance(other, Polygon):
            intersection = self.sh_polygon.boundary.intersection(
                other.sh_polygon.boundary
            )
            return [Point(p.x, p.y) for p in intersection.geoms]
        else:
            raise TypeError("The other object must be a Polygon or a Curve")

    def _plot_element(self, axes: plt.Axes, z_order: int):
        # Create a polygon patch for the fill
        if self.fill:
            kwargs = {
                "alpha": self.fill_alpha,
                "zorder": z_order - 1,
            }
            if self.fill_color is not None:
                kwargs["facecolor"] = self.fill_color
            polygon_fill = MPLPolygon(self.vertices, **kwargs)
            axes.add_patch(polygon_fill)
        # Create a polygon patch for the edge
        if self.edge_color is not None:
            kwargs = {
                "fill": None,
                "linewidth": self.line_width,
                "linestyle": self.line_style,
                "edgecolor": self.edge_color,
                "zorder": z_order,
            }
            polygon_edge = MPLPolygon(self.vertices, **kwargs)
            axes.add_patch(polygon_edge)


@dataclass
class Circle(Polygon):
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
    fill_color : str, optional
        The color of the circle's fill.
        Default depends on the ``figure_style`` configuration.
    edge_color : str, optional
        The color of the circle's edge.
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
    number_of_points : int, optional
        The number of points to use to approximate the circle.
        Default is 100 (covers approximately 99.9% of perfect circle area).
    """

    def __init__(
        self,
        x_center: float,
        y_center: float,
        radius: float,
        fill: bool = "default",
        fill_color: str = "default",
        edge_color: str = "default",
        line_width: float | Literal["default"] = "default",
        line_style: str = "default",
        fill_alpha: float | Literal["default"] = "default",
        number_of_points: int = 100,
    ):
        if number_of_points < 4:
            raise ValueError("The number of points must be greater than or equal to 4")
        self.fill = fill
        self.fill_color = fill_color
        self.edge_color = edge_color
        self.line_width = line_width
        self.line_style = line_style
        self.fill_alpha = fill_alpha
        if radius <= 0:
            raise ValueError("The radius must be positive")
        self.sh_polygon = sh.geometry.Point(x_center, y_center).buffer(
            radius, number_of_points // 4
        )


@dataclass
class Rectangle(Polygon):
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
    fill_color : str, optional
        The color of the rectangle's fill.
        Default depends on the ``figure_style`` configuration.
    edge_color : str, optional
        The color of the rectangle's edge.
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

    def __init__(
        self,
        x_bottom_left: float,
        y_bottom_left: float,
        width: float,
        height: float,
        fill: bool = "default",
        fill_color: str = "default",
        edge_color: str = "default",
        line_width: float | Literal["default"] = "default",
        line_style: str = "default",
        fill_alpha: float | Literal["default"] = "default",
    ):
        if width <= 0:
            raise ValueError("The width must be positive")
        if height <= 0:
            raise ValueError("The height must be positive")

        self.fill = fill
        self.fill_color = fill_color
        self.edge_color = edge_color
        self.line_width = line_width
        self.line_style = line_style
        self.fill_alpha = fill_alpha
        self.sh_polygon = ShPolygon(
            [
                (x_bottom_left, y_bottom_left),
                (x_bottom_left + width, y_bottom_left),
                (x_bottom_left + width, y_bottom_left + height),
                (x_bottom_left, y_bottom_left + height),
            ]
        )

    @classmethod
    def from_center(
        cls,
        x: float,
        y: float,
        width: float,
        height: float,
        fill: bool = "default",
        fill_color: str = "default",
        edge_color: str = "default",
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
        fill_color : str, optional
            The color of the rectangle's fill.
            Default depends on the ``figure_style`` configuration.
        edge_color : str, optional
            The color of the rectangle's edge.
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
            fill_color,
            edge_color,
            line_width,
            line_style,
            fill_alpha,
        )
