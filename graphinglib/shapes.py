from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Literal, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import shapely as sh
import shapely.affinity
import shapely.ops as ops
from matplotlib.patches import Polygon as MPLPolygon
from shapely import LineString
from shapely import Polygon as ShPolygon

from ._animation import (
    _DRAW_SESSION,
    _ArtistEdit,
    _bound_session,
    _combine_edits,
    _edit,
    _stage_live_value,
)
from .data_plotting_1d import Curve
from .exceptions import (
    IncompatibleArgumentsError,
    InvalidParameterError,
    InvalidParameterTypeError,
    UnsupportedFeatureError,
)
from .graph_elements import Plottable, Point
from .inherit import INHERIT, Inherit, resolve_or, strip_inherit

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


def _bind_arrow_endpoints(element, artist, locate, *properties):
    session = _DRAW_SESSION.get()
    if session is None:
        return
    patch = artist.arrow_patch
    assert patch is not None

    def prepare():
        target, origin = locate()
        if not all(
            isinstance(value, Real) and isfinite(value) for value in (*target, *origin)
        ):
            raise InvalidParameterError(
                "Animated arrow and line endpoints must be finite."
            )

        def move(points):
            artist.xy, origin = points
            artist.set_position(origin)

        return _combine_edits(
            [
                _edit(move, (artist.xy, artist.get_position()), (target, origin)),
                _edit(
                    patch.set_visible,
                    patch.get_visible(),
                    tuple(target) != tuple(origin),
                ),
            ]
        )

    patch.set_visible(tuple(artist.xy) != tuple(artist.get_position()))
    session.bind_derived(element, properties, prepare)

    def prepare_color(value):
        edge, face = patch.get_edgecolor(), patch.get_facecolor()

        def restore():
            patch.set_edgecolor(edge)
            patch.set_facecolor(face)

        return _ArtistEdit(lambda: patch.set_color(value), restore)

    session.bind_update(element, "_color", prepare_color)
    session.bind_property(element, "_alpha", patch, "alpha")
    session.bind_property(element, "_width", patch, "linewidth")


def _bind_arrow_style(element, annotation, get_style, *properties):
    session = _DRAW_SESSION.get()
    if session is None:
        return
    patch = annotation.arrow_patch
    assert patch is not None

    def prepare():
        style = get_style()
        existing_style = patch.get_arrowstyle()
        existing_options = annotation.arrowprops

        def apply():
            patch.set_arrowstyle(style)
            annotation.arrowprops = {**existing_options, "arrowstyle": style}

        def undo():
            patch.set_arrowstyle(existing_style)
            annotation.arrowprops = existing_options

        return _ArtistEdit(apply, undo)

    session.bind_derived(element, properties, prepare)


@dataclass
class Arrow(Plottable):
    """This class implements an arrow object.

    Parameters
    ----------
    pointA : tuple[float, float]
        Point A of the arrow. If the arrow is single-sided, refers to the tail.
    pointB : tuple[float, float]
        Point B of the arrow. If the arrow is single-sided, refers to the head.
    color : str
        Color of the arrow.
        Default depends on the ``figure_style`` configuration.
    width : float, optional
        Arrow line width.
        Typical range is ``0.5`` to ``4``.
        Default depends on the ``figure_style`` configuration.
    head_size : float, optional
        Scales the size of the arrow head.
        Typical range is ``0.5`` to ``3``.
        Default depends on the ``figure_style`` configuration.
    style : ``Literal["->", "-|>", "-[", "]->", "simple", "fancy", "wedge"] | Inherit``, optional
        The style of the arrow. For a visual explanation of all available styles, see the gallery
        `Arrow Styles <https://graphinglib.org/latest/examples/arrow_styles.html>`_ example.
        Values are ``"->"``, ``"-|>"``, ``"-["``, ``"]->"``, ``"simple"``, ``"fancy"``, and ``"wedge"``.
        Default depends on the ``figure_style`` configuration.

        .. warning::
            The ``style`` parameter differs slightly from matplotlib's ``arrowstyle`` since GraphingLib also uses a
            ``two_sided`` parameter to explicitly define whether the arrow is single or double-sided.
    shrink : float
        Fraction of the total length of the arrow to shrink from both ends.
        Range is ``0`` to ``0.5``. A value of ``0.5`` means the arrow is no longer visible.
        Defaults to 0.
    alpha : float
        Opacity of the arrow.
        Range is ``0`` (transparent) to ``1`` (opaque).
        Default depends on the ``figure_style`` configuration.
    two_sided : bool
        If ``True``, an arrow is shown at both head and tail. Defaults to ``False``.

    Notes
    -----
    Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
    (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
    values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).
    """

    def __init__(
        self,
        pointA: tuple[float, float],
        pointB: tuple[float, float],
        color: str | Inherit = INHERIT,
        width: float | Inherit = INHERIT,
        head_size: float | Inherit = INHERIT,
        shrink: float = 0,
        style: Literal["->", "-|>", "-[", "]->", "simple", "fancy", "wedge"]
        | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
        two_sided: bool = False,
    ):
        """This class implements an arrow object.

        Parameters
        ----------
        pointA : tuple[float, float]
            Point A of the arrow. If the arrow is single-sided, refers to the tail.
        pointB : tuple[float, float]
            Point B of the arrow. If the arrow is double-sided, refers to the head.
        color : str
            Color of the arrow. Default depends on the ``figure_style`` configuration.
        width : float, optional
            Arrow line width.
            Typical range is ``0.5`` to ``4``.
            Default depends on the ``figure_style`` configuration.
        head_size : float, optional
            Scales the size of the arrow head.
            Typical range is ``0.5`` to ``3``.
            Default depends on the ``figure_style`` configuration.
        style : ``Literal["->", "-|>", "-[", "]->", "simple", "fancy", "wedge"] | Inherit``, optional
            The style of the arrow. For a visual explanation of all available styles, see the gallery
            `Arrow Styles <https://graphinglib.org/latest/examples/arrow_styles.html>`_ example.
            Values are ``"->"``, ``"-|>"``, ``"-["``, ``"]->"``, ``"simple"``, ``"fancy"``, and ``"wedge"``.
            Default depends on the ``figure_style`` configuration.

            .. warning::
                The ``style`` parameter differs slightly from matplotlib's ``arrowstyle`` since GraphingLib also uses a
                ``two_sided`` parameter to explicitly define whether the arrow is single or double-sided.
        shrink : float
            Fraction of the total length of the arrow to shrink from both ends.
            Range is ``0`` to ``0.5``. A value of ``0.5`` means the arrow is no longer visible.
            Defaults to 0.
        alpha : float
            Opacity of the arrow.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.
        two_sided : bool
            If ``True``, the arrow is double-sided. Defaults to ``False``.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).
        """
        self._pointA = pointA
        self._pointB = pointB
        self._color = color
        self._width = width
        self._head_size = head_size
        self._shrink = shrink
        self.style = style
        self._alpha = alpha
        self._two_sided = two_sided

    @property
    def pointA(self):
        return self._pointA

    @pointA.setter
    def pointA(self, value):
        self._pointA = _stage_live_value(self, "_pointA", value)

    @property
    def pointB(self):
        return self._pointB

    @pointB.setter
    def pointB(self, value):
        self._pointB = _stage_live_value(self, "_pointB", value)

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        self._color = _stage_live_value(self, "_color", value)

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = _stage_live_value(self, "_width", value)

    @property
    def head_size(self):
        return self._head_size

    @head_size.setter
    def head_size(self, value):
        self._head_size = _stage_live_value(self, "_head_size", value)

    @property
    def shrink(self):
        return self._shrink

    @shrink.setter
    def shrink(self, value):
        self._shrink = _stage_live_value(self, "_shrink", value)

    @property
    def style(self):
        return self._style

    @style.setter
    def style(self, value):
        if value not in [
            "->",
            "-|>",
            "-[",
            "]->",
            "simple",
            "fancy",
            "wedge",
            INHERIT,
        ]:
            raise InvalidParameterError(
                "style must be one of '->', '-|>', '-[', ']->', 'simple', 'fancy', "
                f"'wedge', or INHERIT; got {value!r}."
            )
        self._style = _stage_live_value(self, "_style", value)

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = _stage_live_value(self, "_alpha", value)

    @property
    def two_sided(self):
        return self._two_sided

    @two_sided.setter
    def two_sided(self, value):
        self._two_sided = _stage_live_value(self, "_two_sided", value)

    def _shrink_points(self):
        x_length, y_length = (
            self._pointA[0] - self._pointB[0],
            self._pointA[1] - self._pointB[1],
        )
        newA = (
            self._pointB[0] + (1 - self._shrink) * x_length,
            self._pointB[1] + (1 - self._shrink) * y_length,
        )
        newB = (
            self._pointA[0] - (1 - self._shrink) * x_length,
            self._pointA[1] - (1 - self._shrink) * y_length,
        )
        return newA, newB

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.shapes.Arrow` object.
        """
        return deepcopy(self)

    @staticmethod
    def _arrow_style(style, head_size, two_sided):
        if style == INHERIT and head_size != INHERIT:
            style = "simple"
        if two_sided:
            match style:
                case "->":
                    full_style = "<->"
                case "-|>":
                    full_style = "<|-|>"
                case "-[":
                    full_style = "]-["
                case _:
                    raise IncompatibleArgumentsError(
                        "A two-sided arrow requires a head style of '->', '-|>', or "
                        f"'-['; got {style!r}."
                    )
        else:
            full_style = style

        if head_size == 0:
            return "-"
        if head_size != INHERIT:
            head_length, head_width = head_size * 0.4, head_size * 0.2

            # Set specific arrow properties
            match style:
                case "->" | "-|>" | "simple" | "fancy":
                    prop_style_values = (
                        f"head_width={head_width}, head_length={head_length}"
                    )
                case "-[":
                    prop_style_values = f"widthB={head_width}" + (
                        f", widthA={head_width}" if two_sided else ""
                    )
                case "]->":
                    prop_style_values = f"widthA={head_width}"
                case "wedge":
                    prop_style_values = f"tail_width={head_width}"

            return f"{full_style}, {prop_style_values}"
        return full_style

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs):
        session = _DRAW_SESSION.get()
        initial_style = self._style
        initial_head = self._head_size
        if session is not None:
            initial_style = resolve_or(initial_style, "simple")
        arrow_style = self._arrow_style(initial_style, initial_head, self._two_sided)
        props = {
            "arrowstyle": arrow_style,
            "color": self._color,
            "linewidth": self._width,
            "alpha": self._alpha,
        }
        props = strip_inherit(props)

        def locate():
            if session is not None and (
                not isinstance(self._shrink, Real)
                or not isfinite(self._shrink)
                or not 0 <= self._shrink <= 0.5
            ):
                raise InvalidParameterError(
                    "Animated Arrow shrink must be between 0 and 0.5."
                )
            if session is not None and self._shrink == 0.5:
                center = tuple(
                    a / 2 + b / 2
                    for a, b in zip(self._pointA, self._pointB, strict=True)
                )
                return center, center
            origin, target = (
                self._shrink_points()
                if self._shrink != 0
                else (self._pointA, self._pointB)
            )
            return target, origin

        target, origin = locate()
        artist = axes.annotate("", target, origin, zorder=z_order, arrowprops=props)
        _bind_arrow_endpoints(self, artist, locate, "_pointA", "_pointB", "_shrink")

        def read_style():
            head = resolve_or(self._head_size, initial_head)
            if head != INHERIT and (
                not isinstance(head, Real) or not isfinite(head) or head < 0
            ):
                raise InvalidParameterError(
                    "Animated Arrow head_size must be finite and nonnegative."
                )
            if not isinstance(self._two_sided, (bool, np.bool_)):
                raise InvalidParameterTypeError("Arrow.two_sided must be a boolean.")
            return self._arrow_style(
                resolve_or(self._style, initial_style), head, self._two_sided
            )

        _bind_arrow_style(
            self, artist, read_style, "_style", "_head_size", "_two_sided"
        )


@dataclass
class Line(Plottable):
    """This class implements a line object.

    Parameters
    ----------
    pointA : tuple[float, float]
        Point A of the line.
    pointB : tuple[float, float]
        Point B of the line.
    color : str
        Color of the line.
        Default depends on the ``figure_style`` configuration.
    width : float, optional
        Line width.
        Typical range is ``0.5`` to ``4``.
        Default depends on the ``figure_style`` configuration.
    capped_line : bool
        If ``True``, the line is capped on both ends.
        Defaults to ``False``.
    cap_width : float
        Width of the caps.
        Typical range is ``0.5`` to ``4``.
        Default depends on the ``figure_style`` configuration.
    alpha : float
        Opacity of the line.
        Range is ``0`` (transparent) to ``1`` (opaque).
        Default depends on the ``figure_style`` configuration.

    Notes
    -----
    Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
    (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
    values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).
    """

    _pointA: tuple[float, float]
    _pointB: tuple[float, float]
    _color: str | Inherit = INHERIT
    _width: float | Inherit = INHERIT
    _capped_line: bool = False
    _cap_width: float | Inherit = INHERIT
    _alpha: float | Inherit = INHERIT

    def __init__(
        self,
        pointA: tuple[float, float],
        pointB: tuple[float, float],
        color: str | Inherit = INHERIT,
        width: float | Inherit = INHERIT,
        capped_line: bool = False,
        cap_width: float | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
    ):
        self._pointA = pointA
        self._pointB = pointB
        self._color = color
        self._width = width
        self._capped_line = capped_line
        self._cap_width = cap_width
        self._alpha = alpha

    @property
    def pointA(self) -> tuple[float, float]:
        return self._pointA

    @pointA.setter
    def pointA(self, value: tuple[float, float]):
        self._pointA = _stage_live_value(self, "_pointA", value)

    @property
    def pointB(self) -> tuple[float, float]:
        return self._pointB

    @pointB.setter
    def pointB(self, value: tuple[float, float]):
        self._pointB = _stage_live_value(self, "_pointB", value)

    @property
    def color(self) -> str | Inherit:
        return self._color

    @color.setter
    def color(self, value: str | Inherit):
        self._color = _stage_live_value(self, "_color", value)

    @property
    def width(self) -> float | Inherit:
        return self._width

    @width.setter
    def width(self, value: float | Inherit):
        self._width = _stage_live_value(self, "_width", value)

    @property
    def capped_line(self) -> bool:
        return self._capped_line

    @capped_line.setter
    def capped_line(self, value: bool):
        self._capped_line = _stage_live_value(self, "_capped_line", value)

    @property
    def cap_width(self) -> float | Inherit:
        return self._cap_width

    @cap_width.setter
    def cap_width(self, value: float | Inherit):
        self._cap_width = _stage_live_value(self, "_cap_width", value)

    @property
    def alpha(self) -> float | Inherit:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float | Inherit):
        self._alpha = _stage_live_value(self, "_alpha", value)

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.shapes.Line` object.
        """
        return deepcopy(self)

    @staticmethod
    def _arrow_style(capped_line, cap_width):
        if capped_line:
            return f"|-|, widthA={cap_width / 2}, widthB={cap_width / 2}"
        return "-"

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs):
        initial_cap_width = resolve_or(self._cap_width, 1)
        style = self._arrow_style(self._capped_line, initial_cap_width)
        props = {
            "arrowstyle": style,
            "color": self._color,
            "linewidth": self._width,
            "alpha": self._alpha,
        }
        props = strip_inherit(props)
        artist = axes.annotate(
            "",
            self._pointA,
            self._pointB,
            zorder=z_order,
            arrowprops=props,
        )
        _bind_arrow_endpoints(
            self, artist, lambda: (self._pointA, self._pointB), "_pointA", "_pointB"
        )

        def read_style():
            cap = resolve_or(self._cap_width, initial_cap_width)
            if not isinstance(cap, Real) or not isfinite(cap) or cap < 0:
                raise InvalidParameterError(
                    "Animated Line cap_width must be finite and nonnegative."
                )
            if not isinstance(self._capped_line, (bool, np.bool_)):
                raise InvalidParameterTypeError("Line.capped_line must be a boolean.")
            return self._arrow_style(self._capped_line and cap > 0, cap)

        _bind_arrow_style(self, artist, read_style, "_capped_line", "_cap_width")


class Polygon(Plottable):
    """This class implements a Polygon object.

    Parameters
    ----------
    vertices : list[tuple[float, float]]
        List of coordinates that define the polygon.
    fill : bool, optional
        Whether the polygon should be filled or not.
        Default depends on the ``figure_style`` configuration.
    edge_color : str, optional
        The color of the polygon's edge.
        Default depends on the ``figure_style`` configuration.
    fill_color : str, optional
        The color of the polygon's fill.
        Default depends on the ``figure_style`` configuration.
    line_width : float, optional
        The width of the line.
        Typical range is ``0.5`` to ``4``.
        Default depends on the ``figure_style`` configuration.
    line_style : str, optional
        The style of the line.
        Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
        ``"dotted"``.
        Default depends on the ``figure_style`` configuration.
    fill_alpha : float, optional
        The alpha value of the fill.
        Range is ``0`` (transparent) to ``1`` (opaque).
        Default depends on the ``figure_style`` configuration.

    Notes
    -----
    Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
    (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
    values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).
    """

    def __init__(
        self,
        vertices: Sequence[tuple[float, float]],
        fill: bool | Inherit = INHERIT,
        edge_color: str | Inherit = INHERIT,
        fill_color: str | Inherit = INHERIT,
        line_width: float | Inherit = INHERIT,
        line_style: str | Inherit = INHERIT,
        fill_alpha: float | Inherit = INHERIT,
    ):
        self._fill = fill
        self._edge_color = edge_color
        self._fill_color = fill_color
        self._line_width = line_width
        self._line_style = line_style
        self._fill_alpha = fill_alpha
        self._sh_polygon = ShPolygon(vertices)

    @property
    def fill(self):
        return self._fill

    @fill.setter
    def fill(self, value):
        self._fill = _stage_live_value(self, "_fill", value)

    @property
    def fill_color(self):
        return self._fill_color

    @fill_color.setter
    def fill_color(self, value):
        self._fill_color = _stage_live_value(self, "_fill_color", value)

    @property
    def edge_color(self):
        return self._edge_color

    @edge_color.setter
    def edge_color(self, value):
        self._edge_color = _stage_live_value(self, "_edge_color", value)

    @property
    def line_width(self):
        return self._line_width

    @line_width.setter
    def line_width(self, value):
        self._line_width = _stage_live_value(self, "_line_width", value)

    @property
    def line_style(self):
        return self._line_style

    @line_style.setter
    def line_style(self, value):
        self._line_style = _stage_live_value(self, "_line_style", value)

    @property
    def fill_alpha(self):
        return self._fill_alpha

    @fill_alpha.setter
    def fill_alpha(self, value):
        self._fill_alpha = _stage_live_value(self, "_fill_alpha", value)

    @property
    def vertices(self):
        return np.array(self._sh_polygon.exterior.coords)

    @vertices.setter
    def vertices(self, value):
        self._sh_polygon = _stage_live_value(self, "_sh_polygon", ShPolygon(value))

    @property
    def area(self):
        return self._sh_polygon.area

    @property
    def perimeter(self):
        return self._sh_polygon.length

    def __contains__(self, point: Point) -> bool:
        return self._sh_polygon.contains(sh.geometry.Point(point._x, point._y))

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.shapes.Polygon` object.
        """
        return deepcopy(self)

    def get_centroid_coordinates(self) -> tuple[float, float]:
        """Returns the center coordinates of the polygon.

        Returns
        -------
        tuple[float, float]
            The center coordinates of the polygon.
        """
        return self._sh_polygon.centroid.coords[0]

    def create_centroid_point(self) -> Point:
        """Returns the center point of the polygon.

        Returns
        -------
        :class:`~graphinglib.graph_elements.Point`
            The center point of the polygon.
        """
        return Point(*self.get_centroid_coordinates())

    def create_intersection(self, other: Self, copy_style: bool = False) -> Polygon:
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
            new_poly._sh_polygon = self._sh_polygon.intersection(other._sh_polygon)
            return new_poly
        else:
            return Polygon(
                list(self._sh_polygon.intersection(other._sh_polygon).exterior.coords)
            )

    def create_union(self, other: Self, copy_style: bool = False) -> Polygon:
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
            new_poly._sh_polygon = self._sh_polygon.union(other._sh_polygon)
            return new_poly
        else:
            return Polygon(
                list(self._sh_polygon.union(other._sh_polygon).exterior.coords)
            )

    def create_difference(self, other: Self, copy_style: bool = False) -> Polygon:
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
            new_poly._sh_polygon = self._sh_polygon.difference(other._sh_polygon)
            return new_poly
        else:
            return Polygon(
                list(self._sh_polygon.difference(other._sh_polygon).exterior.coords)
            )

    def create_symmetric_difference(
        self, other: Self, copy_style: bool = False
    ) -> list[Polygon]:
        """
        Returns the symmetric difference of the polygon with another polygon.

        In general, this can create more than one polygon, so the result is returned as a list of polygons even if there is only one.

        Parameters
        ----------
        other : :class:`~graphinglib.shapes.Polygon`
            The other polygon to find the symmetric difference with.
        copy_style : bool, optional
            If ``True``, the current polygon's parameters are copied to the new polygon. If ``False``, the new polygon will have default parameters. Default is ``False``.

        Returns
        -------
        list[:class:`~graphinglib.shapes.Polygon`]
            A list of polygons resulting from the symmetric difference.
        """
        multi_poly = self._sh_polygon.symmetric_difference(other._sh_polygon)
        if multi_poly.geom_type == "MultiPolygon":
            polygons = [
                Polygon(list(p.exterior.coords)) for p in list(multi_poly.geoms)
            ]
        else:
            polygons = [Polygon(list(multi_poly.exterior.coords))]
        if copy_style:
            for polygon in polygons:
                polygon._fill = self._fill
                polygon._fill_color = self._fill_color
                polygon._edge_color = self._edge_color
                polygon._line_width = self._line_width
                polygon._line_style = self._line_style
                polygon._fill_alpha = self._fill_alpha
        return polygons

    def translate(self, dx: float, dy: float) -> None:
        """
        Translates the polygon by the specified amount.

        Parameters
        ----------
        dx : float
            The amount to move the polygon in the x direction.
        dy : float
            The amount to move the polygon in the y direction.
        """
        self._sh_polygon = _stage_live_value(
            self,
            "_sh_polygon",
            sh.affinity.translate(self._sh_polygon, xoff=dx, yoff=dy),
        )

    def rotate(
        self,
        angle: float,
        center: Optional[tuple[float, float]] = None,
        use_rad: bool = False,
    ) -> None:
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
        self._sh_polygon = _stage_live_value(
            self,
            "_sh_polygon",
            sh.affinity.rotate(
                self._sh_polygon, angle, origin=center, use_radians=use_rad
            ),
        )

    def scale(
        self,
        x_scale: float,
        y_scale: float,
        center: Optional[tuple[float, float]] = None,
    ) -> None:
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
        self._sh_polygon = _stage_live_value(
            self,
            "_sh_polygon",
            sh.affinity.scale(
                self._sh_polygon, xfact=x_scale, yfact=y_scale, origin=center
            ),
        )

    def skew(
        self,
        x_skew: float,
        y_skew: float,
        center: Optional[tuple[float, float]] = None,
        use_rad: bool = False,
    ) -> None:
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
        self._sh_polygon = _stage_live_value(
            self,
            "_sh_polygon",
            sh.affinity.skew(
                self._sh_polygon,
                xs=x_skew,
                ys=y_skew,
                origin=center,
                use_radians=use_rad,
            ),
        )

    def split(self, curve: Curve, copy_style: bool = False) -> list[Polygon]:
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
            raise InvalidParameterTypeError(
                f"curve must be a Curve; got {type(curve).__name__}."
            )
        sh_curve = LineString([(x, y) for x, y in zip(curve._x_data, curve._y_data)])
        split_sh_polygons = ops.split(self._sh_polygon, sh_curve)
        split_sh_polygons = [
            p.simplify(0.001 * p.length) for p in list(split_sh_polygons.geoms)
        ]
        polygons = [Polygon(list(p.exterior.coords)) for p in split_sh_polygons]
        if copy_style:
            for polygon in polygons:
                polygon._fill = self._fill
                polygon._fill_color = self._fill_color
                polygon._edge_color = self._edge_color
                polygon._line_width = self._line_width
                polygon._line_style = self._line_style
                polygon._fill_alpha = self._fill_alpha
        return polygons

    def linear_transformation(self, matrix: np.ndarray) -> None:
        """
        Applies a transformation matrix to the polygon.

        Parameters
        ----------
        transform : numpy.ndarray
            The transformation matrix to apply. The matrix should be a 2x2 matrix for 2D transformations.
        """
        new_points = np.dot(self.vertices, matrix)
        self._sh_polygon = _stage_live_value(self, "_sh_polygon", ShPolygon(new_points))

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
        intersection = self._sh_polygon.boundary.intersection(
            other._sh_polygon.boundary
        )
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
            other_points = [(x, y) for x, y in zip(other._x_data, other._y_data)]
            other_boundary = LineString(other_points)

            intersection = self._sh_polygon.boundary.intersection(other_boundary)
            return [Point(p.x, p.y) for p in intersection.geoms]
        elif isinstance(other, Polygon):
            intersection = self._sh_polygon.boundary.intersection(
                other._sh_polygon.boundary
            )
            return [Point(p.x, p.y) for p in intersection.geoms]
        else:
            raise InvalidParameterTypeError(
                f"other must be a Polygon or a Curve; got {type(other).__name__}."
            )

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs):
        session = _DRAW_SESSION.get()
        filled = resolve_or(self._fill, True)
        outlined = self._edge_color is not None
        if filled or session is not None:
            params = {
                "alpha": self._fill_alpha,
                "zorder": z_order,
            }
            if self._fill_color is not None:
                params["facecolor"] = self._fill_color
            polygon_fill = MPLPolygon(self.vertices, **strip_inherit(params))
            if filled:
                axes.add_patch(polygon_fill)
            else:
                # Keep runtime-only artists from affecting initial autoscaling.
                polygon_fill.set_visible(False)
                axes.add_artist(polygon_fill)
        if outlined or session is not None:
            params = {
                "fill": None,
                "linewidth": self._line_width,
                "linestyle": self._line_style,
                "edgecolor": self._edge_color,
                "zorder": z_order,
            }
            polygon_edge = MPLPolygon(self.vertices, **strip_inherit(params))
            if outlined:
                axes.add_patch(polygon_edge)
            else:
                polygon_edge.set_visible(False)
                axes.add_artist(polygon_edge)
        if session is None:
            return

        def prepare_geometry(shape: ShPolygon) -> _ArtistEdit:
            vertices = np.asarray(shape.exterior.coords)
            if not np.isfinite(vertices).all():
                raise InvalidParameterError("Animated polygon vertices must be finite.")
            return _combine_edits(
                [
                    _edit(patch.set_xy, patch.get_xy(), vertices)
                    for patch in (polygon_fill, polygon_edge)
                ]
            )

        def prepare_edge(color) -> _ArtistEdit:
            return _combine_edits(
                [
                    _edit(
                        polygon_edge.set_edgecolor, polygon_edge.get_edgecolor(), color
                    ),
                    _edit(
                        polygon_edge.set_visible,
                        polygon_edge.get_visible(),
                        color is not None,
                    ),
                ]
            )

        def fill_visibility(value) -> bool:
            if not isinstance(value, (bool, np.bool_)):
                raise InvalidParameterTypeError("Polygon.fill must be a boolean.")
            return bool(value)

        initial_face = plt.rcParams["patch.facecolor"]
        session.bind_update(self, "_sh_polygon", prepare_geometry)
        session.bind_property(self, "_fill", polygon_fill, "visible", fill_visibility)
        session.bind_property(
            self,
            "_fill_color",
            polygon_fill,
            "facecolor",
            lambda value: initial_face if value is None else value,
        )
        session.bind_property(self, "_fill_alpha", polygon_fill, "alpha")
        session.bind_update(self, "_edge_color", prepare_edge)
        session.bind_property(self, "_line_width", polygon_edge, "linewidth")
        session.bind_property(self, "_line_style", polygon_edge, "linestyle")


def _round_shape_resolution(shape: Circle | Ellipse, value: int) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 4:
        raise InvalidParameterError(
            f"number_of_points must be an integer of at least 4; got {value}."
        )
    resolution = int(value) // 4
    session = _bound_session(shape)
    if session is not None:
        vertices = shape.vertices[:-1]
        count = len(vertices)
        if count != resolution * 4:
            if count < 4 or count % 4 or shape._sh_polygon.interiors:
                raise UnsupportedFeatureError(
                    "Live sampling resolution requires a Circle or Ellipse outline."
                )
            # A round shape's quarter turns recover the affine transform of
            # the unit buffer, including rotations and nonuniform scaling.
            origin = (vertices[0] + vertices[count // 2]) / 2
            u = vertices[0] - origin
            v = origin - vertices[count // 4]
            angles = np.arange(count) * (2 * np.pi / count)
            predicted = (
                origin + np.cos(angles)[:, None] * u - np.sin(angles)[:, None] * v
            )
            if not np.allclose(predicted - origin, vertices - origin):
                raise UnsupportedFeatureError(
                    "Live sampling resolution requires a Circle or Ellipse outline."
                )
            disk = sh.geometry.Point(0, 0).buffer(1, resolution)
            polygon = sh.affinity.affine_transform(
                disk, (u[0], v[0], u[1], v[1], origin[0], origin[1])
            )
            shape._sh_polygon = _stage_live_value(shape, "_sh_polygon", polygon)
        previous = shape._num_points
        session.remember(
            shape, "_num_points", lambda: setattr(shape, "_num_points", previous)
        )
    return int(value)


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
        Typical range is ``0.5`` to ``4``.
        Default depends on the ``figure_style`` configuration.
    line_style : str, optional
        The style of the line.
        Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
        ``"dotted"``.
        Default depends on the ``figure_style`` configuration.
    fill_alpha : float, optional
        The alpha value of the fill.
        Range is ``0`` (transparent) to ``1`` (opaque).
        Default depends on the ``figure_style`` configuration.
    number_of_points : int, optional
        The number of points to use to approximate the circle.
        Default is 100 (covers approximately 99.9% of perfect circle area).

    Notes
    -----
    Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
    (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
    values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).
    """

    def __init__(
        self,
        x_center: float,
        y_center: float,
        radius: float,
        fill: bool | Inherit = INHERIT,
        fill_color: str | Inherit = INHERIT,
        edge_color: str | Inherit = INHERIT,
        line_width: float | Inherit = INHERIT,
        line_style: str | Inherit = INHERIT,
        fill_alpha: float | Inherit = INHERIT,
        number_of_points: int = 100,
    ):
        self.number_of_points = number_of_points
        self._fill = fill
        self._fill_color = fill_color
        self._edge_color = edge_color
        self._line_width = line_width
        self._line_style = line_style
        self._fill_alpha = fill_alpha
        self._sh_polygon = sh.geometry.Point(x_center, y_center).buffer(
            1, self._num_points // 4
        )
        self.radius = radius

    @property
    def x_center(self):
        return self.get_centroid_coordinates()[0]

    @x_center.setter
    def x_center(self, value):
        self.translate(value - self.x_center, 0)

    @property
    def y_center(self):
        return self.get_centroid_coordinates()[1]

    @y_center.setter
    def y_center(self, value):
        self.translate(0, value - self.y_center)

    @property
    def radius(self):
        return self._sh_polygon.exterior.length / (2 * np.pi)

    @radius.setter
    def radius(self, value):
        if value <= 0:
            raise InvalidParameterError(f"radius must be positive; got {value}.")
        self._sh_polygon = _stage_live_value(
            self,
            "_sh_polygon",
            sh.geometry.Point(self.x_center, self.y_center).buffer(
                value, self._num_points // 4
            ),
        )

    @property
    def diameter(self):
        return 2 * self.radius

    @diameter.setter
    def diameter(self, value):
        self.radius = value / 2

    @property
    def number_of_points(self):
        return self._num_points

    @number_of_points.setter
    def number_of_points(self, value):
        self._num_points = _round_shape_resolution(self, value)

    @property
    def circumference(self):
        return self._sh_polygon.exterior.length


@dataclass
class Ellipse(Polygon):
    """This class implements an Ellipse object with a given center, two radii and an optional rotation angle.

    Parameters
    ----------
    x_center : float
        The x coordinate of the center of the ellipse.
    y_center : float
        The y coordinate of the center of the ellipse.
    x_radius : float
        The radius of the ellipse along the x axis.
    y_radius : float
        The radius of the ellipse along the y axis.
    angle : float, optional
        The rotation angle of the ellipse in degrees.
        Defaults to 0.
    fill : bool, optional
        Whether the ellipse should be filled or not.
        Default depends on the ``figure_style`` configuration.
    fill_color : str, optional
        The color of the ellipse's fill.
        Default depends on the ``figure_style`` configuration.
    edge_color : str, optional
        The color of the ellipse's edge.
        Default depends on the ``figure_style`` configuration.
    line_width : float, optional
        The width of the line.
        Typical range is ``0.5`` to ``4``.
        Default depends on the ``figure_style`` configuration.
    line_style : str, optional
        The style of the line.
        Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
        ``"dotted"``.
        Default depends on the ``figure_style`` configuration.
    fill_alpha : float, optional
        The alpha value of the fill.
        Range is ``0`` (transparent) to ``1`` (opaque).
        Default depends on the ``figure_style`` configuration.
    number_of_points : int, optional
        The number of points to use to approximate the ellipse.
        Default is 100 (covers approximately 99.9% of perfect ellipse area).

    Notes
    -----
    Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
    (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
    values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).
    """

    def __init__(
        self,
        x_center: float,
        y_center: float,
        x_radius: float,
        y_radius: float,
        angle: float = 0,
        fill: bool | Inherit = INHERIT,
        fill_color: str | Inherit = INHERIT,
        edge_color: str | Inherit = INHERIT,
        line_width: float | Inherit = INHERIT,
        line_style: str | Inherit = INHERIT,
        fill_alpha: float | Inherit = INHERIT,
        number_of_points: int = 100,
    ):
        self.number_of_points = number_of_points
        self._fill = fill
        self._fill_color = fill_color
        self._edge_color = edge_color
        self._line_width = line_width
        self._line_style = line_style
        self._fill_alpha = fill_alpha
        self._x_radius = 1
        self._y_radius = 1
        self._angle = 0
        self._sh_polygon = sh.geometry.Point(x_center, y_center).buffer(
            1, self._num_points // 4
        )
        self.x_radius = x_radius
        self.y_radius = y_radius
        self.angle = angle

    def _rebuild(
        self,
        x_center: float,
        y_center: float,
        x_radius: float,
        y_radius: float,
        angle: float,
    ):
        polygon = sh.geometry.Point(x_center, y_center).buffer(1, self._num_points // 4)
        polygon = sh.affinity.scale(
            polygon, xfact=x_radius, yfact=y_radius, origin=(x_center, y_center)
        )
        if angle != 0:
            polygon = sh.affinity.rotate(polygon, angle, origin=(x_center, y_center))
        session = _bound_session(self)
        if session is not None:
            previous = self._x_radius, self._y_radius, self._angle

            def restore():
                self._x_radius, self._y_radius, self._angle = previous

            session.remember(self, "ellipse_dimensions", restore)
        self._sh_polygon = _stage_live_value(self, "_sh_polygon", polygon)
        self._x_radius, self._y_radius, self._angle = x_radius, y_radius, angle

    @property
    def x_center(self):
        return self.get_centroid_coordinates()[0]

    @x_center.setter
    def x_center(self, value):
        self.translate(value - self.x_center, 0)

    @property
    def y_center(self):
        return self.get_centroid_coordinates()[1]

    @y_center.setter
    def y_center(self, value):
        self.translate(0, value - self.y_center)

    @property
    def x_radius(self):
        return self._x_radius

    @x_radius.setter
    def x_radius(self, value):
        if value <= 0:
            raise InvalidParameterError(f"x_radius must be positive; got {value}.")
        self._rebuild(self.x_center, self.y_center, value, self._y_radius, self._angle)

    @property
    def y_radius(self):
        return self._y_radius

    @y_radius.setter
    def y_radius(self, value):
        if value <= 0:
            raise InvalidParameterError(f"y_radius must be positive; got {value}.")
        self._rebuild(self.x_center, self.y_center, self._x_radius, value, self._angle)

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        self._rebuild(
            self.x_center, self.y_center, self._x_radius, self._y_radius, value
        )

    @property
    def number_of_points(self):
        return self._num_points

    @number_of_points.setter
    def number_of_points(self, value):
        self._num_points = _round_shape_resolution(self, value)

    @property
    def width(self):
        """Returns the width of the ellipse along the x axis."""
        return 2 * self._x_radius

    @width.setter
    def width(self, value):
        """Controls the width of the ellipse along the x axis."""
        if value <= 0:
            raise InvalidParameterError(f"width must be positive; got {value}.")
        self.x_radius = value / 2

    @property
    def height(self):
        """Returns the height of the ellipse along the y axis."""
        return 2 * self._y_radius

    @height.setter
    def height(self, value):
        """Controls the height of the ellipse along the y axis."""
        if value <= 0:
            raise InvalidParameterError(f"height must be positive; got {value}.")
        self.y_radius = value / 2

    @property
    def circumference(self):
        return self._sh_polygon.exterior.length


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
        Typical range is ``0.5`` to ``4``.
        Default depends on the ``figure_style`` configuration.
    line_style : str, optional
        The style of the line.
        Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
        ``"dotted"``.
        Default depends on the ``figure_style`` configuration.
    fill_alpha : float, optional
        The alpha value of the fill.
        Range is ``0`` (transparent) to ``1`` (opaque).
        Default depends on the ``figure_style`` configuration.

    Notes
    -----
    Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
    (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
    values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).
    """

    def __init__(
        self,
        x_bottom_left: float,
        y_bottom_left: float,
        width: float,
        height: float,
        fill: bool | Inherit = INHERIT,
        fill_color: str | Inherit = INHERIT,
        edge_color: str | Inherit = INHERIT,
        line_width: float | Inherit = INHERIT,
        line_style: str | Inherit = INHERIT,
        fill_alpha: float | Inherit = INHERIT,
    ):
        self._fill = fill
        self._fill_color = fill_color
        self._edge_color = edge_color
        self._line_width = line_width
        self._line_style = line_style
        self._fill_alpha = fill_alpha
        self._sh_polygon = ShPolygon(
            [
                (x_bottom_left, y_bottom_left),
                (x_bottom_left + 1, y_bottom_left),
                (x_bottom_left + 1, y_bottom_left + 1),
                (x_bottom_left, y_bottom_left + 1),
            ]
        )
        self.width = width
        self.height = height

    @property
    def x_bottom_left(self):
        return self.vertices[0][0]

    @x_bottom_left.setter
    def x_bottom_left(self, value):
        self._sh_polygon = _stage_live_value(
            self,
            "_sh_polygon",
            ShPolygon(
                [
                    (value, self.y_bottom_left),
                    (value + self.width, self.y_bottom_left),
                    (value + self.width, self.y_bottom_left + self.height),
                    (value, self.y_bottom_left + self.height),
                ]
            ),
        )

    @property
    def y_bottom_left(self):
        return self.vertices[0][1]

    @y_bottom_left.setter
    def y_bottom_left(self, value):
        self._sh_polygon = _stage_live_value(
            self,
            "_sh_polygon",
            ShPolygon(
                [
                    (self.x_bottom_left, value),
                    (self.x_bottom_left + self.width, value),
                    (self.x_bottom_left + self.width, value + self.height),
                    (self.x_bottom_left, value + self.height),
                ]
            ),
        )

    @property
    def width(self):
        return self.vertices[1][0] - self.vertices[0][0]

    @width.setter
    def width(self, value):
        if value <= 0:
            raise InvalidParameterError(f"width must be positive; got {value}.")
        self._sh_polygon = _stage_live_value(
            self,
            "_sh_polygon",
            ShPolygon(
                [
                    (self.x_bottom_left, self.y_bottom_left),
                    (self.x_bottom_left + value, self.y_bottom_left),
                    (self.x_bottom_left + value, self.y_bottom_left + self.height),
                    (self.x_bottom_left, self.y_bottom_left + self.height),
                ]
            ),
        )

    @property
    def height(self):
        return self.vertices[2][1] - self.vertices[1][1]

    @height.setter
    def height(self, value):
        if value <= 0:
            raise InvalidParameterError(f"height must be positive; got {value}.")
        self._sh_polygon = _stage_live_value(
            self,
            "_sh_polygon",
            ShPolygon(
                [
                    (self.x_bottom_left, self.y_bottom_left),
                    (self.x_bottom_left + self.width, self.y_bottom_left),
                    (self.x_bottom_left + self.width, self.y_bottom_left + value),
                    (self.x_bottom_left, self.y_bottom_left + value),
                ]
            ),
        )

    @property
    def center(self):
        return self.get_centroid_coordinates()

    @center.setter
    def center(self, value):
        x, y = value
        self._sh_polygon = _stage_live_value(
            self,
            "_sh_polygon",
            ShPolygon(
                [
                    (x - self.width / 2, y - self.height / 2),
                    (x + self.width / 2, y - self.height / 2),
                    (x + self.width / 2, y + self.height / 2),
                    (x - self.width / 2, y + self.height / 2),
                ]
            ),
        )

    @classmethod
    def from_center(
        cls,
        x: float,
        y: float,
        width: float,
        height: float,
        fill: bool | Inherit = INHERIT,
        fill_color: str | Inherit = INHERIT,
        edge_color: str | Inherit = INHERIT,
        line_width: float | Inherit = INHERIT,
        line_style: str | Inherit = INHERIT,
        fill_alpha: float | Inherit = INHERIT,
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
            Typical range is ``0.5`` to ``4``.
            Default depends on the ``figure_style`` configuration.
        line_style : str, optional
            The style of the line.
            Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
            ``"dotted"``.
            Default depends on the ``figure_style`` configuration.
        fill_alpha : float, optional
            The alpha value of the fill.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).
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
