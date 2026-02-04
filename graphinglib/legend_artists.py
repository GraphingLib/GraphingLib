from matplotlib.artist import Artist
from matplotlib.collections import LineCollection
from matplotlib.colors import is_color_like
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerLineCollection
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Polygon, Patch
from matplotlib.transforms import Transform
from matplotlib.typing import ColorType
from numpy import array, full_like
from typing import Any, Literal, Optional, Protocol, Sequence, runtime_checkable


class HandlerMultipleLines(HandlerLineCollection):
    """
    Custom Handler for `LineCollection <https://matplotlib.org/stable/api/collections_api.html#matplotlib.collections.LineCollection>`_ objects.

    .. seealso:: The Matplotlib documentation on `legend handlers <https://matplotlib.org/stable/api/legend_handler_api.html>`_.
    """

    def create_artists(
        self,
        legend: Legend,
        orig_handle: Artist,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Transform,
    ) -> list[Line2D]:
        numlines = len(orig_handle.get_segments())
        xdata, _ = self.get_xdata(legend, xdescent, ydescent, width, height, fontsize)
        lines = []
        ydata = full_like(xdata, height / (numlines + 1))
        for i in range(numlines):
            line = Line2D(xdata, ydata * (numlines - i) - ydescent)
            self.update_prop(line, orig_handle, legend)
            try:
                color = orig_handle.get_colors()[i]
            except IndexError:
                color = orig_handle.get_colors()[0]
            try:
                dashes = orig_handle.get_dashes()[i]
            except IndexError:
                dashes = orig_handle.get_dashes()[0]
            if dashes[1] is not None:
                line.set_dashes(dashes[1])
            line.set_color(color)
            line.set_transform(trans)
            line.set_linewidth(2)
            lines.append(line)
        return lines


class HandlerMultipleVerticalLines(HandlerLineCollection):
    """
    Custom handler for :class:`~graphinglib.legend_artists.VerticalLineCollection` objects.

    .. seealso:: The Matplotlib documentation on `legend handlers <https://matplotlib.org/stable/api/legend_handler_api.html>`_.
    """

    def create_artists(
        self,
        legend: Legend,
        orig_handle: Artist,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Transform,
    ) -> list[Line2D]:
        numlines = len(orig_handle.get_segments())
        lines = []
        xdata = array([width / (numlines + 1), width / (numlines + 1)])
        ydata = array([0, height])
        for i in range(numlines):
            line = Line2D(xdata * (numlines - i) - xdescent, ydata - ydescent)
            self.update_prop(line, orig_handle, legend)
            try:
                color = orig_handle.get_colors()[i]
            except IndexError:
                color = orig_handle.get_colors()[0]
            try:
                dashes = orig_handle.get_dashes()[i]
            except IndexError:
                dashes = orig_handle.get_dashes()[0]
            if dashes[1] is not None:
                line.set_dashes(dashes[1])
            line.set_color(color)
            line.set_transform(trans)
            line.set_linewidth(2)
            lines.append(line)
        return lines


class VerticalLineCollection(LineCollection):
    """
    Dummy class for vertical `LineCollection <https://matplotlib.org/stable/api/collections_api.html#matplotlib.collections.LineCollection>`_.
    """

    pass


def histogram_legend_artist(
    legend: Legend,
    orig_handle: Artist,
    xdescent: float,
    ydescent: float,
    width: float,
    height: float,
    fontsize: float,
) -> Polygon:
    """
    The custom :class:`~graphinglib.data_plotting_1d.Histogram` legend artist.
    """
    xy = array(
        [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0], [0, 4, 4, 2.5, 2.5, 5, 5, 1.5, 1.5, 0, 0]]
    ).T
    xy[:, 0] = width * xy[:, 0] / 4 + xdescent
    xy[:, 1] = height * xy[:, 1] / 5 - ydescent
    patch = Polygon(xy)
    return patch


@runtime_checkable
class LegendElement(Protocol):
    """
    This class implements a legend element that can be used to create custom legend entries for the
    :meth:`~graphinglib.SmartFigure.set_custom_legend` method. It should not be used on its own and must be subclassed
    to create specific legend elements that implement the `handle` property, which returns a Matplotlib artist. All
    parameters are also available as properties.
    """

    @property
    def handle(self) -> Artist:
        """
        Returns the Matplotlib artist that represents this legend element.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, value: str) -> None:
        self._label = value

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError("Alpha value must be a number.")
        if not (0 <= value <= 1):
            raise ValueError("Alpha value must be between 0 and 1.")
        self._alpha = value

    def _color_setter(self, attr: str, value: ColorType) -> None:
        if value is not None:
            if not is_color_like(value):
                raise ValueError(f"'{value}' is not a valid color.")
        setattr(self, f"_{attr}", value)

    def _number_setter(self, attr: str, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError(f"'{value}' is not a valid number.")
        if value < 0:
            raise ValueError(f"'{value}' cannot be negative.")
        setattr(self, f"_{attr}", value)

    def _line_style_setter(
        self,
        attr: str,
        value: Literal["-", "--", "-.", ":", "solid", "dashed", "dashdot", "dotted"]
        | tuple[float, Sequence],
    ) -> None:
        if isinstance(value, str):
            if value not in [
                "-",
                "--",
                "-.",
                ":",
                "solid",
                "dashed",
                "dashdot",
                "dotted",
            ]:
                raise ValueError(f"'{value}' is not a valid line style.")
        elif isinstance(value, tuple):
            if len(value) != 2 or not all(
                isinstance(x, (int, float)) for x in [value[0], *value[1]]
            ):
                raise ValueError(f"'{value}' is not a valid line style tuple.")
        else:
            raise TypeError(f"'{value}' is not a valid line style type.")
        setattr(self, f"_{attr}", value)


class LegendLine(LegendElement):
    """
    This class implements a legend line wrapping the
    `Line2D <https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D>`_ object
    for creating custom legend entries with the :meth:`~graphinglib.SmartFigure.set_custom_legend` method. All
    parameters are also available as properties.

    Parameters
    ----------
    label : str
        The label for the legend line.
    color : ColorType
        The color of the line, which can be `any color format supported by Matplotlib
        <https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def>`_.
    gap_color : ColorType, optional
        The color of the gaps in the line (for dashed lines), which can be `any color format supported by Matplotlib
        <https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def>`_.
    line_width : float, optional
        The width of the line in points.
        Defaults to ``2.0``.
    line_style : {"-", "--", "-.", ":", "solid", "dashed", "dashdot", "dotted"} or tuple of float and sequence, optional
        The style of the line, which can be `any pattern supported by Matplotlib
        <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch.set_linestyle>`_.
        Defaults to ``"-"`` (solid line).
    alpha : float, optional
        The transparency level of the line, between ``0`` (fully transparent) and ``1`` (fully opaque).
        Defaults to ``1.0``.
    """

    def __init__(
        self,
        label: str,
        color: ColorType,
        gap_color: Optional[ColorType] = None,
        line_width: float = 2.0,
        line_style: Literal[
            "-", "--", "-.", ":", "solid", "dashed", "dashdot", "dotted"
        ]
        | tuple[float, Sequence] = "-",
        alpha: float = 1.0,
    ) -> None:
        self.label = label
        self.color = color
        self.gap_color = gap_color
        self.line_width = line_width
        self.line_style = line_style
        self.alpha = alpha

    @property
    def handle(self) -> Line2D:
        """
        Returns the Matplotlib Line2D artist that represents this legend line.
        """
        return Line2D(
            [],
            [],
            label=self._label,
            color=self._color,
            gapcolor=self._gap_color,
            linewidth=self._line_width,
            linestyle=self._line_style,
            alpha=self._alpha,
        )

    @property
    def color(self) -> ColorType:
        return self._color

    @color.setter
    def color(self, value: ColorType) -> None:
        self._color_setter("color", value)

    @property
    def gap_color(self) -> ColorType:
        return self._gap_color

    @gap_color.setter
    def gap_color(self, value: ColorType) -> None:
        self._color_setter("gap_color", value)

    @property
    def line_width(self) -> float:
        return self._line_width

    @line_width.setter
    def line_width(self, value: float) -> None:
        self._number_setter("line_width", value)

    @property
    def line_style(
        self,
    ) -> (
        Literal["-", "--", "-.", ":", "solid", "dashed", "dashdot", "dotted"]
        | tuple[float, Sequence]
    ):
        return self._line_style

    @line_style.setter
    def line_style(
        self,
        value: Literal["-", "--", "-.", ":", "solid", "dashed", "dashdot", "dotted"]
        | tuple[float, Sequence],
    ) -> None:
        self._line_style_setter("line_style", value)


class LegendMarker(LegendElement):
    """
    This class implements a legend marker wrapping the `Line2D
    <https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D>`_ object with a
    marker style set for creating custom legend entries with the :meth:`~graphinglib.SmartFigure.set_custom_legend`
    method. All parameters are also available as properties.

    Parameters
    ----------
    label : str
        The label for the legend line.
    face_color : ColorType
        The color of the marker, which can be `any color format supported by Matplotlib
        <https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def>`_.
    face_color_alt : ColorType, optional
        The alternative face color of the marker (for markers with two colors), which can be `any color format
        supported by Matplotlib
        <https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def>`_.
    edge_color : ColorType, optional
        The color of the marker edge, which can be `any color format supported by Matplotlib
        <https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def>`_.
    edge_width : float, optional
        The width of the marker edge in points.
        Defaults to ``1.0``.
    marker_size : float, optional
        The size of the marker in points.
        Defaults to ``6.0``.
    marker_style : Any, optional
        The style of the marker, which can be `any marker style supported by Matplotlib
        <https://matplotlib.org/stable/api/markers_api.html#matplotlib.markers.MarkerStyle>`_.
        Defaults to ``"o"`` (circle).
    fill_style : {"full", "left", "right", "bottom", "top"}, optional
        The fill style of the marker.
        Defaults to ``"full"``.
    alpha : float, optional
        The transparency level of the marker, between ``0`` (fully transparent) and ``1`` (fully opaque).
        Defaults to ``1.0``.
    """

    def __init__(
        self,
        label: str,
        face_color: Optional[ColorType] = None,
        face_color_alt: Optional[ColorType] = None,
        edge_color: Optional[ColorType] = None,
        edge_width: float = 1.0,
        marker_size: float = 6.0,
        marker_style: Any = "o",
        fill_style: Literal["full", "left", "right", "bottom", "top"] = "full",
        alpha: float = 1.0,
    ) -> None:
        self.label = label
        self.face_color = face_color
        self.face_color_alt = face_color_alt
        self.edge_color = edge_color
        self.edge_width = edge_width
        self.marker_size = marker_size
        self.marker_style = marker_style
        self.fill_style = fill_style
        self.alpha = alpha

    @property
    def handle(self) -> Line2D:
        """
        Returns the Matplotlib Line2D artist that represents this legend marker.
        """
        return Line2D(
            [],
            [],
            linestyle="none",
            label=self._label,
            markerfacecolor=self._face_color
            if self._face_color is not None
            else "none",
            markerfacecoloralt=self._face_color_alt
            if self._face_color_alt is not None
            else "none",
            markeredgecolor=self._edge_color
            if self._edge_color is not None
            else "none",
            markeredgewidth=self._edge_width,
            markersize=self._marker_size,
            marker=self._marker_style,
            fillstyle=(self._fill_style if self._fill_style is not None else "none"),
            alpha=self._alpha,
        )

    @property
    def face_color(self) -> ColorType:
        return self._face_color

    @face_color.setter
    def face_color(self, value: ColorType) -> None:
        self._color_setter("face_color", value)

    @property
    def face_color_alt(self) -> ColorType:
        return self._face_color_alt

    @face_color_alt.setter
    def face_color_alt(self, value: ColorType) -> None:
        self._color_setter("face_color_alt", value)

    @property
    def edge_color(self) -> ColorType:
        return self._edge_color

    @edge_color.setter
    def edge_color(self, value: ColorType) -> None:
        self._color_setter("edge_color", value)

    @property
    def edge_width(self) -> float:
        return self._edge_width

    @edge_width.setter
    def edge_width(self, value: float) -> None:
        self._number_setter("edge_width", value)

    @property
    def marker_size(self) -> float:
        return self._marker_size

    @marker_size.setter
    def marker_size(self, value: float) -> None:
        self._number_setter("marker_size", value)

    @property
    def marker_style(self) -> Any:
        return self._marker_style

    @marker_style.setter
    def marker_style(self, value: Any) -> None:
        try:
            MarkerStyle(value)  # Validate the marker style
        except Exception:
            raise ValueError(f"'{value}' is not a valid marker style.")
        self._marker_style = value

    @property
    def fill_style(self) -> Literal["full", "left", "right", "bottom", "top"]:
        return self._fill_style

    @fill_style.setter
    def fill_style(
        self, value: Literal["full", "left", "right", "bottom", "top"]
    ) -> None:
        if value is not None:
            if value not in MarkerStyle.fillstyles:
                raise ValueError(f"'{value}' is not a valid fill style.")
        self._fill_style = value


class LegendPatch(LegendElement):
    """
    This class implements a legend patch wrapping the
    `Patch <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch>`_ object
    for creating custom legend entries with the :meth:`~graphinglib.SmartFigure.set_custom_legend` method. All
    parameters are also available as properties.

    Parameters
    ----------
    label : str
        The label for the legend patch.
    face_color : ColorType
        The face color of the patch, which can be `any color format supported by Matplotlib
        <https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def>`_.
    edge_color : ColorType, optional
        The edge color of the patch, which can be `any color format supported by Matplotlib
        <https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def>`_.
    line_width : float, optional
        The width of the patch edge and hatch (if present) in points.
        Defaults to ``1.0``.
    line_style : {"-", "--", "-.", ":", "solid", "dashed", "dashdot", "dotted"} or tuple of float and sequence, optional
        The style of the patch edge, which can be `any pattern supported by Matplotlib
        <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch.set_linestyle>`_.
        Defaults to ``"-"`` (solid line).
    hatch : {"/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"}, optional
        The hatch pattern of the patch, which can be `any hatch pattern supported by Matplotlib
        <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch.set_hatch>`_.
    alpha : float, optional
        The transparency level of the patch, between ``0`` (fully transparent) and ``1`` (fully opaque).
        Defaults to ``1.0``.
    """

    def __init__(
        self,
        label: str,
        face_color: Optional[ColorType] = None,
        edge_color: Optional[ColorType] = None,
        line_width: float = 1.0,
        line_style: Literal[
            "-", "--", "-.", ":", "solid", "dashed", "dashdot", "dotted"
        ]
        | tuple[float, Sequence] = "-",
        hatch: Literal["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"] = None,
        alpha: float = 1.0,
    ) -> None:
        self.label = label
        self.face_color = face_color
        self.edge_color = edge_color
        self.line_width = line_width
        self.line_style = line_style
        self.hatch = hatch
        self.alpha = alpha

    @property
    def handle(self) -> Patch:
        """
        Returns the Matplotlib Patch artist that represents this legend patch.
        """
        return Patch(
            label=self._label,
            facecolor=(self._face_color if self._face_color is not None else "none"),
            edgecolor=(self._edge_color if self._edge_color is not None else "none"),
            linewidth=self._line_width,
            linestyle=self._line_style,
            hatch=self._hatch,
            alpha=self._alpha,
            fill=(self._face_color is not None),
        )

    @property
    def face_color(self) -> ColorType:
        return self._face_color

    @face_color.setter
    def face_color(self, value: ColorType) -> None:
        self._color_setter("face_color", value)

    @property
    def edge_color(self) -> ColorType:
        return self._edge_color

    @edge_color.setter
    def edge_color(self, value: ColorType) -> None:
        self._color_setter("edge_color", value)

    @property
    def line_width(self) -> float:
        return self._line_width

    @line_width.setter
    def line_width(self, value: float) -> None:
        self._number_setter("line_width", value)

    @property
    def line_style(
        self,
    ) -> (
        Literal["-", "--", "-.", ":", "solid", "dashed", "dashdot", "dotted"]
        | tuple[float, Sequence]
    ):
        return self._line_style

    @line_style.setter
    def line_style(
        self,
        value: Literal["-", "--", "-.", ":", "solid", "dashed", "dashdot", "dotted"]
        | tuple[float, Sequence],
    ) -> None:
        self._line_style_setter("line_style", value)

    @property
    def hatch(self) -> Literal["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]:
        return self._hatch

    @hatch.setter
    def hatch(
        self, value: Literal["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
    ) -> None:
        if value is not None:
            # This logic is adapted from matplotlib's hatch validation
            valid_hatch_patterns = set(r"-+|/\xXoO.*")
            invalids = set(value).difference(valid_hatch_patterns)
            if invalids:
                raise ValueError(f"Invalid hatch pattern(s): {invalids}")
        self._hatch = value
