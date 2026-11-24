"""Artist bindings for one SmartFigure animation."""

from collections.abc import Callable, Collection, Mapping
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain
from math import isfinite
from numbers import Integral, Real
from typing import Any, Literal, Protocol, cast

import numpy as np
from matplotlib import rcParams
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.collections import (
    LineCollection,
    PathCollection,
    PolyCollection,
)
from matplotlib.colorbar import Colorbar
from matplotlib.container import ErrorbarContainer
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.offsetbox import DrawingArea
from matplotlib.patches import Polygon
from matplotlib.text import Text as TextArtist
from matplotlib.ticker import FixedLocator, FuncFormatter, Locator, MultipleLocator
from numpy.typing import ArrayLike

from .exceptions import (
    GraphingLibError,
    InvalidParameterError,
    PlottingError,
    UnsupportedFeatureError,
)
from .inherit import is_inherit
from .legend_artists import _reference_line_style, _ReferenceLegendSwatch


@dataclass
class _ArtistEdit:
    apply: Callable[[], None]
    undo: Callable[[], None]


def _edit(setter: Callable[[Any], None], old: Any, new: Any) -> _ArtistEdit:
    return _ArtistEdit(lambda: setter(new), lambda: setter(old))


def _combine_edits(edits: list[_ArtistEdit]) -> _ArtistEdit:
    def apply() -> None:
        for edit in edits:
            edit.apply()

    def undo() -> None:
        for edit in reversed(edits):
            edit.undo()

    return _ArtistEdit(apply, undo)


def _attach_artists(axes: Axes, artists: tuple[Artist, ...]) -> _ArtistEdit:
    def attach() -> None:
        for artist in artists:
            axes.add_artist(artist)

    def detach() -> None:
        for artist in artists:
            if artist.axes is axes:
                artist.remove()

    return _ArtistEdit(attach, detach)


_ERROR_STYLE_PROPERTIES = (
    "_errorbars_color",
    "_errorbars_line_width",
    "_cap_width",
    "_cap_thickness",
)


def _subplot_visibility_entries(
    value: bool | list[bool | None], slots: Collection[int], subplot_count: int
) -> dict[int, bool]:
    if isinstance(value, bool):
        return dict.fromkeys(slots, value)
    if len(value) > subplot_count:
        raise InvalidParameterError("Too many visibility entries for this SmartFigure.")
    entries = {index: flag for index, flag in enumerate(value) if flag is not None}
    if any(index not in slots for index in entries):
        raise InvalidParameterError(
            "Set axis properties on a nested SmartFigure directly."
        )
    return entries


_AXIS_TICK_FIELDS = (
    ("major_locator", "isDefault_majloc"),
    ("minor_locator", "isDefault_minloc"),
    ("major_formatter", "isDefault_majfmt"),
    ("minor_formatter", "isDefault_minfmt"),
)


def _axis_tick_keys(dimension: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    prefix = f"{dimension}_" if dimension else ""
    return (
        (prefix + "ticks", prefix + "tick_spacing", prefix + "tick_labels"),
        ("minor_" + prefix + "ticks", "minor_" + prefix + "tick_spacing"),
    )


def _updated_tick_config(
    previous: dict[str, Any],
    changes: dict[str, Any],
    *,
    reset: bool,
    dimensions: tuple[str, ...],
) -> dict[str, Any]:
    config = {} if reset else previous.copy()
    for dimension in dimensions:
        for keys in _axis_tick_keys(dimension):
            if any(key in changes for key in keys):
                for key in keys:
                    config.pop(key, None)
    return {**config, **changes}


def _save_axis_ticks(axis: Axis) -> list[tuple[str, str, Any, bool]]:
    return [
        (field, flag, getattr(axis, f"get_{field}")(), getattr(axis, flag))
        for field, flag in _AXIS_TICK_FIELDS
    ]


def _restore_axis_ticks(
    axis: Axis, ticks: list[tuple[str, str, Any, bool]], *, custom_only: bool = False
) -> None:
    for field, flag, value, is_default in ticks:
        if not custom_only or not is_default:
            getattr(axis, f"set_{field}")(value)
            setattr(axis, flag, is_default)


def _axis_removal(axes: Axes, remove: bool) -> _ArtistEdit:
    def write(hidden: bool) -> None:
        if hidden:
            axes.set_axis_off()
        else:
            axes.set_axis_on()

    return _edit(write, not axes.axison, remove)


def _tick_removal(axis: Axis, dimension: str) -> Callable[[bool], _ArtistEdit]:
    sides = ("bottom", "top") if dimension == "x" else ("left", "right")
    keys = (*sides, *(f"label{side}" for side in sides))
    attributes = ("tick1line", "tick2line", "label1", "label2")
    groups: tuple[Literal["major"], Literal["minor"]] = ("major", "minor")

    def read() -> dict[Literal["major", "minor"], dict[str, bool]]:
        flags = {}
        for which in groups:
            params = axis.get_tick_params(which=which)
            if not all(key in params for key in keys):
                # Matplotlib may omit flags which still use the Tick defaults.
                get_ticks = (
                    axis.get_major_ticks if which == "major" else axis.get_minor_ticks
                )
                tick = get_ticks(1)[0]
            else:
                flags[which] = {key: params[key] for key in keys}
                continue
            flags[which] = {
                key: params[key] if key in params else getattr(tick, attr).get_visible()
                for key, attr in zip(keys, attributes, strict=True)
            }
        return flags

    initial = read()
    hidden = {which: dict.fromkeys(keys, False) for which in groups}

    def write(flags: dict[Literal["major", "minor"], dict[str, bool]]) -> None:
        for which, params in flags.items():
            axis.set_tick_params(which=which, **params)

    def prepare(remove: bool) -> _ArtistEdit:
        return _edit(write, read(), hidden if remove else initial)

    return prepare


class _TickStyle(Protocol):
    hidden: bool
    suppressed_labels: tuple[str, ...]

    def prepare(self, style: list[dict[str, Any]], hide: bool) -> _ArtistEdit: ...


class _NativeTickStyle:
    """Keep axis tick styles and visibility in sync without resetting ticks."""

    groups: tuple[Literal["major"], Literal["minor"]] = ("major", "minor")

    def __init__(self, axis: Axis, dimension: str) -> None:
        self.axis = axis
        sides = ("bottom", "top") if dimension == "x" else ("left", "right")
        self.side_keys = (*sides, *(f"label{side}" for side in sides))
        self.defaults = self.read()
        self.hidden = False
        self.suppressed_labels: tuple[str, ...] = ()

    def read(self) -> list[dict[str, Any]]:
        snapshots = []
        for ticks in (self.axis.get_major_ticks, self.axis.get_minor_ticks):
            tick = ticks(1)[0]
            snapshots.append(
                {
                    "length": tick.tick1line.get_markersize(),
                    "width": tick.tick1line.get_markeredgewidth(),
                    "color": tick.tick1line.get_markeredgecolor(),
                    "direction": tick.get_tickdir(),
                    "pad": tick.get_pad(),
                    "labelsize": tick.label1.get_fontsize(),
                    "labelcolor": tick.label1.get_color(),
                    "labelrotation": cast(Any, tick)._labelrotation,
                    **dict(
                        zip(
                            self.side_keys,
                            (
                                tick.tick1line.get_visible(),
                                tick.tick2line.get_visible(),
                                tick.label1.get_visible(),
                                tick.label2.get_visible(),
                            ),
                            strict=True,
                        )
                    ),
                }
            )
        return snapshots

    def prepare(self, style: list[dict[str, Any]], hide: bool) -> _ArtistEdit:
        previous = self.read()
        hidden = self.hidden
        offset = self.axis.get_offset_text()
        old_offset = offset.get_color(), offset.get_visible()
        excluded = {
            "bottom",
            "top",
            "left",
            "right",
            "labelbottom",
            "labeltop",
            "labelleft",
            "labelright",
        } - set(self.side_keys)
        requested = []
        for index, (default, overrides) in enumerate(
            zip(self.defaults, style, strict=True)
        ):
            suppressed = (
                self.side_keys if hide else self.suppressed_labels if index == 0 else ()
            )
            requested.append(
                default
                | {
                    key: value
                    for key, value in overrides.items()
                    if key not in excluded
                }
                | dict.fromkeys(suppressed, False)
            )

        def write(params: list[dict[str, Any]]) -> None:
            for which, config in zip(self.groups, params, strict=True):
                self.axis.set_tick_params(which=which, **config)

        def apply() -> None:
            write(requested)
            offset.set_color(style[1].get("labelcolor", requested[0]["labelcolor"]))
            offset.set_visible(any(requested[0][key] for key in self.side_keys[2:]))
            self.hidden = hide

        def undo() -> None:
            write(previous)
            offset.set_color(old_offset[0])
            offset.set_visible(old_offset[1])
            self.hidden = hidden

        return _ArtistEdit(apply, undo)


class _WCSTickStyle:
    def __init__(self, coordinate: Any, dimension: str) -> None:
        self.coordinate = coordinate
        self.sides = ("bottom", "top") if dimension == "x" else ("left", "right")
        self.defaults = self.read_style()
        self.hidden = False
        self.suppressed_labels: tuple[str, ...] = ()

    def read_style(self) -> tuple[dict[str, Any], dict[str, Any], float]:
        # CoordinateHelper has setters but no public getters for these styles.
        ticks, labels = self.coordinate._ticks, self.coordinate._ticklabels
        return (
            {
                "size": ticks.get_ticksize(),
                "width": ticks.get_linewidth(),
                "color": ticks.get_color(),
                "direction": "out" if ticks.get_tick_out() else "in",
            },
            {
                "size": labels.get_fontsize(),
                "color": labels.get_color(),
                "pad": labels.get_pad(),
                "rotation": labels.get_rotation(),
            },
            ticks.get_minor_ticksize(),
        )

    def prepare(self, style: list[dict[str, Any]], hide: bool) -> _ArtistEdit:
        coordinate = self.coordinate
        previous = self.read_style()
        old_positions = (
            coordinate.get_ticks_position(),
            coordinate.get_ticklabel_position(),
        )
        old_visible = coordinate.get_ticks_visible(), coordinate.get_ticklabel_visible()
        hidden = self.hidden
        major, minor = style
        default_ticks, default_labels, default_minor = self.defaults
        ticks = default_ticks | {
            target: major[key]
            for key, target in (
                ("length", "size"),
                ("width", "width"),
                ("color", "color"),
                ("direction", "direction"),
            )
            if key in major
        }
        labels = default_labels | {
            target: major[key]
            for key, target in (
                ("labelsize", "size"),
                ("labelcolor", "color"),
                ("pad", "pad"),
                ("labelrotation", "rotation"),
            )
            if key in major
        }
        positions = (
            [side[0] for side in self.sides if major.get(side)],
            [
                side[0]
                for side in self.sides
                if major.get(f"label{side}")
                and f"label{side}" not in self.suppressed_labels
            ],
        )

        def write(
            tick_style: dict[str, Any], label_style: dict[str, Any], minor_size: float
        ) -> None:
            coordinate.set_ticks(**tick_style)
            coordinate.set_ticklabel(**label_style)
            coordinate.tick_params(which="minor", length=minor_size)

        def write_positions(
            ticks: list[str],
            labels: list[str],
            ticks_visible: bool,
            labels_visible: bool,
        ) -> None:
            coordinate.set_ticks_position(ticks)
            coordinate.set_ticklabel_position(labels)
            coordinate.set_ticks_visible(ticks_visible)
            coordinate.set_ticklabel_visible(labels_visible)

        def apply() -> None:
            write(ticks, labels, minor.get("length", default_minor))
            write_positions(
                *positions,
                bool(positions[0]) and not hide,
                bool(positions[1]) and not hide,
            )
            self.hidden = hide

        def undo() -> None:
            write(*previous)
            write_positions(*old_positions, *old_visible)
            self.hidden = hidden

        return _ArtistEdit(apply, undo)


def _draw_live_errorbars(
    axes: Axes, x: Any, y: Any, *, show: bool = True, **params: Any
) -> ErrorbarContainer:
    cap_width = params.get("capsize")
    if cap_width is None:
        cap_width = rcParams["errorbar.capsize"]
    hide_caps = (
        axes.name != "polar"
        and cap_width <= 0
        and (params.get("xerr") is not None or params.get("yerr") is not None)
    )
    if hide_caps:
        # Keep the native cap artists and legend entries available for later frames.
        params["capsize"] = 1
    errors = (params.get("xerr"), params.get("yerr"))
    if not show:
        # Reserve the configured artists without including hidden error
        # extents in the initial axes limits.
        params["xerr"] = None if errors[0] is None else 0
        params["yerr"] = None if errors[1] is None else 0
    handle = axes.errorbar(x, y, **params)
    if hide_caps:
        for cap in handle.lines[1]:
            cap.set_markersize(0)
            cap.set_visible(False)
    if not show:
        _, caps, segments = handle.lines
        directions = [
            (is_x, err) for is_x, err in zip((True, False), errors) if err is not None
        ]
        x, y = np.asarray(x), np.asarray(y)
        for index, ((is_x, err), segment) in enumerate(
            zip(directions, segments, strict=True)
        ):
            error = _error_array(err, x.size, "x_error" if is_x else "y_error")
            assert error is not None
            binding = _ErrorArtists(segment, tuple(caps[2 * index : 2 * index + 2]))
            _error_update(
                binding, x if is_x else y, y if is_x else x, error, is_x=is_x
            ).apply()
            segment.set_visible(False)
        for cap in caps:
            cap.set_visible(False)
    return handle


def _reserve_errorbar_directions(
    axes: Axes,
    existing: ErrorbarContainer | None,
    *,
    color: Any,
    zorder: float,
    cap_width: float | None,
) -> ErrorbarContainer | None:
    if axes.name == "polar":
        return None
    missing_x = existing is None or not existing.has_xerr
    missing_y = existing is None or not existing.has_yerr
    if not (missing_x or missing_y):
        return None
    cap_width = rcParams["errorbar.capsize"] if cap_width is None else cap_width
    segments, caps = [], []
    for missing, marker in ((missing_x, "|"), (missing_y, "_")):
        if missing:
            segments.append(
                LineCollection([], colors=color, zorder=zorder, visible=False)
            )
            caps.extend(
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker=marker,
                    color=color,
                    markersize=2 * cap_width,
                    zorder=zorder,
                    visible=False,
                )
                for _ in range(2)
            )
    return ErrorbarContainer(
        (None, tuple(caps), tuple(segments)),  # ty: ignore[invalid-argument-type] (fmt="none")
        has_xerr=missing_x,
        has_yerr=missing_y,
    )


def _error_style_edits(
    handle: ErrorbarContainer,
    *,
    color: Any,
    line_width: float | None,
    cap_width: float | None,
    cap_thickness: float | None,
    alpha: float | None,
    show: bool,
    directions: tuple[bool, bool] | None = None,
    reserved_axes: Axes | None = None,
    reserved_data: tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]
    | None = None,
) -> list[_ArtistEdit]:
    if not isinstance(show, (bool, np.bool_)):
        raise InvalidParameterError("show_errorbars must be a boolean.")
    _, caps, segments = handle.lines
    for name, value in (
        ("line width", line_width),
        ("cap width", cap_width),
        ("cap thickness", cap_thickness),
    ):
        if value is not None and (
            not isinstance(value, Real) or not isfinite(value) or value < 0
        ):
            raise InvalidParameterError(
                f"Animated errorbar {name} must be a finite, nonnegative number."
            )
    if cap_width and (
        not caps or reserved_axes is None and any(cap.axes is None for cap in caps)
    ):
        raise UnsupportedFeatureError(
            "This plot has no live errorbar cap artists. Use a nonempty initial "
            "dataset on a Cartesian plot to animate cap width."
        )

    edits = []
    present = (
        [
            value
            for value, exists in zip(
                directions, (handle.has_xerr, handle.has_yerr), strict=True
            )
            if exists
        ]
        if directions is not None
        else [True] * len(segments)
    )
    for segment, exists in zip(segments, present, strict=True):
        if color is not None:
            edits.append(_edit(segment.set_color, segment.get_color(), color))
        if line_width is not None:
            edits.append(
                _edit(segment.set_linewidth, segment.get_linewidth(), line_width)
            )
        edits.append(_edit(segment.set_alpha, segment.get_alpha(), alpha))
        edits.append(_edit(segment.set_visible, segment.get_visible(), show and exists))
    for index, cap in enumerate(caps):
        exists = present[index // 2] if len(caps) == 2 * len(segments) else True
        if color is not None:
            edits.append(
                _edit(cap.set_markeredgecolor, cap.get_markeredgecolor(), color)
            )
        if cap_thickness is not None:
            edits.append(
                _edit(cap.set_markeredgewidth, cap.get_markeredgewidth(), cap_thickness)
            )
        size = cap.get_markersize()
        if cap_width is not None:
            edits.append(_edit(cap.set_markersize, size, 2 * cap_width))
        if cap.axes is not None or reserved_axes is not None:
            edits.append(
                _edit(
                    cap.set_visible,
                    cap.get_visible(),
                    show
                    and exists
                    and (size > 0 if cap_width is None else cap_width > 0),
                )
            )
        edits.append(_edit(cap.set_alpha, cap.get_alpha(), alpha))
    if reserved_axes is not None and show and any(present) and segments[0].axes is None:
        edits.insert(0, _attach_artists(reserved_axes, (*segments, *caps)))
    if reserved_data is not None and show:
        x, y, dx, dy = reserved_data
        inputs = [
            values
            for exists, values in (
                (handle.has_xerr, (x, y, dx, True)),
                (handle.has_yerr, (y, x, dy, False)),
            )
            if exists
        ]
        for index, (changing, fixed, error, is_x) in enumerate(inputs):
            if error is not None:
                edits.append(
                    _error_update(
                        _ErrorArtists(segments[index], caps[2 * index : 2 * index + 2]),
                        changing,
                        fixed,
                        error,
                        is_x=is_x,
                    )
                )
    return edits


@dataclass
class _ErrorArtists:
    segments: LineCollection
    caps: tuple[Line2D, ...]


def _error_update(
    artists: _ErrorArtists,
    changing: np.ndarray,
    fixed: np.ndarray,
    error: np.ndarray,
    *,
    is_x: bool,
) -> _ArtistEdit:
    lower, upper = np.broadcast_to(error, (2, changing.size)).astype(float)
    low, high = changing - lower, changing + upper
    start, end = (
        ((low, fixed), (high, fixed)) if is_x else ((fixed, low), (fixed, high))
    )
    segments = np.stack((np.column_stack(start), np.column_stack(end)), axis=1)

    old_segments = [path.vertices for path in artists.segments.get_paths()]
    old_caps = tuple(cap.get_data() for cap in artists.caps)

    def update(new_segments: Any, cap_data: Any) -> None:
        artists.segments.set_segments(new_segments)
        for cap, data in zip(artists.caps, cap_data):
            cap.set_data(data)

    return _ArtistEdit(
        lambda: update(segments, (start, end)),
        lambda: update(old_segments, old_caps),
    )


@dataclass
class _PointArtists:
    points: Line2D | PathCollection
    x_error: _ErrorArtists | None
    y_error: _ErrorArtists | None
    bands: "_CurveBands | None" = None
    flexible_errors: bool = False


@dataclass
class _CurveBands:
    lower: Line2D | None
    upper: Line2D | None
    error_fill: PolyCollection | None

    def edits(
        self, x: np.ndarray, y: np.ndarray, error: np.ndarray | None
    ) -> list[_ArtistEdit]:
        edits = []
        if self.lower is not None and self.upper is not None:
            low, high = _error_bounds(y, error)
            edits.append(_edit(self.lower.set_data, self.lower.get_data(), (x, low)))
            edits.append(_edit(self.upper.set_data, self.upper.get_data(), (x, high)))
            if self.error_fill is not None:
                edits.append(_fill_edit(self.error_fill, x, high, low))
        return edits


def _error_bounds(
    y: np.ndarray, error: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    if error is None:
        return y, y
    lower, upper = np.broadcast_to(error, (2, y.size))
    return y - lower, y + upper


def _fill_edit(
    artist: PolyCollection,
    x: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    where: np.ndarray | None = None,
) -> _ArtistEdit:
    mask = np.isfinite(x) & np.isfinite(y1) & np.isfinite(y2)
    if where is not None:
        mask &= where
    changes = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    polygons = []
    for start, stop in changes.reshape(-1, 2):
        a, b, c = x[start:stop], y1[start:stop], y2[start:stop]
        polygons.append(np.column_stack((np.r_[a, a[::-1]], np.r_[b, c[::-1]])))
    before = [path.vertices for path in artist.get_paths()]
    return _edit(artist.set_verts, before, polygons)


_Binding = _PointArtists | TextArtist | str | None


def _error_array(value: ArrayLike | None, size: int, name: str) -> np.ndarray | None:
    if value is None:
        return None
    error = np.asarray(value)
    if error.shape not in ((), (size,), (2, size)):
        raise InvalidParameterError(
            f"{name} must be a scalar, a length-{size} array, or shape (2, {size})."
        )
    if (
        error.dtype.kind not in "iuf"
        or not np.isfinite(error).all()
        or (error < 0).any()
    ):
        raise InvalidParameterError(
            f"{name} must contain finite, nonnegative real values."
        )
    return error.copy()


class _ColorbarOwner(Protocol):
    _color_bar_params: dict[str, Any]


class _AnimatedColorbarLocator(FixedLocator):
    """Mark tick positions explicitly assigned by an animation callback."""


@dataclass(frozen=True)
class _LegendFilter:
    owner: object
    property_name: str
    is_hidden: Callable[[], bool]


class _LegendRows:
    def __init__(
        self, legend: Legend, shown: bool, filters: list[tuple[_LegendFilter, ...]]
    ) -> None:
        self.legend = legend
        self.shown = shown
        self.columns = cast(Any, legend)._legend_handle_box.get_children()
        self.rows = [row for column in self.columns for row in column.get_children()]
        self.filters = [
            rules
            for rules, handle in zip(filters, legend.legend_handles, strict=True)
            if handle is not None
        ]

    def show(self, shown: bool) -> None:
        self.shown = shown
        self.legend.set_visible(shown and any(row.get_visible() for row in self.rows))

    def set_rows(self, visible: list[bool]) -> None:
        for row, show in zip(self.rows, visible, strict=True):
            row.set_visible(show)
        for column in self.columns:
            column.set_visible(any(row.get_visible() for row in column.get_children()))
        self.show(self.shown)

    def refresh(self) -> None:
        self.set_rows(
            [not any(rule.is_hidden() for rule in rules) for rules in self.filters]
        )

    def prepare(self) -> _ArtistEdit:
        previous = [row.get_visible() for row in self.rows]
        return _ArtistEdit(self.refresh, lambda: self.set_rows(previous))


class _AnimationSession:
    def __init__(self, *, reserve_legend_entries: bool = False) -> None:
        self._reserve_legend_entries = reserve_legend_entries
        self._artists: dict[int, list[_Binding]] = {}
        self._properties: dict[tuple[int, str], list[Callable[[Any], _ArtistEdit]]] = {}
        self._axes_method_artists: dict[int, list[tuple[Artist, ...]]] = {}
        self._updates: dict[tuple[int, str], list[_ArtistEdit]] = {}
        self._dependents: dict[
            tuple[int, str], dict[int, Callable[[], _ArtistEdit]]
        ] = {}
        self._point_links: dict[
            int, dict[int, dict[int, Callable[[], _ArtistEdit]]]
        ] = {}
        self._dirty_dependents: dict[int, Callable[[], _ArtistEdit]] = {}
        self._post_artist_prepares: set[int] = set()
        self._initially_hidden_artists: list[Artist] = []
        self._legend_rows: dict[Legend, _LegendRows] = {}
        self._initial_colorbar_ticks: dict[int, Any] = {}
        self._legend_sources: dict[
            int, tuple[object, Callable[[], str | None], tuple[str, ...]]
        ] = {}
        self._reference_legend_sources: dict[
            int, tuple[tuple[str, ...], Callable[[], list[dict[str, Any]]]]
        ] = {}
        self._rollback: dict[tuple[int, str], Callable[[], None]] = {}
        self._cleanup: list[Callable[[], None]] = []
        self.failure: BaseException | None = None

    def remember(self, element: object, name: str, restore: Callable[[], None]) -> None:
        self._rollback.setdefault((id(element), name), restore)

    def has_frame_update(self, element: object, name: str) -> bool:
        return (id(element), name) in self._rollback

    def on_finish(self, callback: Callable[[], None]) -> None:
        self._cleanup.append(callback)

    def mark_present(self, element: object) -> None:
        self._artists.setdefault(id(element), [])

    def bind_axes_method(self, element: object, artists: tuple[Artist, ...]) -> None:
        self.mark_present(element)
        self._axes_method_artists.setdefault(id(element), []).append(artists)

    def queue_axes_method_properties(
        self, element: object, index: int | None, properties: dict[str, Any]
    ) -> None:
        appearances = self._axes_method_artists.get(id(element), ())
        if not appearances or any(not artists for artists in appearances):
            raise UnsupportedFeatureError(
                "The Axes method must return at least one Matplotlib artist per appearance."
            )
        if index is not None and (
            not isinstance(index, Integral) or isinstance(index, bool)
        ):
            raise InvalidParameterError("The artist index must be an integer or None.")

        updates = {}
        for artists in appearances:
            if index is None:
                selected = artists
            else:
                try:
                    selected = (artists[index],)
                except IndexError as exc:
                    raise InvalidParameterError(
                        f"Artist index {index} is out of range for this Axes method."
                    ) from exc
            for artist in selected:
                for name, value in properties.items():
                    getter = getattr(artist, f"get_{name}", None)
                    setter = getattr(artist, f"set_{name}", None)
                    if not callable(getter) or not callable(setter):
                        raise UnsupportedFeatureError(
                            f"{type(artist).__name__}.{name} needs a Matplotlib getter and setter."
                        )
                    before = getter()
                    if name in ("facecolor", "edgecolor"):
                        # Patches and collections keep the paint separately from applied alpha.
                        before = getattr(artist, f"_original_{name}", before)
                    if isinstance(before, (np.ndarray, list, tuple, dict)):
                        before = deepcopy(before)
                    updates[(id(element), f"artist/{id(artist)}/{name}")] = [
                        _edit(setter, before, value)
                    ]

        self._updates.update(updates)

    def bind_update(
        self, element: object, name: str, prepare: Callable[[Any], _ArtistEdit]
    ) -> None:
        self.mark_present(element)
        self._properties.setdefault((id(element), name), []).append(prepare)

    def bind_property(
        self,
        element: object,
        name: str,
        artist: Artist,
        mpl_name: str,
        convert: Callable[[Any], Any] | None = None,
    ) -> None:
        getter = getattr(artist, f"get_{mpl_name}")
        setter = getattr(artist, f"set_{mpl_name}")

        def prepare(value: Any) -> _ArtistEdit:
            return _edit(setter, getter(), convert(value) if convert else value)

        self.bind_update(element, name, prepare)

    def queue_property(
        self, element: object, name: str, value: Any, restore: Callable[[], None]
    ) -> None:
        prepares = self._properties.get((id(element), name), ())
        if not prepares and (id(element), name) not in self._dependents:
            raise UnsupportedFeatureError(
                f"{type(element).__name__}.{name.removeprefix('_')} cannot change in this animation."
            )
        if is_inherit(value):
            raise UnsupportedFeatureError(
                "Use an explicit style value during animation."
            )
        self._stage(element, name, [prepare(value) for prepare in prepares], restore)

    def bind_axis_text(
        self,
        element: object,
        name: str,
        read: Callable[[], str | None],
        write: Callable[[str], Any],
    ) -> None:
        def prepare(value: str | None) -> _ArtistEdit:
            if value is not None and not isinstance(value, str):
                raise InvalidParameterError(
                    "Animated axis text must be a string or None."
                )
            return _edit(write, read() or "", value or "")

        self.bind_update(element, name, prepare)

    def bind_resolved_axis_text(
        self,
        sources: list[tuple[object, str]],
        resolve: Callable[[], str | None],
        read: Callable[[], str | None],
        write: Callable[[str], Any],
    ) -> None:
        def prepare() -> _ArtistEdit:
            value = resolve()
            if value is not None and not isinstance(value, str):
                raise InvalidParameterError(
                    "Animated axis text must be a string or None."
                )
            return _edit(write, read() or "", value or "")

        for element, name in sources:
            self.bind_derived(element, (name,), prepare)

    def bind_axis_visibility(
        self,
        element: object,
        name: str,
        targets: dict[int, Callable[[bool], _ArtistEdit]],
        subplot_count: int,
    ) -> None:
        if not targets:
            return

        def prepare(value: bool | list[bool | None]) -> _ArtistEdit:
            entries = _subplot_visibility_entries(value, targets, subplot_count)
            return _combine_edits(
                [targets[index](flag) for index, flag in entries.items()]
            )

        self.bind_update(element, name, prepare)

    def bind_tick_style(
        self,
        element: object,
        dimension: str,
        targets: Mapping[int, _TickStyle],
        subplot_count: int,
        read_config: Callable[[], dict[str, dict[str, Any]]],
        read_visibility: Callable[[], bool | list[bool | None]],
    ) -> None:
        if not targets:
            return
        prefix = f"{dimension} " if dimension else ""
        visibility = f"_remove_{dimension}_ticks" if dimension else "_remove_ticks"

        def prepare() -> _ArtistEdit:
            entries = {}
            if self.has_frame_update(element, visibility):
                entries = _subplot_visibility_entries(
                    read_visibility(), targets, subplot_count
                )
            config = read_config()
            groups = [config[prefix + group] for group in _NativeTickStyle.groups]
            return _combine_edits(
                [
                    target.prepare(groups, entries.get(index, target.hidden))
                    for index, target in targets.items()
                ]
            )

        self.bind_derived(element, ("_tick_params", visibility), prepare)

    def bind_axis_ticks(
        self,
        element: object,
        targets: list[tuple[Axis, str]],
        read_config: Callable[[], dict[str, Any]],
    ) -> None:
        if not targets:
            return
        # Shared axes use the same Ticker objects. Update each group once.
        unique = {id(axis.major): (axis, dimension) for axis, dimension in targets}
        current = read_config()

        def prepare() -> _ArtistEdit:
            previous = current
            config = read_config()
            snapshots: list[tuple[Axis, list[tuple[str, str, Any, bool]]]] = []

            def apply() -> None:
                nonlocal current
                for axis, dimension in unique.values():
                    major, minor = _axis_tick_keys(dimension)
                    update_major = any(
                        config.get(key) is not previous.get(key) for key in major
                    )
                    update_minor = any(
                        config.get(key) is not previous.get(key) for key in minor
                    )
                    if not (update_major or update_minor):
                        continue
                    saved = _save_axis_ticks(axis)
                    snapshots.append((axis, saved))
                    # Obtain defaults from the scale in effect after the viewport edit.
                    cast(Any, axis)._scale.set_default_locators_and_formatters(axis)
                    for _, flag in _AXIS_TICK_FIELDS:
                        setattr(axis, flag, True)
                    if update_major:
                        configure(axis, major, minor=False)
                    if update_minor:
                        configure(axis, minor, minor=True)
                    if not update_major or not update_minor:
                        untouched = "minor_" if update_major else "major_"
                        _restore_axis_ticks(
                            axis,
                            [
                                entry
                                for entry in saved
                                if entry[0].startswith(untouched)
                            ],
                        )
                current = config

            def configure(axis: Axis, keys: tuple[str, ...], *, minor: bool) -> None:
                positions, spacing = (config.get(key) for key in keys[:2])
                set_locator = (
                    axis.set_minor_locator if minor else axis.set_major_locator
                )
                if positions is not None:
                    pos = np.asarray(positions, dtype=float)
                    if pos.ndim != 1 or not np.isfinite(pos).all():
                        raise InvalidParameterError(
                            "Animated ticks must be one-dimensional finite numbers."
                        )
                    set_locator(FixedLocator(pos.tolist()))
                elif spacing is not None:
                    if (
                        not isinstance(spacing, Real)
                        or not isfinite(spacing)
                        or spacing <= 0
                    ):
                        raise InvalidParameterError(
                            "Animated tick spacing must be finite and positive."
                        )
                    set_locator(MultipleLocator(spacing))
                if not minor:
                    labels = config.get(keys[2])
                    if callable(labels):
                        if positions is not None:
                            axis.set_ticklabels([labels(value) for value in positions])
                        else:
                            axis.set_major_formatter(
                                FuncFormatter(lambda value, _: labels(value))
                            )
                    elif labels is not None:
                        axis.set_ticklabels(labels)

            def undo() -> None:
                nonlocal current
                for axis, saved in reversed(snapshots):
                    _restore_axis_ticks(axis, saved)
                current = previous

            return _ArtistEdit(apply, undo)

        self._post_artist_prepares.add(id(prepare))
        self.bind_derived(element, ("_ticks",), prepare)

    def bind_axis_view(
        self,
        element: object,
        name: str,
        inversion_name: str,
        scale_name: str,
        dimension: str,
        targets: list[tuple[int, Axes, bool]],
        subplot_count: int,
    ) -> None:
        """Bind limits, direction, and scale to the same shared viewport."""
        if not targets:
            return
        initial: dict[frozenset[Axes], tuple[float, float]] = {}
        directions: dict[frozenset[Axes], bool | None] = {}

        def read(axes: Axes) -> tuple[float, float]:
            return axes.get_xlim() if dimension == "x" else axes.get_ylim()

        def write(axes: Axes, limits: tuple[float, float], *, emit: bool) -> None:
            if dimension == "x":
                axes.set_xlim(*limits, emit=emit, auto=False)
            else:
                axes.set_ylim(*limits, emit=emit, auto=False)

        def mpl_axis(axes: Axes) -> Axis:
            return axes.xaxis if dimension == "x" else axes.yaxis

        def write_scale(axes: Axes, scale: Any) -> None:
            if dimension == "x":
                axes.set_xscale(scale)
            else:
                axes.set_yscale(scale)

        def prepare_scale_edit(
            axes: Axes, siblings: frozenset[Axes], scale: str
        ) -> _ArtistEdit:
            before_scale = cast(Any, mpl_axis(axes))._scale
            ticks = [(axis, _save_axis_ticks(axis)) for axis in map(mpl_axis, siblings)]

            def set_state(new_scale, *, custom_only: bool) -> None:
                write_scale(axes, new_scale)
                for axis, state in ticks:
                    _restore_axis_ticks(axis, state, custom_only=custom_only)

            return _ArtistEdit(
                lambda: set_state(scale, custom_only=True),
                lambda: set_state(before_scale, custom_only=False),
            )

        def read_entries(property_name: str, reset_all: bool) -> dict[int, Any]:
            if not self.has_frame_update(element, property_name):
                return {}
            value = getattr(element, property_name)
            if not isinstance(value, list):
                if value is None and not reset_all:
                    return {}
                return {index: value for index, _, _ in targets}
            if len(value) > subplot_count:
                raise InvalidParameterError(
                    f"Too many {dimension}-axis entries for this SmartFigure."
                )
            available = {index for index, _, _ in targets}
            result = {
                index: item for index, item in enumerate(value) if item is not None
            }
            if result.keys() - available:
                raise InvalidParameterError(
                    "Set axis properties on a nested SmartFigure directly."
                )
            return result

        def orient(
            bounds: tuple[float, float], direction: bool | None
        ) -> tuple[float, float]:
            if direction is not None and (bounds[0] > bounds[1]) != direction:
                return bounds[::-1]
            return bounds

        def prepare() -> _ArtistEdit:
            grouped: dict[frozenset[Axes], list[tuple[int, Axes, bool]]] = {}
            for index, axes, initially_inverted in targets:
                shared = (
                    axes.get_shared_x_axes()
                    if dimension == "x"
                    else axes.get_shared_y_axes()
                )
                siblings = frozenset(shared.get_siblings(axes))
                grouped.setdefault(siblings, []).append(
                    (index, axes, initially_inverted)
                )
            limits = read_entries(name, reset_all=True)
            inversions = read_entries(inversion_name, reset_all=False)
            scales = read_entries(scale_name, reset_all=False)
            changes: list[
                tuple[frozenset[Axes], Axes, tuple[float, float], bool | None]
            ] = []
            old_limits: dict[Axes, tuple[float, float]] = {}
            old_directions: dict[frozenset[Axes], bool | None] = {}
            scale_edits: dict[Axes, _ArtistEdit] = {}

            for siblings, members in grouped.items():
                axes = members[0][1]
                previous = read(axes)
                initial.setdefault(siblings, previous)
                before = directions.setdefault(
                    siblings, True if any(flag for _, _, flag in members) else None
                )
                requested = [
                    limits[index] for index, _, _ in members if index in limits
                ]
                flags = {
                    inversions[index] for index, _, _ in members if index in inversions
                }
                if len(flags) > 1:
                    raise InvalidParameterError(
                        "Shared axes must receive matching inversion settings within a frame."
                    )
                scale_flags = {
                    scales[index] for index, _, _ in members if index in scales
                }
                if len(scale_flags) > 1:
                    raise InvalidParameterError(
                        "Shared axes must receive matching scale settings within a frame."
                    )
                if not requested and not flags and not scale_flags:
                    continue
                direction = next(iter(flags), before)
                current_scale = mpl_axis(axes).get_scale()
                scale = (
                    ("log" if next(iter(scale_flags)) else "linear")
                    if scale_flags
                    else current_scale
                )

                # Partial limits use logical endpoint order for an explicitly
                # oriented axis, independent of callback assignment order.
                force_order = bool(flags) or direction is True
                base = tuple(sorted(previous)) if force_order else previous
                options = []
                for item in requested:
                    if item is None:
                        options.append(orient(initial[siblings], direction))
                        continue
                    low, high = (
                        prior if bound is None else bound
                        for bound, prior in zip(item, base, strict=True)
                    )
                    if (
                        not all(
                            isinstance(bound, Real) and isfinite(bound)
                            for bound in (low, high)
                        )
                        or low == high
                    ):
                        raise InvalidParameterError(
                            "Animated axis limits must be distinct finite numbers."
                        )
                    bounds = float(low), float(high)
                    options.append(orient(bounds, direction) if force_order else bounds)
                bounds = options[0] if options else orient(previous, direction)
                if any(option != bounds for option in options):
                    raise InvalidParameterError(
                        "Shared axes must receive matching limits within a frame."
                    )
                if scale == "log" and min(bounds) <= 0:
                    raise InvalidParameterError(
                        "Animated logarithmic axis limits must be positive."
                    )
                if scale != current_scale:
                    if current_scale not in ("linear", "log"):
                        raise UnsupportedFeatureError(
                            "Only linear and logarithmic axes can switch scales during animation."
                        )
                    scale_edits[axes] = prepare_scale_edit(axes, siblings, scale)
                changes.append((siblings, axes, bounds, direction))
                old_limits.update((sibling, read(sibling)) for sibling in siblings)
                old_directions[siblings] = before

            applied_scales = []

            def apply() -> None:
                for siblings, axes, bounds, direction in changes:
                    if axes in scale_edits:
                        edit = scale_edits[axes]
                        applied_scales.append(edit)
                        edit.apply()
                    write(axes, bounds, emit=True)
                    directions[siblings] = direction

            def undo() -> None:
                for edit in reversed(applied_scales):
                    edit.undo()
                for axes, bounds in old_limits.items():
                    write(axes, bounds, emit=False)
                directions.update(old_directions)

            return _ArtistEdit(apply, undo)

        self.bind_derived(element, (name, inversion_name, scale_name), prepare)

    def bind_derived(
        self,
        element: object,
        properties: tuple[str, ...],
        prepare: Callable[[], _ArtistEdit],
    ) -> None:
        self.mark_present(element)
        for name in properties:
            # One artist can have independent data, color, and label updates.
            self._dependents.setdefault((id(element), name), {})[id(prepare)] = prepare

    def link_point_source(
        self, owner: object, property_name: str, old: object | None, new: object | None
    ) -> None:
        if old is new:
            return
        owner_id = id(owner)
        if old is not None:
            links = self._point_links.get(id(old))
            if links is not None:
                links.pop(owner_id, None)
                if not links:
                    del self._point_links[id(old)]
        if new is not None:
            prepares = self._dependents[(owner_id, property_name)]
            self._point_links.setdefault(id(new), {})[owner_id] = prepares

    def can_reserve_legend_entry(self, handle: object) -> bool:
        return self._reserve_legend_entries and id(handle) in self._legend_sources

    def _reserved_legend_filters(self, handle: object) -> tuple[_LegendFilter, ...]:
        if not self.can_reserve_legend_entry(handle):
            return ()
        element, read_label, properties = self._legend_sources[id(handle)]
        if read_label() is not None:
            return ()

        def is_hidden() -> bool:
            return not read_label()

        return tuple(_LegendFilter(element, name, is_hidden) for name in properties)

    def bind_legend_source(
        self,
        element: object,
        handle: Artist | ErrorbarContainer,
        read_label: Callable[[], str | None],
        properties: tuple[str, ...] = ("_label",),
    ) -> None:
        self._legend_sources[id(handle)] = (element, read_label, properties)

        def prepare() -> _ArtistEdit:
            label = read_label()
            if label is not None and not isinstance(label, str):
                raise InvalidParameterError(
                    "Animated legend labels must be strings or None."
                )
            return _edit(handle.set_label, handle.get_label(), label)

        self.bind_derived(element, properties, prepare)

    def bind_reference_legend(
        self,
        handle: LineCollection,
        properties: tuple[str, ...],
        read_styles: Callable[[], list[dict[str, Any]]],
    ) -> None:
        self._reference_legend_sources[id(handle)] = (properties, read_styles)

    def bind_legend_visibility(
        self,
        element: object,
        targets: dict[int, Legend | None],
        subplot_count: int,
    ) -> None:
        def prepare(value: bool | list[bool | None]) -> _ArtistEdit:
            entries = _subplot_visibility_entries(value, targets, subplot_count)
            edits = []
            for index, show in entries.items():
                legend = targets[index]
                if legend is not None:
                    rows = self._legend_rows[legend]
                    edits.append(_edit(rows.show, rows.shown, show))
            return _combine_edits(edits)

        self.bind_update(element, "_show_legend", prepare)

    def apply_initial_visibility(self) -> None:
        for artist in self._initially_hidden_artists:
            artist.set_visible(False)
        self._initially_hidden_artists.clear()
        for rows in self._legend_rows.values():
            rows.refresh()

    def bind_colorbar(
        self,
        element: _ColorbarOwner,
        bar: Colorbar,
        *,
        initially_visible: bool,
        default_locator: Callable[[], Locator] | None = None,
    ) -> None:
        self.bind_property(element, "_show_color_bar", bar.ax, "visible")
        if not initially_visible:
            self._initially_hidden_artists.append(bar.ax)
        label = (
            bar.ax.yaxis.label if bar.orientation == "vertical" else bar.ax.xaxis.label
        )
        self.bind_property(element, "_color_bar_label", label, "text")
        initial_locator = bar.locator
        self._initial_colorbar_ticks.setdefault(
            id(element), element._color_bar_params.get("ticks")
        )

        def prepare_ticks(ticks: list[float] | None) -> _ArtistEdit:
            previous = bar.locator

            def apply() -> None:
                if ticks is not None:
                    bar.locator = _AnimatedColorbarLocator(ticks)
                else:
                    bar.locator = (
                        default_locator() if default_locator else initial_locator
                    )

            def undo() -> None:
                bar.locator = previous

            return _ArtistEdit(apply, undo)

        self.bind_update(element, "_color_bar_ticks", prepare_ticks)

    def queue_colorbar_params(
        self,
        element: _ColorbarOwner,
        label: str | None,
        position: str | None,
        reserve: bool,
        params: dict[str, Any],
    ) -> None:
        if position is not None or reserve or params.keys() - {"ticks"}:
            raise UnsupportedFeatureError(
                "During animation set_color_bar_params supports label and ticks."
            )
        if label is not None and not isinstance(label, str):
            raise InvalidParameterError("The colorbar label must be a string.")
        if label is None and "ticks" not in params:
            return
        values = params.get("ticks")
        ticks = None
        if values is not None:
            try:
                if np.iscomplexobj(values):
                    raise ValueError("Complex ticks")
                array = np.asarray(values, dtype=float)
            except (TypeError, ValueError) as exc:
                raise InvalidParameterError(
                    "Colorbar ticks must be finite numbers."
                ) from exc
            if array.ndim != 1 or not np.isfinite(array).all():
                raise InvalidParameterError(
                    "Colorbar ticks must be a one-dimensional sequence of finite numbers."
                )
            ticks = array.tolist()
        previous = element._color_bar_params
        updated = previous.copy()
        if label is not None:
            self.queue_property(element, "_color_bar_label", label, lambda: None)
            updated["label"] = label
        if "ticks" in params:
            self.queue_property(element, "_color_bar_ticks", ticks, lambda: None)
            if ticks is None:
                initial = self._initial_colorbar_ticks[id(element)]
                if initial is None:
                    updated.pop("ticks", None)
                else:
                    updated["ticks"] = initial
            else:
                updated["ticks"] = ticks

        def restore() -> None:
            element._color_bar_params = previous

        self.remember(element, "_color_bar_params", restore)
        element._color_bar_params = updated

    def bind_legend(
        self,
        legend: Legend,
        handles: list[Any],
        filters: list[tuple[_LegendFilter, ...]],
        *,
        initially_visible: bool = True,
    ) -> None:
        filters = [
            (*rules, *self._reserved_legend_filters(handle))
            for rules, handle in zip(filters, handles, strict=True)
        ]
        rows = _LegendRows(legend, initially_visible, filters)
        self._legend_rows[legend] = rows
        prepare = rows.prepare
        for rules in filters:
            for rule in rules:
                self.bind_derived(rule.owner, (rule.property_name,), prepare)
        texts = iter(legend.get_texts())
        groups = {
            id(children[0]): children
            for area in legend.findobj(match=DrawingArea)
            if (children := area.get_children())
        }
        for handle, swatch in zip(handles, legend.legend_handles, strict=True):
            if swatch is None:
                # Matplotlib keeps a null handle for unsupported entries and skips their text.
                continue
            text = next(texts)
            source = self._legend_sources.get(id(handle))
            if source is None:
                continue
            element, read_label, properties = source

            def prepare(text=text, read_label=read_label) -> _ArtistEdit:
                return _edit(text.set_text, text.get_text(), read_label() or "")

            self.bind_derived(element, properties, prepare)
            swatch_group = groups.get(id(swatch), [])
            if reference := self._reference_legend_sources.get(id(handle)):
                self._bind_reference_swatches(element, swatch_group, *reference)
            elif isinstance(handle, ErrorbarContainer) and (
                handle.has_xerr or handle.has_yerr
            ):
                self._bind_errorbar_swatch(element, handle, swatch_group)
            else:
                self._bind_legend_swatch(element, handle, swatch, legend.markerscale)

    def _bind_reference_swatches(
        self,
        element: object,
        group: list[Artist],
        properties: tuple[str, ...],
        read_styles: Callable[[], list[dict[str, Any]]],
    ) -> None:
        swatches = [
            swatch for swatch in group if isinstance(swatch, _ReferenceLegendSwatch)
        ]
        if not swatches:
            return

        def prepare() -> _ArtistEdit:
            before = [
                (swatch.get_data(), swatch.get_visible(), _reference_line_style(swatch))
                for swatch in swatches
            ]

            def apply() -> None:
                styles = read_styles()[: len(swatches)]
                for index, swatch in enumerate(swatches):
                    swatch.update_sample(index, styles)

            def undo() -> None:
                for swatch, (data, visible, style) in zip(
                    swatches, before, strict=True
                ):
                    swatch.set_data(*data)
                    swatch.set(visible=visible, **style)

            return _ArtistEdit(apply, undo)

        self._post_artist_prepares.add(id(prepare))
        self.bind_derived(element, properties, prepare)

    def _bind_errorbar_swatch(
        self, element: object, source: ErrorbarContainer, swatches: list[Artist]
    ) -> None:
        line, caps, segments = source.lines
        originals = [*segments, *caps, line, line]
        # HandlerErrorbar returns segments, caps, the line, and its marker.
        if len(originals) != len(swatches) or any(
            target is None or not isinstance(swatch, type(target))
            for target, swatch in zip(originals, swatches)
        ):
            return
        properties = (
            *_ERROR_STYLE_PROPERTIES,
            "_show_errorbars",
            "point_data",
            "_color",
            "_line_width",
            "_line_style",
            "_alpha",
        )
        for index, (target, swatch) in enumerate(zip(originals, swatches, strict=True)):
            if index < len(segments):
                fields = ("alpha", "color", "linewidth", "visible")
            elif index < len(segments) + len(caps):
                fields = (
                    "alpha",
                    "markeredgecolor",
                    "markeredgewidth",
                    "markersize",
                    "visible",
                )
            elif index == len(originals) - 2:
                fields = ("alpha", "color", "linewidth", "linestyle")
            else:
                fields = ("alpha", "color")
            if target is not None:
                self._bind_swatch_fields(element, target, swatch, properties, fields)

    def _bind_legend_swatch(
        self, element: object, source: object, swatch: Artist, marker_scale: float
    ) -> None:
        if isinstance(source, ErrorbarContainer) and not (
            source.has_xerr or source.has_yerr
        ):
            source = source.lines[0]

        if isinstance(source, Line2D) and isinstance(swatch, Line2D):
            properties = ("_color", "_line_style", "_line_width", "_alpha")
            fields = ("alpha", "color", "linestyle", "linewidth")
        elif isinstance(source, PathCollection) and isinstance(swatch, PathCollection):
            properties = (
                "point_data",
                "_alpha",
                "_face_color",
                "_edge_color",
                "_color_map",
                "_color_map_range",
                "_marker_style",
                "_marker_size",
                "_marker_edge_width",
            )
            fields = ("alpha", "paths", "facecolor", "edgecolor", "linewidth", "sizes")
        elif isinstance(source, Polygon) and isinstance(swatch, Polygon):
            properties = ("_face_color", "_edge_color", "_alpha", "_line_width")
            fields = ("facecolor", "edgecolor", "linewidth")
        else:
            return

        self._bind_swatch_fields(
            element, source, swatch, properties, fields, marker_scale
        )

    def _bind_swatch_fields(
        self,
        element: object,
        source: Artist,
        swatch: Artist,
        properties: tuple[str, ...],
        fields: tuple[str, ...],
        marker_scale: float = 1,
    ) -> None:
        supported = tuple(
            prop
            for prop in properties
            if (id(element), prop) in self._properties
            or (id(element), prop) in self._dependents
        )
        if not supported:
            return

        def prepare() -> _ArtistEdit:
            setters = [getattr(swatch, f"set_{field}") for field in fields]
            original = [getattr(swatch, f"get_{field}")() for field in fields]

            def apply() -> None:
                for field, set_value in zip(fields, setters):
                    value = getattr(source, f"get_{field}")()
                    if field == "sizes":
                        value = np.asarray(value) * marker_scale**2
                    set_value(value)

            def undo() -> None:
                for set_value, value in zip(setters, original):
                    set_value(value)

            return _ArtistEdit(apply, undo)

        self._post_artist_prepares.add(id(prepare))
        self.bind_derived(element, supported, prepare)

    def block_curve_updates(
        self,
        curve: object,
        reason: str,
    ) -> None:
        self._artists.setdefault(id(curve), []).append(reason)

    def has_binding(self, element: object) -> bool:
        return id(element) in self._artists or id(element) in self._point_links

    def reject_structural_update(self, element: object, message: str) -> None:
        if self.has_binding(element):
            raise UnsupportedFeatureError(message)

    def bind_curve(
        self,
        curve: object,
        handle: ErrorbarContainer | Line2D,
        *,
        bands: _CurveBands | None = None,
        reserved_errors: ErrorbarContainer | None = None,
    ) -> None:
        data_line = handle if isinstance(handle, Line2D) else handle.lines[0]
        if not isinstance(data_line, Line2D):
            self.block_curve_updates(curve, "This curve has no line artist to update.")
            return
        self._bind_points(
            curve,
            data_line,
            handle if isinstance(handle, ErrorbarContainer) else None,
            bands=bands,
            reserved_errors=reserved_errors,
        )

    def bind_scatter(
        self,
        scatter: object,
        points: PathCollection,
        errors: ErrorbarContainer | None,
        reserved_errors: ErrorbarContainer | None = None,
    ) -> None:
        self._bind_points(scatter, points, errors, reserved_errors=reserved_errors)

    def _bind_points(
        self,
        element: object,
        points: Line2D | PathCollection,
        handle: ErrorbarContainer | None,
        *,
        bands: _CurveBands | None = None,
        reserved_errors: ErrorbarContainer | None = None,
    ) -> None:
        by_direction = {}
        if handle is not None:
            _, caps, segments = handle.lines
            directions = [
                is_x
                for is_x, present in ((True, handle.has_xerr), (False, handle.has_yerr))
                if present
            ]
            if len(segments) != len(directions) or len(caps) not in (
                0,
                2 * len(directions),
            ):
                self._artists.setdefault(id(element), []).append(None)
                return
            for index, (is_x, collection) in enumerate(
                zip(directions, segments, strict=True)
            ):
                by_direction[is_x] = _ErrorArtists(
                    collection, caps[2 * index : 2 * index + 2]
                )
        self._artists.setdefault(id(element), []).append(
            _PointArtists(
                points,
                by_direction.get(True),
                by_direction.get(False),
                bands,
                reserved_errors is not None
                or (
                    points.axes is not None
                    and points.axes.name != "polar"
                    and len(by_direction) == 2
                ),
            )
        )

    def bind_text(self, text: object, artist: TextArtist) -> None:
        self._artists.setdefault(id(text), []).append(artist)

    def queue_points(
        self,
        element: object,
        x: np.ndarray,
        y: np.ndarray,
        dx: np.ndarray | None,
        dy: np.ndarray | None,
        *,
        show_errorbars: bool,
        restore: Callable[[], None],
    ) -> None:
        bindings = self._artists.get(id(element), ())
        if x.dtype.kind not in "biuf" or y.dtype.kind not in "biuf":
            raise InvalidParameterError(
                "Animated plot coordinates must be real numeric values."
            )
        updates: list[_ArtistEdit] = []
        for binding in bindings:
            if not isinstance(binding, _PointArtists):
                raise UnsupportedFeatureError(
                    binding
                    if isinstance(binding, str)
                    else "This plot has an unsupported errorbar layout."
                )
            configured = (
                show_errorbars
                or binding.x_error is not None
                or binding.y_error is not None
            )
            horizontal = dx if configured else None
            vertical = dy if configured else None
            if not binding.flexible_errors and (
                (binding.x_error is None) != (horizontal is None)
                or (binding.y_error is None) != (vertical is None)
            ):
                raise UnsupportedFeatureError(
                    "An animation cannot add or remove errorbar directions."
                )
            if isinstance(binding.points, Line2D):
                line = binding.points
                updates.append(_edit(line.set_data, line.get_data(), (x, y)))
            else:
                offsets = np.column_stack((x, y))
                updates.append(
                    _edit(
                        binding.points.set_offsets,
                        binding.points.get_offsets(),
                        offsets,
                    )
                )
            if binding.bands is not None:
                updates.extend(binding.bands.edits(x, y, dy))
            if binding.x_error is not None and horizontal is not None:
                updates.append(
                    _error_update(binding.x_error, x, y, horizontal, is_x=True)
                )
            if binding.y_error is not None and vertical is not None:
                updates.append(
                    _error_update(binding.y_error, y, x, vertical, is_x=False)
                )
        self._stage(element, "point_data", updates, restore)

    def queue_text(
        self,
        text: object,
        x: float,
        y: float,
        content: str,
        restore: Callable[[], None],
    ) -> None:
        updates: list[_ArtistEdit] = []
        for artist in self._artists.get(id(text), ()):
            if isinstance(artist, TextArtist):
                updates.extend(
                    (
                        _edit(artist.set_position, artist.get_position(), (x, y)),
                        _edit(artist.set_text, artist.get_text(), content),
                    )
                )
        self._stage(text, "text_data", updates, restore)

    def _stage(
        self,
        element: object,
        name: str,
        updates: list[_ArtistEdit],
        restore: Callable[[], None],
    ) -> None:
        key = id(element), name
        if updates:
            self._updates[key] = updates
        dependents = self._dependents.get(key)
        if dependents:
            self._dirty_dependents.update(dependents)
        links = self._point_links.get(id(element)) if name == "point_data" else None
        if links:
            for prepares in links.values():
                self._dirty_dependents.update(prepares)
        if updates or dependents or links:
            self.remember(element, name, restore)

    def draw_frame(self, update: Callable[[int], object], index: int) -> None:
        token = _FRAME_SESSION.set(self)
        applied = []
        try:
            update(index)
            # Derive dependent artist edits from the final frame inputs.
            dependent_edits, post_artist_edits = [], []
            for key, prepare in self._dirty_dependents.items():
                edits = (
                    post_artist_edits
                    if key in self._post_artist_prepares
                    else dependent_edits
                )
                edits.append(prepare())
            for updates in chain(
                self._updates.values(), (dependent_edits, post_artist_edits)
            ):
                for edit in updates:
                    applied.append(edit)
                    try:
                        edit.apply()
                    except GraphingLibError:
                        raise
                    except (TypeError, ValueError) as exc:
                        raise PlottingError(
                            f"Could not apply animation frame {index} ({exc})."
                        ) from exc
        except BaseException as exc:
            self.failure = exc
            for edit in reversed(applied):
                try:
                    edit.undo()
                except Exception as rollback_error:  # noqa: BLE001 - Continue restoring other artists.
                    exc.add_note(
                        f"Could not restore an animation artist: {rollback_error}"
                    )
            for restore in reversed(self._rollback.values()):
                restore()
            raise
        finally:
            self._updates.clear()
            self._dirty_dependents.clear()
            self._rollback.clear()
            _FRAME_SESSION.reset(token)

    def clear(self) -> None:
        self._artists.clear()
        self._axes_method_artists.clear()
        self._properties.clear()
        self._dependents.clear()
        self._point_links.clear()
        self._legend_sources.clear()
        self._initially_hidden_artists.clear()
        self._legend_rows.clear()
        self._initial_colorbar_ticks.clear()
        self._reference_legend_sources.clear()
        self._post_artist_prepares.clear()
        self._dirty_dependents.clear()
        for callback in reversed(self._cleanup):
            callback()
        self._cleanup.clear()
        self._updates.clear()
        self._rollback.clear()
        self.failure = None


_DRAW_SESSION: ContextVar[_AnimationSession | None] = ContextVar(
    "animation_draw", default=None
)
_FRAME_SESSION: ContextVar[_AnimationSession | None] = ContextVar(
    "animation_frame", default=None
)


def _bound_session(element: object) -> _AnimationSession | None:
    session = _FRAME_SESSION.get()
    return session if session is not None and session.has_binding(element) else None


def _stage_live_value[T](element: object, name: str, value: T) -> T:
    session = _bound_session(element)
    if session is not None:
        previous = getattr(element, name, None)
        session.queue_property(
            element, name, value, lambda: setattr(element, name, previous)
        )
    return value


def _reject_live_update(element: object, message: str) -> None:
    if _bound_session(element) is not None:
        raise UnsupportedFeatureError(message)
