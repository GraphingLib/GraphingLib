"""Artist bindings for one SmartFigure animation."""

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from itertools import chain
from math import isfinite
from numbers import Real
from typing import Any

import numpy as np
from matplotlib import rcParams
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import (
    LineCollection,
    PathCollection,
    PolyCollection,
)
from matplotlib.container import ErrorbarContainer
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.offsetbox import DrawingArea
from matplotlib.patches import Polygon
from matplotlib.text import Text as TextArtist
from numpy.typing import ArrayLike

from .exceptions import (
    GraphingLibError,
    InvalidParameterError,
    PlottingError,
    UnsupportedFeatureError,
)
from .inherit import is_inherit


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


_ERROR_STYLE_PROPERTIES = (
    "_errorbars_color",
    "_errorbars_line_width",
    "_cap_width",
    "_cap_thickness",
)


def _draw_live_errorbars(
    axes: Axes, x: Any, y: Any, **params: Any
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
    handle = axes.errorbar(x, y, **params)
    if hide_caps:
        for cap in handle.lines[1]:
            cap.set_markersize(0)
            cap.set_visible(False)
    return handle


def _error_style_edits(
    handle: ErrorbarContainer,
    *,
    color: Any,
    line_width: float | None,
    cap_width: float | None,
    cap_thickness: float | None,
    alpha: float | None,
) -> list[_ArtistEdit]:
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
    if cap_width and (not caps or any(cap.axes is None for cap in caps)):
        raise UnsupportedFeatureError(
            "This plot has no live errorbar cap artists. Use a nonempty initial "
            "dataset on a Cartesian plot to animate cap width."
        )

    edits = []
    for segment in segments:
        if color is not None:
            edits.append(_edit(segment.set_color, segment.get_color(), color))
        if line_width is not None:
            edits.append(
                _edit(segment.set_linewidth, segment.get_linewidth(), line_width)
            )
        edits.append(_edit(segment.set_alpha, segment.get_alpha(), alpha))
    for cap in caps:
        if color is not None:
            edits.append(
                _edit(cap.set_markeredgecolor, cap.get_markeredgecolor(), color)
            )
        if cap_thickness is not None:
            edits.append(
                _edit(cap.set_markeredgewidth, cap.get_markeredgewidth(), cap_thickness)
            )
        if cap_width is not None:
            edits.extend(
                (
                    _edit(cap.set_markersize, cap.get_markersize(), 2 * cap_width),
                    _edit(cap.set_visible, cap.get_visible(), cap_width > 0),
                )
            )
        edits.append(_edit(cap.set_alpha, cap.get_alpha(), alpha))
    return edits


@dataclass
class _ErrorArtists:
    segments: LineCollection
    caps: tuple[Line2D, ...]


@dataclass
class _PointArtists:
    points: Line2D | PathCollection
    x_error: _ErrorArtists | None
    y_error: _ErrorArtists | None
    bands: "_CurveBands | None" = None


@dataclass
class _CurveBands:
    lower: Line2D | None
    upper: Line2D | None
    error_fill: PolyCollection | None
    area_fill: PolyCollection | None
    area_bounds: tuple[float, float] | None

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
        if self.area_fill is not None and self.area_bounds is not None:
            within = (x >= self.area_bounds[0]) & (x <= self.area_bounds[1])
            edits.append(_fill_edit(self.area_fill, x, y, np.zeros_like(y), within))
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


class _AnimationSession:
    def __init__(self) -> None:
        self._artists: dict[int, list[_Binding]] = {}
        self._properties: dict[tuple[int, str], list[Callable[[Any], _ArtistEdit]]] = {}
        self._updates: dict[tuple[int, str], list[_ArtistEdit]] = {}
        self._dependents: dict[
            tuple[int, str], dict[int, Callable[[], _ArtistEdit]]
        ] = {}
        self._dirty_dependents: dict[int, Callable[[], _ArtistEdit]] = {}
        self._legend_swatch_prepares: set[int] = set()
        self._legend_sources: dict[
            int, tuple[object, Callable[[], str | None], tuple[str, ...]]
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

    def bind_axis_limits(
        self,
        element: object,
        name: str,
        dimension: str,
        targets: list[tuple[int, Axes, bool]],
        subplot_count: int,
    ) -> None:
        """Bind explicit limits to the existing axes drawn by this figure or twin."""
        if not targets:
            return
        initial: dict[Axes, tuple[float, float]] = {}

        def read(axes: Axes) -> tuple[float, float]:
            return axes.get_xlim() if dimension == "x" else axes.get_ylim()

        def write(axes: Axes, limits: tuple[float, float], *, emit: bool) -> None:
            if dimension == "x":
                axes.set_xlim(*limits, emit=emit, auto=False)
            else:
                axes.set_ylim(*limits, emit=emit, auto=False)

        def prepare(value: Any) -> _ArtistEdit:
            if isinstance(value, list):
                if len(value) > subplot_count:
                    raise InvalidParameterError(
                        f"Too many {dimension}-axis limit entries for this SmartFigure."
                    )
                available = {index for index, _, _ in targets}
                if any(
                    item is not None and index not in available
                    for index, item in enumerate(value)
                ):
                    raise InvalidParameterError(
                        "Set limits on a nested SmartFigure directly."
                    )

            groups: dict[frozenset[Axes], tuple[Axes, tuple[float, float]]] = {}
            old: dict[Axes, tuple[float, float]] = {}
            for index, axes, invert in targets:
                initial.setdefault(axes, read(axes))
                if isinstance(value, list):
                    if index >= len(value) or value[index] is None:
                        continue
                    requested = value[index]
                else:
                    requested = value
                if requested is None:
                    bounds = initial[axes]
                else:
                    previous = read(axes)
                    if invert:
                        previous = previous[::-1]
                    effective = tuple(
                        prior if bound is None else bound
                        for bound, prior in zip(requested, previous)
                    )
                    low, high = effective
                    if (
                        not all(
                            isinstance(bound, Real) and isfinite(bound)
                            for bound in effective
                        )
                        or low == high
                    ):
                        raise InvalidParameterError(
                            "Animated axis limits must be distinct finite numbers."
                        )
                    scale = axes.get_xscale() if dimension == "x" else axes.get_yscale()
                    if scale == "log" and min(low, high) <= 0:
                        raise InvalidParameterError(
                            "Animated logarithmic axis limits must be positive."
                        )
                    bounds = (float(low), float(high))
                    if invert and bounds[0] < bounds[1]:
                        bounds = bounds[::-1]

                shared = (
                    axes.get_shared_x_axes()
                    if dimension == "x"
                    else axes.get_shared_y_axes()
                )
                siblings = frozenset(shared.get_siblings(axes))
                if siblings in groups and groups[siblings][1] != bounds:
                    raise InvalidParameterError(
                        "Shared axes must receive matching limits within a frame."
                    )
                groups[siblings] = axes, bounds
                for sibling in siblings:
                    old.setdefault(sibling, read(sibling))

            def apply() -> None:
                for axes, bounds in groups.values():
                    write(axes, bounds, emit=True)

            def undo() -> None:
                for axes, bounds in old.items():
                    write(axes, bounds, emit=False)

            return _ArtistEdit(apply, undo)

        self.bind_update(element, name, prepare)

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

    def bind_legend(self, legend: Legend, handles: list[Any]) -> None:
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
            if isinstance(handle, ErrorbarContainer) and (
                handle.has_xerr or handle.has_yerr
            ):
                self._bind_errorbar_swatch(element, handle, groups.get(id(swatch), []))
            else:
                self._bind_legend_swatch(element, handle, swatch, legend.markerscale)

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
            "_color",
            "_line_width",
            "_line_style",
            "_alpha",
        )
        for index, (target, swatch) in enumerate(zip(originals, swatches, strict=True)):
            if index < len(segments):
                fields = ("alpha", "color", "linewidth")
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

        self._legend_swatch_prepares.add(id(prepare))
        self.bind_derived(element, supported, prepare)

    def bind_point_dependent(
        self,
        elements: tuple[object, object],
        artist: PolyCollection,
        coordinates: Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> None:
        def prepare() -> _ArtistEdit:
            return _fill_edit(artist, *coordinates())

        for element in elements:
            self.bind_derived(element, ("point_data",), prepare)

    def block_curve_updates(
        self,
        curve: object,
        reason: str,
    ) -> None:
        self._artists.setdefault(id(curve), []).append(reason)

    def has_binding(self, element: object) -> bool:
        return id(element) in self._artists

    def reject_structural_update(self, element: object, message: str) -> None:
        if self.has_binding(element):
            raise UnsupportedFeatureError(message)

    def bind_curve(
        self,
        curve: object,
        handle: ErrorbarContainer | Line2D,
        *,
        bands: _CurveBands,
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
        )

    def bind_scatter(
        self, scatter: object, points: PathCollection, errors: ErrorbarContainer | None
    ) -> None:
        self._bind_points(scatter, points, errors)

    def _bind_points(
        self,
        element: object,
        points: Line2D | PathCollection,
        handle: ErrorbarContainer | None,
        *,
        bands: _CurveBands | None = None,
    ) -> None:
        if handle is None:
            self._artists.setdefault(id(element), []).append(
                _PointArtists(points, None, None, bands)
            )
            return
        _, caps, segments = handle.lines
        directions = int(handle.has_xerr) + int(handle.has_yerr)
        if len(segments) != directions or len(caps) not in (0, 2 * directions):
            self._artists.setdefault(id(element), []).append(None)
            return
        # Matplotlib builds the horizontal error artists before the vertical ones.
        errors = [
            _ErrorArtists(collection, tuple(caps[2 * i : 2 * i + 2]))
            for i, collection in enumerate(segments)
        ]
        x_error = errors[0] if handle.has_xerr else None
        y_error = errors[-1] if handle.has_yerr else None
        self._artists.setdefault(id(element), []).append(
            _PointArtists(points, x_error, y_error, bands)
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
        horizontal = dx if show_errorbars else None
        vertical = dy if show_errorbars else None
        updates: list[_ArtistEdit] = []
        for binding in bindings:
            if not isinstance(binding, _PointArtists):
                raise UnsupportedFeatureError(
                    binding
                    if isinstance(binding, str)
                    else "This plot has an unsupported errorbar layout."
                )
            if (binding.x_error is None) != (horizontal is None) or (
                binding.y_error is None
            ) != (vertical is None):
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
                    self._error_update(binding.x_error, x, y, horizontal, is_x=True)
                )
            if binding.y_error is not None and vertical is not None:
                updates.append(
                    self._error_update(binding.y_error, y, x, vertical, is_x=False)
                )
        self._stage(element, "point_data", updates, restore)

    @staticmethod
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
        if updates or dependents:
            self.remember(element, name, restore)

    def draw_frame(self, update: Callable[[int], object], index: int) -> None:
        token = _FRAME_SESSION.set(self)
        applied = []
        try:
            update(index)
            # Derive dependent artist edits from the final frame inputs.
            dependent_edits, swatch_edits = [], []
            for key, prepare in self._dirty_dependents.items():
                edits = (
                    swatch_edits
                    if key in self._legend_swatch_prepares
                    else dependent_edits
                )
                edits.append(prepare())
            for updates in chain(
                self._updates.values(), (dependent_edits, swatch_edits)
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
        self._properties.clear()
        self._dependents.clear()
        self._legend_sources.clear()
        self._legend_swatch_prepares.clear()
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
