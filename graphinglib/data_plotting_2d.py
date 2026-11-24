from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite, nextafter
from numbers import Integral, Real
from typing import Any, Callable, Literal, Optional, Protocol, cast, runtime_checkable

import contourpy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PathCollection, QuadMesh
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap, Normalize, is_color_like, to_rgba
from matplotlib.contour import QuadContourSet
from matplotlib.figure import Figure
from matplotlib.image import AxesImage, imread
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath
from matplotlib.quiver import Quiver
from matplotlib.streamplot import StreamplotSet
from matplotlib.ticker import FixedLocator, MaxNLocator
from matplotlib.transforms import Bbox, nonsingular
from numpy.typing import ArrayLike
from scipy.interpolate import griddata

from ._animation import (
    _DRAW_SESSION,
    _AnimatedColorbarLocator,
    _AnimationSession,
    _ArtistEdit,
    _attach_artists,
    _bound_session,
    _combine_edits,
    _edit,
    _stage_live_value,
)
from .exceptions import InvalidParameterError, PlottingError
from .graph_elements import Plottable
from .inherit import INHERIT, Inherit, Styled, is_inherit, resolve_or, strip_inherit
from .tools import _require_optional_dependency

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

try:  # Optional dependency: pypdfium2
    import pypdfium2 as pdfium

    _PYPDFIUM2_AVAILABLE = True
except ImportError:
    _PYPDFIUM2_AVAILABLE = False


def _require_pypdfium2(feature: str = "this feature") -> None:
    """Raise a clear error when a pdf-extra feature is used without the optional dependency installed."""
    _require_optional_dependency(_PYPDFIUM2_AVAILABLE, feature, "pdf", "pypdfium2")


HAS_PYPDFIUM2 = _PYPDFIUM2_AVAILABLE


@runtime_checkable
class Plottable2D(Plottable, Protocol):
    """
    Dummy class to allow type hinting of Plottable2D objects.
    """

    pass


@dataclass
class Heatmap(Plottable2D):
    """
    The class implements heatmaps.

    Parameters
    ----------
    image : ArrayLike | str
        Image to display. If an array of values is given, the 2D array will be interpreted as the values of the image.
        If a str if given, the corresponding file will be read as an image.
    x_axis_range, y_axis_range : tuple[float, float], optional
        The range of x and y values used for the axes as tuples containing the start and end of the range. These
        values are ignored when ``x_mesh`` and ``y_mesh`` are provided. ``x_axis_range`` and ``y_axis_range`` can be
        set independently of one another; the axis left unset defaults to pixel-index coordinates based on the shape
        of ``image``.
    x_mesh, y_mesh : ArrayLike, optional
        Mesh grids defining the coordinates of the heatmap values. When provided, the heatmap is plotted using
        ``pcolormesh`` instead of ``imshow``.
    color_map : str, Colormap
        The color map to use for the :class:`~graphinglib.data_plotting_2d.Heatmap`. Can either be specified as a
        string (named colormap from Matplotlib) or a Colormap object.
        Examples include ``"viridis"``, ``"plasma"``, and ``"coolwarm"``.
        Default depends on the ``figure_style`` configuration.
    color_map_range: tuple[float, float], optional
        The data range covered by the color map, given as ``(minimum, maximum)``.
    show_color_bar : bool
        Whether or not to display the color bar next to the plot.
        Defaults to ``True``.
    alpha : float
        Opacity value of the :class:`~graphinglib.data_plotting_2d.Heatmap`.
        Range is ``0`` (transparent) to ``1`` (opaque).
        Defaults to 1.0.
    aspect_ratio : str or float
        Aspect ratio of the axes. This value is ignored when ``x_mesh`` and ``y_mesh`` are provided.
        Values include ``"auto"``, ``"equal"``, or a positive float.
        Default depends on the ``figure_style`` configuration.
    origin_position : str
        Position of the origin of the axes (upper left or lower left corner). This value is ignored when ``x_mesh`` and
        ``y_mesh`` are provided.
        Values are ``"upper"`` and ``"lower"``.
        Default depends on the ``figure_style`` configuration.
    interpolation : str
        Interpolation method to be applied to the image. This value is ignored when ``x_mesh`` and ``y_mesh`` are
        provided.
        Values include ``"none"``, ``"nearest"``, ``"bilinear"``, ``"bicubic"``, ``"spline16"``,
        ``"spline36"``, ``"hanning"``, ``"hamming"``, ``"hermite"``, ``"kaiser"``, ``"quadric"``,
        ``"catrom"``, ``"gaussian"``, ``"bessel"``, ``"mitchell"``, ``"sinc"``, and ``"lanczos"``.
        Defaults to ``"none"``.

        .. seealso::

            For other interpolation methods, refer to
            `Interpolations for imshow <https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html>`_.
    norm : str or Normalize, optional
        Normalization of the colormap. Default is ``None``.
    """

    def __init__(
        self,
        image: ArrayLike | str,
        x_axis_range: Optional[tuple[float, float]] = None,
        y_axis_range: Optional[tuple[float, float]] = None,
        x_mesh: Optional[ArrayLike] = None,
        y_mesh: Optional[ArrayLike] = None,
        color_map: str | Colormap | Inherit = INHERIT,
        color_map_range: Optional[tuple[float, float]] = None,
        show_color_bar: bool | Inherit = INHERIT,
        alpha: float = 1.0,
        aspect_ratio: str | float | Inherit = INHERIT,
        origin_position: str | Inherit = INHERIT,
        interpolation: str = "none",
        norm: Optional[str | Normalize] = None,
    ) -> None:
        """
        The class implements heatmaps.

        Parameters
        ----------
        image : ArrayLike | str
            Image to display. If an array of values is given, the 2D array will be interpreted as the values of the
            image. If a str if given, the corresponding file will be read as an image.
        x_axis_range, y_axis_range : tuple[float, float], optional
            The range of x and y values used for the axes as tuples containing the start and end of the range. These
            values are ignored when ``x_mesh`` and ``y_mesh`` are provided. ``x_axis_range`` and ``y_axis_range`` can
            be set independently of one another; the axis left unset defaults to pixel-index coordinates based on
            the shape of ``image``.
        x_mesh, y_mesh : ArrayLike, optional
            Mesh grids defining the coordinates of the heatmap values. When provided, the heatmap is plotted using
            ``pcolormesh`` instead of ``imshow``.
        color_map : str, Colormap
            The color map to use for the :class:`~graphinglib.data_plotting_2d.Heatmap`. Can either be specified as a
            string (named colormap from Matplotlib) or a Colormap object.
            Examples include ``"viridis"``, ``"plasma"``, and ``"coolwarm"``.
            Default depends on the ``figure_style`` configuration.
        color_map_range: tuple[float, float], optional
            The data range covered by the color map, given as ``(minimum, maximum)``.
        show_color_bar : bool
            Whether or not to display the color bar next to the plot.
            Defaults to ``True``.
        alpha : float
            Opacity value of the :class:`~graphinglib.data_plotting_2d.Heatmap`.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Defaults to 1.0.
        aspect_ratio : str or float
            Aspect ratio of the axes. This value is ignored when ``x_mesh`` and ``y_mesh`` are provided.
            Values include ``"auto"``, ``"equal"``, or a positive float.
            Default depends on the ``figure_style`` configuration.
        origin_position : str
            Position of the origin of the axes (upper left or lower left corner). This value is ignored when ``x_mesh``
            and ``y_mesh`` are provided.
            Values are ``"upper"`` and ``"lower"``.
            Default depends on the ``figure_style`` configuration.
        interpolation : str
            Interpolation method to be applied to the image. This value is ignored when ``x_mesh`` and ``y_mesh`` are
            provided.
            Values include ``"none"``, ``"nearest"``, ``"bilinear"``, ``"bicubic"``, ``"spline16"``,
            ``"spline36"``, ``"hanning"``, ``"hamming"``, ``"hermite"``, ``"kaiser"``, ``"quadric"``,
            ``"catrom"``, ``"gaussian"``, ``"bessel"``, ``"mitchell"``, ``"sinc"``, and ``"lanczos"``.
            Defaults to ``"none"``.

            .. seealso::

                For other interpolation methods, refer to
                `Interpolations for imshow <https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html>`_.
        norm : str or Normalize, optional
            Normalization of the colormap. Default is ``None``.
        """
        self.show_color_bar = show_color_bar
        self.image = image
        self.x_axis_range = x_axis_range
        self.y_axis_range = y_axis_range
        self.x_mesh = x_mesh
        self.y_mesh = y_mesh
        self.color_map = color_map
        self.color_map_range = color_map_range
        self.alpha = alpha
        self.aspect_ratio = aspect_ratio
        self.origin_position = origin_position
        self.interpolation = interpolation
        self._norm = norm

        self._color_bar_params: dict = {}
        self._reserve_color_bar = False

    @classmethod
    def from_function(
        cls,
        func: Callable[[ArrayLike, ArrayLike], ArrayLike],
        x_axis_range: tuple[float, float],
        y_axis_range: tuple[float, float],
        color_map: str | Colormap | Inherit = INHERIT,
        color_map_range: Optional[tuple[float, float]] = None,
        show_color_bar: bool = True,
        alpha: float = 1.0,
        aspect_ratio: str | float | Inherit = INHERIT,
        origin_position: str | Inherit = INHERIT,
        interpolation: str = "none",
        number_of_points: tuple[int, int] = (50, 50),
        norm: Optional[str | Normalize] = None,
    ) -> Self:
        """
        Creates a heatmap from a function.

        Parameters
        ----------
        func : Callable[[ArrayLike, ArrayLike], ArrayLike]
            Function to be plotted. Works with regular functions and lambda functions.
        x_axis_range, y_axis_range : tuple[float, float], optional
            The range of x and y values used for the axes as tuples containing the start and end of the range.
        color_map : str, Colormap
            The color map to use for the :class:`~graphinglib.data_plotting_2d.Heatmap`. Can either be specified as a
            string (named colormap from Matplotlib) or a Colormap object.
            Examples include ``"viridis"``, ``"plasma"``, and ``"coolwarm"``.
            Default depends on the ``figure_style`` configuration.
        color_map_range: tuple[float, float], optional
            The data range covered by the color map, given as ``(minimum, maximum)``.
        show_color_bar : bool
            Whether or not to display the color bar next to the plot.
            Defaults to ``True``.
        alpha : float
            Opacity value of the :class:`~graphinglib.data_plotting_2d.Heatmap`.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Defaults to 1.0.
        aspect_ratio : str or float
            Aspect ratio of the axes.
            Values include ``"auto"``, ``"equal"``, or a positive float.
            Default depends on the ``figure_style`` configuration.
        origin_position : str
            Position of the origin of the axes (upper left or lower left corner).
            Values are ``"upper"`` and ``"lower"``.
            Default depends on the ``figure_style`` configuration.
        interpolation : str
            Interpolation method to be applied to the image.
            Values include ``"none"``, ``"nearest"``, ``"bilinear"``, ``"bicubic"``, ``"spline16"``,
            ``"spline36"``, ``"hanning"``, ``"hamming"``, ``"hermite"``, ``"kaiser"``, ``"quadric"``,
            ``"catrom"``, ``"gaussian"``, ``"bessel"``, ``"mitchell"``, ``"sinc"``, and ``"lanczos"``.
            Defaults to ``"none"``.

            .. seealso::
                For other interpolation methods, refer to
                `Interpolations for imshow <https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html>`_.

        number_of_points : tuple[int, int]
            Number of points in the x and y coordinates.
            Defaults to ``(50, 50)``.
        norm : str or Normalize, optional
            Normalization of the colormap. Default is ``None``.

        Returns
        -------
        A :class:`~graphinglib.data_plotting_2d.Heatmap` object created from a function.
        """
        x = np.linspace(x_axis_range[0], x_axis_range[1], number_of_points[0])
        y = np.linspace(y_axis_range[0], y_axis_range[1], number_of_points[1])
        x_grid, y_grid = np.meshgrid(x, y)
        z = func(x_grid, y_grid)
        return cls(
            image=z,
            x_axis_range=x_axis_range,
            y_axis_range=y_axis_range,
            color_map=color_map,
            color_map_range=color_map_range,
            show_color_bar=show_color_bar,
            alpha=alpha,
            aspect_ratio=aspect_ratio,
            origin_position=origin_position,
            interpolation=interpolation,
            norm=norm,
        )

    @classmethod
    def from_points(
        cls,
        points: ArrayLike,
        values: ArrayLike,
        x_axis_range: tuple[float, float],
        y_axis_range: tuple[float, float],
        grid_interpolation: str = "nearest",
        fill_value: float = np.nan,
        color_map: str | Colormap | Inherit = INHERIT,
        color_map_range: Optional[tuple[float, float]] = None,
        show_color_bar: bool = True,
        alpha: float = 1.0,
        aspect_ratio: str | float | Inherit = INHERIT,
        origin_position: str | Inherit = INHERIT,
        interpolation: str = "none",
        number_of_points: tuple[int, int] = (50, 50),
        norm: Optional[str | Normalize] = None,
    ) -> Self:
        """
        Creates a heatmap by interpolating unevenly distributed data points on a grid.

        Parameters
        ----------
        points : ArrayLike
            The list or array of points at which values are known.
        values : ArrayLike
            The list or array of values at given points.
        x_axis_range, y_axis_range : tuple[float, float], optional
            The range of x and y values used for the axes as tuples containing the start and end of the range.
        grid_interpolation : str
            Interpolation method to be used when interpolating the uneavenly distributed data on a grid.
            Values are ``"nearest"``, ``"linear"``, and ``"cubic"``.
        color_map : str, Colormap
            The color map to use for the :class:`~graphinglib.data_plotting_2d.Heatmap`. Can either be specified as a
            string (named colormap from Matplotlib) or a Colormap object.
            Examples include ``"viridis"``, ``"plasma"``, and ``"coolwarm"``.
            Default depends on the ``figure_style`` configuration.
        color_map_range: tuple[float, float], optional
            The data range covered by the color map, given as ``(minimum, maximum)``.
        show_color_bar : bool
            Whether or not to display the color bar next to the plot.
            Defaults to ``True``.
        alpha : float
            Opacity value of the :class:`~graphinglib.data_plotting_2d.Heatmap`.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Defaults to 1.0.
        aspect_ratio : str or float
            Aspect ratio of the axes.
            Values include ``"auto"``, ``"equal"``, or a positive float.
            Default depends on the ``figure_style`` configuration.
        origin_position : str
            Position of the origin of the axes (upper left or lower left corner).
            Values are ``"upper"`` and ``"lower"``.
            Default depends on the ``figure_style`` configuration.
        interpolation : str
            Interpolation method to be applied to the image.
            Values include ``"none"``, ``"nearest"``, ``"bilinear"``, ``"bicubic"``, ``"spline16"``,
            ``"spline36"``, ``"hanning"``, ``"hamming"``, ``"hermite"``, ``"kaiser"``, ``"quadric"``,
            ``"catrom"``, ``"gaussian"``, ``"bessel"``, ``"mitchell"``, ``"sinc"``, and ``"lanczos"``.
            Defaults to ``"none"``.

            .. seealso::
                For other interpolation methods, refer to
                `Interpolations for imshow <https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html>`_.

        number_of_points : tuple[int, int]
            Number of points in the x and y coordinates.
            Defaults to ``(50, 50)``.
        norm : str or Normalize, optional
            Normalization of the colormap. Default is ``None``.

        Returns
        -------
        A :class:`~graphinglib.data_plotting_2d.Heatmap` object created from data points.
        """
        x = np.linspace(x_axis_range[0], x_axis_range[1], number_of_points[0])
        y = np.linspace(y_axis_range[0], y_axis_range[1], number_of_points[1])
        x_grid, y_grid = np.meshgrid(x, y)
        try:
            grid = griddata(
                points,
                values,
                (x_grid, y_grid),
                method=grid_interpolation,
                fill_value=fill_value,
            )
        except Exception as exc:
            raise PlottingError(
                f"Could not interpolate the data onto a grid ({exc}). Check that points "
                "and values are compatible and that enough points were provided."
            ) from exc
        return cls(
            image=grid,
            x_axis_range=x_axis_range,
            y_axis_range=y_axis_range,
            color_map=color_map,
            color_map_range=color_map_range,
            show_color_bar=show_color_bar,
            alpha=alpha,
            aspect_ratio=aspect_ratio,
            origin_position=origin_position,
            interpolation=interpolation,
            norm=norm,
        )

    @classmethod
    def from_pdf(
        cls,
        path: str,
        page: int = 0,
        dpi: float = 200,
        grayscale: bool = False,
        x_axis_range: Optional[tuple[float, float]] = None,
        y_axis_range: Optional[tuple[float, float]] = None,
        color_map: str | Colormap | Inherit = INHERIT,
        color_map_range: Optional[tuple[float, float]] = None,
        show_color_bar: bool = True,
        alpha: float = 1.0,
        aspect_ratio: str | float | Inherit = INHERIT,
        origin_position: str | Inherit = INHERIT,
        interpolation: str = "none",
        norm: Optional[str | Normalize] = None,
    ) -> Self:
        """
        Creates a heatmap by rasterizing a page of a PDF file.

        This requires the optional ``graphinglib[pdf]`` extra (installs ``pypdfium2``).

        Parameters
        ----------
        path : str
            Path to the PDF file to open.
        page : int
            Index of the page to rasterize.
            Defaults to ``0``.
        dpi : float
            Resolution used to rasterize the page, in dots per inch.
            Defaults to ``200``.
        grayscale : bool
            Whether to rasterize the page directly to a single-channel grayscale array, so that
            ``color_map`` is applied to it like a regular data heatmap. When ``False``, the page is
            kept as an RGB image, matching how :class:`~graphinglib.data_plotting_2d.Heatmap`
            already displays other image file formats; in that case, ``color_map`` and
            ``show_color_bar`` have no effect, since an RGB image never gets a color bar (same
            behavior as loading any other image file).
            Defaults to ``False``.
        x_axis_range, y_axis_range : tuple[float, float], optional
            The range of x and y values used for the axes as tuples containing the start and end of the range.
        color_map : str, Colormap
            The color map to use for the :class:`~graphinglib.data_plotting_2d.Heatmap`. Only has an effect when
            ``grayscale`` is ``True``. Can either be specified as a string (named colormap from Matplotlib) or a
            Colormap object.
            Examples include ``"viridis"``, ``"plasma"``, and ``"coolwarm"``.
            Defaults to ``"gray"`` when ``grayscale`` is ``True``; otherwise, default depends on the
            ``figure_style`` configuration.
        color_map_range: tuple[float, float], optional
            The data range covered by the color map, given as ``(minimum, maximum)``.
        show_color_bar : bool
            Whether or not to display the color bar next to the plot. Only has an effect when
            ``grayscale`` is ``True``.
            Defaults to ``True``.
        alpha : float
            Opacity value of the :class:`~graphinglib.data_plotting_2d.Heatmap`.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Defaults to 1.0.
        aspect_ratio : str or float
            Aspect ratio of the axes.
            Values include ``"auto"``, ``"equal"``, or a positive float.
            Default depends on the ``figure_style`` configuration.
        origin_position : str
            Position of the origin of the axes (upper left or lower left corner).
            Values are ``"upper"`` and ``"lower"``.
            Default depends on the ``figure_style`` configuration.
        interpolation : str
            Interpolation method to be applied to the image.
            Values include ``"none"``, ``"nearest"``, ``"bilinear"``, ``"bicubic"``, ``"spline16"``,
            ``"spline36"``, ``"hanning"``, ``"hamming"``, ``"hermite"``, ``"kaiser"``, ``"quadric"``,
            ``"catrom"``, ``"gaussian"``, ``"bessel"``, ``"mitchell"``, ``"sinc"``, and ``"lanczos"``.
            Defaults to ``"none"``.

            .. seealso::
                For other interpolation methods, refer to
                `Interpolations for imshow <https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html>`_.

        norm : str or Normalize, optional
            Normalization of the colormap. Default is ``None``.

        Returns
        -------
        A :class:`~graphinglib.data_plotting_2d.Heatmap` object created from a page of a PDF file.
        """
        _require_pypdfium2("Heatmap.from_pdf")
        with pdfium.PdfDocument(path) as pdf:
            bitmap = pdf[page].render(
                scale=dpi / 72, rev_byteorder=True, grayscale=grayscale
            )
            image = bitmap.to_numpy()
        if grayscale and is_inherit(color_map):
            color_map = "gray"
        return cls(
            image=image,
            x_axis_range=x_axis_range,
            y_axis_range=y_axis_range,
            color_map=color_map,
            color_map_range=color_map_range,
            show_color_bar=show_color_bar,
            alpha=alpha,
            aspect_ratio=aspect_ratio,
            origin_position=origin_position,
            interpolation=interpolation,
            norm=norm,
        )

    @property
    def image(self) -> np.ndarray:
        return self._image

    @image.setter
    def image(self, image: ArrayLike | str) -> None:
        from_file = isinstance(image, str)
        if isinstance(image, str):
            try:
                data = imread(image)
            except FileNotFoundError:
                raise
            except Exception as exc:
                raise PlottingError(
                    f"Could not read {image!r} as an image ({exc})."
                ) from exc
        else:
            data = np.asanyarray(image)
        is_rgb = data.ndim == 3 and data.shape[-1] in (3, 4)
        if data.ndim != 2 and not is_rgb:
            raise InvalidParameterError(
                "image must be a 2D array of values or a 3D array of RGB(A) pixels "
                f"(last axis of size 3 or 4), but got an array of shape {data.shape}."
            )
        session = _bound_session(self)
        if session is not None:
            data = _stage_live_value(self, "_image", data.copy())
        elif from_file or is_rgb:
            # RGB(A) pixels and file-backed images disable the colorbar for static plots.
            self._show_color_bar = False
        self._image = data

    def set_data(
        self,
        image: ArrayLike | str,
        *,
        x_mesh: ArrayLike | None = None,
        y_mesh: ArrayLike | None = None,
    ) -> Self:
        """Assign image values and optionally replace their mesh coordinates.

        During animation, assign coordinates and values that match by the end
        of the callback. With no new coordinates, the current grid is retained.
        """
        x = None if x_mesh is None else np.asanyarray(x_mesh)
        y = None if y_mesh is None else np.asanyarray(y_mesh)
        self.image = image
        if x is not None:
            self.x_mesh = x
        if y is not None:
            self.y_mesh = y
        return self

    @property
    def x_axis_range(self) -> Optional[tuple[float, float]]:
        return self._x_axis_range

    @x_axis_range.setter
    def x_axis_range(self, x_axis_range: Optional[tuple[float, float]]) -> None:
        self._x_axis_range = _stage_live_value(self, "_x_axis_range", x_axis_range)

    @property
    def y_axis_range(self) -> Optional[tuple[float, float]]:
        return self._y_axis_range

    @y_axis_range.setter
    def y_axis_range(self, y_axis_range: Optional[tuple[float, float]]) -> None:
        self._y_axis_range = _stage_live_value(self, "_y_axis_range", y_axis_range)

    @property
    def x_mesh(self) -> Optional[ArrayLike]:
        return self._x_mesh

    @x_mesh.setter
    def x_mesh(self, x_mesh: Optional[ArrayLike]) -> None:
        self._x_mesh = _stage_live_value(
            self, "_x_mesh", None if x_mesh is None else np.asanyarray(x_mesh)
        )

    @property
    def y_mesh(self) -> Optional[ArrayLike]:
        return self._y_mesh

    @y_mesh.setter
    def y_mesh(self, y_mesh: Optional[ArrayLike]) -> None:
        self._y_mesh = _stage_live_value(
            self, "_y_mesh", None if y_mesh is None else np.asanyarray(y_mesh)
        )

    @property
    def color_map(self) -> Styled[str | Colormap]:
        return self._color_map

    @color_map.setter
    def color_map(self, color_map: Styled[str | Colormap]) -> None:
        self._color_map = _stage_live_value(self, "_color_map", color_map)

    @property
    def color_map_range(self) -> tuple[float, float] | None:
        return self._color_map_range

    @color_map_range.setter
    def color_map_range(self, color_map_range: tuple[float, float] | None) -> None:
        self._color_map_range = _stage_live_value(
            self, "_color_map_range", color_map_range
        )

    @property
    def show_color_bar(self) -> Styled[bool]:
        return self._show_color_bar

    @show_color_bar.setter
    def show_color_bar(self, show_color_bar: Styled[bool]) -> None:
        self._show_color_bar = _stage_live_value(
            self, "_show_color_bar", show_color_bar
        )

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        self._alpha = _stage_live_value(self, "_alpha", alpha)

    @property
    def aspect_ratio(self) -> Styled[str | float]:
        return self._aspect_ratio

    @aspect_ratio.setter
    def aspect_ratio(self, aspect_ratio: Styled[str | float]) -> None:
        self._aspect_ratio = _stage_live_value(self, "_aspect_ratio", aspect_ratio)

    @property
    def origin_position(self) -> Styled[str]:
        return self._origin_position

    @origin_position.setter
    def origin_position(self, origin_position: Styled[str]) -> None:
        self._origin_position = _stage_live_value(
            self, "_origin_position", origin_position
        )

    @property
    def interpolation(self) -> str:
        return self._interpolation

    @interpolation.setter
    def interpolation(self, interpolation: str) -> None:
        self._interpolation = _stage_live_value(self, "_interpolation", interpolation)

    @property
    def color_bar_params(self) -> dict:
        return self._color_bar_params

    @property
    def _xy_range(self) -> Optional[tuple[float, float, float, float]]:
        if self._x_axis_range is None and self._y_axis_range is None:
            return None
        num_rows, num_cols = self._image.shape[:2]
        x_range = (
            self._x_axis_range
            if self._x_axis_range is not None
            else (-0.5, num_cols - 0.5)
        )
        if self._y_axis_range is not None:
            y_range = self._y_axis_range
        elif self._origin_position == "lower":
            y_range = (-0.5, num_rows - 0.5)
        else:
            y_range = (num_rows - 0.5, -0.5)
        return (
            float(x_range[0]),
            float(x_range[1]),
            float(y_range[0]),
            float(y_range[1]),
        )

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.data_plotting_2d.Heatmap`.
        """
        return deepcopy(self)

    def set_color_bar_params(
        self,
        label: Optional[str] = None,
        position: Optional[str] = None,
        *,
        reserve: bool = False,
        **color_bar_params,
    ) -> None:
        """
        Sets the color bar parameters.

        Parameters
        ----------
        label : str, optional
            Label of the color bar.
        position : str, optional
            Position of the color bar relative to the ``Figure``. It can be "left",
            "right", "top" or "bottom". This also determines the orientation of the
            color bar (vertical if the color bar is plotted on the "left" or "right",
            horizontal otherwise). If None, the color bar is plotted on the right
            side of the ``Figure``.
            Values are ``"left"``, ``"right"``, ``"top"``, and ``"bottom"``.
        reserve : bool, optional
            Allocate space for a color bar at animation setup, even when
            ``show_color_bar=False``. Set ``show_color_bar=True`` in a callback
            to reveal it. Applies to data with a color scale. Defaults to ``False``.
        **color_bar_params:
            Additional keyword arguments are passed to ``plt.colorbar``.

        Notes
        -----
        During animation, update the label or major tick positions on an
        existing colorbar. Use ``label=""`` to clear its label and
        ``ticks=None`` to restore its tick rule from setup.
        """
        if (session := _bound_session(self)) is not None:
            session.queue_colorbar_params(self, label, position, reserve, color_bar_params)
            return
        self._color_bar_params = color_bar_params
        self._reserve_color_bar = reserve
        if label is not None:
            self._color_bar_params["label"] = label
        if position is not None:
            self._color_bar_params["location"] = position

    @staticmethod
    def _check_live_pixels(data: np.ndarray, initial_ndim: int) -> None:
        if data.ndim != initial_ndim:
            raise InvalidParameterError(
                "An animated Heatmap must keep its initial scalar or RGB(A) data mode."
            )
        if not min(data.shape[:2]) or data.dtype.kind not in "biuf":
            raise InvalidParameterError(
                "Animated Heatmap pixels must be a nonempty real numeric array."
            )

    def _prepare_live_mesh(
        self, axes: plt.Axes, mesh: QuadMesh, shading: str, *, update_geometry: bool
    ) -> _ArtistEdit:
        previous = mesh.get_array()
        if (
            not update_geometry
            and previous is not None
            and previous.shape[:2] == self._image.shape[:2]
        ):
            return _edit(mesh.set_array, previous, np.ma.masked_invalid(self._image))
        try:
            # Follow the same conversion and center-to-corner rules as pcolormesh.
            x, y, values, render_mode = axes._pcolorargs(  # ty: ignore[unresolved-attribute]
                "pcolormesh", self._x_mesh, self._y_mesh, self._image, shading=shading
            )
        except (TypeError, ValueError) as exc:
            raise InvalidParameterError(
                f"The Heatmap mesh must match its image data ({exc})."
            ) from exc
        vertices = np.stack((x, y), axis=-1)
        if not np.isfinite(vertices).all():
            raise InvalidParameterError("Animated Heatmap mesh corners must be finite.")
        bounds = Bbox.null()
        bounds.update_from_data_xy(vertices.reshape(-1, 2))
        artist = cast(Any, mesh)  # No public QuadMesh geometry setter.

        def set_mesh(state) -> None:
            artist._coordinates, artist._shading, artist._bbox, artist._paths, data = (
                state
            )
            artist.set_array(data)

        before = (
            mesh.get_coordinates(),
            artist._shading,
            artist._bbox,
            artist._paths,
            previous,
        )
        return _edit(set_mesh, before, (vertices, render_mode, bounds, None, values))

    def _bind_live_geometry(
        self,
        session: _AnimationSession,
        axes: plt.Axes,
        raster: AxesImage,
        mesh: QuadMesh,
        *,
        raster_at_setup: bool,
    ) -> None:
        shading = plt.rcParams["pcolor.shading"]
        initial_ndim = self._image.ndim
        initial_origin = raster.origin
        initial_extent = tuple(raster.get_extent()) if raster_at_setup else None

        def prepare_extent() -> _ArtistEdit:
            xr, yr = self._x_axis_range, self._y_axis_range
            for limits in (xr, yr):
                if limits is not None and (
                    len(limits) != 2
                    or not np.isfinite(limits).all()
                    or limits[0] == limits[1]
                ):
                    raise InvalidParameterError(
                        "Animated raster extents must have distinct finite endpoints."
                    )
            baseline = initial_extent
            if baseline is None:
                rows, columns = self._image.shape[:2]
                origin = resolve_or(self._origin_position, initial_origin)
                y = (-0.5, rows - 0.5) if origin == "lower" else (rows - 0.5, -0.5)
                baseline = (-0.5, columns - 0.5, *y)
            extent = (
                *(baseline[:2] if xr is None else xr),
                *(baseline[2:] if yr is None else yr),
            )
            previous_extent = tuple(raster.get_extent())
            sticky_x, sticky_y = (
                list(raster.sticky_edges.x),
                list(raster.sticky_edges.y),
            )
            previous_baseline = initial_extent

            def set_extent(value) -> None:
                # Native extent updates report data bounds even with autoscaling disabled.
                limits = axes.dataLim
                points = limits.get_points().copy()
                minpos = cast(Any, limits).minpos.copy()
                ignore = axes.ignore_existing_data_limits
                try:
                    raster.set_extent(value)
                finally:
                    limits.set_points(points)
                    cast(Any, limits).minpos[:] = minpos
                    axes.ignore_existing_data_limits = ignore

            def apply() -> None:
                nonlocal initial_extent
                set_extent(extent)
                initial_extent = baseline

            def undo() -> None:
                nonlocal initial_extent
                set_extent(previous_extent)
                raster.sticky_edges.x[:] = sticky_x
                raster.sticky_edges.y[:] = sticky_y
                initial_extent = previous_baseline

            return _ArtistEdit(apply, undo)

        def prepare_origin(value: str) -> _ArtistEdit:
            if value not in ("lower", "upper"):
                raise InvalidParameterError(
                    "Heatmap origin_position must be 'lower' or 'upper'."
                )

            def set_origin(origin: Literal["lower", "upper"]) -> None:
                raster.origin = origin
                raster.stale = True

            return _edit(set_origin, raster.origin, value)

        def prepare() -> _ArtistEdit:
            x_set, y_set = self._x_mesh is not None, self._y_mesh is not None
            grid_changed = any(
                session.has_frame_update(self, field)
                for field in ("_x_mesh", "_y_mesh")
            )
            if x_set != y_set and grid_changed:
                raise InvalidParameterError(
                    "An animated Heatmap requires both mesh coordinates or neither."
                )
            self._check_live_pixels(self._image, initial_ndim)
            pixels_changed = session.has_frame_update(self, "_image")
            edits = []
            if x_set and y_set:
                activating = not mesh.get_visible()
                if mesh.axes is None:
                    edits.append(_attach_artists(axes, (mesh,)))
                if activating or pixels_changed or grid_changed:
                    edits.append(
                        self._prepare_live_mesh(
                            axes,
                            mesh,
                            shading,
                            update_geometry=activating or grid_changed,
                        )
                    )
                if activating:
                    edits.extend(
                        (
                            _edit(mesh.set_visible, False, True),
                            _edit(raster.set_visible, True, False),
                        )
                    )
            else:
                activating = not raster.get_visible()
                if raster not in axes.images:
                    edits.append(_attach_artists(axes, (raster,)))
                if activating or pixels_changed:
                    edits.append(
                        _edit(raster.set_data, raster.get_array(), self._image)
                    )
                if activating or any(
                    session.has_frame_update(self, field)
                    for field in ("_x_axis_range", "_y_axis_range")
                ):
                    edits.append(prepare_extent())
                if activating:
                    edits.extend(
                        (
                            _edit(raster.set_visible, False, True),
                            _edit(mesh.set_visible, True, False),
                        )
                    )
            return _combine_edits(edits)

        session.bind_derived(
            self,
            ("_image", "_x_mesh", "_y_mesh", "_x_axis_range", "_y_axis_range"),
            prepare,
        )
        session.bind_property(self, "_interpolation", raster, "interpolation")
        session.bind_update(self, "_origin_position", prepare_origin)

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified
        `Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_.
        """
        params = {
            "cmap": self._color_map,
            "alpha": self._alpha,
            "norm": self._norm,
        }
        if self._color_map_range:
            params["vmin"] = min(self._color_map_range)
            params["vmax"] = max(self._color_map_range)
        use_pcolormesh = self._x_mesh is not None and self._y_mesh is not None
        if use_pcolormesh:
            params = strip_inherit(params)
            image = axes.pcolormesh(
                self._x_mesh,
                self._y_mesh,
                self._image,
                zorder=z_order,
                **params,
            )
        else:
            params.update(
                {
                    "aspect": self._aspect_ratio,
                    "origin": self._origin_position,
                    "interpolation": self._interpolation,
                    "extent": self._xy_range,
                }
            )

            params = strip_inherit(params)
            image = axes.imshow(
                self._image,
                zorder=z_order,
                **params,
            )
        session = _DRAW_SESSION.get()
        show_colorbar = resolve_or(self._show_color_bar, True)
        bar = None
        if show_colorbar or (
            session is not None and self._reserve_color_bar and self._image.ndim == 2
        ):
            fig = axes.get_figure()
            assert fig is not None
            bar = fig.colorbar(image, ax=axes, **self._color_bar_params)
        if session is not None:
            if bar is not None:
                session.bind_colorbar(self, bar, initially_visible=show_colorbar)
            # The colorbar keeps its original mappable; both rendering artists
            # share that artist's normalization and receive the same colormap.
            common: dict[str, Any] = {
                "alpha": image.get_alpha(),
                "cmap": image.get_cmap(),
                "norm": image.norm,
                "visible": False,
                "zorder": z_order,
            }
            if isinstance(image, AxesImage):
                raster = image
                mesh = QuadMesh(
                    np.array([[[0, 0], [1, 0]], [[0, 1], [1, 1]]]),
                    edgecolors="none",
                    antialiased=False,
                    snap=plt.rcParams["pcolormesh.snap"],
                    **common,
                )
            else:
                mesh = image
                raster = AxesImage(
                    axes,
                    extent=(0, 1, 0, 1),
                    origin=cast(
                        Any,
                        resolve_or(self._origin_position, plt.rcParams["image.origin"]),
                    ),
                    interpolation=resolve_or(
                        self._interpolation, plt.rcParams["image.interpolation"]
                    ),
                    **common,
                )
                raster.set_data([[0]])
            for artist in (raster, mesh):
                session.bind_property(self, "_alpha", artist, "alpha")

            if self._image.ndim == 2:
                initial_clim = image.get_clim()
                for artist in (raster, mesh):
                    session.bind_property(self, "_color_map", artist, "cmap")

                def prepare_range(value: tuple[float, float] | None) -> _ArtistEdit:
                    if value is not None and (
                        len(value) != 2
                        or not np.isfinite(value).all()
                        or value[0] == value[1]
                    ):
                        raise InvalidParameterError(
                            "An animated Heatmap color_map_range must contain two distinct finite numbers."
                        )

                    def set_limits(limits: tuple[float, float]) -> None:
                        # Notify colorbars after both endpoints are available, even on older Matplotlib.
                        norm = image.norm
                        with norm.callbacks.blocked(signal="changed"):
                            image.set_clim(limits)
                        norm.callbacks.process("changed")

                    limits = initial_clim if value is None else tuple(sorted(value))
                    return _edit(set_limits, image.get_clim(), limits)

                if self._norm is None or type(self._norm) is Normalize:
                    session.bind_update(self, "_color_map_range", prepare_range)

            def prepare_aspect(value: str | float) -> _ArtistEdit:
                valid = (
                    value in ("equal", "auto")
                    if isinstance(value, str)
                    else isinstance(value, Real) and isfinite(value) and value > 0
                )
                if not valid:
                    raise InvalidParameterError(
                        "Animated Heatmap aspect_ratio must be 'equal', 'auto', or a positive number."
                    )
                return _edit(axes.set_aspect, axes.get_aspect(), value)

            session.bind_update(self, "_aspect_ratio", prepare_aspect)
            self._bind_live_geometry(
                session,
                axes,
                raster,
                mesh,
                raster_at_setup=isinstance(image, AxesImage),
            )


class _LiveQuiver(Quiver):
    """Update locations as well as offsets used by Matplotlib's arrow renderer."""

    def set_grid(
        self, x: np.ndarray, y: np.ndarray, u: np.ndarray, v: np.ndarray
    ) -> None:
        # Quiver.set_offsets only changes placement. XY and N are also read
        # when making the arrow paths and validating the component arrays.
        self.X, self.Y = x, y
        self.XY = np.column_stack((x, y))
        self.N = len(x)
        self.set_offsets(self.XY)
        self.set_UVC(u, v)

    def _angles_lengths(self, *args, eps=1):
        # Matplotlib derives eps from the data limits in screen-angle mode.
        # They can be zero at the origin or infinite after an empty setup.
        if not isfinite(eps) or eps <= 0:
            eps = 0.001
        return super()._angles_lengths(*args, eps=eps)  # ty: ignore[unresolved-attribute]

    def _make_verts(self, *args):
        # Older Matplotlib passes (U, V, angles). Newer versions prepend XY.
        u, v = args[-3:-1]
        if self.scale is not None or np.any(
            np.logical_not(self.Umask) & ((u != 0) | (v != 0))
        ):
            return super()._make_verts(*args)  # ty: ignore[unresolved-attribute]
        # An empty, masked, or zero field has no auto scale. Render that
        # frame without freezing the scale used by the first nonzero field.
        self.scale = 1
        try:
            return super()._make_verts(*args)  # ty: ignore[unresolved-attribute]
        finally:
            self.scale = None


def _quiver_data(
    axes: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    expand_grid: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Match Quiver's handling of coordinate vectors and pre-expanded meshes.
    try:
        x = np.asarray(axes.convert_xunits(x), dtype=float).ravel()
        y = np.asarray(axes.convert_yunits(y), dtype=float).ravel()
    except (TypeError, ValueError) as exc:
        raise InvalidParameterError(
            "Animated VectorField positions must be real numbers after unit conversion."
        ) from exc
    u, v = np.atleast_1d(u, v)
    if max(u.ndim, v.ndim) > 2:
        raise InvalidParameterError(
            "VectorField components must be scalar, one-dimensional or two-dimensional."
        )
    mesh_component = v if u.ndim == 1 and v.ndim == 2 else u
    rows, columns = (
        (1, mesh_component.size) if mesh_component.ndim == 1 else mesh_component.shape
    )
    if expand_grid and x.size == columns and y.size == rows:
        x, y = (item.ravel() for item in np.meshgrid(x, y))
    if x.size != y.size or u.size not in (1, x.size) or v.size not in (1, x.size):
        raise InvalidParameterError(
            "VectorField coordinates must be paired or span the component grid. "
            "Each component must be a scalar or match the number of positions."
        )
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise InvalidParameterError("Animated VectorField positions must be finite.")

    def component(values: np.ndarray) -> np.ndarray:
        if np.asarray(values).dtype.kind not in "biuf":
            raise InvalidParameterError("VectorField components must be real numbers.")
        flattened = np.ma.asarray(values).ravel()
        return np.ma.array(
            np.broadcast_to(flattened.data, x.size),
            mask=np.broadcast_to(np.ma.getmaskarray(flattened), x.size),
            copy=True,
        )

    return x.copy(), y.copy(), component(u), component(v)


def _quiver_number(value: float, name: str, *, allow_zero: bool = False) -> float:
    if (
        isinstance(value, Real)
        and isfinite(value)
        and (value > 0 or value == 0 and allow_zero)
    ):
        return float(value)
    requirement = "nonnegative" if allow_zero else "positive"
    raise InvalidParameterError(
        f"Animated VectorField {name} must be a finite {requirement} number."
    )


@dataclass
class VectorField(Plottable2D):
    """
    This class implements vector fields.

    Parameters
    ----------
    x_data, y_data : ArrayLike
        x and y coordinates of the vectors.
    u_data, v_data : ArrayLike
        Magnitudes in the x and y coordinates.
    arrow_width : float
        Width of the arrow shaft. Acts as a multiplier for the standard arrow width.
        Typical range is ``0.5`` to ``4``.
        Default depends on the ``figure_style`` configuration.
    arrow_head_size : float
        Size of the arrow head. Acts as a multiplier for the standard arrow head size.
        Typical range is ``0.5`` to ``4``.
        Default depends on the ``figure_style`` configuration.
    scale : float
        Scaling of the arrow lengths. If ``None``, the arrows will be automatically scaled to look nice. Use 1 for no scaling.
    angle_in_data_coords : bool
        Whether to use the screen coordinates or the data coordinates to determine the vector directions.
        Defaults to ``True``.
    color : str
        Color of the vector arrows.
        Default depends on the ``figure_style`` configuration.

    Notes
    -----
    Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
    (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
    values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).
    """

    def __init__(
        self,
        x_data: ArrayLike,
        y_data: ArrayLike,
        u_data: ArrayLike,
        v_data: ArrayLike,
        arrow_width: float | Inherit = INHERIT,
        arrow_head_size: float | Inherit = INHERIT,
        scale: Optional[float] = None,
        make_angles_axes_independent: bool = False,
        color: str | Inherit = INHERIT,
    ) -> None:
        """
        This class implements vector fields.

        Parameters
        ----------
        x_data, y_data : ArrayLike
            x and y coordinates of the vectors.
        u_data, v_data : ArrayLike
            Magnitudes in the x and y coordinates.
        arrow_width : float
            Width of the arrow shaft. Acts as a multiplier for the standard arrow width.
            Typical range is ``0.5`` to ``4``.
            Default depends on the ``figure_style`` configuration.
        arrow_head_size : float
            Size of the arrow head. Acts as a multiplier for the standard arrow head size.
            Typical range is ``0.5`` to ``4``.
            Default depends on the ``figure_style`` configuration.
        scale : float
            Scaling of the arrow lengths. If ``None``, the arrows will be automatically scaled to look nice. Use 1 for no scaling.
            Default is ``None``.
        make_angles_axes_independent : bool
            Whether to use the screen coordinates or the data coordinates to
            determine the vector directions. If ``True``, vectors with u = v will
            always appear as 45 degree angles with respect to the screen, regardless
            of the axes limits. If ``False``, the vectors will scale with the
            aspect ratio of the axes.
            Defaults to ``False``.
        color : str
            Color of the vector arrows.
            Default depends on the ``figure_style`` configuration.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).
        """
        self._x_data = np.asarray(x_data)
        self._y_data = np.asarray(y_data)
        self._u_data = np.asanyarray(u_data)
        self._v_data = np.asanyarray(v_data)
        self._arrow_width = arrow_width
        self._arrow_head_size = arrow_head_size
        self._scale = scale
        self._make_angles_axes_independent = make_angles_axes_independent

        self._color = color

    @classmethod
    def from_function(
        cls,
        func: Callable[[ArrayLike, ArrayLike], tuple[ArrayLike, ArrayLike]],
        x_axis_range: tuple[float, float],
        y_axis_range: tuple[float, float],
        number_of_arrows_x: int = 10,
        number_of_arrows_y: int = 10,
        arrow_width: float | Inherit = INHERIT,
        arrow_head_size: float | Inherit = INHERIT,
        scale: Optional[float] = None,
        make_angles_axes_independent: bool = False,
        color: str | Inherit = INHERIT,
    ) -> Self:
        """
        Creates a :class:`~graphinglib.data_plotting_2d.VectorField` from a function.

        Parameters
        ----------
        func : Callable[[ArrayLike, ArrayLike], tuple[ArrayLike, ArrayLike]]
            Function to be plotted. Works with regular functions and lambda functions.
        x_axis_range, y_axis_range : tuple[float, float], optional
            The range of x and y values used for the axes as tuples containing the start and end of the range.
        number_of_arrows_x, number_of_arrows_y : int
            Number of arrows to plot in the x and y direction. Defaults to 10.
        arrow_width : float
            Width of the arrow shaft. Acts as a multiplier for the standard arrow width.
            Typical range is ``0.5`` to ``4``.
            Default depends on the ``figure_style`` configuration.
        arrow_head_size : float
            Size of the arrow head. Acts as a multiplier for the standard arrow head size.
            Typical range is ``0.5`` to ``4``.
            Default depends on the ``figure_style`` configuration.
        scale : float
            Scaling of the arrow lengths. If ``None``, the arrows will be automatically scaled to look nice. Use 1 for no scaling.
            Default is ``None``.
        make_angles_axes_independent : bool
            Whether to use the screen coordinates or the data coordinates to
            determine the vector directions. If ``True``, vectors with u = v will
            always appear as 45 degree angles with respect to the screen, regardless
            of the axes limits. If ``False``, the vectors will scale with the
            aspect ratio of the axes.
            Defaults to ``False``.
        color : str
            Color of the vector arrows.
            Default depends on the ``figure_style`` configuration.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).

        Returns
        -------
        A :class:`~graphinglib.data_plotting_2d.VectorField` object from a function.
        """
        x = np.linspace(x_axis_range[0], x_axis_range[-1], number_of_arrows_x)
        y = np.linspace(y_axis_range[0], y_axis_range[-1], number_of_arrows_y)
        x_grid, y_grid = np.meshgrid(x, y)
        u, v = func(x_grid, y_grid)
        return cls(
            x_grid,
            y_grid,
            u,
            v,
            arrow_width,
            arrow_head_size,
            scale,
            make_angles_axes_independent,
            color,
        )

    @property
    def x_data(self) -> ArrayLike:
        return self._x_data

    @x_data.setter
    def x_data(self, x_data: ArrayLike) -> None:
        coordinates = (
            np.asarray(x_data)
            if _bound_session(self) is None
            else np.array(x_data, copy=True)
        )
        self._x_data = _stage_live_value(self, "_x_data", coordinates)

    @property
    def y_data(self) -> ArrayLike:
        return self._y_data

    @y_data.setter
    def y_data(self, y_data: ArrayLike) -> None:
        coordinates = (
            np.asarray(y_data)
            if _bound_session(self) is None
            else np.array(y_data, copy=True)
        )
        self._y_data = _stage_live_value(self, "_y_data", coordinates)

    @property
    def u_data(self) -> ArrayLike:
        return self._u_data

    @u_data.setter
    def u_data(self, u_data: ArrayLike) -> None:
        self._u_data = _stage_live_value(self, "_u_data", np.asanyarray(u_data))

    @property
    def v_data(self) -> ArrayLike:
        return self._v_data

    @v_data.setter
    def v_data(self, v_data: ArrayLike) -> None:
        self._v_data = _stage_live_value(self, "_v_data", np.asanyarray(v_data))

    def set_data(
        self,
        u_data: ArrayLike,
        v_data: ArrayLike,
        *,
        x_data: ArrayLike | None = None,
        y_data: ArrayLike | None = None,
    ) -> Self:
        """Assign both vector components and optionally replace their positions.

        Positions may be paired arrays or coordinate vectors for a rectangular
        grid. Supply a two-dimensional component to specify a new rectangular
        grid with one-dimensional coordinate vectors. When positions are omitted,
        scalar components broadcast to all current positions during playback.
        """
        x = None if x_data is None else np.asarray(x_data)
        y = None if y_data is None else np.asarray(y_data)
        u, v = np.asanyarray(u_data), np.asanyarray(v_data)
        self.u_data, self.v_data = u, v
        if x is not None:
            self.x_data = x
        if y is not None:
            self.y_data = y
        return self

    @property
    def arrow_width(self) -> Styled[float]:
        return self._arrow_width

    @arrow_width.setter
    def arrow_width(self, arrow_width: float) -> None:
        self._arrow_width = _stage_live_value(self, "_arrow_width", arrow_width)

    @property
    def arrow_head_size(self) -> Styled[float]:
        return self._arrow_head_size

    @arrow_head_size.setter
    def arrow_head_size(self, arrow_head_size: float) -> None:
        self._arrow_head_size = _stage_live_value(
            self, "_arrow_head_size", arrow_head_size
        )

    @property
    def scale(self) -> Optional[float]:
        return self._scale

    @scale.setter
    def scale(self, scale: Optional[float]) -> None:
        self._scale = _stage_live_value(self, "_scale", scale)

    @property
    def make_angles_axes_independent(self) -> bool:
        return self._make_angles_axes_independent

    @make_angles_axes_independent.setter
    def make_angles_axes_independent(self, value: bool) -> None:
        self._make_angles_axes_independent = _stage_live_value(
            self, "_make_angles_axes_independent", value
        )

    @property
    def color(self) -> Styled[str]:
        return self._color

    @color.setter
    def color(self, color: str) -> None:
        self._color = _stage_live_value(self, "_color", color)

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.data_plotting_2d.VectorField`.
        """
        return deepcopy(self)

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified
        `Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_.
        """
        if self._make_angles_axes_independent:
            angle = "uv"
        else:
            angle = "xy"
        arrow_width = resolve_or(self._arrow_width, 1)
        arrow_head_size = resolve_or(self._arrow_head_size, 1)
        session = _DRAW_SESSION.get()
        if session is not None:
            _quiver_number(arrow_width, "arrow_width")
            _quiver_number(arrow_head_size, "arrow_head_size", allow_zero=True)
            if self._scale is not None:
                _quiver_number(self._scale, "scale")
        params = {
            "angles": angle,
            "width": 0.005 * arrow_width,
            "headwidth": 4 * arrow_head_size / arrow_width,
            "headlength": 4 * arrow_head_size / arrow_width,
            "headaxislength": 4 * arrow_head_size / arrow_width,
            "color": self._color,
            "scale": 1 / self._scale if self._scale is not None else None,
            "scale_units": "xy",
        }
        params = strip_inherit(params)
        if session is None:
            axes.quiver(
                self._x_data,
                self._y_data,
                self._u_data,
                self._v_data,
                zorder=z_order,
                **params,
            )
            return

        axes.xaxis.update_units(self._x_data)
        axes.yaxis.update_units(self._y_data)
        raw_positions = self._x_data, self._y_data
        current = _quiver_data(axes, *raw_positions, self._u_data, self._v_data)
        artist = _LiveQuiver(axes, *current, zorder=z_order, **params)
        axes.add_collection(artist)

        def prepare_field() -> _ArtistEdit:
            new_positions = self._x_data, self._y_data
            keep_grid = all(
                old is new
                for old, new in zip(raw_positions, new_positions, strict=True)
            )
            # Scalars or flattened components can replace the velocities of a
            # rectangular grid without changing the existing positions.
            new = _quiver_data(
                axes,
                *(current[:2] if keep_grid else new_positions),
                self._u_data,
                self._v_data,
                expand_grid=not keep_grid,
            )

            def set_field(state) -> None:
                nonlocal current, raw_positions
                data, positions = state
                artist.set_grid(*data)
                current, raw_positions = data, positions

            return _edit(set_field, (current, raw_positions), (new, new_positions))

        def attr_edit(name: str, value: Any) -> _ArtistEdit:
            return _edit(
                lambda new: setattr(artist, name, new), getattr(artist, name), value
            )

        def prepare_heads() -> _ArtistEdit:
            width = _quiver_number(
                resolve_or(self._arrow_width, arrow_width), "arrow_width"
            )
            size = _quiver_number(
                resolve_or(self._arrow_head_size, arrow_head_size),
                "arrow_head_size",
                allow_zero=True,
            )
            head = 4 * size / width
            return _combine_edits(
                [
                    attr_edit("width", 0.005 * width),
                    attr_edit("headwidth", head),
                    attr_edit("headlength", head),
                    attr_edit("headaxislength", head),
                ]
            )

        def prepare_angles(value: bool) -> _ArtistEdit:
            if not isinstance(value, (bool, np.bool_)):
                raise InvalidParameterError(
                    "VectorField.make_angles_axes_independent must be a boolean."
                )
            return attr_edit("angles", "uv" if value else "xy")

        session.bind_derived(
            self, ("_x_data", "_y_data", "_u_data", "_v_data"), prepare_field
        )
        session.bind_derived(self, ("_arrow_width", "_arrow_head_size"), prepare_heads)
        session.bind_update(self, "_make_angles_axes_independent", prepare_angles)
        session.bind_update(
            self,
            "_scale",
            lambda value: attr_edit(
                "scale", None if value is None else 1 / _quiver_number(value, "scale")
            ),
        )
        session.bind_property(self, "_color", artist, "facecolor")


@dataclass
class _ContourGeometry:
    levels: np.ndarray
    values: np.ndarray
    paths: list[MplPath]
    data_range: tuple[float, float] | None


@dataclass
class _ContourFrame:
    geometry: _ContourGeometry
    filled: bool
    clim: tuple[float, float]
    cmap: Colormap
    alpha: float | None
    widths: np.ndarray | None


class _LegacyContourCollections:
    """Reuse one native PathCollection per contour level on Matplotlib 3.7."""

    def __init__(self, contour: Any) -> None:
        self.contour = contour
        self.pool: list[PathCollection] = list(contour.collections)
        self.zorder = contour._contour_zorder

    def paths(self) -> list[MplPath]:
        return [
            MplPath.make_compound_path(*collection.get_paths())
            for collection in self.contour.collections
        ]

    def set_paths(self, paths: list[MplPath]) -> None:
        contour = self.contour
        while len(self.pool) < len(paths):
            artist = PathCollection(
                [],
                transform=contour.get_transform(),
                zorder=self.zorder,
                label="_nolegend_",
            )
            self.pool.append(artist)
            contour.axes.add_collection(artist, autolim=False)
        contour.collections[:] = self.pool[: len(paths)]
        for collection, path in zip(contour.collections, paths, strict=True):
            collection.set_paths([path])
            collection.set_visible(True)
        for collection in self.pool[len(paths) :]:
            collection.set_visible(False)
        contour.allsegs = [
            [path.vertices] if np.asarray(path.vertices).size else [] for path in paths
        ]
        contour.allkinds = [
            [path.codes] if np.asarray(path.vertices).size else [] for path in paths
        ]

    def style(self, state: _ContourFrame, dashes: list[str]) -> None:
        if state.widths is not None:
            self.contour.tlinewidths = [(width,) for width in state.widths]
        for index, collection in enumerate(self.contour.collections):
            collection.set_alpha(state.alpha)
            if state.filled:
                collection.set_edgecolor("none")
            elif state.widths is not None:
                collection.set_facecolor("none")
                collection.set_linewidth(state.widths[index])
                collection.set_linestyle(dashes[index])

    def truncate(self, count: int) -> None:
        for collection in reversed(self.pool[count:]):
            if collection.axes is not None:
                collection.remove()
        del self.pool[count:]


def _contour_levels(
    specification: int | ArrayLike,
    data_range: tuple[float, float],
    filled: bool,
) -> np.ndarray:
    if isinstance(specification, Integral):
        if specification < 1:
            raise InvalidParameterError(
                "The requested Contour level count must be positive."
            )
        lo, hi = data_range
        ticks = np.asarray(
            MaxNLocator(int(specification) + 1, min_n_ticks=1).tick_values(lo, hi)
        )
        # Keep the nearest outside levels, matching Matplotlib's automatic levels.
        first = max(np.searchsorted(ticks, lo, side="left") - 1, 0)
        last = np.searchsorted(ticks, hi, side="right") + 1
        return ticks[first:last] if len(ticks[first:last]) >= 3 else ticks

    levels = np.asarray(specification)
    if (
        levels.ndim != 1
        or levels.dtype.kind not in "biuf"
        or levels.size < (2 if filled else 1)
        or not np.isfinite(levels).all()
        or not np.all(levels[1:] > levels[:-1])
    ):
        raise InvalidParameterError(
            "Contour levels must be a strictly increasing sequence of finite numbers "
            "with at least two entries for filled contours or one for contour lines."
        )
    return levels.astype(float)


def _contour_bar_boundaries(levels: np.ndarray) -> np.ndarray:
    if len(levels) == 1:
        # Matplotlib's line colorbar needs a span even for a single contour.
        return np.asarray(nonsingular(levels[0], levels[0], expander=0.05))
    return levels


def _set_contour_bar_mode(colorbar: Colorbar, filled: bool) -> None:
    # Colorbar.update_normal() uses the mode captured at construction.
    bar = cast(Any, colorbar)
    if bar._filled == filled:
        return
    if filled:
        for line in bar.lines:
            line.remove()
        bar.lines = []
    else:
        if bar.solids is not None:
            bar.solids.remove()
            bar.solids = None
        for patch in bar.solids_patches:
            patch.remove()
        bar.solids_patches = []
        bar.dividers.set_segments([])
    bar._filled = filled


def _contour_path_groups(
    x: np.ndarray | None,
    y: np.ndarray | None,
    grid: np.ma.MaskedArray,
    levels: np.ndarray,
    *,
    filled: bool,
    algorithm: str,
    corner_mask: bool,
) -> list[MplPath]:
    if x is None or y is None:
        x, y = np.arange(grid.shape[1]), np.arange(grid.shape[0])
    try:
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Contour positions must be finite.")
        generator = contourpy.contour_generator(
            x,
            y,
            grid,
            name=algorithm,
            corner_mask=corner_mask,
            line_type=contourpy.LineType.SeparateCode,
            fill_type=contourpy.FillType.OuterCode,
        )
    except (TypeError, ValueError) as exc:
        raise InvalidParameterError(f"Invalid animated Contour grid ({exc}).") from exc

    if filled:
        lower = levels[:-1].copy()
        if grid.count() and grid.min() == lower[0]:
            lower[0] = nextafter(lower[0], -np.inf)
        groups = (generator.filled(lo, hi) for lo, hi in zip(lower, levels[1:]))
    else:
        groups = (generator.lines(level) for level in levels)
    return [
        MplPath(np.concatenate(vertices), np.concatenate(codes))
        if vertices
        else MplPath(np.empty((0, 2)))
        for vertices, codes in cast(
            "Iterable[tuple[list[np.ndarray], list[np.ndarray]]]", groups
        )
    ]


def _contour_line_widths(value: ArrayLike, count: int) -> np.ndarray:
    widths = np.atleast_1d(value)
    if (
        widths.ndim != 1
        or not widths.size
        or widths.dtype.kind not in "biuf"
        or not np.isfinite(widths).all()
        or np.any(widths < 0)
    ):
        raise InvalidParameterError(
            "Contour line widths must be finite and nonnegative."
        )
    return np.resize(widths.astype(float), count)


@dataclass
class Contour(Plottable2D):
    """
    This class implements contour plots.

    Parameters
    ----------
    z_data : ArrayLike
        Data for each point of the mesh.
    x_mesh, y_mesh : ArrayLike, optional
        Mesh grids defining the coordinates of the contour values. If not provided, a mesh grid will be created based on
        the shape of ``z_data``.
    levels : int | ArrayLike
        If `levels` is an integer, it defines the number of levels to use in the contour.
        If `levels` is an array, it defines the value of each contour level.
        Typical range is ``5`` to ``20`` when given as an integer.
        Default depends on the ``figure_style`` configuration.
    color_map : str or Colormap
        The color map to use for the :class:`~graphinglib.data_plotting_2d.Contour`. Can either be specified as a
        string (named colormap from Matplotlib) or a Colormap object.
        Examples include ``"viridis"``, ``"plasma"``, and ``"coolwarm"``.
        Default depends on the ``figure_style`` configuration.
    color_map_range: tuple[float, float], optional
        The data range covered by the color map, given as ``(minimum, maximum)``.
    show_color_bar : bool
        Whether or not to display the color bar next to the plot.
        Default depends on the ``figure_style`` configuration.
    filled : bool
        Wheter or not to fill the contour with color.
        Default depends on the ``figure_style`` configuration.
    alpha : float
        Opacity of the filled contour.
        Range is ``0`` (transparent) to ``1`` (opaque).
        Default depends on the ``figure_style`` configuration.
    line_widths : float | ArrayLike
        If the contour is not filled, the width of the contour lines. If an array is provided, it defines the line width
        for each contour level.
        Typical range is ``0.5`` to ``3`` points.
        Default depends on the ``figure_style`` configuration.
    """

    _z_data: np.ndarray
    _x_mesh: np.ndarray | None
    _y_mesh: np.ndarray | None
    _levels: Styled[int | ArrayLike] = INHERIT
    _color_map: str | Colormap | Inherit = INHERIT
    _show_color_bar: bool | Inherit = INHERIT
    _filled: bool | Inherit = INHERIT
    _alpha: float | Inherit = INHERIT
    _line_widths: float | ArrayLike | Inherit = INHERIT

    def __init__(
        self,
        z_data: ArrayLike,
        x_mesh: Optional[ArrayLike] = None,
        y_mesh: Optional[ArrayLike] = None,
        levels: int | ArrayLike | Inherit = INHERIT,
        color_map: str | Colormap | Inherit = INHERIT,
        color_map_range: Optional[tuple[float, float]] = None,
        show_color_bar: bool | Inherit = INHERIT,
        filled: bool | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
        line_widths: float | ArrayLike | Inherit = INHERIT,
    ) -> None:
        """
        This class implements contour plots.

        Parameters
        ----------
        z_data : ArrayLike
            Data for each point of the mesh.
        x_mesh, y_mesh : ArrayLike, optional
            Mesh grids defining the coordinates of the contour values. If not provided, a mesh grid will be created
            based on the shape of ``z_data``.
        levels : int | ArrayLike
            If `levels` is an integer, it defines the number of levels to use in the contour.
            If `levels` is an array, it defines the value of each contour level.
            Typical range is ``5`` to ``20`` when given as an integer.
            Default depends on the ``figure_style`` configuration.
        color_map : str or Colormap
            The color map to use for the :class:`~graphinglib.data_plotting_2d.Contour`. Can either be specified as a
            string (named colormap from Matplotlib) or a Colormap object.
            Examples include ``"viridis"``, ``"plasma"``, and ``"coolwarm"``.
            Default depends on the ``figure_style`` configuration.
        color_map_range: tuple[float, float], optional
            The data range covered by the color map, given as ``(minimum, maximum)``.
        show_color_bar : bool
            Whether or not to display the color bar next to the plot.
            Default depends on the ``figure_style`` configuration.
        filled : bool
            Wheter or not to fill the contour with color.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacity of the filled contour.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.
        line_widths : float | ArrayLike
            If the contour is not filled, the width of the contour lines. If an array is provided, it defines the line
            width for each contour level.
            Typical range is ``0.5`` to ``3`` points.
            Default depends on the ``figure_style`` configuration.
        """
        self.z_data = z_data
        self.x_mesh = x_mesh
        self.y_mesh = y_mesh
        self._levels = levels
        self._color_map = color_map
        self._color_map_range = color_map_range
        self._show_color_bar = show_color_bar
        self._filled = filled
        self._alpha = alpha
        self._line_widths = line_widths

        self._color_bar_params: dict = {}
        self._reserve_color_bar = False

    @classmethod
    def from_function(
        cls,
        func: Callable[[ArrayLike, ArrayLike], ArrayLike],
        x_axis_range: tuple[float, float],
        y_axis_range: tuple[float, float],
        levels: int | ArrayLike | Inherit = INHERIT,
        color_map: str | Colormap | Inherit = INHERIT,
        color_map_range: Optional[tuple[float, float]] = None,
        show_color_bar: bool | Inherit = INHERIT,
        filled: bool | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
        line_widths: float | ArrayLike | Inherit = INHERIT,
        number_of_points: tuple[int, int] = (500, 500),
    ) -> Self:
        """
        Creates a Contour object from a function.

        Parameters
        ----------
        func : Callable[[ArrayLike, ArrayLike], ArrayLike]
            Function to be plotted. Works with regular functions and lambda functions.
        x_axis_range, y_axis_range : tuple[float, float], optional
            The range of x and y values used for the axes as tuples containing the start and end of the range.
        levels : int | ArrayLike
            If `levels` is an integer, it defines the number of levels to use in the contour.
            If `levels` is an array, it defines the value of each contour level.
            Typical range is ``5`` to ``20`` when given as an integer.
            Default depends on the ``figure_style`` configuration.
        color_map : str or Colormap
            The color map to use for the :class:`~graphinglib.data_plotting_2d.Contour`. Can either be specified as a
            string (named colormap from Matplotlib) or a Colormap object.
            Examples include ``"viridis"``, ``"plasma"``, and ``"coolwarm"``.
            Default depends on the ``figure_style`` configuration.
        color_map_range: tuple[float, float], optional
            The data range covered by the color map, given as ``(minimum, maximum)``.
        show_color_bar : bool
            Whether or not to display the color bar next to the plot.
            Default depends on the ``figure_style`` configuration.
        filled : bool
            Wheter or not to fill the contour with color.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacity of the filled contour.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.
        line_widths : float | ArrayLike
            If the contour is not filled, the width of the contour lines. If an array is provided, it defines the line
            width for each contour level.
            Typical range is ``0.5`` to ``3`` points.
            Default depends on the ``figure_style`` configuration.
        number_of_points : tuple[int, int]
            Number of points in the x and y coordinates.
            Defaults to ``(50, 50)``.

        Returns
        -------
        A :class:`~graphinglib.data_plotting_2d.Contour` object from a function.
        """
        x = np.linspace(x_axis_range[0], x_axis_range[1], number_of_points[0])
        y = np.linspace(y_axis_range[0], y_axis_range[1], number_of_points[1])
        x_mesh, y_mesh = np.meshgrid(x, y)
        z_data = func(x_mesh, y_mesh)
        return cls(
            z_data,
            x_mesh,
            y_mesh,
            levels,
            color_map,
            color_map_range,
            show_color_bar,
            filled,
            alpha,
            line_widths,
        )

    @property
    def x_mesh(self) -> np.ndarray | None:
        return self._x_mesh

    @x_mesh.setter
    def x_mesh(self, x_mesh: ArrayLike | None) -> None:
        self._x_mesh = _stage_live_value(
            self, "_x_mesh", None if x_mesh is None else np.asarray(x_mesh)
        )

    @property
    def y_mesh(self) -> np.ndarray | None:
        return self._y_mesh

    @y_mesh.setter
    def y_mesh(self, y_mesh: ArrayLike | None) -> None:
        self._y_mesh = _stage_live_value(
            self, "_y_mesh", None if y_mesh is None else np.asarray(y_mesh)
        )

    @property
    def z_data(self) -> np.ndarray:
        return self._z_data

    @z_data.setter
    def z_data(self, z_data: ArrayLike) -> None:
        self._z_data = _stage_live_value(self, "_z_data", np.asanyarray(z_data))

    def set_data(
        self,
        z_data: ArrayLike,
        *,
        x_mesh: ArrayLike | None = None,
        y_mesh: ArrayLike | None = None,
    ) -> Self:
        """Assign scalar values and optionally replace their coordinates.

        Omit the coordinates to retain the current mesh. Set both mesh
        properties to None to use the row and column indices.
        """
        grid = np.asanyarray(z_data)
        x = None if x_mesh is None else np.asarray(x_mesh)
        y = None if y_mesh is None else np.asarray(y_mesh)
        self.z_data = grid
        if x is not None:
            self.x_mesh = x
        if y is not None:
            self.y_mesh = y
        return self

    @property
    def levels(self) -> Styled[int | ArrayLike]:
        return self._levels

    @levels.setter
    def levels(self, levels: Styled[int | ArrayLike]) -> None:
        self._levels = _stage_live_value(self, "_levels", levels)

    @property
    def color_map(self) -> Styled[str | Colormap]:
        return self._color_map

    @color_map.setter
    def color_map(self, color_map: Styled[str | Colormap]) -> None:
        self._color_map = _stage_live_value(self, "_color_map", color_map)

    @property
    def color_map_range(self) -> tuple[float, float] | None:
        return self._color_map_range

    @color_map_range.setter
    def color_map_range(self, color_map_range: tuple[float, float] | None) -> None:
        self._color_map_range = _stage_live_value(
            self, "_color_map_range", color_map_range
        )

    @property
    def show_color_bar(self) -> Styled[bool]:
        return self._show_color_bar

    @show_color_bar.setter
    def show_color_bar(self, show_color_bar: Styled[bool]) -> None:
        self._show_color_bar = _stage_live_value(
            self, "_show_color_bar", show_color_bar
        )

    @property
    def filled(self) -> Styled[bool]:
        return self._filled

    @filled.setter
    def filled(self, filled: Styled[bool]) -> None:
        self._filled = _stage_live_value(self, "_filled", filled)

    @property
    def alpha(self) -> Styled[float]:
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: Styled[float]) -> None:
        self._alpha = _stage_live_value(self, "_alpha", alpha)

    @property
    def line_widths(self) -> Styled[float | ArrayLike]:
        return self._line_widths

    @line_widths.setter
    def line_widths(self, line_widths: Styled[float | ArrayLike]) -> None:
        self._line_widths = _stage_live_value(self, "_line_widths", line_widths)

    @property
    def color_bar_params(self) -> dict:
        return self._color_bar_params

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.data_plotting_2d.Contour`.
        """
        return deepcopy(self)

    def set_color_bar_params(
        self,
        label: Optional[str] = None,
        position: Optional[str] = None,
        *,
        reserve: bool = False,
        **color_bar_params,
    ) -> None:
        """
        Sets the color bar parameters.

        Parameters
        ----------
        label : str, optional
            Label of the color bar.
        position : str, optional
            Position of the color bar relative to the ``Figure``. It can be "left",
            "right", "top" or "bottom". This also determines the orientation of the
            color bar (vertical if the color bar is plotted on the "left" or "right",
            horizontal otherwise). If None, the color bar is plotted on the right
            side of the ``Figure``.
            Values are ``"left"``, ``"right"``, ``"top"``, and ``"bottom"``.
        reserve : bool, optional
            Allocate space for a color bar at animation setup, even when
            ``show_color_bar=False``. Set ``show_color_bar=True`` in a callback
            to reveal it. Applies to data with a color scale. Defaults to ``False``.
        **color_bar_params:
            Additional keyword arguments are passed to ``plt.colorbar``.

        Notes
        -----
        During animation, update the label or major tick positions on an
        existing colorbar. Use ``label=""`` to clear its label and
        ``ticks=None`` to restore its tick rule from setup.
        """
        if (session := _bound_session(self)) is not None:
            session.queue_colorbar_params(self, label, position, reserve, color_bar_params)
            return
        self._color_bar_params = color_bar_params
        self._reserve_color_bar = reserve
        if label is not None:
            self._color_bar_params["label"] = label
        if position is not None:
            self._color_bar_params["location"] = position

    def _bind_live_contour(
        self,
        session: _AnimationSession,
        contour: QuadContourSet,
        bar: Colorbar | None,
        filled: bool,
    ) -> None:
        # Stubs include the pre-3.8 contour interface and omit its collection state.
        cont = cast(Any, contour)
        legacy = None if hasattr(cont, "set_paths") else _LegacyContourCollections(cont)
        algorithm = plt.rcParams["contour.algorithm"]
        corner_mask = algorithm != "mpl2005" and plt.rcParams["contour.corner_mask"]
        initial_levels = np.asarray(cont.levels).copy()
        initial_level_request = resolve_or(self._levels, 7)
        default_width = plt.rcParams["contour.linewidth"]
        if default_width is None:
            default_width = plt.rcParams["lines.linewidth"]
        initial_widths = resolve_or(self._line_widths, default_width)
        levels_follow_ticks = self._color_bar_params.get("ticks") is None
        initial_cmap, initial_alpha = cont.get_cmap(), cont.get_alpha()
        limits = cont.get_clim()
        if not all(value is not None for value in limits):
            limits = initial_levels[0], initial_levels[-1]
        initial_grid = np.ma.masked_invalid(self._z_data)
        data_range = (
            (float(initial_grid.min()), float(initial_grid.max()))
            if initial_grid.count()
            else None
        )
        geometry = _ContourGeometry(
            initial_levels,
            np.asarray(cont.layers),
            legacy.paths() if legacy is not None else list(cont.get_paths()),
            data_range,
        )
        current = _ContourFrame(
            geometry,
            filled,
            cast("tuple[float, float]", limits),
            initial_cmap,
            initial_alpha,
            None
            if filled
            else np.asarray(
                [width[0] for width in cont.tlinewidths]
                if legacy is not None
                else cont.get_linewidths()
            ),
        )

        def changed(name: str) -> bool:
            return session.has_frame_update(self, name)

        def prepare_geometry(fill: bool) -> _ContourGeometry:
            values = self._z_data
            if (
                values.ndim != 2
                or min(values.shape) < 2
                or values.dtype.kind not in "biuf"
            ):
                raise InvalidParameterError(
                    "Animated Contour data must be a real, two-dimensional grid with at least two rows and columns."
                )
            grid = np.ma.masked_invalid(values)
            x = y = None
            if self._x_mesh is not None and self._y_mesh is not None:
                x = np.asarray(cont.axes.convert_xunits(self._x_mesh))
                y = np.asarray(cont.axes.convert_yunits(self._y_mesh))
            data_range = (
                (float(grid.min()), float(grid.max()))
                if grid.count()
                else current.geometry.data_range
            )
            specification = resolve_or(self._levels, initial_level_request)
            first_finite_grid = (
                current.geometry.data_range is None and data_range is not None
            )
            levels = current.geometry.levels
            if changed("_levels") or (
                first_finite_grid and isinstance(specification, Integral)
            ):
                levels = _contour_levels(specification, data_range or (0.0, 1.0), fill)
            if fill and levels.size < 2:
                raise InvalidParameterError(
                    "Filled contours require at least two levels."
                )
            values = 0.5 * levels[:-1] + 0.5 * levels[1:] if fill else levels
            paths = _contour_path_groups(
                x,
                y,
                grid,
                levels,
                filled=fill,
                algorithm=algorithm,
                corner_mask=corner_mask,
            )
            return _ContourGeometry(levels, values, paths, data_range)

        def set_frame(
            state: _ContourFrame, geometry_changed: bool, bar_changed: bool
        ) -> None:
            nonlocal current
            geometry = state.geometry
            with (
                cont.callbacks.blocked(signal="changed"),
                cont.norm.callbacks.blocked(signal="changed"),
            ):
                if geometry_changed:
                    if legacy is None:
                        cont.set_paths(geometry.paths)
                    else:
                        legacy.set_paths(geometry.paths)
                        cont.set_array(geometry.values)
                    cont.levels = geometry.levels
                    cont._levels = geometry.levels
                    cont.layers = cont.cvalues = geometry.values
                if cont.filled != state.filled:
                    cont.filled = state.filled
                    if legacy is None:
                        cont.set_facecolor(None if state.filled else "none")
                        cont.set_edgecolor("none" if state.filled else None)
                lo, hi = geometry.data_range or (0.0, 1.0)
                tolerance = 1e-15 * (lo - hi)
                dashes = (
                    [
                        cont.negative_linestyles
                        if getattr(state.cmap, "monochrome", False)
                        and level < tolerance
                        else "solid"
                        for level in geometry.levels
                    ]
                    if not state.filled
                    else []
                )
                if legacy is not None:
                    legacy.style(state, dashes)
                cont.set_cmap(state.cmap)
                cont.set_clim(*state.clim)
                cont.set_alpha(state.alpha)
                if legacy is None and state.widths is not None:
                    cont.set_linewidth(state.widths)
                    cont.set_linestyle(dashes)
            if bar is not None and bar_changed:
                _set_contour_bar_mode(bar, state.filled)
                bar.boundaries = _contour_bar_boundaries(geometry.levels).tolist()
                bar.values = geometry.values.tolist()
                if levels_follow_ticks and not isinstance(
                    bar.locator, _AnimatedColorbarLocator
                ):
                    bar.locator = FixedLocator(geometry.levels.tolist(), nbins=10)
                cont.changed()
            else:
                with cont.callbacks.blocked(signal="changed"):
                    cont.changed()
            current = state

        def prepare() -> _ArtistEdit:
            fill = resolve_or(self._filled, filled)
            if not isinstance(fill, (bool, np.bool_)):
                raise InvalidParameterError("Contour.filled must be a boolean.")
            geometry_changed = fill != current.filled or any(
                changed(name) for name in ("_z_data", "_x_mesh", "_y_mesh", "_levels")
            )
            geometry = prepare_geometry(fill) if geometry_changed else current.geometry
            levels = geometry.levels
            clim = current.clim
            levels_changed = levels is not current.geometry.levels
            if levels_changed or changed("_color_map_range"):
                explicit = self._color_map_range
                if explicit is None:
                    clim = float(levels[0]), float(levels[-1])
                else:
                    bounds = np.asarray(explicit)
                    if (
                        bounds.shape != (2,)
                        or bounds.dtype.kind not in "iuf"
                        or not np.isfinite(bounds).all()
                        or bounds[0] == bounds[1]
                    ):
                        raise InvalidParameterError(
                            "Contour color_map_range must contain two distinct finite numbers."
                        )
                    clim = float(bounds.min()), float(bounds.max())
            state = _ContourFrame(
                geometry,
                fill,
                clim,
                plt.get_cmap(resolve_or(self._color_map, initial_cmap)),
                resolve_or(self._alpha, initial_alpha),
                None
                if fill
                else _contour_line_widths(
                    resolve_or(self._line_widths, initial_widths), len(levels)
                ),
            )
            previous = current
            bar_changed = (
                levels_changed
                or fill != previous.filled
                or any(
                    changed(name)
                    for name in (
                        "_color_map_range",
                        "_color_map",
                        "_alpha",
                        "_line_widths",
                    )
                )
            )
            pool_size = 0 if legacy is None else len(legacy.pool)

            def undo() -> None:
                try:
                    set_frame(previous, geometry_changed, bar_changed)
                finally:
                    if legacy is not None:
                        legacy.truncate(pool_size)

            return _ArtistEdit(lambda: set_frame(state, geometry_changed, bar_changed), undo)

        properties = (
            "_z_data",
            "_x_mesh",
            "_y_mesh",
            "_levels",
            "_alpha",
            "_color_map",
            "_color_map_range",
            "_filled",
            "_line_widths",
        )
        session.bind_derived(self, properties, prepare)

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified
        `Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_.
        """
        session = _DRAW_SESSION.get()
        if session is not None and (
            self._z_data.ndim != 2
            or min(self._z_data.shape) < 2
            or self._z_data.dtype.kind not in "biuf"
        ):
            raise InvalidParameterError(
                "Animated Contour data must be a real, two-dimensional grid with at least two rows and columns."
            )
        if self._x_mesh is None or self._y_mesh is None:
            x_mesh, y_mesh = np.meshgrid(
                np.arange(self._z_data.shape[1]),
                np.arange(self._z_data.shape[0]),
            )
        else:
            x_mesh = self._x_mesh
            y_mesh = self._y_mesh
        filled = resolve_or(self._filled, True)
        params = {
            "levels": self._levels,
            "cmap": self._color_map,
            "alpha": self._alpha,
            "linewidths": self._line_widths if not filled else None,
        }
        if self._color_map_range is not None:
            params["vmin"] = min(self._color_map_range)
            params["vmax"] = max(self._color_map_range)

        params = strip_inherit(params)
        if session is not None:
            levels = params.get("levels")
            if (
                levels is None or isinstance(levels, Integral)
            ) and not np.ma.masked_invalid(self._z_data).count():
                params["levels"] = _contour_levels(
                    7 if levels is None else levels, (0.0, 1.0), filled
                )
        if filled:
            cont = axes.contourf(
                x_mesh,
                y_mesh,
                self._z_data,
                zorder=z_order,
                **params,
            )
        else:
            cont = axes.contour(
                x_mesh,
                y_mesh,
                self._z_data,
                zorder=z_order,
                **params,
            )
        self.handle = cont
        bar = None
        show_colorbar = resolve_or(self._show_color_bar, True)
        if show_colorbar or (session is not None and self._reserve_color_bar):
            fig = axes.get_figure()
            assert fig is not None
            levels = np.asarray(cont.levels)
            if session is not None and not filled and levels.size == 1:
                # Only the colorbar needs a nonzero span for a single line.
                live_contour = cast(Any, cont)
                original = live_contour._levels
                live_contour._levels = _contour_bar_boundaries(levels)
                try:
                    bar = fig.colorbar(cont, ax=axes, **self._color_bar_params)
                finally:
                    live_contour._levels = original
            else:
                bar = fig.colorbar(cont, ax=axes, **self._color_bar_params)
        if session is not None:
            session.mark_present(self)
            if bar is not None:
                session.bind_colorbar(
                    self,
                    bar,
                    initially_visible=show_colorbar,
                    default_locator=(
                        lambda: FixedLocator(
                            np.asarray(cont.levels).tolist(), nbins=10
                        )
                    )
                    if self._color_bar_params.get("ticks") is None
                    else None,
                )
            self._bind_live_contour(session, cont, bar, filled)


@dataclass
class Stream(Plottable2D):
    """
    This class implements stream plots.

    Parameters
    ----------
    x_data, y_data : ArrayLike
        x and y coordinates of the vectors as a mesh grid.
    u_data, v_data : ArrayLike
        Magnitudes of the vectors for each point of the mesh grid.
    density : float or tuple[float, float]
        Density of stream lines. Can be specified independently for the x and y coordinates
        by specifying a density tuple instead. Defaults to 1.
        Typical range is ``0.5`` to ``3``.
    line_width : float or ArrayLike
        Width of the stream lines. An array specifies widths at grid points.
        Default depends on the ``figure_style`` configuration.
        Typical range is ``0.5`` to ``3`` points.
    color : str or ArrayLike
        Color of the stream lines. If an array of intensities is provided, the values are mapped to the specified color map.
        Default depends on the ``figure_style`` configuration.
    color_map : str or Colormap
        Color map of the stream lines, to be used in combination with the color parameter to specify intensity.
        Examples include ``"viridis"``, ``"plasma"``, and ``"coolwarm"``.
        Default depends on the ``figure_style`` configuration.
    arrow_size : float
        Arrow size multiplier. Default depends on the ``figure_style`` configuration.
        Typical range is ``0.5`` to ``3``.

    Notes
    -----
    Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
    (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
    values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``). They may also be arrays of
    intensity values, which are mapped through ``color_map``.
    """

    def __init__(
        self,
        x_data: ArrayLike,
        y_data: ArrayLike,
        u_data: ArrayLike,
        v_data: ArrayLike,
        density: float | tuple[float, float] = 1,
        line_width: float | ArrayLike | Inherit = INHERIT,
        color: str | ArrayLike | Inherit = INHERIT,
        color_map: str | Colormap | Inherit = INHERIT,
        arrow_size: float | Inherit = INHERIT,
    ) -> None:
        """
        This class implements stream plots.

        Parameters
        ----------
        x_data, y_data : ArrayLike
            x and y coordinates of the vectors as a mesh grid.
        u_data, v_data : ArrayLike
            Magnitudes of the vectors for each point of the mesh grid.
        density : float or tuple[float, float]
            Density of stream lines. Can be specified independently for the x and y coordinates
            by specifying a density tuple instead. Defaults to 1.
            Typical range is ``0.5`` to ``3``.
        line_width : float or ArrayLike
            Width of the stream lines. Default depends on the ``figure_style`` configuration.
            Typical range is ``0.5`` to ``3`` points.
        color : str or ArrayLike
            Color of the stream lines. If an array of intensities is provided, the values are mapped to the specified color map.
            Default depends on the ``figure_style`` configuration.
        color_map : str or Colormap
            Color map of the stream lines, to be used in combination with the color parameter to specify intensity.
            Examples include ``"viridis"``, ``"plasma"``, and ``"coolwarm"``.
            Default depends on the ``figure_style`` configuration.
        arrow_size : float
            Arrow size multiplier. Default depends on the ``figure_style`` configuration.
            Typical range is ``0.5`` to ``3``.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``). They may also be arrays of
        intensity values, which are mapped through ``color_map``.
        """
        self._x_data = np.asarray(x_data)
        self._y_data = np.asarray(y_data)
        self._u_data = np.ma.asarray(u_data)
        self._v_data = np.ma.asarray(v_data)
        self._density = density
        self._line_width = line_width
        self._color = color
        self._color_map = color_map
        self._arrow_size = arrow_size

    @classmethod
    def from_function(
        cls,
        func: Callable[[ArrayLike, ArrayLike], tuple[ArrayLike, ArrayLike]],
        x_axis_range: tuple[float, float],
        y_axis_range: tuple[float, float],
        number_of_points_x: int = 30,
        number_of_points_y: int = 30,
        density: float | tuple[float, float] = 1,
        line_width: float | ArrayLike | Inherit = INHERIT,
        color: str | ArrayLike | Inherit = INHERIT,
        color_map: str | Colormap | Inherit = INHERIT,
        arrow_size: float | Inherit = INHERIT,
    ) -> Self:
        """
        Creates a :class:`~graphinglib.data_plotting_2d.Stream` from a function.

        Parameters
        ----------
        func : Callable[[ArrayLike, ArrayLike], [ArrayLike, ArrayLike]]
            Function to be plotted. Works with regular functions and lambda functions.
        x_axis_range, y_axis_range : tuple[float, float], optional
            The range of x and y values used for the axes as tuples containing the start and end of the range.
        number_of_points_x, number_of_points_y : int
            Number of points to fill the x and y ranges.
            Defaults to 30.
        density : float or tuple[float, float]
            Density of stream lines. Can be specified independently for the x and y coordinates
            by specifying a density tuple instead. Defaults to 1.
            Typical range is ``0.5`` to ``3``.
        line_width : float or ArrayLike
            Width of the stream lines. Default depends on the ``figure_style`` configuration.
            Typical range is ``0.5`` to ``3`` points.
        color : str
            Color of the stream lines. Default depends on the ``figure_style`` configuration.
        color_map : str or Colormap
            Color map of the stream lines. Default depends on the ``figure_style`` configuration.
            Examples include ``"viridis"``, ``"plasma"``, and ``"coolwarm"``.
        arrow_size : float
            Arrow size multiplier. Default depends on the ``figure_style`` configuration.
            Typical range is ``0.5`` to ``3``.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``). They may also be arrays of
        intensity values, which are mapped through ``color_map``.

        Returns
        -------
        A :class:`~graphinglib.data_plotting_2d.Stream` object from a function.
        """
        x = np.linspace(x_axis_range[0], x_axis_range[1], number_of_points_x)
        y = np.linspace(y_axis_range[0], y_axis_range[1], number_of_points_y)
        x_grid, y_grid = np.meshgrid(x, y)
        u, v = func(x_grid, y_grid)
        return cls(x, y, u, v, density, line_width, color, color_map, arrow_size)

    @property
    def x_data(self) -> np.ndarray:
        return self._x_data

    @x_data.setter
    def x_data(self, value: ArrayLike) -> None:
        self._x_data = _stage_live_value(self, "_x_data", np.array(value, copy=True))

    @property
    def y_data(self) -> np.ndarray:
        return self._y_data

    @y_data.setter
    def y_data(self, value: ArrayLike) -> None:
        self._y_data = _stage_live_value(self, "_y_data", np.array(value, copy=True))

    @property
    def u_data(self) -> np.ndarray:
        return self._u_data

    @u_data.setter
    def u_data(self, value: ArrayLike) -> None:
        self._u_data = _stage_live_value(self, "_u_data", np.ma.array(value, copy=True))

    @property
    def v_data(self) -> np.ndarray:
        return self._v_data

    @v_data.setter
    def v_data(self, value: ArrayLike) -> None:
        self._v_data = _stage_live_value(self, "_v_data", np.ma.array(value, copy=True))

    def set_data(
        self,
        u_data: ArrayLike,
        v_data: ArrayLike,
        *,
        x_data: ArrayLike | None = None,
        y_data: ArrayLike | None = None,
    ) -> Self:
        """Assign both velocity components and optionally replace the coordinate grid.

        Components must match the grid. During animation the final field is
        integrated once per occurrence, even when several setters change it.
        """
        x = self._x_data if x_data is None else np.array(x_data, copy=True)
        y = self._y_data if y_data is None else np.array(y_data, copy=True)
        u, v = np.ma.array(u_data, copy=True), np.ma.array(v_data, copy=True)
        session = _bound_session(self)
        if session is not None:
            previous = self._x_data, self._y_data, self._u_data, self._v_data

            def restore() -> None:
                self._x_data, self._y_data, self._u_data, self._v_data = previous

            session.queue_property(self, "field", None, restore)
        self._x_data, self._y_data, self._u_data, self._v_data = x, y, u, v
        return self

    @property
    def density(self) -> float | tuple[float, float]:
        return self._density

    @density.setter
    def density(self, value: float | tuple[float, float]) -> None:
        self._density = _stage_live_value(self, "_density", value)

    @property
    def arrow_size(self) -> Styled[float]:
        return self._arrow_size

    @arrow_size.setter
    def arrow_size(self, value: Styled[float]) -> None:
        self._arrow_size = _stage_live_value(self, "_arrow_size", value)

    @property
    def color_map(self) -> Styled[str | Colormap]:
        return self._color_map

    @color_map.setter
    def color_map(self, value: Styled[str | Colormap]) -> None:
        self._color_map = _stage_live_value(self, "_color_map", value)

    @property
    def color(self) -> Styled[str | ArrayLike]:
        """Color of the streamlines and their arrowheads."""
        return self._color

    @color.setter
    def color(self, color: Styled[str | ArrayLike]) -> None:
        self._color = _stage_live_value(self, "_color", color)

    @property
    def line_width(self) -> Styled[float | ArrayLike]:
        """Width of the streamlines and their arrowheads."""
        return self._line_width

    @line_width.setter
    def line_width(self, line_width: Styled[float | ArrayLike]) -> None:
        self._line_width = _stage_live_value(self, "_line_width", line_width)

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.data_plotting_2d.Stream`.
        """
        return deepcopy(self)

    def _field(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self._x_data, self._y_data, self._u_data, self._v_data

    def _draw_stream(
        self, axes: plt.Axes, z_order: int, params: dict[str, Any]
    ) -> StreamplotSet:
        x, y, u, v = self._field()
        try:
            return axes.streamplot(x, y, u, v, zorder=z_order, **params)
        except Exception as exc:
            raise PlottingError(
                f"Could not draw the Stream ({exc}). Check that the grid is evenly "
                "spaced and that u and v have shape (len(y), len(x))."
            ) from exc

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs) -> None:
        params = strip_inherit(
            {
                "density": self._density,
                "linewidth": self._line_width,
                "cmap": self._color_map,
                "arrowsize": self._arrow_size,
                "color": self._color,
            }
        )
        session = _DRAW_SESSION.get()
        patch_start = len(axes.patches) if session is not None else 0
        result = self._draw_stream(axes, z_order, params)
        if session is None:
            return

        lines = result.lines
        pool = [
            patch
            for patch in axes.patches[patch_start:]
            if isinstance(patch, FancyArrowPatch)
        ]
        field = self._field()
        density = self._density
        scratch = None
        initial_color = params.get("color")
        initial_colors = np.asarray(lines.get_color())
        default_color = tuple(initial_colors[0]) if initial_colors.size else "C0"
        default_width = params.get("linewidth")
        if default_width is None:
            default_width = plt.rcParams["lines.linewidth"]
        initial_map = params.get("cmap") or plt.rcParams["image.cmap"]
        initial_size = params.get("arrowsize", 1.0)
        lo, hi = lines.get_clim()
        color_range = (lo, hi) if lo is not None and hi is not None else (0.0, 1.0)

        def prepare() -> _ArtistEdit:
            nonlocal scratch
            color = resolve_or(self._color, initial_color)
            if color is None:
                color = default_color
            widths = np.asarray(
                resolve_or(self._line_width, default_width), dtype=float
            )
            cmap = resolve_or(self._color_map, initial_map)
            size = resolve_or(self._arrow_size, initial_size)
            new_field = self._field()
            new_density = self._density
            if not np.isfinite(size) or size < 0:
                raise InvalidParameterError(
                    "Stream arrow_size must be finite and nonnegative."
                )
            if (
                not np.isfinite(new_density).all()
                or (np.asarray(new_density) <= 0).any()
            ):
                raise InvalidParameterError(
                    "Stream density must be finite and positive."
                )
            if not np.isfinite(widths).all() or (widths < 0).any():
                raise InvalidParameterError(
                    "Stream line widths must be finite and nonnegative."
                )
            width = float(widths.item()) if widths.ndim == 0 else widths
            uniform_color = is_color_like(color)
            same_field = all(
                before is after for before, after in zip(field, new_field)
            ) and np.array_equal(density, new_density)
            if (
                same_field
                and uniform_color
                and isinstance(width, float)
                and lines.get_array() is None
                and np.size(lines.get_linewidth()) == 1
            ):
                # Uniform styling does not require integrating the vector field again.
                return _stream_style_edit(lines, pool, color, width, size)

            if scratch is None:
                scratch = Figure().add_axes((0, 0, 1, 1))
            generated = self._draw_stream(
                scratch,
                z_order,
                {
                    "density": new_density,
                    "linewidth": width,
                    "cmap": cmap,
                    "arrowsize": size,
                    "color": to_rgba(cast(Any, color))
                    if uniform_color
                    else np.ma.asarray(color),
                    "norm": Normalize(*color_range),
                },
            )
            scratch_arrows = list(scratch.patches)
            try:
                previous_lines = _stream_line_state(lines)
                next_lines = _stream_line_state(generated.lines)
                previous_arrows = [_StreamArrow.capture(patch) for patch in pool]
                next_arrows = [
                    _StreamArrow.capture(cast(FancyArrowPatch, patch))
                    for patch in scratch_arrows
                ]
            finally:
                generated.lines.remove()
                for arrow in scratch_arrows:
                    arrow.remove()

            previous_field, previous_density = field, density

            def apply() -> None:
                nonlocal field, density
                _set_stream_line_state(lines, next_lines)
                for index, arrow in enumerate(next_arrows):
                    if index == len(pool):
                        patch = FancyArrowPatch(
                            *arrow.endpoints,
                            arrowstyle="-|>",
                            transform=axes.transData,
                            zorder=z_order,
                        )
                        # add_artist preserves the manual view and the axes' data limits.
                        axes.add_artist(patch)
                        pool.append(patch)
                    arrow.apply(pool[index])
                for patch in pool[len(next_arrows) :]:
                    patch.set_visible(False)
                field, density = new_field, new_density

            def undo() -> None:
                nonlocal field, density
                _set_stream_line_state(lines, previous_lines)
                for patch in pool[len(previous_arrows) :]:
                    patch.remove()
                del pool[len(previous_arrows) :]
                for previous, patch in zip(previous_arrows, pool):
                    previous.apply(patch)
                field, density = previous_field, previous_density

            return _ArtistEdit(apply, undo)

        session.bind_derived(
            self,
            (
                "field",
                "_x_data",
                "_y_data",
                "_u_data",
                "_v_data",
                "_density",
                "_color",
                "_line_width",
                "_color_map",
                "_arrow_size",
            ),
            prepare,
        )


@dataclass
class _StreamArrow:
    endpoints: tuple[Any, Any]
    edge: Any
    face: Any
    width: float
    scale: float
    visible: bool

    @classmethod
    def capture(cls, patch: FancyArrowPatch) -> Self:
        # FancyArrowPatch exposes set_positions, but has no public endpoint getter.
        ends = cast(Any, patch)._posA_posB
        return cls(
            (ends[0], ends[1]),
            patch.get_edgecolor(),
            patch.get_facecolor(),
            patch.get_linewidth(),
            patch.get_mutation_scale(),
            patch.get_visible(),
        )

    def apply(self, patch: FancyArrowPatch) -> None:
        patch.set_positions(*self.endpoints)
        patch.set_edgecolor(self.edge)
        patch.set_facecolor(self.face)
        patch.set_linewidth(self.width)
        patch.set_mutation_scale(self.scale)
        patch.set_visible(self.visible)


def _stream_line_state(lines: LineCollection) -> tuple[Any, ...]:
    return (
        lines.get_segments(),
        lines.get_array(),
        lines.get_cmap(),
        lines.norm,
        lines.get_color(),
        lines.get_linewidth(),
    )


def _set_stream_line_state(lines: LineCollection, state: tuple[Any, ...]) -> None:
    segments, mapped, cmap, norm, color, width = state
    lines.set_segments(segments)
    lines.set_cmap(cmap)
    lines.set_norm(norm)
    lines.set_array(mapped)
    lines.set_color(color)
    lines.set_linewidth(width)


def _stream_style_edit(
    lines: LineCollection,
    arrows: list[FancyArrowPatch],
    color: Any,
    width: float,
    size: float,
) -> _ArtistEdit:
    original_line = lines.get_color(), lines.get_linewidth()
    original_arrows = [_StreamArrow.capture(arrow) for arrow in arrows]

    def apply() -> None:
        lines.set_color(color)
        lines.set_linewidth(width)
        for arrow in arrows:
            arrow.set_color(color)
            arrow.set_linewidth(width)
            arrow.set_mutation_scale(10 * size)

    def undo() -> None:
        old_color, old_width = original_line
        lines.set_color(old_color)
        lines.set_linewidth(old_width)
        for state, arrow in zip(original_arrows, arrows):
            state.apply(arrow)

    return _ArtistEdit(apply, undo)
