from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, cast, runtime_checkable

import contourpy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap, Normalize, is_color_like, to_rgba
from matplotlib.figure import Figure
from matplotlib.image import imread
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath
from matplotlib.streamplot import StreamplotSet
from numpy.typing import ArrayLike
from scipy.interpolate import griddata

from ._animation import (
    _DRAW_SESSION,
    _ArtistEdit,
    _bound_session,
    _edit,
    _reject_live_update,
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
            data = np.asarray(image)
        is_rgb = data.ndim == 3 and data.shape[-1] in (3, 4)
        if data.ndim != 2 and not is_rgb:
            raise InvalidParameterError(
                "image must be a 2D array of values or a 3D array of RGB(A) pixels "
                f"(last axis of size 3 or 4), but got an array of shape {data.shape}."
            )
        session = _bound_session(self)
        if session is not None:
            data = data.copy()
            previous = self._image
            session.queue_heatmap(self, data, lambda: setattr(self, "_image", previous))
        elif from_file or is_rgb:
            # RGB(A) pixels and file-backed images disable the colorbar for static plots.
            self._show_color_bar = False
        self._image = data

    @property
    def x_axis_range(self) -> Optional[tuple[float, float]]:
        return self._x_axis_range

    @x_axis_range.setter
    def x_axis_range(self, x_axis_range: Optional[tuple[float, float]]) -> None:
        self._check_animation_geometry()
        self._x_axis_range = x_axis_range

    @property
    def y_axis_range(self) -> Optional[tuple[float, float]]:
        return self._y_axis_range

    @y_axis_range.setter
    def y_axis_range(self, y_axis_range: Optional[tuple[float, float]]) -> None:
        self._check_animation_geometry()
        self._y_axis_range = y_axis_range

    @property
    def x_mesh(self) -> Optional[ArrayLike]:
        return self._x_mesh

    @x_mesh.setter
    def x_mesh(self, x_mesh: Optional[ArrayLike]) -> None:
        self._check_animation_geometry()
        self._x_mesh = None if x_mesh is None else np.asarray(x_mesh)

    @property
    def y_mesh(self) -> Optional[ArrayLike]:
        return self._y_mesh

    @y_mesh.setter
    def y_mesh(self, y_mesh: Optional[ArrayLike]) -> None:
        self._check_animation_geometry()
        self._y_mesh = None if y_mesh is None else np.asarray(y_mesh)

    def _check_animation_geometry(self) -> None:
        session = _bound_session(self)
        if session is not None:
            session.reject_structural_update(
                self, "Configure the Heatmap grid and color scale before animating."
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
        self._check_animation_geometry()
        self._show_color_bar = show_color_bar

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
        _reject_live_update(
            self, "Configure the Heatmap aspect and origin before animating."
        )
        self._aspect_ratio = aspect_ratio

    @property
    def origin_position(self) -> Styled[str]:
        return self._origin_position

    @origin_position.setter
    def origin_position(self, origin_position: Styled[str]) -> None:
        _reject_live_update(
            self, "Configure the Heatmap aspect and origin before animating."
        )
        self._origin_position = origin_position

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
        **color_bar_params:
            Additional keyword arguments are passed to ``plt.colorbar`` call.
        """
        _reject_live_update(
            self, "Configure the Heatmap colorbar layout before animating."
        )
        self._color_bar_params = color_bar_params
        if label is not None:
            self._color_bar_params["label"] = label
        if position is not None:
            self._color_bar_params["location"] = position

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
        if resolve_or(self._show_color_bar, True):
            fig = axes.get_figure()
            assert fig is not None
            fig.colorbar(image, ax=axes, **self._color_bar_params)
        session = _DRAW_SESSION.get()
        if session is not None:
            session.bind_heatmap(self, image, self._image.shape)
            session.bind_property(self, "_alpha", image, "alpha")
            if self._image.ndim == 2:
                session.bind_property(self, "_color_map", image, "cmap")

                def prepare_range(value: tuple[float, float] | None) -> _ArtistEdit:
                    if (
                        value is None
                        or len(value) != 2
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

                    return _edit(set_limits, image.get_clim(), tuple(sorted(value)))

                if self._norm is None or type(self._norm) is Normalize:
                    session.bind_update(self, "_color_map_range", prepare_range)
            if not use_pcolormesh:
                session.bind_property(self, "_interpolation", image, "interpolation")


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
        self._u_data = np.asarray(u_data)
        self._v_data = np.asarray(v_data)
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
        _reject_live_update(
            self, "Configure the VectorField grid and arrow geometry before animating."
        )
        self._x_data = np.asarray(x_data)

    @property
    def y_data(self) -> ArrayLike:
        return self._y_data

    @y_data.setter
    def y_data(self, y_data: ArrayLike) -> None:
        _reject_live_update(
            self, "Configure the VectorField grid and arrow geometry before animating."
        )
        self._y_data = np.asarray(y_data)

    @property
    def u_data(self) -> ArrayLike:
        return self._u_data

    @u_data.setter
    def u_data(self, u_data: ArrayLike) -> None:
        self._u_data = _stage_live_value(self, "_u_data", np.asarray(u_data))

    @property
    def v_data(self) -> ArrayLike:
        return self._v_data

    @v_data.setter
    def v_data(self, v_data: ArrayLike) -> None:
        self._v_data = _stage_live_value(self, "_v_data", np.asarray(v_data))

    @property
    def arrow_width(self) -> Styled[float]:
        return self._arrow_width

    @arrow_width.setter
    def arrow_width(self, arrow_width: float) -> None:
        _reject_live_update(
            self, "Configure the VectorField grid and arrow geometry before animating."
        )
        self._arrow_width = arrow_width

    @property
    def arrow_head_size(self) -> Styled[float]:
        return self._arrow_head_size

    @arrow_head_size.setter
    def arrow_head_size(self, arrow_head_size: float) -> None:
        _reject_live_update(
            self, "Configure the VectorField grid and arrow geometry before animating."
        )
        self._arrow_head_size = arrow_head_size

    @property
    def scale(self) -> Optional[float]:
        return self._scale

    @scale.setter
    def scale(self, scale: Optional[float]) -> None:
        _reject_live_update(
            self, "Configure the VectorField grid and arrow geometry before animating."
        )
        self._scale = scale

    @property
    def make_angles_axes_independent(self) -> bool:
        return self._make_angles_axes_independent

    @make_angles_axes_independent.setter
    def make_angles_axes_independent(self, value: bool) -> None:
        _reject_live_update(
            self, "Configure the VectorField grid and arrow geometry before animating."
        )
        self._make_angles_axes_independent = value

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
        artist = axes.quiver(
            self._x_data,
            self._y_data,
            self._u_data,
            self._v_data,
            zorder=z_order,
            **params,
        )
        session = _DRAW_SESSION.get()
        if session is not None:
            u, v = np.ma.asarray(self._u_data), np.ma.asarray(self._v_data)
            expected = u.shape, v.shape

            def prepare(_value: np.ndarray) -> _ArtistEdit:
                old_u, old_v = u, v

                def set_vectors(new_u: np.ndarray, new_v: np.ndarray) -> None:
                    nonlocal u, v
                    artist.set_UVC(new_u, new_v)
                    u, v = new_u, new_v

                def apply() -> None:
                    new_u, new_v = (
                        np.ma.asarray(self._u_data),
                        np.ma.asarray(self._v_data),
                    )
                    if (new_u.shape, new_v.shape) != expected:
                        raise InvalidParameterError(
                            "An animated VectorField must retain its component shapes."
                        )
                    set_vectors(new_u, new_v)

                return _ArtistEdit(apply, lambda: set_vectors(old_u, old_v))

            session.bind_update(self, "_u_data", prepare)
            session.bind_update(self, "_v_data", prepare)
            session.bind_property(self, "_color", artist, "facecolor")


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
        _reject_live_update(
            self, "Configure Contour grid, levels and colorbar before animating."
        )
        self._x_mesh = None if x_mesh is None else np.asarray(x_mesh)

    @property
    def y_mesh(self) -> np.ndarray | None:
        return self._y_mesh

    @y_mesh.setter
    def y_mesh(self, y_mesh: ArrayLike | None) -> None:
        _reject_live_update(
            self, "Configure Contour grid, levels and colorbar before animating."
        )
        self._y_mesh = None if y_mesh is None else np.asarray(y_mesh)

    @property
    def z_data(self) -> np.ndarray:
        return self._z_data

    @z_data.setter
    def z_data(self, z_data: ArrayLike) -> None:
        self._z_data = _stage_live_value(self, "_z_data", np.asarray(z_data))

    @property
    def levels(self) -> Styled[int | ArrayLike]:
        return self._levels

    @levels.setter
    def levels(self, levels: Styled[int | ArrayLike]) -> None:
        _reject_live_update(
            self, "Configure Contour grid, levels and colorbar before animating."
        )
        self._levels = levels

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
        _reject_live_update(
            self, "Configure Contour grid, levels and colorbar before animating."
        )
        self._color_map_range = color_map_range

    @property
    def show_color_bar(self) -> Styled[bool]:
        return self._show_color_bar

    @show_color_bar.setter
    def show_color_bar(self, show_color_bar: Styled[bool]) -> None:
        _reject_live_update(
            self, "Configure Contour grid, levels and colorbar before animating."
        )
        self._show_color_bar = show_color_bar

    @property
    def filled(self) -> Styled[bool]:
        return self._filled

    @filled.setter
    def filled(self, filled: Styled[bool]) -> None:
        _reject_live_update(
            self, "Configure Contour grid, levels and colorbar before animating."
        )
        self._filled = filled

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
        **color_bar_params:
            Additional keyword arguments are passed to ``plt.colorbar`` call.
        """
        _reject_live_update(self, "Configure the Contour colorbar before animating.")
        self._color_bar_params = color_bar_params
        if label is not None:
            self._color_bar_params["label"] = label
        if position is not None:
            self._color_bar_params["location"] = position

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified
        `Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_.
        """
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
        if resolve_or(self._show_color_bar, True):
            fig = axes.get_figure()
            assert fig is not None
            fig.colorbar(cont, ax=axes, **self._color_bar_params)
        session = _DRAW_SESSION.get()
        if session is not None:
            session.mark_present(self)
            # Matplotlib 3.8 moved all levels into a single collection with a public path API.
            if hasattr(cont, "set_paths"):
                shape = self._z_data.shape
                levels = np.asarray(cont.levels).copy()
                algorithm = plt.rcParams["contour.algorithm"]
                corner_mask = (
                    algorithm != "mpl2005" and plt.rcParams["contour.corner_mask"]
                )

                def prepare(values: np.ndarray) -> _ArtistEdit:
                    if values.shape != shape or values.dtype.kind not in "biuf":
                        raise InvalidParameterError(
                            "An animated Contour must retain its grid shape and real numeric values."
                        )
                    grid = np.ma.masked_invalid(values)
                    generator = contourpy.contour_generator(
                        x_mesh,
                        y_mesh,
                        grid,
                        name=algorithm,
                        corner_mask=corner_mask,
                        line_type=contourpy.LineType.SeparateCode,
                        fill_type=contourpy.FillType.OuterCode,
                    )
                    if filled:
                        lower = levels[:-1].copy()
                        if grid.count() and grid.min() == lower[0]:
                            lower[0] -= 1
                        groups = (
                            generator.filled(lo, hi)
                            for lo, hi in zip(lower, levels[1:])
                        )
                    else:
                        groups = (generator.lines(level) for level in levels)
                    paths = [
                        MplPath(np.concatenate(vertices), np.concatenate(codes))
                        if vertices
                        else MplPath(np.empty((0, 2)))
                        for vertices, codes in cast(
                            "Iterable[tuple[list[np.ndarray], list[np.ndarray]]]",
                            groups,
                        )
                    ]
                    return _edit(cont.set_paths, cont.get_paths(), paths)

                session.bind_update(self, "_z_data", prepare)
                session.bind_property(self, "_alpha", cont, "alpha")
                session.bind_property(self, "_color_map", cont, "cmap")
                if not filled:
                    session.bind_property(self, "_line_widths", cont, "linewidth")


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
