from matplotlib.artist import Artist
from matplotlib.collections import LineCollection
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerLineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.transforms import Transform
from numpy import array, full_like


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
            try:
                lw = orig_handle.get_linewidths()[i]
            except IndexError:
                lw = orig_handle.get_linewidths()[0]
            if dashes[1] is not None:
                line.set_dashes(dashes[1])
            line.set_color(color)
            line.set_transform(trans)
            line.set_linewidth(lw)
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
            try:
                lw = orig_handle.get_linewidths()[i]
            except IndexError:
                lw = orig_handle.get_linewidths()[0]
            if dashes[1] is not None:
                line.set_dashes(dashes[1])
            line.set_color(color)
            line.set_transform(trans)
            line.set_linewidth(lw)
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
