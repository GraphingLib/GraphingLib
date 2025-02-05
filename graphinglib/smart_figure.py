from shutil import which
from typing import Literal, Optional
from warnings import warn

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Polygon
from matplotlib.gridspec import GridSpec

from graphinglib.file_manager import (
    FileLoader,
    FileUpdater,
    get_default_style,
)
from graphinglib.graph_elements import GraphingException, Plottable
from graphinglib.legend_artists import (
    HandlerMultipleLines,
    HandlerMultipleVerticalLines,
    VerticalLineCollection,
    histogram_legend_artist,
)


class SmartFigure:
    def __init__(
        self,
        num_rows: int = 1,
        num_cols: int = 1,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        elements: Optional[list[Plottable]] = None,
    ) -> None:
        self._num_rows = num_rows
        self._num_cols = num_cols
        self._x_label = x_label
        self._y_label = y_label
        self._elements = elements
        self._figure = None
        self._subfigs = None

    def show(
        self
    ) -> None:
        self._figure = plt.figure(layout="constrained")
        self._prepare_figure()
        plt.show()
        plt.rcParams.update(plt.rcParamsDefault)

    def _prepare_figure(
        self,
    ) -> None:
        gridspec = self._figure.add_gridspec(self._num_rows, self._num_cols)
        self._subfigs = [[None]*self._num_cols for _ in range(self._num_rows)]

        for i, element in enumerate(self._elements):
            row_i, col_i = divmod(i, self._num_cols)

            if isinstance(element, Plottable):
                subfig = self._figure.add_subfigure(gridspec[row_i, col_i])
                ax = subfig.add_subplot()       # creates the Axes object used by Plottable objects
                # if self._x_label is not None:
                #     ax.set_xlabel(" ")
                    # ax.set_xlabel(self._x_label)
                element._plot_element(ax, 2)
                self._subfigs[row_i][col_i] = subfig
            elif isinstance(element, SmartFigure):
                subfig = self._figure.add_subfigure(gridspec[row_i, col_i])
                print(subfig.get_axes())
                element._figure = subfig        # associates the current subfigure with the nested SmartFigure
                element._prepare_figure()
                self._subfigs[row_i][col_i] = subfig
            elif element is not None:
                raise GraphingException(f"Unsupported element type: {type(element).__name__}")
        
        if len(self._elements) == 1:
            # A single plottable is plotted
            ax.set_xlabel(self._x_label)
            ax.set_ylabel(self._y_label)
        else:
            self._figure.supxlabel(self._x_label, fontsize=plt.rcParams["font.size"])
            self._figure.supylabel(self._y_label, fontsize=plt.rcParams["font.size"])

        # print(self._figure.get_axes())

        # if self._x_label is not None:
        #     self._figure.supxlabel(self._x_label)
        #     self._figure.text(0.5, 0.04, self._x_label, ha="center")
        # if self._y_label is not None:
        #     self._figure.supylabel(self._y_label)
            # self._figure.text(0.04, 0.5, self._y_label, va="center", rotation='vertical')
    
        # ax = self._figure.get_axes()
        # ax.set_xlabel(self._xlabel)
        # ax.set_ylabel(self._ylabel)


import graphinglib as gl


# Create a random SmartFigure which has two Curve objects of any shape
curve_1 = gl.Curve.from_function(lambda x: x**2, x_min=0, x_max=1, label="Curve 1")
curve_2 = gl.Curve.from_function(lambda x: x**3, x_min=0, x_max=1, label="Curve 2", color="red")
elements = [curve_1, curve_2]
sf_1 = SmartFigure(2, 1, elements=elements, x_label="xlab", y_label="ylab")

sf_2 = SmartFigure(1, 1, elements=[gl.Scatter([0, 1], [5, 10], label="testi")], x_label="xxx", y_label="yyy")

elements = [
    gl.Curve([0, 1], [0, 1], label="first curve", color="orange"),
    sf_2,
    gl.Curve([0, 1, 2], [2, 1, 2], label="second curve", color="green"),
    sf_1

]
sf = SmartFigure(2, 2, elements=elements, x_label="main mama x", y_label="main mama y")
sf.show()
