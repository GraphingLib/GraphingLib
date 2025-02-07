from shutil import which
from typing import Literal, Optional
from warnings import warn
from string import ascii_lowercase

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Polygon
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import ScaledTranslation
from matplotlib.figure import SubFigure

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

import numpy as np

class SmartFigure:
    def __init__(
        self,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        remove_x_ticks: bool = False,
        remove_y_ticks: bool = False,
        reference_labels: bool = True,
        reflabel_loc: str = "outside",
        num_rows: int = 1,
        num_cols: int = 1,
        elements: Optional[list[Plottable]] = None,
        width_padding: bool = None,
        height_padding: bool = None
    ) -> None:
        self._x_label = x_label
        self._y_label = y_label
        self._remove_x_ticks = remove_x_ticks
        self._remove_y_ticks = remove_y_ticks
        self._reference_labels = reference_labels
        self._reflabel_loc = reflabel_loc
        self._num_rows = num_rows
        self._num_cols = num_cols
        self._elements = elements
        self._figure = None
        self._reference_label_i = None
        self._width_padding = width_padding
        self._height_padding = height_padding

    def __len__(self) -> int:
        return len(self._elements)
    
    def show(
        self
    ) -> None:
        self._figure = plt.figure(constrained_layout=True)
        self._reference_label_i = 0
        self._prepare_figure()
        plt.show()
        plt.rcParams.update(plt.rcParamsDefault)

    def _prepare_figure(
        self,
    ) -> None:
        if len(self) != self._num_rows*self._num_cols:
            raise GraphingException("Number of elements does not match the grid size")
        
        # If nested figures are plotted, the wspace and hspace parameters must be scaled to the parent's gridspec to 
        # appear similar to their form when they are plotted individually
        if isinstance(self._figure, SubFigure):
            h_scale_factor, w_scale_factor = self._get_subfigure_weight_factors(self._figure)
        else:
            w_scale_factor, h_scale_factor = 1, 1
        gridspec = self._figure.add_gridspec(
            self._num_rows,
            self._num_cols,
            wspace=self._width_padding / w_scale_factor if self._width_padding is not None else None,
            hspace=self._height_padding / h_scale_factor if self._height_padding is not None else None,
        )

        # Plottable and subfigure plotting
        for i, element in enumerate(self._elements):
            row_i, col_i = divmod(i, self._num_cols)

            if isinstance(element, SmartFigure):
                subfig = self._figure.add_subfigure(gridspec[row_i, col_i])
                element._figure = subfig        # associates the current subfigure with the nested SmartFigure
                element._reference_label_i = self._reference_label_i
                element._prepare_figure()
                self._reference_label_i = element._reference_label_i

            elif isinstance(element, (Plottable, list)):
                current_elements = element if isinstance(element, list) else [element]
                subfig = self._figure.add_subfigure(gridspec[row_i, col_i])
                ax = subfig.add_subplot()
                for current_element in current_elements:
                    if isinstance(current_element, Plottable):
                        current_element._plot_element(
                            ax,
                            2,
                        )
                    else:
                        raise GraphingException(f"Unsupported element type: {type(current_element).__name__}")
                if self._reference_labels and len(self) > 1:
                    ax.text(
                        0,
                        1,
                        ascii_lowercase[self._reference_label_i] + ")",
                        transform=ax.transAxes + self._get_reflabel_translation(),
                    )
                    self._reference_label_i += 1
                if self._remove_x_ticks:
                    ax.get_xaxis().set_visible(False)
                if self._remove_y_ticks:
                    ax.get_yaxis().set_visible(False)

            elif isinstance(element, list):
                pass
            elif element is not None:
                raise GraphingException(f"Unsupported element type: {type(element).__name__}")
        
        # Axes labels
        if len(self._elements) == 1:
            ax = self._figure.get_axes()[0]
            ax.set_xlabel(self._x_label)
            ax.set_ylabel(self._y_label)
        else:
            self._figure.supxlabel(self._x_label, fontsize=plt.rcParams["font.size"])
            self._figure.supylabel(self._y_label, fontsize=plt.rcParams["font.size"])

    def _get_reflabel_translation(
            self,
    ) -> ScaledTranslation:
        if self._reflabel_loc == "outside":
            return ScaledTranslation(-5 / 72, 10 / 72, self._figure.dpi_scale_trans)
        elif self._reflabel_loc == "inside":
            return ScaledTranslation(10 / 72, -15 / 72, self._figure.dpi_scale_trans)
        else:
            raise ValueError("Invalid reference label location. Please specify either 'inside' or 'outside'.")

    @staticmethod
    def _get_subfigure_weight_factors(
            subfigure: SubFigure,
    ) -> np.ndarray:
        parent_gridspec = subfigure._subplotspec.get_gridspec()
        if isinstance(parent_gridspec.figure, SubFigure):
            return (
                np.array([parent_gridspec.nrows, parent_gridspec.ncols]) 
                * SmartFigure._get_subfigure_weight_factors(parent_gridspec.figure)
            )
        else:
            return np.array([parent_gridspec.nrows, parent_gridspec.ncols])








import graphinglib as gl
import numpy as np


def rc():
    return gl.Curve.from_function(lambda x: np.random.random()*x**(np.random.random()*5), x_min=0, x_max=np.random.randint(1, 11))

# Create a random SmartFigure which has two Curve objects of any shape
curve_1 = gl.Curve.from_function(lambda x: x**2, x_min=0, x_max=1, label="Curve 1")
curve_2 = gl.Curve.from_function(lambda x: x**10, x_min=0, x_max=1, label="Curve 2", color="red")
elements = [curve_1, curve_2]
sf_1 = SmartFigure(num_rows=2, num_cols=1, elements=elements, x_label="xlab", y_label="ylab")

sf_2 = SmartFigure(num_rows=1, num_cols=1, elements=[gl.Scatter([0, 1], [5, 10], label="testi")], x_label="xxx", y_label="yyy")

two_by_two = SmartFigure(num_rows=2, num_cols=2, elements=[rc() for _ in range(4)], remove_x_ticks=True, remove_y_ticks=True, reference_labels=False, height_padding=0.5, width_padding=0.9)

orange_curve = gl.Curve([0, 2], [0, 1], label="first curve", color="orange")
green_curve = gl.Curve([0, 1, 2], [2, 1, 2], label="second curve", color="green")
cs = [green_curve, orange_curve]
other = SmartFigure(num_rows=3, num_cols=1, elements=[rc(), rc(), two_by_two], remove_x_ticks=True, remove_y_ticks=True, reference_labels=False)
elements = [
    cs,# two_by_two,
    two_by_two,# None,
    other,# cs,
    two_by_two
]
sf = SmartFigure(num_rows=2, num_cols=2, elements=elements, x_label="main mama x", y_label="main mama y", remove_x_ticks=True, remove_y_ticks=True, reference_labels=False,
    # height_padding=0.5, width_padding=0.5,
)
# two_by_two.show()
sf.show()


# two_by_two.show()
# other.show()


# remove tick spacing
# add subplot weights
# better subplot spacing conversion to account for wspace and hspace of parent figures
