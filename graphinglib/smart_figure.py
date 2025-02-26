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
from numpy.typing import ArrayLike

class SmartFigure:
    def __init__(
        self,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        size: tuple[float, float] = None,
        title: Optional[str] = None,
        x_lim: Optional[tuple[float, float]] = None,
        y_lim: Optional[tuple[float, float]] = None,
        log_scale_x: bool = False,
        log_scale_y: bool = False,
        remove_axes: bool = False,
        remove_x_ticks: bool = False,
        remove_y_ticks: bool = False,
        reference_labels: bool = True,
        reflabel_loc: str = "outside",
        num_rows: int = 1,
        num_cols: int = 1,
        elements: Optional[list[Plottable]] = None,
        width_padding: float = None,
        height_padding: float = None,
        width_ratios: ArrayLike = None,
        height_ratios: ArrayLike = None,
        share_x: bool = False,
        share_y: bool = False,
    ) -> None:
        self._x_label = x_label
        self._y_label = y_label
        self._size = size
        self._title = title
        self._x_lim = x_lim
        self._y_lim = y_lim
        self._log_scale_x = log_scale_x
        self._log_scale_y = log_scale_y
        self._remove_axes = remove_axes
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
        self._width_ratios = width_ratios
        self._height_ratios = height_ratios
        self._share_x = share_x
        self._share_y = share_y

    def __len__(self) -> int:
        return len(self._elements)
    
    def show(
        self,
    ) -> None:
        self._initialize_parent_smart_figure()
        plt.show()
        plt.rcParams.update(plt.rcParamsDefault)

    def save(
            self,
            file_name: str,
            dpi: Optional[int] = None,
    ) -> None:
        self._initialize_parent_smart_figure()
        plt.savefig(file_name, bbox_inches="tight", dpi=dpi if dpi is not None else "figure")
        plt.close()
        plt.rcParams.update(plt.rcParamsDefault)
    
    def _initialize_parent_smart_figure(
            self,
    ) -> None:
        self._figure = plt.figure(constrained_layout=True, figsize=self._size)
        self._figure.set_constrained_layout_pads(w_pad=0, h_pad=0)
        self._reference_label_i = 0
        main_gridspec = self._prepare_figure()

        # Create an artificial axis to add padding around the figure
        ax_dummy = self._figure.add_subplot(main_gridspec[:, :])
        ax_dummy.set_zorder(-1)
        ax_dummy.set_navigate(False)
        ax_dummy.tick_params(colors=(0,0,0,0), axis="both", direction="in", labelright=True, labeltop=True, labelsize=0)
        ax_dummy.spines["top"].set_visible(False)
        ax_dummy.spines["right"].set_visible(False)
        ax_dummy.spines["left"].set_visible(False)
        ax_dummy.spines["bottom"].set_visible(False)
        ax_dummy.set_xticks([0.5])
        ax_dummy.set_yticks([0.5])
        ax_dummy.set_xticklabels([" "])
        ax_dummy.set_yticklabels([" "])

    def _prepare_figure(
        self,
    ) -> GridSpec:
        if len(self) != self._num_rows*self._num_cols:
            raise GraphingException("Number of elements does not match the grid size")
        
        gridspec = self._figure.add_gridspec(
            self._num_rows,
            self._num_cols,
            wspace=self._width_padding,
            hspace=self._height_padding,
            width_ratios=self._width_ratios,
            height_ratios=self._height_ratios,
        )

        # Plottable and subfigure plotting
        ax = None   # keep track of the last Axis object, needed for sharing axes
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
                ax = subfig.add_subplot(
                    sharex=ax if self._share_x else None,
                    sharey=ax if self._share_y else None,
                )

                # Plotting loop
                for current_element in current_elements:
                    if isinstance(current_element, Plottable):
                        current_element._plot_element(
                            ax,
                            2,
                        )
                    else:
                        raise GraphingException(f"Unsupported element type: {type(current_element).__name__}")

                # Add reference label
                if self._reference_labels and len(self) > 1:
                    ax.text(
                        0,
                        1,
                        ascii_lowercase[self._reference_label_i] + ")",
                        transform=ax.transAxes + self._get_reflabel_translation(),
                    )
                    self._reference_label_i += 1
                
                # If axes are shared, manually remove ticklabels from unnecessary plots
                if self._share_x and row_i != (self._num_rows - 1):
                    ax.tick_params(labelbottom=False)
                if self._share_y and col_i != 0:
                    ax.tick_params(labelleft=False)

                # Remove ticks
                if self._remove_x_ticks:
                    ax.get_xaxis().set_visible(False)
                if self._remove_y_ticks:
                    ax.get_yaxis().set_visible(False)
                
                # Axes limits
                if self._x_lim:
                    ax.set_xlim(*self._x_lim)
                if self._y_lim:
                    ax.set_ylim(*self._y_lim)

                # Logarithmic scale
                if self._log_scale_x:
                    ax.set_xscale("log")
                if self._log_scale_y:
                    ax.set_yscale("log")

                # Remove axes
                if self._remove_axes:
                    ax.axis("off")

            elif element is not None:
                raise GraphingException(f"Unsupported element type: {type(element).__name__}")

        # Axes labels
        if self._num_cols == 1 and self._num_rows == 1:
            if ax is not None:  # makes sure an element was plotted and that an axis was created
                ax.set_xlabel(self._x_label)
                ax.set_ylabel(self._y_label)
        else:
            self._figure.supxlabel(self._x_label, fontsize=plt.rcParams["font.size"])
            self._figure.supylabel(self._y_label, fontsize=plt.rcParams["font.size"])

        # Title
        if self._title:
            self._figure.suptitle(self._title, fontdict={"fontsize": "medium"})

        return gridspec

    def _get_reflabel_translation(
            self,
    ) -> ScaledTranslation:
        if self._reflabel_loc == "outside":
            return ScaledTranslation(-5 / 72, 10 / 72, self._figure.dpi_scale_trans)
        elif self._reflabel_loc == "inside":
            return ScaledTranslation(10 / 72, -15 / 72, self._figure.dpi_scale_trans)
        else:
            raise ValueError("Invalid reference label location. Please specify either 'inside' or 'outside'.")







import graphinglib as gl
import numpy as np


def rc():
    return gl.Curve.from_function(lambda x: np.random.random()*x**(np.random.random()*5), x_min=0, x_max=np.random.randint(1, 11))

# Create a random SmartFigure which has two Curve objects of any shape
curve_1 = gl.Curve.from_function(lambda x: x**2, x_min=0, x_max=2, label="Curve 1")
curve_2 = gl.Curve.from_function(lambda x: x**4, x_min=0, x_max=2, label="Curve 2", color="red")
elements = [curve_1, curve_2]

sf_1 = SmartFigure(num_rows=2, num_cols=1, elements=elements, x_label="xlab", y_label="ylab", height_padding=0.1, share_x=False)
sf_2 = SmartFigure(num_rows=1, num_cols=1, elements=[gl.Scatter([0, 1], [5, 10], label="testi")], x_label="xxx", y_label="yyy")
two_by_two = SmartFigure(num_rows=2, num_cols=2, elements=[rc() for _ in range(4)], remove_x_ticks=True, remove_y_ticks=True, reference_labels=True, height_padding=0.05, width_padding=0.1, y_label="Two by two y", x_label="Two by two x", share_x=True, share_y=True, size=(3,3), width_ratios=[3,5], height_ratios=[1,2], title="two by two plot")
simple_sf = SmartFigure(elements=[curve_1], remove_x_ticks=False, remove_y_ticks=True)
orange_curve = gl.Curve([0, 2], [0, 1], label="first curve", color="orange")
green_curve = gl.Curve([0, 1, 2], [2, 1, 2], label="second curve", color="green")
cs = [green_curve, orange_curve]
other = SmartFigure(num_rows=2, num_cols=1, elements=[rc(), rc()], remove_x_ticks=True, remove_y_ticks=True, reference_labels=False, height_padding=0.1, width_padding=0.1)
elements = [
    elements,# two_by_two,
    two_by_two,# None,
    cs,
    simple_sf
]
sf = SmartFigure(num_rows=2, num_cols=2, elements=elements, x_label="Mama x", y_label="Mama y", remove_x_ticks=False, remove_y_ticks=False, reference_labels=False,
    height_padding=0.05, width_padding=0.03, share_x=False, width_ratios=(0.5,2), title="Main Mama Figure", remove_axes=True,
    # master_height_padding=0, master_width_padding=0,
    # size=(14.5,8.1)
    size=(7,7)
)
# two_by_two.show()
# sf.save("zdev/test5.png", dpi=300)
two_by_two.show()
sf.show()


# custom axis label spacing and positionning
# remove x/y margins
# custom ticks
