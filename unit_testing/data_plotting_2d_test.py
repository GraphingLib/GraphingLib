import unittest

import numpy as np

from graphinglib.data_plotting_2d import Contour, Heatmap
from graphinglib.figure import Figure


class TestHeatmap(unittest.TestCase):
    # TODO: Write tests for Heatmap
    ...


class TestContour(unittest.TestCase):
    def test_init(self):
        x = np.linspace(0, 3 * np.pi, 200)
        y = np.linspace(0, 3 * np.pi, 200)
        xx, yy = np.meshgrid(x, y)
        z = np.sin(xx) + np.cos(yy)

        contour = Contour(
            x_mesh=xx,
            y_mesh=yy,
            z_data=z,
        )

        self.assertIsInstance(contour, Contour)

    def test_from_function(self):
        contour = Contour.from_function(
            func=lambda x, y: np.sin(x) + np.cos(y),
            x_axis_range=(0, 3 * np.pi),
            y_axis_range=(0, 3 * np.pi),
        )

        self.assertIsInstance(contour, Contour)

    def test_defaults_are_covered(self):
        contour = Contour.from_function(
            func=lambda x, y: np.sin(x) + np.cos(y),
            x_axis_range=(0, 3 * np.pi),
            y_axis_range=(0, 3 * np.pi),
        )
        fig = Figure(figure_style="plain")
        fig.add_elements(contour)
        fig._prepare_figure()

        fig_2 = Figure(figure_style="horrible")
        fig_2.add_elements(contour)
        fig_2._prepare_figure()
