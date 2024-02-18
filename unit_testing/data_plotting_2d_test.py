import unittest

import numpy as np
from matplotlib import pyplot as plt

from graphinglib.data_plotting_2d import Contour, Heatmap
from graphinglib.figure import Figure


class TestHeatmap(unittest.TestCase):
    def test_init_and_plot(self):
        array_of_data = np.random.rand(10, 10)
        heatmap = Heatmap(
            image=array_of_data,
            x_axis_range=(0, 10),
            y_axis_range=(0, 11),
            color_map="viridis",
            show_color_bar=True,
            alpha_value=0.5,
            aspect_ratio="auto",
            origin_position="upper",
            interpolation="nearest",
        )
        fig, ax = plt.subplots()
        heatmap._plot_element(ax, 0)
        self.assertEqual(ax.get_xlim(), (0, 10))
        self.assertEqual(ax.get_ylim(), (0, 11))
        self.assertEqual(ax.images[0].get_cmap().name, "viridis")
        self.assertEqual(ax.images[0].get_alpha(), 0.5)
        self.assertEqual(ax.images[0].get_interpolation(), "nearest")
        self.assertTrue(ax.images[0].get_visible())
        self.assertEqual(len(fig.axes), 2)
        self.assertEqual(
            fig.axes[1].get_ylim(),
            (min(array_of_data.flatten()), max(array_of_data.flatten())),
        )

    def test_from_function(self):
        heatmap = Heatmap.from_function(
            func=lambda x, y: np.sin(x) + np.cos(y),
            x_axis_range=(0, 3 * np.pi),
            y_axis_range=(0, 3 * np.pi),
            color_map="viridis",
            show_color_bar=True,
            alpha_value=0.5,
            aspect_ratio="auto",
            origin_position="upper",
            interpolation="nearest",
        )
        fig, ax = plt.subplots()
        heatmap._plot_element(ax, 0)
        self.assertEqual(ax.get_xlim(), (0, 3 * np.pi))
        self.assertEqual(ax.get_ylim(), (0, 3 * np.pi))
        self.assertEqual(ax.images[0].get_cmap().name, "viridis")
        self.assertEqual(ax.images[0].get_alpha(), 0.5)
        self.assertEqual(ax.images[0].get_interpolation(), "nearest")
        self.assertTrue(ax.images[0].get_visible())
        self.assertEqual(len(fig.axes), 2)

    def test_from_points(self):
        # make list of random (x,y) points
        x = np.random.rand(10)
        y = np.random.rand(10)
        points = np.array([x, y]).T
        # make list of random z values
        z = np.random.rand(10)

        heatmap = Heatmap.from_points(
            points=points,
            values=z,
            x_axis_range=(0, 10),
            y_axis_range=(0, 11),
            color_map="viridis",
            show_color_bar=True,
            alpha_value=0.5,
            aspect_ratio="auto",
            origin_position="upper",
            interpolation="nearest",
        )
        fig, ax = plt.subplots()
        heatmap._plot_element(ax, 0)
        self.assertEqual(ax.get_xlim(), (0, 10))
        self.assertEqual(ax.get_ylim(), (0, 11))
        self.assertEqual(ax.images[0].get_cmap().name, "viridis")
        self.assertEqual(ax.images[0].get_alpha(), 0.5)
        self.assertEqual(ax.images[0].get_interpolation(), "nearest")
        self.assertTrue(ax.images[0].get_visible())
        self.assertEqual(len(fig.axes), 2)
        self.assertEqual(fig.axes[1].get_ylim(), (min(z), max(z)))


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


if __name__ == "__main__":
    unittest.main()
