import unittest
from unittest.mock import patch

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba

from graphinglib.data_plotting_2d import Contour, Heatmap, Stream, VectorField
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
            alpha=0.5,
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
            alpha=0.5,
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
        np.random.seed(0)
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
            alpha=0.5,
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
        self.assertAlmostEqual(fig.axes[1].get_ylim()[0], min(z), places=3)
        self.assertAlmostEqual(fig.axes[1].get_ylim()[1], max(z), places=3)

    def test_copy(self):
        array_of_data = np.random.rand(10, 10)
        heatmap = Heatmap(
            image=array_of_data,
            x_axis_range=(0, 10),
            y_axis_range=(0, 11),
            color_map="viridis",
            show_color_bar=True,
            alpha=0.5,
            aspect_ratio="auto",
            origin_position="upper",
            interpolation="nearest",
        )
        heatmap_copy = heatmap.copy()
        self.assertIsInstance(heatmap_copy, Heatmap)
        self.assertEqual(heatmap_copy._image.tolist(), array_of_data.tolist())
        self.assertEqual(heatmap_copy._x_axis_range, (0, 10))
        self.assertEqual(heatmap_copy._y_axis_range, (0, 11))
        self.assertEqual(heatmap_copy._color_map, "viridis")
        self.assertEqual(heatmap_copy._show_color_bar, True)
        self.assertEqual(heatmap_copy._alpha, 0.5)
        self.assertEqual(heatmap_copy._aspect_ratio, "auto")
        self.assertEqual(heatmap_copy._origin_position, "upper")
        self.assertEqual(heatmap_copy._interpolation, "nearest")

    def test_with_custom_mesh_pcolormesh(self):
        # Test that providing x_mesh and y_mesh uses pcolormesh instead of imshow
        array_of_data = np.random.rand(10, 10)
        x = np.linspace(0, 10, 11)
        y = np.linspace(0, 10, 11)
        x_mesh, y_mesh = np.meshgrid(x, y)

        heatmap = Heatmap(
            image=array_of_data,
            x_mesh=x_mesh,
            y_mesh=y_mesh,
            color_map="plasma",
            color_map_range=(20, 80),
            show_color_bar=True,
            alpha=0.8,
        )

        fig, ax = plt.subplots()
        heatmap._plot_element(ax, 0)

        # Check that pcolormesh was used (creates a QuadMesh collection)
        self.assertEqual(len(ax.collections), 1)
        self.assertEqual(ax.collections[0].get_cmap().name, "plasma")
        self.assertEqual(ax.collections[0].get_alpha(), 0.8)
        # Color bar should still be created
        self.assertEqual(len(fig.axes), 2)
        self.assertEqual(ax.collections[0].get_clim(), (20, 80))

    def test_image_setter_array_conversion(self):
        heatmap = Heatmap(image=np.zeros((2, 2)))
        heatmap.image = [[1, 2], [3, 4]]
        self.assertIsInstance(heatmap._image, np.ndarray)
        self.assertEqual(heatmap._image.shape, (2, 2))

    def test_image_setter_reads_file_path(self):
        heatmap = Heatmap(image=np.zeros((2, 2)))
        fake_image = np.ones((3, 3))
        with patch("graphinglib.data_plotting_2d.imread", return_value=fake_image):
            heatmap.image = "dummy.png"
        self.assertTrue(np.array_equal(heatmap._image, fake_image))
        self.assertFalse(heatmap._show_color_bar)

    def test_init_with_file_image_disables_color_bar(self):
        fake_image = np.ones((3, 3))
        with patch("graphinglib.data_plotting_2d.imread", return_value=fake_image):
            heatmap = Heatmap(image="dummy.png", show_color_bar=True)
        self.assertTrue(np.array_equal(heatmap._image, fake_image))
        self.assertFalse(heatmap._show_color_bar)


class TestVectorField(unittest.TestCase):
    def test_init(self):
        x = np.linspace(0, 3 * np.pi, 20)
        y = np.linspace(0, 3 * np.pi, 20)
        xx, yy = np.meshgrid(x, y)
        u = np.sin(xx) * np.cos(yy)
        v = -np.cos(xx) * np.sin(yy)

        # Create a VectorField object
        vector_field = VectorField(
            x_data=xx,
            y_data=yy,
            u_data=u,
            v_data=v,
            arrow_width=1,
            arrow_head_size=1,
            scale=1,
            make_angles_axes_independent=True,
            color="black",
        )

        self.assertIsInstance(vector_field, VectorField)
        self.assertListEqual(vector_field._x_data.tolist(), xx.tolist())
        self.assertListEqual(vector_field._y_data.tolist(), yy.tolist())
        self.assertListEqual(vector_field._u_data.tolist(), u.tolist())
        self.assertListEqual(vector_field._v_data.tolist(), v.tolist())
        self.assertEqual(vector_field._arrow_width, 1)
        self.assertEqual(vector_field._arrow_head_size, 1)
        self.assertEqual(vector_field._scale, 1)
        self.assertEqual(vector_field._make_angles_axes_independent, True)
        self.assertEqual(vector_field._color, "black")

    def test_from_function(self):
        vector_field = VectorField.from_function(
            func=lambda x, y: (np.sin(x), np.cos(y)),
            x_axis_range=(0, 3 * np.pi),
            y_axis_range=(0, 3 * np.pi),
            number_of_arrows_x=20,
            number_of_arrows_y=20,
            arrow_width=1,
            arrow_head_size=1,
            scale=1,
            make_angles_axes_independent=True,
            color="black",
        )

        self.assertIsInstance(vector_field, VectorField)
        self.assertEqual(vector_field._arrow_width, 1)
        self.assertEqual(vector_field._arrow_head_size, 1)
        self.assertEqual(vector_field._scale, 1)
        self.assertEqual(vector_field._make_angles_axes_independent, True)
        self.assertEqual(vector_field._color, "black")

    def test_plot_element(self):
        x = np.linspace(0, 3 * np.pi, 20)
        y = np.linspace(0, 3 * np.pi, 20)
        xx, yy = np.meshgrid(x, y)
        u = np.sin(xx) * np.cos(yy)
        v = -np.cos(xx) * np.sin(yy)

        # Create a VectorField object
        vector_field = VectorField(
            x_data=xx,
            y_data=yy,
            u_data=u,
            v_data=v,
            arrow_width=1,
            arrow_head_size=1,
            scale=1,
            make_angles_axes_independent=True,
            color="black",
        )

        _, ax = plt.subplots()
        vector_field._plot_element(ax, 0)
        self.assertEqual(len(ax.collections), 1)
        self.assertListEqual(
            ax.collections[0].get_facecolors()[0].tolist(), [0, 0, 0, 1]
        )
        self.assertEqual(ax.collections[0].width, 1 * 0.005)
        self.assertEqual(ax.collections[0].headwidth, 4 * 1 / 1)
        self.assertEqual(ax.collections[0].headlength, 4)
        self.assertEqual(ax.collections[0].headaxislength, 4)
        self.assertEqual(ax.collections[0].angles, "uv")

    def test_copy(self):
        x = np.linspace(0, 3 * np.pi, 20)
        y = np.linspace(0, 3 * np.pi, 20)
        xx, yy = np.meshgrid(x, y)
        u = np.sin(xx) * np.cos(yy)
        v = -np.cos(xx) * np.sin(yy)

        # Create a VectorField object
        vector_field = VectorField(
            x_data=xx,
            y_data=yy,
            u_data=u,
            v_data=v,
            arrow_width=1,
            arrow_head_size=1,
            make_angles_axes_independent=True,
            color="black",
        )
        vector_field_copy = vector_field.copy()
        self.assertIsInstance(vector_field_copy, VectorField)
        self.assertListEqual(vector_field_copy._x_data.tolist(), xx.tolist())
        self.assertListEqual(vector_field_copy._y_data.tolist(), yy.tolist())
        self.assertListEqual(vector_field_copy._u_data.tolist(), u.tolist())
        self.assertListEqual(vector_field_copy._v_data.tolist(), v.tolist())
        self.assertEqual(vector_field_copy._arrow_width, 1)
        self.assertEqual(vector_field_copy._arrow_head_size, 1)
        self.assertEqual(vector_field_copy._make_angles_axes_independent, True)
        self.assertEqual(vector_field_copy._color, "black")


class TestContour(unittest.TestCase):
    def test_init(self):
        x = np.linspace(0, 3 * np.pi, 200)
        y = np.linspace(0, 3 * np.pi, 200)
        xx, yy = np.meshgrid(x, y)
        z = np.sin(xx) + np.cos(yy)

        contour = Contour(
            z_data=z,
            x_mesh=xx,
            y_mesh=yy,
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

    def test_copy(self):
        x = np.linspace(0, 3 * np.pi, 200)
        y = np.linspace(0, 3 * np.pi, 200)
        xx, yy = np.meshgrid(x, y)
        z = np.sin(xx) + np.cos(yy)

        contour = Contour(
            z_data=z,
            x_mesh=xx,
            y_mesh=yy,
        )
        contour_copy = contour.copy()
        self.assertIsInstance(contour_copy, Contour)
        self.assertListEqual(contour_copy._x_mesh.tolist(), xx.tolist())
        self.assertListEqual(contour_copy._y_mesh.tolist(), yy.tolist())
        self.assertListEqual(contour_copy._z_data.tolist(), z.tolist())

    def test_without_mesh(self):
        contour = Contour(
            z_data=np.random.rand(10, 20),
        )
        fig = Figure(figure_style="plain")
        fig.add_elements(contour)
        fig._prepare_figure()


class TestStream(unittest.TestCase):
    def test_init(self):
        x_grid, y_grid = np.meshgrid(np.linspace(0, 11, 30), np.linspace(0, 11, 30))
        u, v = (np.cos(x_grid * 0.2), np.sin(y_grid * 0.3))

        stream = Stream(
            x_data=x_grid,
            y_data=y_grid,
            u_data=u,
            v_data=v,
            density=0.7,
            line_width=2,
            color="black",
            color_map="viridis",
            arrow_size=2,
        )

        self.assertIsInstance(stream, Stream)
        self.assertListEqual(stream._x_data.tolist(), x_grid.tolist())
        self.assertListEqual(stream._y_data.tolist(), y_grid.tolist())
        self.assertListEqual(stream._u_data.tolist(), u.tolist())
        self.assertListEqual(stream._v_data.tolist(), v.tolist())
        self.assertEqual(stream._density, 0.7)
        self.assertEqual(stream._line_width, 2)
        self.assertEqual(stream._color, "black")
        self.assertEqual(stream._color_map, "viridis")
        self.assertEqual(stream._arrow_size, 2)

    def test_from_function(self):
        stream = Stream.from_function(
            func=lambda x, y: (np.sin(x), np.cos(y)),
            x_axis_range=(0, 3 * np.pi),
            y_axis_range=(0, 3 * np.pi),
            number_of_points_x=20,
            number_of_points_y=20,
            density=0.7,
            line_width=2,
            color="black",
            color_map="viridis",
            arrow_size=2,
        )

        self.assertIsInstance(stream, Stream)
        self.assertListEqual(
            stream._x_data.tolist(), np.linspace(0, 3 * np.pi, 20).tolist()
        )
        self.assertListEqual(
            stream._y_data.tolist(), np.linspace(0, 3 * np.pi, 20).tolist()
        )
        self.assertEqual(stream._density, 0.7)
        self.assertEqual(stream._line_width, 2)
        self.assertEqual(stream._color, "black")
        self.assertEqual(stream._color_map, "viridis")
        self.assertEqual(stream._arrow_size, 2)

    def test_plot_element(self):
        x_grid, y_grid = np.meshgrid(np.linspace(0, 11, 30), np.linspace(0, 11, 30))
        u, v = (np.cos(x_grid * 0.2), np.sin(y_grid * 0.3))

        stream = Stream(
            x_data=x_grid,
            y_data=y_grid,
            u_data=u,
            v_data=v,
            density=0.7,
            line_width=2,
            color="black",
            color_map="viridis",
            arrow_size=2,
        )

        _, ax = plt.subplots()
        stream._plot_element(ax, 0)
        self.assertEqual(len(ax.collections), 1)
        self.assertEqual(ax.collections[0].get_linewidth(), 2)
        self.assertListEqual(
            ax.collections[0].get_color().tolist()[0], [i for i in to_rgba("k")]
        )
        self.assertEqual(ax.collections[0].get_cmap().name, "viridis")

    def test_copy(self):
        x_grid, y_grid = np.meshgrid(np.linspace(0, 11, 30), np.linspace(0, 11, 30))
        u, v = (np.cos(x_grid * 0.2), np.sin(y_grid * 0.3))

        stream = Stream(
            x_data=x_grid,
            y_data=y_grid,
            u_data=u,
            v_data=v,
            density=0.7,
            line_width=2,
            color="black",
            color_map="viridis",
            arrow_size=2,
        )
        stream_copy = stream.copy()
        self.assertIsInstance(stream_copy, Stream)
        self.assertListEqual(stream_copy._x_data.tolist(), x_grid.tolist())
        self.assertListEqual(stream_copy._y_data.tolist(), y_grid.tolist())
        self.assertListEqual(stream_copy._u_data.tolist(), u.tolist())
        self.assertListEqual(stream_copy._v_data.tolist(), v.tolist())
        self.assertEqual(stream_copy._density, 0.7)
        self.assertEqual(stream_copy._line_width, 2)
        self.assertEqual(stream_copy._color, "black")
        self.assertEqual(stream_copy._color_map, "viridis")
        self.assertEqual(stream_copy._arrow_size, 2)


if __name__ == "__main__":
    unittest.main()
