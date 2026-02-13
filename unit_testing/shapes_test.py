import unittest

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba

from graphinglib.data_plotting_1d import Curve
from graphinglib.shapes import Arrow, Circle, Ellipse, Line, Polygon, Rectangle


class TestCircle(unittest.TestCase):
    def test_init(self):
        circle = Circle(2, 2, 1)

        self.assertAlmostEqual(circle.area, np.pi, places=2)
        self.assertAlmostEqual(circle.perimeter, 2 * np.pi, places=2)

    def test_init_invalid_radius(self):
        with self.assertRaises(ValueError):
            Circle(0, 0, 0)
        with self.assertRaises(ValueError):
            Circle(0, 0, -1)

    def test_radius_setter_invalid(self):
        circle = Circle(0, 0, 1)
        with self.assertRaises(ValueError):
            circle.radius = 0
        with self.assertRaises(ValueError):
            circle.radius = -1

    def test_init_invalid_points(self):
        with self.assertRaises(ValueError):
            Circle(0, 0, 1, number_of_points=3)

    def test_number_of_points_setter_invalid(self):
        circle = Circle(0, 0, 1)
        with self.assertRaises(ValueError):
            circle.number_of_points = 3


class TestEllipse(unittest.TestCase):
    def test_init(self):
        ellipse = Ellipse(2, 2, 2, 1)

        self.assertAlmostEqual(ellipse.x_center, 2, places=2)
        self.assertAlmostEqual(ellipse.y_center, 2, places=2)
        self.assertAlmostEqual(ellipse.x_radius, 2, places=2)
        self.assertAlmostEqual(ellipse.y_radius, 1, places=2)
        self.assertAlmostEqual(ellipse.angle, 0, places=2)
        self.assertAlmostEqual(ellipse.width, 4, places=2)
        self.assertAlmostEqual(ellipse.height, 2, places=2)

    def test_init_with_angle(self):
        ellipse = Ellipse(0, 0, 2, 1, angle=45)

        self.assertAlmostEqual(ellipse.angle, 45, places=2)

    def test_init_invalid_radii(self):
        with self.assertRaises(ValueError):
            Ellipse(2, 2, -1, 1)

        with self.assertRaises(ValueError):
            Ellipse(2, 2, 1, -1)

        with self.assertRaises(ValueError):
            Ellipse(2, 2, 0, 1)

    def test_init_invalid_points(self):
        with self.assertRaises(ValueError):
            Ellipse(2, 2, 1, 1, number_of_points=3)

    def test_number_of_points_setter_invalid(self):
        ellipse = Ellipse(0, 0, 2, 1)
        with self.assertRaises(ValueError):
            ellipse.number_of_points = 3

    def test_x_center_setter(self):
        ellipse = Ellipse(0, 0, 2, 1)
        ellipse.x_center = 5

        self.assertAlmostEqual(ellipse.x_center, 5, places=2)

    def test_y_center_setter(self):
        ellipse = Ellipse(0, 0, 2, 1)
        ellipse.y_center = 3

        self.assertAlmostEqual(ellipse.y_center, 3, places=2)

    def test_x_radius_setter(self):
        ellipse = Ellipse(0, 0, 2, 1)
        ellipse.x_radius = 3

        self.assertAlmostEqual(ellipse.x_radius, 3, places=2)
        self.assertAlmostEqual(ellipse.width, 6, places=2)

    def test_y_radius_setter(self):
        ellipse = Ellipse(0, 0, 2, 1)
        ellipse.y_radius = 2

        self.assertAlmostEqual(ellipse.y_radius, 2, places=2)
        self.assertAlmostEqual(ellipse.height, 4, places=2)

    def test_x_radius_setter_invalid(self):
        ellipse = Ellipse(0, 0, 2, 1)

        with self.assertRaises(ValueError):
            ellipse.x_radius = -1

        with self.assertRaises(ValueError):
            ellipse.x_radius = 0

    def test_y_radius_setter_invalid(self):
        ellipse = Ellipse(0, 0, 2, 1)

        with self.assertRaises(ValueError):
            ellipse.y_radius = -1

        with self.assertRaises(ValueError):
            ellipse.y_radius = 0

    def test_width_setter(self):
        ellipse = Ellipse(0, 0, 2, 1)
        ellipse.width = 6

        self.assertAlmostEqual(ellipse.width, 6, places=2)
        self.assertAlmostEqual(ellipse.x_radius, 3, places=2)

    def test_height_setter(self):
        ellipse = Ellipse(0, 0, 2, 1)
        ellipse.height = 4

        self.assertAlmostEqual(ellipse.height, 4, places=2)
        self.assertAlmostEqual(ellipse.y_radius, 2, places=2)

    def test_width_setter_invalid(self):
        ellipse = Ellipse(0, 0, 2, 1)

        with self.assertRaises(ValueError):
            ellipse.width = -1

        with self.assertRaises(ValueError):
            ellipse.width = 0

    def test_height_setter_invalid(self):
        ellipse = Ellipse(0, 0, 2, 1)

        with self.assertRaises(ValueError):
            ellipse.height = -1

        with self.assertRaises(ValueError):
            ellipse.height = 0

    def test_angle_setter(self):
        ellipse = Ellipse(0, 0, 2, 1)
        ellipse.angle = 30

        self.assertAlmostEqual(ellipse.angle, 30, places=2)

    def test_angle_rotation(self):
        ellipse1 = Ellipse(0, 0, 2, 1, angle=0)
        ellipse2 = Ellipse(0, 0, 2, 1, angle=45)

        # Both ellipses should have the same area (shape shouldn't change on rotation)
        self.assertAlmostEqual(ellipse1.area, ellipse2.area, places=1)

    def test_copy(self):
        ellipse = Ellipse(2, 3, 2, 1, fill_color="red", edge_color="blue", angle=45)
        ellipse_copy = ellipse.copy()

        self.assertAlmostEqual(ellipse_copy.x_center, ellipse.x_center, places=2)
        self.assertAlmostEqual(ellipse_copy.y_center, ellipse.y_center, places=2)
        self.assertAlmostEqual(ellipse_copy.x_radius, ellipse.x_radius, places=2)
        self.assertAlmostEqual(ellipse_copy.y_radius, ellipse.y_radius, places=2)
        self.assertAlmostEqual(ellipse_copy.angle, ellipse.angle, places=2)
        self.assertEqual(ellipse_copy._fill_color, ellipse._fill_color)
        self.assertEqual(ellipse_copy._edge_color, ellipse._edge_color)

    def test_plotting(self):
        ellipse = Ellipse(
            2,
            2,
            2,
            1,
            fill=True,
            fill_color="red",
            fill_alpha=0.5,
            edge_color="blue",
            line_width=2,
            line_style="-",
            angle=30,
        )

        fig, ax = plt.subplots()
        ellipse._plot_element(ax, 0)
        plt.close(fig)


class TestRectangle(unittest.TestCase):
    def test_init(self):
        rectangle = Rectangle(0, 0, 1, 1)

        self.assertAlmostEqual(rectangle.area, 1, places=2)
        self.assertAlmostEqual(rectangle.perimeter, 4, places=2)

    def test_init_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            Rectangle(0, 0, 0, 1)
        with self.assertRaises(ValueError):
            Rectangle(0, 0, 1, 0)
        with self.assertRaises(ValueError):
            Rectangle(0, 0, -1, 1)
        with self.assertRaises(ValueError):
            Rectangle(0, 0, 1, -1)

    def test_setters_reject_non_positive_dimensions(self):
        rectangle = Rectangle(0, 0, 1, 1)
        with self.assertRaises(ValueError):
            rectangle.width = 0
        with self.assertRaises(ValueError):
            rectangle.width = -1
        with self.assertRaises(ValueError):
            rectangle.height = 0
        with self.assertRaises(ValueError):
            rectangle.height = -1


class TestArrow(unittest.TestCase):
    def test_init(self):
        arrow = Arrow(
            pointA=(3, 3),
            pointB=(4, 4),
            color="blue",
            width=2,
            head_size=3,
            shrink=0.1,
            style="-|>",
            alpha=0.4,
            two_sided=True,
        )

        self.assertEqual(arrow._pointA[0], 3)
        self.assertEqual(arrow._pointA[1], 3)
        self.assertEqual(arrow._pointB[0], 4)
        self.assertEqual(arrow._pointB[1], 4)
        self.assertEqual(arrow._color, "blue")
        self.assertEqual(arrow._width, 2)
        self.assertEqual(arrow._head_size, 3)
        self.assertEqual(arrow._shrink, 0.1)
        self.assertEqual(arrow._style, "-|>")
        self.assertEqual(arrow._alpha, 0.4)
        self.assertEqual(arrow._two_sided, True)

    def test_shrink_points(self):
        arrow = Arrow(
            pointA=(3, 3),
            pointB=(4, 4),
            shrink=0.1,
        )
        shrinkedA, shrinkedB = arrow._shrink_points()
        self.assertEqual(shrinkedA[0], 3.1)
        self.assertEqual(shrinkedA[1], 3.1)
        self.assertEqual(shrinkedB[0], 3.9)
        self.assertEqual(shrinkedB[1], 3.9)

        arrow2 = Arrow(
            pointA=(3, 4),
            pointB=(4, 3),
            shrink=0.2,
        )
        shrinkedA, shrinkedB = arrow2._shrink_points()
        self.assertEqual(shrinkedA[0], 3.2)
        self.assertEqual(shrinkedA[1], 3.8)
        self.assertEqual(shrinkedB[0], 3.8)
        self.assertEqual(shrinkedB[1], 3.2)

        arrow3 = Arrow(
            pointA=(4, 3),
            pointB=(3, 4),
            shrink=0.3,
        )
        shrinkedA, shrinkedB = arrow3._shrink_points()
        self.assertEqual(shrinkedA[0], 3.7)
        self.assertEqual(shrinkedA[1], 3.3)
        self.assertEqual(shrinkedB[0], 3.3)
        self.assertEqual(shrinkedB[1], 3.7)

        arrow4 = Arrow(
            pointA=(4, 4),
            pointB=(3, 3),
            shrink=0.4,
        )
        shrinkedA, shrinkedB = arrow4._shrink_points()
        self.assertEqual(shrinkedA[0], 3.6)
        self.assertEqual(shrinkedA[1], 3.6)
        self.assertEqual(shrinkedB[0], 3.4)
        self.assertEqual(shrinkedB[1], 3.4)

    def test_styles(self):
        valid_styles = ["->", "-|>", "-[", "]->", "simple", "fancy", "wedge"]
        for style in valid_styles:
            arrow = Arrow(
                pointA=(0, 0),
                pointB=(1, 1),
                color="blue",
                width=3,
                head_size=1,
                style=style,
                alpha=1,
            )
            self.assertEqual(arrow._style, style)
        with self.assertRaises(ValueError):
            Arrow(
                pointA=(0, 0),
                pointB=(1, 1),
                color="blue",
                width=3,
                style="<->",
            )

        # Check errors at plotting time
        _, ax = plt.subplots()
        valid_two_sided_styles = ["->", "-|>", "-["]
        for style in valid_two_sided_styles:
            arrow = Arrow(
                pointA=(0, 0),
                pointB=(1, 1),
                color="blue",
                width=3,
                head_size=1,
                style=style,
                two_sided=True,
                alpha=1,
            )
            arrow._plot_element(ax, 0)
        invalid_two_sided_styles = ["]->", "simple", "fancy", "wedge"]
        for style in invalid_two_sided_styles:
            arrow = Arrow(
                pointA=(0, 0),
                pointB=(1, 1),
                color="blue",
                width=3,
                head_size=1,
                style=style,
                two_sided=True,
                alpha=1,
            )
            with self.assertRaises(ValueError):
                arrow._plot_element(ax, 0)

    def test_plotting(self):
        arrow = Arrow(
            pointA=(3, 3),
            pointB=(4, 4),
            color="blue",
            width=2,
            head_size=3,
            shrink=0.1,
            style="fancy",
            alpha=1.0,
            two_sided=False,
        )

        _, ax = plt.subplots()
        arrow._plot_element(ax, 0)

        for child in ax.get_children():
            if isinstance(child, plt.Annotation):
                self.assertEqual(child.xy, (3.9, 3.9))
                self.assertEqual(child.xyann, (3.1, 3.1))
                self.assertEqual(child.arrow_patch.get_edgecolor(), to_rgba("blue"))
                self.assertEqual(child.arrow_patch.get_linewidth(), 2)
                self.assertEqual(child.arrow_patch.get_alpha(), 1.0)
        plt.close()

    def test_copy(self):
        arrow = Arrow(
            pointA=(3, 3),
            pointB=(4, 4),
            color="blue",
            width=2,
            head_size=3,
            shrink=0.1,
            style="-[",
            alpha=0.85,
            two_sided=True,
        )
        arrow_copy = arrow.copy()
        self.assertEqual(arrow._pointA, arrow_copy._pointA)
        self.assertEqual(arrow._pointB, arrow_copy._pointB)
        self.assertEqual(arrow._color, arrow_copy._color)
        self.assertEqual(arrow._width, arrow_copy._width)
        self.assertEqual(arrow._head_size, arrow_copy._head_size)
        self.assertEqual(arrow._shrink, arrow_copy._shrink)
        self.assertEqual(arrow._style, arrow_copy._style)
        self.assertEqual(arrow._alpha, arrow_copy._alpha)
        self.assertEqual(arrow._two_sided, arrow_copy._two_sided)


class TestLine(unittest.TestCase):
    def test_init(self):
        line = Line(
            pointA=(3, 3),
            pointB=(4, 4),
            color="blue",
            width=2,
            capped_line=True,
            cap_width=3,
            alpha=0.2,
        )

        self.assertEqual(line._pointA[0], 3)
        self.assertEqual(line._pointA[1], 3)
        self.assertEqual(line._pointB[0], 4)
        self.assertEqual(line._pointB[1], 4)
        self.assertEqual(line._color, "blue")
        self.assertEqual(line._width, 2)
        self.assertEqual(line._capped_line, True)
        self.assertEqual(line._cap_width, 3)
        self.assertEqual(line._alpha, 0.2)

    def test_plotting(self):
        line = Line(
            pointA=(3, 3),
            pointB=(4, 4),
            color="blue",
            width=2,
            capped_line=True,
            cap_width=3,
            alpha=1.0,
        )

        _, ax = plt.subplots()
        line._plot_element(ax, 0)

        for child in ax.get_children():
            if isinstance(child, plt.Annotation):
                self.assertEqual(child.xy, (3, 3))
                self.assertEqual(child.xyann, (4, 4))
                self.assertEqual(child.arrow_patch.get_edgecolor(), to_rgba("blue"))
                self.assertEqual(child.arrow_patch.get_linewidth(), 2)
        plt.close()

    def test_copy(self):
        line = Line(
            pointA=(3, 3),
            pointB=(4, 4),
            color="blue",
            width=2,
            capped_line=True,
            cap_width=3,
            alpha=0.3,
        )
        line_copy = line.copy()
        self.assertEqual(line._pointA, line_copy._pointA)
        self.assertEqual(line._pointB, line_copy._pointB)
        self.assertEqual(line._color, line_copy._color)
        self.assertEqual(line._width, line_copy._width)
        self.assertEqual(line._capped_line, line_copy._capped_line)
        self.assertEqual(line._cap_width, line_copy._cap_width)
        self.assertEqual(line._alpha, line_copy._alpha)


class TestPolygon(unittest.TestCase):
    def test_init(self):
        vertices = [
            (0, 0),
            (1, 0),
            (1, 1),
            (0.5, 1.5),
            (0, 1),
            (-0.5, 2),
            (-0.8, 1.3),
            (-1, 0),
            (0, 0),
        ]
        polygon = Polygon(vertices, line_width=2)

        for i in range(len(vertices) - 1):
            self.assertEqual(polygon.vertices[i][0], vertices[i][0])
            self.assertEqual(polygon.vertices[i][1], vertices[i][1])
        self.assertEqual(polygon._line_width, 2)

    def test_create_centroid_point(self):
        vertices = [
            (0, 0),
            (1, 0),
            (1, 1),
            (0.5, 1.5),
            (0, 1),
            (-0.5, 2),
            (-0.8, 1.3),
            (-1, 0),
        ]
        polygon = Polygon(vertices, line_width=2)
        centroid = polygon.create_centroid_point()

        self.assertAlmostEqual(centroid._x, -0.00476190476190, places=6)
        self.assertAlmostEqual(centroid._y, 0.70317460317460, places=6)

    def test_get_centroid_coordinates(self):
        vertices = [
            (0, 0),
            (1, 0),
            (1, 1),
            (0.5, 1.5),
            (0, 1),
            (-0.5, 2),
            (-0.8, 1.3),
            (-1, 0),
        ]
        polygon = Polygon(vertices, line_width=2)
        centroid = polygon.get_centroid_coordinates()

        self.assertAlmostEqual(centroid[0], -0.00476190476190, places=6)
        self.assertAlmostEqual(centroid[1], 0.70317460317460, places=6)

    def test_create_intersection(self):
        vertices_1 = [
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1),
            (0, 0),
        ]
        vertices_2 = [
            (0.5, 0.5),
            (1.5, 0.5),
            (1.5, 1.5),
            (0.5, 1.5),
            (0.5, 0.5),
        ]
        polygon_1 = Polygon(vertices_1, line_width=2)
        polygon_2 = Polygon(vertices_2, line_width=2)

        intersection = polygon_1.create_intersection(polygon_2)

        self.assertEqual(len(intersection.vertices), 5)
        self.assertListEqual(list(intersection.vertices[0]), list((1, 1)))
        self.assertListEqual(list(intersection.vertices[1]), list((1, 0.5)))
        self.assertListEqual(list(intersection.vertices[2]), list((0.5, 0.5)))
        self.assertListEqual(list(intersection.vertices[3]), list((0.5, 1)))
        self.assertListEqual(list(intersection.vertices[4]), list((1, 1)))

    def test_create_union(self):
        vertices_1 = [
            (1, 0),
            (1, 0),
            (1, 1),
            (0, 1),
            (0, 0),
        ]
        vertices_2 = [
            (0.5, 0.5),
            (1.5, 0.5),
            (1.5, 1.5),
            (0.5, 1.5),
            (0.5, 0.5),
        ]

        polygon_1 = Polygon(vertices_1, line_width=2)
        polygon_2 = Polygon(vertices_2, line_width=2)

        union = polygon_1.create_union(polygon_2)

        self.assertEqual(len(union.vertices), 9)
        self.assertListEqual(list(union.vertices[0]), list((1, 0)))
        self.assertListEqual(list(union.vertices[1]), list((0, 0)))
        self.assertListEqual(list(union.vertices[2]), list((0, 1)))
        self.assertListEqual(list(union.vertices[3]), list((0.5, 1)))
        self.assertListEqual(list(union.vertices[4]), list((0.5, 1.5)))
        self.assertListEqual(list(union.vertices[5]), list((1.5, 1.5)))
        self.assertListEqual(list(union.vertices[6]), list((1.5, 0.5)))
        self.assertListEqual(list(union.vertices[7]), list((1, 0.5)))
        self.assertListEqual(list(union.vertices[8]), list((1, 0)))

    def test_create_difference(self):
        vertices_1 = [
            (1, 0),
            (1, 1),
            (0, 1),
            (0, 0),
            (1, 0),
        ]
        vertices_2 = [
            (0.5, 0.5),
            (1.5, 0.5),
            (1.5, 1.5),
            (0.5, 1.5),
            (0.5, 0.5),
        ]

        polygon_1 = Polygon(vertices_1, line_width=2)
        polygon_2 = Polygon(vertices_2, line_width=2)

        difference = polygon_1.create_difference(polygon_2)

        self.assertEqual(len(difference.vertices), 7)
        self.assertListEqual(list(difference.vertices[0]), list((1, 0)))
        self.assertListEqual(list(difference.vertices[1]), list((0, 0)))
        self.assertListEqual(list(difference.vertices[2]), list((0, 1)))
        self.assertListEqual(list(difference.vertices[3]), list((0.5, 1)))
        self.assertListEqual(list(difference.vertices[4]), list((0.5, 0.5)))
        self.assertListEqual(list(difference.vertices[5]), list((1, 0.5)))
        self.assertListEqual(list(difference.vertices[6]), list((1, 0)))

    def test_translate(self):
        vertices = [
            (0, 0),
            (1, 0),
            (1, 1),
            (0.5, 1.5),
            (0, 1),
            (-0.5, 2),
            (-0.8, 1.3),
            (-1, 0),
        ]
        polygon = Polygon(vertices, line_width=2)
        polygon.translate(1, 0)

        for i in range(len(vertices) - 1):
            self.assertEqual(polygon.vertices[i][0], vertices[i][0] + 1)
            self.assertEqual(polygon.vertices[i][1], vertices[i][1])

    def test_rotate(self):
        vertices = [
            (0, 0),
            (1, 0),
            (1, 1),
            (0.5, 1.5),
            (0, 1),
            (-0.5, 2),
            (-0.8, 1.3),
            (-1, 0),
        ]
        polygon = Polygon(vertices, line_width=2)
        polygon.rotate(45, center=(0, 0))

        predicted_vertices = [
            (0, 0),
            (np.sqrt(2) / 2, np.sqrt(2) / 2),
            (0, np.sqrt(2)),
            (-np.sqrt(2) / 2, np.sqrt(2)),
            (-np.sqrt(2) / 2, np.sqrt(2) / 2),
            (-5 * np.sqrt(2) / 4, 3 * np.sqrt(2) / 4),
            (-1.05 * np.sqrt(2), np.sqrt(2) / 4),
            (-np.sqrt(2) / 2, -np.sqrt(2) / 2),
            (0, 0),
        ]

        self.assertAlmostEqual(
            polygon.vertices[0][0], predicted_vertices[0][0], places=15
        )
        self.assertAlmostEqual(
            polygon.vertices[0][1], predicted_vertices[0][1], places=15
        )
        self.assertAlmostEqual(
            polygon.vertices[1][0], predicted_vertices[1][0], places=15
        )
        self.assertAlmostEqual(
            polygon.vertices[1][1], predicted_vertices[1][1], places=15
        )
        self.assertAlmostEqual(
            polygon.vertices[2][0], predicted_vertices[2][0], places=15
        )
        self.assertAlmostEqual(
            polygon.vertices[2][1], predicted_vertices[2][1], places=15
        )
        self.assertAlmostEqual(
            polygon.vertices[3][0], predicted_vertices[3][0], places=15
        )
        self.assertAlmostEqual(
            polygon.vertices[3][1], predicted_vertices[3][1], places=15
        )
        self.assertAlmostEqual(
            polygon.vertices[4][0], predicted_vertices[4][0], places=15
        )
        self.assertAlmostEqual(
            polygon.vertices[4][1], predicted_vertices[4][1], places=15
        )
        self.assertAlmostEqual(
            polygon.vertices[5][0], predicted_vertices[5][0], places=15
        )
        self.assertAlmostEqual(
            polygon.vertices[5][1], predicted_vertices[5][1], places=15
        )
        self.assertAlmostEqual(
            polygon.vertices[6][0], predicted_vertices[6][0], places=15
        )
        self.assertAlmostEqual(
            polygon.vertices[6][1], predicted_vertices[6][1], places=15
        )
        self.assertAlmostEqual(
            polygon.vertices[7][0], predicted_vertices[7][0], places=15
        )
        self.assertAlmostEqual(
            polygon.vertices[7][1], predicted_vertices[7][1], places=15
        )
        self.assertAlmostEqual(
            polygon.vertices[8][0], predicted_vertices[8][0], places=15
        )
        self.assertAlmostEqual(
            polygon.vertices[8][1], predicted_vertices[8][1], places=15
        )

    def test_scale(self):
        vertices = [
            (0, 0),
            (1, 0),
            (1, 1),
            (0.5, 1.5),
            (0, 1),
            (-0.5, 2),
            (-0.8, 1.3),
            (-1, 0),
            (0, 0),
        ]
        polygon = Polygon(vertices, line_width=2)
        polygon.scale(2, 2, center=(0, 0))

        for i in range(len(vertices) - 1):
            self.assertEqual(polygon.vertices[i][0], vertices[i][0] * 2)
            self.assertEqual(polygon.vertices[i][1], vertices[i][1] * 2)

    def test_skew(self):
        vertices = [
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1),
            (0, 0),
        ]
        polygon = Polygon(vertices, line_width=2)
        polygon.skew(45, 45, center=(0, 0))

        self.assertListEqual(list(polygon.vertices[0]), list((0, 0)))
        self.assertListEqual(list(polygon.vertices[1]), list((1, 0.9999999999999999)))
        self.assertListEqual(list(polygon.vertices[2]), list((2, 2)))
        self.assertListEqual(list(polygon.vertices[3]), list((0.9999999999999999, 1)))
        self.assertListEqual(list(polygon.vertices[4]), list((0, 0)))

    def test_split(self):
        vertices = [
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1),
            (0, 0),
        ]
        polygon = Polygon(vertices, line_width=2)
        curve = Curve.from_function(lambda x: x, 0, 1)

        poly_1, poly_2 = polygon.split(curve)

        self.assertIn(list((0, 0)), poly_1.vertices)
        self.assertIn(list((0, 1)), poly_1.vertices)
        self.assertIn(list((1, 1)), poly_1.vertices)

        self.assertIn(list((0, 0)), poly_2.vertices)
        self.assertIn(list((1, 0)), poly_2.vertices)
        self.assertIn(list((1, 1)), poly_2.vertices)

    def test_apply_transform(self):
        vertices = [
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1),
            (0, 0),
        ]
        polygon = Polygon(vertices, line_width=2)
        transform_matrix_2x2 = np.array([[2, 0], [0, 2]])
        polygon.linear_transformation(transform_matrix_2x2)

        self.assertIn(list((0, 0)), polygon.vertices)
        self.assertIn(list((2, 0)), polygon.vertices)
        self.assertIn(list((2, 2)), polygon.vertices)
        self.assertIn(list((0, 2)), polygon.vertices)

    def test_get_intersection_coordinates(self):
        vertices_1 = [
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1),
            (0, 0),
        ]
        vertices_2 = [
            (0.5, 0.5),
            (1.5, 0.5),
            (1.5, 1.5),
            (0.5, 1.5),
            (0.5, 0.5),
        ]
        polygon_1 = Polygon(vertices_1, line_width=2)
        polygon_2 = Polygon(vertices_2, line_width=2)

        intersection = polygon_1.get_intersection_coordinates(polygon_2)

        self.assertIn((1, 0.5), intersection)
        self.assertIn((0.5, 1), intersection)

    def test_create_intersection_points(self):
        vertices_1 = [
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1),
            (0, 0),
        ]
        vertices_2 = [
            (0.5, 0.5),
            (1.5, 0.5),
            (1.5, 1.5),
            (0.5, 1.5),
            (0.5, 0.5),
        ]
        polygon_1 = Polygon(vertices_1, line_width=2)
        polygon_2 = Polygon(vertices_2, line_width=2)

        intersection = polygon_1.create_intersection_points(polygon_2)

        self.assertEqual(len(intersection), 2)
        self.assertEqual(intersection[0]._x, 0.5)
        self.assertEqual(intersection[0]._y, 1)
        self.assertEqual(intersection[1]._x, 1)
        self.assertEqual(intersection[1]._y, 0.5)


if __name__ == "__main__":
    unittest.main()
