import unittest

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba

from graphinglib.data_plotting_1d import Curve
from graphinglib.graph_elements import Point
from graphinglib.shapes import Arrow, Circle, Line, Polygon, Rectangle


class TestCircle(unittest.TestCase):
    def test_init(self):
        circle = Circle(2, 2, 1)

        self.assertAlmostEqual(circle.get_area(), np.pi, places=2)
        self.assertAlmostEqual(circle.get_perimeter(), 2 * np.pi, places=2)


class TestRectangle(unittest.TestCase):
    def test_init(self):
        rectangle = Rectangle(0, 0, 1, 1)

        self.assertAlmostEqual(rectangle.get_area(), 1, places=2)
        self.assertAlmostEqual(rectangle.get_perimeter(), 4, places=2)


class TestArrow(unittest.TestCase):
    def test_init(self):
        arrow = Arrow(
            pointA=(3, 3),
            pointB=(4, 4),
            color="blue",
            width=2,
            head_size=3,
            shrink=0.1,
            two_sided=True,
        )

        self.assertEqual(arrow.pointA[0], 3)
        self.assertEqual(arrow.pointA[1], 3)
        self.assertEqual(arrow.pointB[0], 4)
        self.assertEqual(arrow.pointB[1], 4)
        self.assertEqual(arrow.color, "blue")
        self.assertEqual(arrow.width, 2)
        self.assertEqual(arrow.head_size, 3)
        self.assertEqual(arrow.shrink, 0.1)
        self.assertEqual(arrow.two_sided, True)

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

    def test_plotting(self):
        arrow = Arrow(
            pointA=(3, 3),
            pointB=(4, 4),
            color="blue",
            width=2,
            head_size=3,
            shrink=0.1,
            two_sided=True,
        )

        _, ax = plt.subplots()
        arrow._plot_element(ax, 0)

        for child in ax.get_children():
            if isinstance(child, plt.Annotation):
                self.assertEqual(child.xy, (3.9, 3.9))
                self.assertEqual(child.xyann, (3.1, 3.1))
                self.assertEqual(child.arrow_patch.get_edgecolor(), to_rgba("blue"))
                self.assertEqual(child.arrow_patch.get_linewidth(), 2)
        plt.close()

    def test_copy(self):
        arrow = Arrow(
            pointA=(3, 3),
            pointB=(4, 4),
            color="blue",
            width=2,
            head_size=3,
            shrink=0.1,
            two_sided=True,
        )
        arrow_copy = arrow.copy()
        self.assertEqual(arrow.pointA, arrow_copy.pointA)
        self.assertEqual(arrow.pointB, arrow_copy.pointB)
        self.assertEqual(arrow.color, arrow_copy.color)
        self.assertEqual(arrow.width, arrow_copy.width)
        self.assertEqual(arrow.head_size, arrow_copy.head_size)
        self.assertEqual(arrow.shrink, arrow_copy.shrink)
        self.assertEqual(arrow.two_sided, arrow_copy.two_sided)


class TestLine(unittest.TestCase):
    def test_init(self):
        line = Line(
            pointA=(3, 3),
            pointB=(4, 4),
            color="blue",
            width=2,
            capped_line=True,
            cap_width=3,
        )

        self.assertEqual(line.pointA[0], 3)
        self.assertEqual(line.pointA[1], 3)
        self.assertEqual(line.pointB[0], 4)
        self.assertEqual(line.pointB[1], 4)
        self.assertEqual(line.color, "blue")
        self.assertEqual(line.width, 2)
        self.assertEqual(line.capped_line, True)
        self.assertEqual(line.cap_width, 3)

    def test_plotting(self):
        line = Line(
            pointA=(3, 3),
            pointB=(4, 4),
            color="blue",
            width=2,
            capped_line=True,
            cap_width=3,
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
        )
        line_copy = line.copy()
        self.assertEqual(line.pointA, line_copy.pointA)
        self.assertEqual(line.pointB, line_copy.pointB)
        self.assertEqual(line.color, line_copy.color)
        self.assertEqual(line.width, line_copy.width)
        self.assertEqual(line.capped_line, line_copy.capped_line)
        self.assertEqual(line.cap_width, line_copy.cap_width)


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
        self.assertEqual(polygon.line_width, 2)

    def test_get_area(self):
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
        self.assertEqual(polygon.get_area(), 2.625)

    def test_get_perimeter(self):
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
        self.assertEqual(polygon.get_perimeter(), 7.609119505505971)

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
