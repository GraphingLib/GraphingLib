import unittest

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba

from graphinglib.data_plotting_1d import Curve
from graphinglib.graph_elements import Point
from graphinglib.shapes import Arrow, Circle, Line, Polygon, Rectangle


class TestCircle(unittest.TestCase):
    def test_radius(self):
        with self.assertRaises(ValueError):
            Circle(0, 0, -1)
        with self.assertRaises(ValueError):
            Circle(0, 0, 0)
        with self.assertRaises(ValueError):
            Circle(5, -8, -2)

    def test_circumference(self):
        self.assertEqual(Circle(0, 0, 1).get_circumference(), 2 * 3.141592653589793)
        self.assertEqual(Circle(0, 0, 2).get_circumference(), 4 * 3.141592653589793)
        self.assertEqual(Circle(0, 0, 3).get_circumference(), 6 * 3.141592653589793)
        # If circle is not centered at origin
        self.assertEqual(Circle(1, 1, 1).get_circumference(), 2 * 3.141592653589793)
        self.assertEqual(Circle(1, 1, 2).get_circumference(), 4 * 3.141592653589793)
        self.assertEqual(Circle(1, 1, 3).get_circumference(), 6 * 3.141592653589793)

    def test_area(self):
        self.assertEqual(Circle(0, 0, 1).get_area(), 3.141592653589793)
        self.assertEqual(Circle(0, 0, 2).get_area(), 4 * 3.141592653589793)
        self.assertEqual(Circle(0, 0, 3).get_area(), 9 * 3.141592653589793)
        # If circle is not centered at origin
        self.assertEqual(Circle(1, 1, 1).get_area(), 3.141592653589793)
        self.assertEqual(Circle(1, 1, 2).get_area(), 4 * 3.141592653589793)
        self.assertEqual(Circle(1, 1, 3).get_area(), 9 * 3.141592653589793)

    def test_contains(self):
        self.assertTrue(Point(0, 0) in Circle(0, 0, 1))
        self.assertTrue(Point(0, 1) in Circle(0, 0, 1))
        self.assertTrue(Point(0, -1) in Circle(0, 0, 1))
        self.assertTrue(Point(1, 0) in Circle(0, 0, 1))
        self.assertTrue(Point(-1, 0) in Circle(0, 0, 1))
        self.assertFalse(Point(1, 1) in Circle(0, 0, 1))
        self.assertFalse(Point(1, -1) in Circle(0, 0, 1))
        self.assertFalse(Point(-1, 1) in Circle(0, 0, 1))
        self.assertFalse(Point(-1, -1) in Circle(0, 0, 1))

    def test_get_coordinates_at_x(self):
        self.assertListEqual(Circle(0, 0, 1).get_coordinates_at_x(0), [(0, 1), (0, -1)])
        self.assertListEqual(Circle(0, 0, 1).get_coordinates_at_x(1), [(1, 0)])
        self.assertListEqual(Circle(0, 0, 1).get_coordinates_at_x(-1), [(-1, 0)])
        self.assertListEqual(
            Circle(0, 0, 1).get_coordinates_at_x(0.5),
            [(0.5, 0.8660254037844386), (0.5, -0.8660254037844386)],
        )
        with self.assertRaises(ValueError):
            Circle(0, 0, 1).get_coordinates_at_x(2)

    def test_create_points_at_x(self):
        circle = Circle(0, 0, 1)
        self.assertEqual(circle.create_points_at_x(0)[0].y, 1)
        self.assertEqual(circle.create_points_at_x(1)[0].y, 0)
        self.assertEqual(circle.create_points_at_x(-1)[0].y, 0)
        self.assertEqual(circle.create_points_at_x(0.5)[0].y, 0.8660254037844386)
        with self.assertRaises(ValueError):
            circle.create_points_at_x(2)

    def test_get_coordinates_at_y(self):
        self.assertListEqual(Circle(0, 0, 1).get_coordinates_at_y(0), [(1, 0), (-1, 0)])
        self.assertListEqual(Circle(0, 0, 1).get_coordinates_at_y(1), [(0, 1)])
        self.assertListEqual(Circle(0, 0, 1).get_coordinates_at_y(-1), [(0, -1)])
        self.assertListEqual(
            Circle(0, 0, 1).get_coordinates_at_y(0.5),
            [(0.8660254037844386, 0.5), (-0.8660254037844386, 0.5)],
        )
        with self.assertRaises(ValueError):
            Circle(0, 0, 1).get_coordinates_at_y(2)

    def test_create_points_at_y(self):
        circle = Circle(0, 0, 1)
        self.assertEqual(circle.create_points_at_y(0)[0].x, 1)
        self.assertEqual(circle.create_points_at_y(1)[0].x, 0)
        self.assertEqual(circle.create_points_at_y(-1)[0].x, 0)
        self.assertEqual(circle.create_points_at_y(0.5)[0].x, 0.8660254037844386)
        with self.assertRaises(ValueError):
            circle.create_points_at_y(2)

    def test_get_coordinates_at_angle(self):
        self.assertEqual(
            Circle(0, 0, 1).get_coordinates_at_angle(0, degrees=True), (1, 0)
        )
        self.assertAlmostEqual(
            Circle(0, 0, 1).get_coordinates_at_angle(90, degrees=True)[0], 0, places=15
        )
        self.assertAlmostEqual(
            Circle(0, 0, 1).get_coordinates_at_angle(90, degrees=True)[1], 1, places=15
        )
        self.assertEqual(
            Circle(0, 0, 1).get_coordinates_at_angle(180, degrees=True)[0], -1
        )
        self.assertEqual(
            Circle(0, 0, 1).get_coordinates_at_angle(270, degrees=True)[1], -1
        )
        self.assertEqual(
            Circle(0, 0, 1).get_coordinates_at_angle(360, degrees=True)[0], 1
        )
        # Test with degrees=False
        self.assertEqual(Circle(0, 0, 1).get_coordinates_at_angle(0), (1, 0))
        self.assertEqual(Circle(0, 0, 1).get_coordinates_at_angle(np.pi / 2)[1], 1)
        self.assertEqual(Circle(0, 0, 1).get_coordinates_at_angle(np.pi)[0], -1)

        # Test with other angle and radius and center
        self.assertEqual(
            Circle(1, 1, 2).get_coordinates_at_angle(0, degrees=True), (3, 1)
        )

    def test_create_point_at_angle(self):
        point = Circle(0, 0, 1).create_point_at_angle(0)
        self.assertEqual(point.x, 1)
        self.assertEqual(point.y, 0)
        point = Circle(0, 0, 1).create_point_at_angle(np.pi / 2)
        self.assertAlmostEqual(point.x, 0, places=15)
        self.assertAlmostEqual(point.y, 1, places=15)

    def test_get_center_coordinates(self):
        self.assertEqual(Circle(0, 0, 1).get_center_coordinates(), (0, 0))
        self.assertEqual(Circle(1, 1, 1).get_center_coordinates(), (1, 1))
        self.assertEqual(Circle(2, 3, 1).get_center_coordinates(), (2, 3))
        self.assertEqual(Circle(2.5, 3.5, 1).get_center_coordinates(), (2.5, 3.5))

    def test_create_center_point(self):
        the_circle = Circle(2.5, 3.5, 8)
        center_point = the_circle.create_center_point()
        self.assertEqual(center_point.x, 2.5)
        self.assertEqual(center_point.y, 3.5)

    def test_copy(self):
        circle = Circle(0, 0, 1)
        circle_copy = circle.copy()
        self.assertEqual(circle.x_center, circle_copy.x_center)
        self.assertEqual(circle.y_center, circle_copy.y_center)
        self.assertEqual(circle.radius, circle_copy.radius)
        self.assertEqual(circle.color, circle_copy.color)
        self.assertEqual(circle.line_width, circle_copy.line_width)
        self.assertEqual(circle.fill, circle_copy.fill)
        self.assertEqual(circle.line_style, circle_copy.line_style)
        self.assertEqual(circle.fill_alpha, circle_copy.fill_alpha)


class TestRectangle(unittest.TestCase):
    def test_width_and_height(self):
        with self.assertRaises(ValueError):
            Rectangle(0, 0, 0, 0)
        with self.assertRaises(ValueError):
            Rectangle(0, 0, -1, 1)
        with self.assertRaises(ValueError):
            Rectangle(0, 0, 1, -1)
        with self.assertRaises(ValueError):
            Rectangle(0, 0, -1, -1)

    def test_from_points(self):
        with self.assertRaises(ValueError):
            Rectangle.from_points(Point(0, 0), Point(0, 1))
        with self.assertRaises(ValueError):
            Rectangle.from_points(Point(0, 0), Point(1, 0))
        with self.assertRaises(ValueError):
            Rectangle.from_points(Point(1, 0), Point(0, 0))
        with self.assertRaises(ValueError):
            Rectangle.from_points(Point(2, 3), Point(4, 3))
        rect = Rectangle.from_points(Point(0, 0), Point(1, 1))
        self.assertEqual(rect.x_bottom_left, 0)
        self.assertEqual(rect.y_bottom_left, 0)
        self.assertEqual(rect.width, 1)
        self.assertEqual(rect.height, 1)
        rect = Rectangle.from_points(Point(1, 3), Point(7, 2))
        self.assertEqual(rect.x_bottom_left, 1)
        self.assertEqual(rect.y_bottom_left, 2)
        self.assertEqual(rect.width, 6)
        self.assertEqual(rect.height, 1)

    def test_from_center(self):
        with self.assertRaises(ValueError):
            Rectangle.from_center(0, 0, 0, 0)
        with self.assertRaises(ValueError):
            Rectangle.from_center(0, 0, -1, 1)
        with self.assertRaises(ValueError):
            Rectangle.from_center(0, 0, 1, -1)
        with self.assertRaises(ValueError):
            Rectangle.from_center(0, 0, -1, -1)
        rect = Rectangle.from_center(0, 0, 1, 1)
        self.assertEqual(rect.x_bottom_left, -0.5)
        self.assertEqual(rect.y_bottom_left, -0.5)
        self.assertEqual(rect.width, 1)
        self.assertEqual(rect.height, 1)
        rect = Rectangle.from_center(1, 3, 6, 1)
        self.assertEqual(rect.x_bottom_left, -2)
        self.assertEqual(rect.y_bottom_left, 2.5)
        self.assertEqual(rect.width, 6)
        self.assertEqual(rect.height, 1)

    def test_contains(self):
        self.assertTrue(Point(0, 0) in Rectangle(0, 0, 1, 1))
        self.assertTrue(Point(0, 1) in Rectangle(0, 0, 1, 1))
        self.assertTrue(Point(1, 0) in Rectangle(0, 0, 1, 1))
        self.assertTrue(Point(1, 1) in Rectangle(0, 0, 1, 1))
        self.assertFalse(Point(0, 2) in Rectangle(0, 0, 1, 1))
        self.assertFalse(Point(2, 0) in Rectangle(0, 0, 1, 1))
        self.assertFalse(Point(2, 2) in Rectangle(0, 0, 1, 1))

    def test_area(self):
        self.assertEqual(Rectangle(0, 0, 1, 1).get_area(), 1)
        self.assertEqual(Rectangle(0, 0, 2, 1).get_area(), 2)
        self.assertEqual(Rectangle(0, 0, 1, 2).get_area(), 2)
        self.assertEqual(Rectangle(0, 0, 2, 2).get_area(), 4)
        self.assertEqual(Rectangle(0, 0, 3, 2).get_area(), 6)
        self.assertEqual(Rectangle(0, 0, 2, 3).get_area(), 6)
        self.assertEqual(Rectangle(0, 0, 3, 3).get_area(), 9)
        self.assertEqual(Rectangle(0, 0, 4, 3).get_area(), 12)
        self.assertEqual(Rectangle(0, 0, 3, 4).get_area(), 12)
        self.assertEqual(Rectangle(0, 0, 4, 4).get_area(), 16)

    def test_get_center_coordinates(self):
        self.assertEqual(Rectangle(0, 0, 1, 1).get_center_coordinates(), (0.5, 0.5))
        self.assertEqual(Rectangle(0, 0, 2, 1).get_center_coordinates(), (1, 0.5))
        self.assertEqual(Rectangle(0, 0, 1, 2).get_center_coordinates(), (0.5, 1))
        self.assertEqual(Rectangle(0, 0, 2, 2).get_center_coordinates(), (1, 1))
        self.assertEqual(Rectangle(0, 0, 3, 2).get_center_coordinates(), (1.5, 1))
        self.assertEqual(Rectangle(0, 0, 2, 3).get_center_coordinates(), (1, 1.5))
        self.assertEqual(Rectangle(0, 0, 3, 3).get_center_coordinates(), (1.5, 1.5))
        self.assertEqual(Rectangle(0, 0, 4, 3).get_center_coordinates(), (2, 1.5))
        self.assertEqual(Rectangle(0, 0, 3, 4).get_center_coordinates(), (1.5, 2))
        self.assertEqual(Rectangle(0, 0, 4, 4).get_center_coordinates(), (2, 2))

    def test_create_center_point(self):
        rect = Rectangle(0, 0, 1, 1)
        center_point = rect.create_center_point()
        self.assertEqual(center_point.x, 0.5)
        self.assertEqual(center_point.y, 0.5)
        rect = Rectangle(0, 0, 2, 1)
        center_point = rect.create_center_point()
        self.assertEqual(center_point.x, 1)
        self.assertEqual(center_point.y, 0.5)

    def test_get_coordinates_at_x(self):
        self.assertListEqual(
            Rectangle(0, 0, 1, 1).get_coordinates_at_x(0.5), [(0.5, 0), (0.5, 1)]
        )
        self.assertListEqual(
            Rectangle(2, 3, 5, 5).get_coordinates_at_x(2.3), [(2.3, 3), (2.3, 8)]
        )

    def test_create_points_at_x(self):
        rect = Rectangle(0, 0, 1, 1)
        points = rect.create_points_at_x(0.5)
        self.assertEqual(points[0].x, 0.5)
        self.assertEqual(points[0].y, 0)
        self.assertEqual(points[1].x, 0.5)
        self.assertEqual(points[1].y, 1)
        rect = Rectangle(2, 3, 5, 5)
        points = rect.create_points_at_x(2.3)
        self.assertEqual(points[0].x, 2.3)
        self.assertEqual(points[0].y, 3)
        self.assertEqual(points[1].x, 2.3)
        self.assertEqual(points[1].y, 8)

    def test_get_coordintes_at_y(self):
        self.assertListEqual(
            Rectangle(0, 0, 1, 1).get_coordinates_at_y(0.5), [(0, 0.5), (1, 0.5)]
        )
        self.assertListEqual(
            Rectangle(2, 3, 5, 5).get_coordinates_at_y(4.3), [(2, 4.3), (7, 4.3)]
        )

    def test_create_points_at_y(self):
        rect = Rectangle(0, 0, 1, 1)
        points = rect.create_points_at_y(0.5)
        self.assertEqual(points[0].x, 0)
        self.assertEqual(points[0].y, 0.5)
        self.assertEqual(points[1].x, 1)
        self.assertEqual(points[1].y, 0.5)
        rect = Rectangle(2, 3, 5, 5)
        points = rect.create_points_at_y(4.3)
        self.assertEqual(points[0].x, 2)
        self.assertEqual(points[0].y, 4.3)
        self.assertEqual(points[1].x, 7)
        self.assertEqual(points[1].y, 4.3)

    def test_copy(self):
        rect = Rectangle(0, 0, 1, 1)
        rect_copy = rect.copy()
        self.assertEqual(rect.x_bottom_left, rect_copy.x_bottom_left)
        self.assertEqual(rect.y_bottom_left, rect_copy.y_bottom_left)
        self.assertEqual(rect.width, rect_copy.width)
        self.assertEqual(rect.height, rect_copy.height)
        self.assertEqual(rect.color, rect_copy.color)
        self.assertEqual(rect.line_width, rect_copy.line_width)
        self.assertEqual(rect.fill, rect_copy.fill)
        self.assertEqual(rect.line_style, rect_copy.line_style)
        self.assertEqual(rect.fill_alpha, rect_copy.fill_alpha)


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

        self.assertAlmostEqual(centroid.x, -0.00476190476190, places=6)
        self.assertAlmostEqual(centroid.y, 0.70317460317460, places=6)

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
        polygon.apply_transform(transform_matrix_2x2)

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
        self.assertEqual(intersection[0].x, 0.5)
        self.assertEqual(intersection[0].y, 1)
        self.assertEqual(intersection[1].x, 1)
        self.assertEqual(intersection[1].y, 0.5)


if __name__ == "__main__":
    unittest.main()
