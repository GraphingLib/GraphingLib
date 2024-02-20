import unittest

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba

from graphinglib.graph_elements import Point
from graphinglib.shapes import Arrow, Circle, Line, Rectangle


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


if __name__ == "__main__":
    unittest.main()
