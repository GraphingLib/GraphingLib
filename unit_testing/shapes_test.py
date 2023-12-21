import unittest

import numpy as np

from graphinglib.graph_elements import Point
from graphinglib.shapes import Circle, Rectangle


class TestCircle(unittest.TestCase):
    def test_radius(self):
        with self.assertRaises(ValueError):
            Circle(0, 0, -1)
        with self.assertRaises(ValueError):
            Circle(0, 0, 0)
        with self.assertRaises(ValueError):
            Circle(5, -8, -2)

    def test_circumference(self):
        self.assertEqual(Circle(0, 0, 1).circumference(), 2 * 3.141592653589793)
        self.assertEqual(Circle(0, 0, 2).circumference(), 4 * 3.141592653589793)
        self.assertEqual(Circle(0, 0, 3).circumference(), 6 * 3.141592653589793)
        # If circle is not centered at origin
        self.assertEqual(Circle(1, 1, 1).circumference(), 2 * 3.141592653589793)
        self.assertEqual(Circle(1, 1, 2).circumference(), 4 * 3.141592653589793)
        self.assertEqual(Circle(1, 1, 3).circumference(), 6 * 3.141592653589793)

    def test_area(self):
        self.assertEqual(Circle(0, 0, 1).area(), 3.141592653589793)
        self.assertEqual(Circle(0, 0, 2).area(), 4 * 3.141592653589793)
        self.assertEqual(Circle(0, 0, 3).area(), 9 * 3.141592653589793)
        # If circle is not centered at origin
        self.assertEqual(Circle(1, 1, 1).area(), 3.141592653589793)
        self.assertEqual(Circle(1, 1, 2).area(), 4 * 3.141592653589793)
        self.assertEqual(Circle(1, 1, 3).area(), 9 * 3.141592653589793)

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

    def test_get_center_point(self):
        self.assertEqual(Circle(0, 0, 1).get_center_point(), (0, 0))
        self.assertEqual(Circle(1, 1, 1).get_center_point(), (1, 1))
        self.assertEqual(Circle(2, 3, 1).get_center_point(), (2, 3))
        self.assertEqual(Circle(2.5, 3.5, 1).get_center_point(), (2.5, 3.5))
        the_circle = Circle(2.5, 3.5, 1)
        center_point = the_circle.get_center_point(as_point_object=True)
        self.assertEqual(center_point.x, 2.5)
        self.assertEqual(center_point.y, 3.5)

    def test_get_equation(self):
        self.assertEqual(Circle(0, 0, 1).get_equation(), "(x - 0)^2 + (y - 0)^2 = 1^2")
        self.assertEqual(Circle(1, 1, 1).get_equation(), "(x - 1)^2 + (y - 1)^2 = 1^2")
        self.assertEqual(Circle(2, 3, 1).get_equation(), "(x - 2)^2 + (y - 3)^2 = 1^2")
        self.assertEqual(
            Circle(2.5, 3.5, 1).get_equation(), "(x - 2.5)^2 + (y - 3.5)^2 = 1^2"
        )

    def test_get_points_at_x(self):
        self.assertListEqual(Circle(0, 0, 1).get_points_at_x(0), [(0, 1), (0, -1)])
        self.assertListEqual(Circle(0, 0, 1).get_points_at_x(1), [(1, 0)])
        self.assertListEqual(Circle(0, 0, 1).get_points_at_x(-1), [(-1, 0)])
        self.assertListEqual(
            Circle(0, 0, 1).get_points_at_x(0.5),
            [(0.5, 0.8660254037844386), (0.5, -0.8660254037844386)],
        )
        with self.assertRaises(ValueError):
            Circle(0, 0, 1).get_points_at_x(2)

    def test_get_points_at_y(self):
        self.assertListEqual(Circle(0, 0, 1).get_points_at_y(0), [(1, 0), (-1, 0)])
        self.assertListEqual(Circle(0, 0, 1).get_points_at_y(1), [(0, 1)])
        self.assertListEqual(Circle(0, 0, 1).get_points_at_y(-1), [(0, -1)])
        self.assertListEqual(
            Circle(0, 0, 1).get_points_at_y(0.5),
            [(0.8660254037844386, 0.5), (-0.8660254037844386, 0.5)],
        )
        with self.assertRaises(ValueError):
            Circle(0, 0, 1).get_points_at_y(2)

    def test_get_point_at_angle(self):
        self.assertEqual(Circle(0, 0, 1).get_point_at_angle(0, degrees=True), (1, 0))
        self.assertAlmostEqual(
            Circle(0, 0, 1).get_point_at_angle(90, degrees=True)[0], 0, places=15
        )
        self.assertAlmostEqual(
            Circle(0, 0, 1).get_point_at_angle(90, degrees=True)[1], 1, places=15
        )
        self.assertEqual(Circle(0, 0, 1).get_point_at_angle(180, degrees=True)[0], -1)
        self.assertEqual(Circle(0, 0, 1).get_point_at_angle(270, degrees=True)[1], -1)
        self.assertEqual(Circle(0, 0, 1).get_point_at_angle(360, degrees=True)[0], 1)
        # Test with degrees=False
        self.assertEqual(Circle(0, 0, 1).get_point_at_angle(0), (1, 0))
        self.assertEqual(Circle(0, 0, 1).get_point_at_angle(np.pi / 2)[1], 1)
        self.assertEqual(Circle(0, 0, 1).get_point_at_angle(np.pi)[0], -1)
        # Test with as_point_object=True
        point = Circle(0, 0, 1).get_point_at_angle(0, as_point_object=True)
        self.assertEqual(point.x, 1)
        self.assertEqual(point.y, 0)
        point = Circle(0, 0, 1).get_point_at_angle(np.pi / 2, as_point_object=True)
        self.assertAlmostEqual(point.x, 0, places=15)
        self.assertAlmostEqual(point.y, 1, places=15)
        # Test with other angle and radius and center
        self.assertEqual(Circle(1, 1, 2).get_point_at_angle(0, degrees=True), (3, 1))

    def test_get_center_point(self):
        self.assertEqual(Circle(0, 0, 1).get_center_point(), (0, 0))
        self.assertEqual(Circle(1, 1, 1).get_center_point(), (1, 1))
        self.assertEqual(Circle(2, 3, 1).get_center_point(), (2, 3))
        self.assertEqual(Circle(2.5, 3.5, 1).get_center_point(), (2.5, 3.5))
        the_circle = Circle(2.5, 3.5, 8)
        center_point = the_circle.get_center_point(as_point_object=True)
        self.assertEqual(center_point.x, 2.5)
        self.assertEqual(center_point.y, 3.5)
