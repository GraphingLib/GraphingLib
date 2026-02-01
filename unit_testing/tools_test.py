import unittest

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from graphinglib.tools import MathematicalObject


class TestMathematicalObject(unittest.TestCase):
    def test_object_not_implemented(self):
        with self.assertRaises(TypeError):
            MathematicalObject()

    def test_valid_children_class(self):
        class TestObject(MathematicalObject):
            def __init__(self, value: int):
                self.value = value

            def __add__(self, other: int) -> Self:
                return TestObject(self.value + other)

            def __sub__(self, other: int) -> Self:
                return TestObject(self.value - other)

            def __mul__(self, other: int) -> Self:
                return TestObject(self.value * other)

            def __truediv__(self, other: int) -> Self:
                return TestObject(self.value / other)

            def __pow__(self, other: int) -> Self:
                return TestObject(self.value ** other)

        obj = TestObject(10)
        self.assertEqual((obj + 5).value, 15)
        self.assertEqual((5 + obj).value, 15)
        self.assertEqual((obj - 3).value, 7)
        self.assertEqual((3 - obj).value, -7)
        self.assertEqual((obj * 2).value, 20)
        self.assertEqual((2 * obj).value, 20)
        self.assertEqual((obj / 2).value, 5)
        self.assertEqual((2 / obj).value, 0.2)
        self.assertEqual((obj ** 2).value, 100)

        obj += 10
        self.assertEqual(obj.value, 20)
        obj -= 5
        self.assertEqual(obj.value, 15)
        obj *= 2
        self.assertEqual(obj.value, 30)
        obj /= 3
        self.assertEqual(obj.value, 10)
        obj **= 3
        self.assertEqual(obj.value, 1000)

    def test_invalid_children_class(self):
        class IncompleteObject(MathematicalObject):
            def __init__(self, value: int):
                self.value = value

            def __add__(self, other: int) -> Self:
                return IncompleteObject(self.value + other)

        obj = IncompleteObject(10)
        with self.assertRaises(NotImplementedError):
            _ = obj - 5
        with self.assertRaises(NotImplementedError):
            _ = obj * 2
        with self.assertRaises(NotImplementedError):
            _ = obj / 2
        with self.assertRaises(NotImplementedError):
            _ = obj ** 2

        with self.assertRaises(NotImplementedError):
            obj -= 5
        with self.assertRaises(NotImplementedError):
            obj *= 2
        with self.assertRaises(NotImplementedError):
            obj /= 2
        with self.assertRaises(NotImplementedError):
            obj **= 2


if __name__ == "__main__":
    unittest.main()
