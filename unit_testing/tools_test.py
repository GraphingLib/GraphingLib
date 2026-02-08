import unittest

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from graphinglib.tools import (
    MathematicalObject,
    _copy_with_overrides,
    get_contrasting_shade,
)


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
                return TestObject(self.value**other)

        obj = TestObject(10)
        self.assertEqual((obj + 5).value, 15)
        self.assertEqual((5 + obj).value, 15)
        self.assertEqual((obj - 3).value, 7)
        self.assertEqual((3 - obj).value, -7)
        self.assertEqual((obj * 2).value, 20)
        self.assertEqual((2 * obj).value, 20)
        self.assertEqual((obj / 2).value, 5)
        self.assertEqual((2 / obj).value, 0.2)
        self.assertEqual((obj**2).value, 100)

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
            _ = obj**2

        with self.assertRaises(NotImplementedError):
            obj -= 5
        with self.assertRaises(NotImplementedError):
            obj *= 2
        with self.assertRaises(NotImplementedError):
            obj /= 2
        with self.assertRaises(NotImplementedError):
            obj **= 2


class TestGetContrastingShade(unittest.TestCase):
    def test_dark_rgb_returns_white(self):
        self.assertEqual(get_contrasting_shade((0, 0, 0)), "white")
        self.assertEqual(get_contrasting_shade((5, 5, 5)), "white")

    def test_light_rgb_returns_black(self):
        self.assertEqual(get_contrasting_shade((255, 255, 255)), "black")
        self.assertEqual(get_contrasting_shade((250, 240, 230)), "black")

    def test_color_strings_use_matplotlib_parsing(self):
        self.assertEqual(get_contrasting_shade("navy"), "white")
        self.assertEqual(get_contrasting_shade("yellow"), "black")


class TestCopyWithOverrides(unittest.TestCase):
    def test_copy_with_overrides(self):
        class Dummy:
            def __init__(self):
                self._value = 1
                self._fixed = 2

            @property
            def value(self):
                return self._value

            @value.setter
            def value(self, value):
                self._value = value

            @property
            def fixed(self):
                return self._fixed

        obj = Dummy()
        copied = _copy_with_overrides(obj, value=3)
        self.assertEqual(copied.value, 3)
        self.assertEqual(obj.value, 1)

        with self.assertRaisesRegex(AttributeError, "Did you mean 'value'"):
            _copy_with_overrides(obj, valu=3)

        with self.assertRaisesRegex(AttributeError, "read-only property"):
            _copy_with_overrides(obj, fixed=4)

        with self.assertRaisesRegex(AttributeError, "has no public writable property"):
            _copy_with_overrides(obj, _value=5)

    def test_copy_with_rejects_method_override(self):
        class Dummy:
            def __init__(self):
                self._value = 1

            @property
            def value(self):
                return self._value

            @value.setter
            def value(self, value):
                self._value = value

            def copy(self):
                return self

        obj = Dummy()
        with self.assertRaisesRegex(
            AttributeError, "has no public writable property 'copy'"
        ):
            _copy_with_overrides(obj, copy="not allowed")


if __name__ == "__main__":
    unittest.main()
