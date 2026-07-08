import unittest

import graphinglib as gl
from graphinglib.exceptions import (
    GraphingException,
    GraphingLibError,
    IncompatibleArgumentsError,
    InvalidOperationError,
    InvalidParameterError,
    InvalidParameterTypeError,
    LayoutError,
    MissingOptionalDependencyError,
    PlottingError,
    StyleFileError,
    StyleNotFoundError,
    UnsupportedFeatureError,
)

# Each leaf paired with the built-in exception it must also derive from (or None).
LEAVES_AND_BUILTINS = [
    (InvalidParameterError, ValueError),
    (InvalidParameterTypeError, TypeError),
    (IncompatibleArgumentsError, ValueError),
    (InvalidOperationError, RuntimeError),
    (LayoutError, RuntimeError),
    (StyleNotFoundError, LookupError),
    (StyleFileError, None),
    (MissingOptionalDependencyError, ImportError),
    (UnsupportedFeatureError, NotImplementedError),
    (PlottingError, RuntimeError),
]


class TestExceptionHierarchy(unittest.TestCase):
    def test_alias_identity(self):
        # GraphingException must stay a fully interchangeable alias of the root.
        self.assertIs(GraphingException, GraphingLibError)

    def test_all_leaves_derive_from_root(self):
        for leaf, _ in LEAVES_AND_BUILTINS:
            with self.subTest(leaf=leaf.__name__):
                self.assertTrue(issubclass(leaf, GraphingLibError))

    def test_leaves_derive_from_matching_builtin(self):
        for leaf, builtin in LEAVES_AND_BUILTINS:
            if builtin is None:
                continue
            with self.subTest(leaf=leaf.__name__):
                self.assertTrue(issubclass(leaf, builtin))

    def test_layout_error_is_an_invalid_operation(self):
        self.assertTrue(issubclass(LayoutError, InvalidOperationError))

    def test_leaf_is_catchable_as_root_and_builtin(self):
        # A leaf must be catchable both by the GraphingLib base and by its built-in, so
        # existing `except GraphingException` and `except ValueError` code keeps working.
        for leaf, builtin in LEAVES_AND_BUILTINS:
            with self.subTest(leaf=leaf.__name__):
                with self.assertRaises(GraphingException):
                    raise leaf("boom")
                if builtin is not None:
                    with self.assertRaises(builtin):
                        raise leaf("boom")

    def test_public_api_exports(self):
        for name in [
            "GraphingException",
            "GraphingLibError",
            "InvalidParameterError",
            "InvalidParameterTypeError",
            "IncompatibleArgumentsError",
            "InvalidOperationError",
            "LayoutError",
            "StyleNotFoundError",
            "StyleFileError",
            "MissingOptionalDependencyError",
            "UnsupportedFeatureError",
            "PlottingError",
        ]:
            with self.subTest(name=name):
                self.assertIn(name, gl.__all__)
                self.assertIs(getattr(gl, name), getattr(gl.exceptions, name))


if __name__ == "__main__":
    unittest.main()
