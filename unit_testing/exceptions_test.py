import unittest

import matplotlib

matplotlib.use("Agg")
import numpy as np

import graphinglib as gl
from graphinglib.file_manager import set_default_style
from graphinglib.tools import _require_optional_dependency
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


def _empty_figure() -> None:
    gl.Figure()._prepare_figure()


def _twin_axis_twice() -> None:
    sf = gl.SmartFigure(elements=[gl.Curve([0, 1], [0, 1])])
    sf.create_twin_axis(is_y=True)
    sf.create_twin_axis(is_y=True)


def _assign_into_leaf() -> None:
    sf = gl.SmartFigure(elements=[gl.Curve([0, 1], [0, 1])])
    sf[0, 0] = gl.SmartFigure()


def _non_converging_fit() -> None:
    noise = gl.Scatter(np.linspace(0, 10, 50), np.random.rand(50))
    gl.FitFromSine(noise, guesses=[1, 1, 1, 1], max_iterations=1)


# (description, callable that misuses the public API, expected class, built-in it must also be)
MISUSE_CASES = [
    (
        "negative radius",
        lambda: gl.Circle(0, 0, radius=-1),
        InvalidParameterError,
        ValueError,
    ),
    (
        "bad image shape",
        lambda: gl.Heatmap([1, 2, 3, 4]),
        InvalidParameterError,
        ValueError,
    ),
    (
        "unknown style name",
        lambda: set_default_style("nope_zzz"),
        InvalidParameterError,
        ValueError,
    ),
    (
        "mismatched x/y",
        lambda: gl.Curve([0, 1, 2], [0, 1]),
        IncompatibleArgumentsError,
        ValueError,
    ),
    (
        "wrong operand type",
        lambda: gl.Curve([0, 1], [0, 1]) + "x",
        InvalidParameterTypeError,
        TypeError,
    ),
    ("nothing to plot", _empty_figure, InvalidOperationError, RuntimeError),
    ("twin axis twice", _twin_axis_twice, InvalidOperationError, RuntimeError),
    ("assign into leaf", _assign_into_leaf, LayoutError, RuntimeError),
    (
        "unknown figure style",
        lambda: gl.Figure(figure_style="nope_zzz")._prepare_figure(),
        StyleNotFoundError,
        LookupError,
    ),
    (
        "unsupported projection",
        lambda: gl.SmartFigure(projection="3d"),
        UnsupportedFeatureError,
        NotImplementedError,
    ),
    (
        "missing optional dep",
        lambda: _require_optional_dependency(False, "PDF export", "pdf", "pypdfium2"),
        MissingOptionalDependencyError,
        ImportError,
    ),
    ("fit does not converge", _non_converging_fit, PlottingError, RuntimeError),
]


class TestPublicMisuseRaisesRightType(unittest.TestCase):
    """End-to-end contract: a public misuse raises the specific GraphingLib type, and it
    stays catchable both as the matching built-in and as GraphingException."""

    def test_misuse_raises_specific_type(self):
        for desc, func, expected, _ in MISUSE_CASES:
            with self.subTest(case=desc):
                with self.assertRaises(expected):
                    func()

    def test_misuse_is_catchable_as_builtin(self):
        for desc, func, _, builtin in MISUSE_CASES:
            with self.subTest(case=desc):
                with self.assertRaises(builtin):
                    func()

    def test_misuse_is_catchable_as_graphing_exception(self):
        for desc, func, _, _ in MISUSE_CASES:
            with self.subTest(case=desc):
                with self.assertRaises(GraphingException):
                    func()


if __name__ == "__main__":
    unittest.main()
