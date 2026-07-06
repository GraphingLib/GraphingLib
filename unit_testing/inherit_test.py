import unittest
from copy import copy, deepcopy

import numpy as np

from graphinglib.inherit import (
    INHERIT,
    Inherit,
    is_inherit,
    resolve_or,
    resolved,
    strip_inherit,
)


class TestInheritSentinel(unittest.TestCase):
    def test_is_inherit(self):
        self.assertTrue(is_inherit(INHERIT))
        self.assertFalse(is_inherit(None))
        self.assertFalse(is_inherit(0))
        self.assertFalse(is_inherit("INHERIT"))

    def test_copy_preserves_identity(self):
        self.assertIs(copy(INHERIT), INHERIT)
        self.assertIs(deepcopy(INHERIT), INHERIT)

    def test_bool_raises(self):
        # Regression test: INHERIT used to be silently truthy, which made
        # `if self._some_flag:` behave as True for unresolved style parameters
        # (e.g. the Histogram normalize bug fixed in PR #688).
        with self.assertRaises(TypeError):
            bool(INHERIT)
        with self.assertRaises(TypeError):
            if INHERIT:
                pass

    def test_equality_is_identity_based(self):
        self.assertTrue(INHERIT == Inherit() or INHERIT != Inherit())
        self.assertTrue(INHERIT == INHERIT)
        self.assertFalse(INHERIT != INHERIT)


class TestResolutionHelpers(unittest.TestCase):
    def test_resolved_passes_through_concrete_values(self):
        self.assertEqual(resolved(2.5), 2.5)
        self.assertEqual(resolved("red"), "red")
        self.assertIsNone(resolved(None))

    def test_resolved_raises_on_inherit(self):
        with self.assertRaises(TypeError):
            resolved(INHERIT)

    def test_resolve_or(self):
        self.assertEqual(resolve_or(2.5, 1.0), 2.5)
        self.assertEqual(resolve_or(INHERIT, 1.0), 1.0)
        self.assertFalse(resolve_or(INHERIT, False))
        self.assertIsNone(resolve_or(None, "default"))

    def test_strip_inherit(self):
        params = {"a": 1, "b": INHERIT, "c": None, "d": "x"}
        self.assertDictEqual(strip_inherit(params), {"a": 1, "c": None, "d": "x"})

    def test_strip_inherit_is_safe_for_array_values(self):
        # Regression test: the old `v != INHERIT` filter idiom broadcast the
        # comparison when v was an ndarray, making the filter raise; strip_inherit
        # uses identity instead.
        params = {"levels": np.array([1, 2, 3]), "cmap": INHERIT}
        result = strip_inherit(params)
        self.assertEqual(list(result.keys()), ["levels"])
        self.assertTrue(np.array_equal(result["levels"], np.array([1, 2, 3])))

    def test_strip_inherit_does_not_mutate_input(self):
        params = {"a": INHERIT}
        strip_inherit(params)
        self.assertIn("a", params)
