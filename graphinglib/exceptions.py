"""
Exception hierarchy for GraphingLib.

Every error GraphingLib raises derives from :class:`GraphingLibError`, so
``except GraphingLibError`` (or its alias ``except GraphingException``) catches anything the
library raises. Most leaf classes *also* derive from the matching built-in exception
(``ValueError``, ``TypeError``, ...), so existing ``except ValueError`` / ``except TypeError``
code keeps working too.

Choosing which to raise (ask top to bottom, raise the first that matches):

============================================  ==================================
The failure is...                             Raise
============================================  ==================================
an argument of the wrong *type*               ``InvalidParameterTypeError``
an argument of the right type, bad *value*    ``InvalidParameterError``
two valid arguments that *conflict*           ``IncompatibleArgumentsError``
an operation invalid for the current *state*  ``InvalidOperationError``
...a SmartFigure grid/layout constraint       ``LayoutError``
a named style/resource that doesn't exist     ``StyleNotFoundError``
a style file that is corrupt/un-updatable     ``StyleFileError``
a missing optional dependency                 ``MissingOptionalDependencyError``
a capability that isn't supported             ``UnsupportedFeatureError``
a failure inside matplotlib/scipy/numpy       ``PlottingError`` (wrap with ``from exc``)
============================================  ==================================

Message style: one sentence ending in a period; state what is wrong, what was received, and
what is valid, e.g. ``"aspect_ratio must be 'equal', 'auto', or a positive float; got -1.0."``.
"""


class GraphingLibError(Exception):
    """
    Base class for every exception raised by GraphingLib.

    Catch this to handle any GraphingLib-specific failure. It is rarely raised directly;
    prefer one of the specific subclasses below so the failure mode is explicit.
    """


# Backward-compatible alias: ``GraphingException`` was the original catch-all base and is
# part of the public API. It remains a fully supported way to catch any GraphingLib error.
GraphingException = GraphingLibError


class InvalidParameterError(GraphingLibError, ValueError):
    """
    A public argument has the right type but an unacceptable **value**.

    Use for out-of-range numbers, unrecognised option strings, malformed tuples, and similar
    (e.g. a negative radius, an unknown line style). Also a ``ValueError``.
    """


class InvalidParameterTypeError(GraphingLibError, TypeError):
    """
    A public argument has the wrong **type**.

    Use when the value is not an instance of any accepted type (e.g. ``num_rows`` given a
    string, an unsupported operand in ``curve + other``). Also a ``TypeError``.

    A wrong *value* of the correct type (e.g. ``num_rows`` given ``0``) is an
    :class:`InvalidParameterError`, not this.
    """


class IncompatibleArgumentsError(GraphingLibError, ValueError):
    """
    Several arguments are individually valid but **conflict** with one another.

    Use for mutually exclusive options set together, co-required options where only one is
    given, or length mismatches between related sequences (e.g. tick positions and tick
    labels of different lengths). Also a ``ValueError``.
    """


class InvalidOperationError(GraphingLibError, RuntimeError):
    """
    The operation is not allowed for the object's current **state or configuration**.

    Use when the arguments are fine but the object cannot honour the call as it stands (e.g.
    plotting a figure with no elements, adding a twin axis that already exists). Also a
    ``RuntimeError``.
    """


class LayoutError(InvalidOperationError):
    """
    A :class:`~graphinglib.SmartFigure` grid/geometry constraint is violated.

    A specialised :class:`InvalidOperationError` for layout problems: overlapping spans,
    children that do not fit the target grid, indexing a leaf as if it were a layout, etc.
    """


class StyleNotFoundError(GraphingLibError, LookupError):
    """
    A named style (or similar resource) could not be found.

    Use when a requested ``figure_style`` does not exist among the built-in or user styles.
    Also a ``LookupError``.
    """


class StyleFileError(GraphingLibError):
    """
    A style file could not be read, parsed, or automatically updated.

    An integrity error about GraphingLib's own configuration data rather than user input;
    this is the one place where directing the user to report the problem is appropriate.
    """


class MissingOptionalDependencyError(GraphingLibError, ImportError):
    """
    A feature requires an optional dependency that is not installed.

    Use when a feature gated behind a ``graphinglib[extra]`` is invoked without the extra.
    The message should name the extra and how to install it. Also an ``ImportError``.
    """


class UnsupportedFeatureError(GraphingLibError, NotImplementedError):
    """
    A requested capability is deliberately not available.

    Use for things GraphingLib knowingly does not support (e.g. a 3D projection, or a
    WCS-only feature used on a non-WCS figure). Also a ``NotImplementedError``.
    """


class PlottingError(GraphingLibError, RuntimeError):
    """
    A matplotlib, scipy, or numpy call failed while building or rendering an element.

    Use to wrap a lower-level exception raised deep inside a dependency (a failed plot call,
    a fit that did not converge, an interpolation error, ...) so the user sees GraphingLib
    context instead of a bare backend error. Always chain the original with
    ``raise PlottingError(...) from exc``. Also a ``RuntimeError``.
    """
