.. _api_ref:

API reference
=============

.. currentmodule:: graphinglib

.. _objects_api:

Figure and MultiFigure
----------------------

.. autosummary::
    :toctree: generated/
    :template: class
    :nosignatures:

    Figure
    MultiFigure

.. autosummary::
    :toctree: generated/
    :template: smart_figure_classes
    :nosignatures:

    SmartFigure
    SmartFigureWCS
    SmartTwinAxis

Plottables
----------

.. autosummary::
    :toctree: generated/
    :template: class
    :nosignatures:

    Arrow
    Circle
    Contour
    Curve
    Ellipse
    FitFromExponential
    FitFromFOTF
    FitFromFunction
    FitFromGaussian
    FitFromLog
    FitFromPolynomial
    FitFromSine
    FitFromSquareRoot
    Heatmap
    Histogram
    Hlines
    Line
    PlottableAxMethod
    Point
    Polygon
    Rectangle
    Scatter
    Stream
    Table
    Text
    VectorField
    Vlines

Legend elements
---------------
.. autosummary::
    :toctree: generated/
    :template: class
    :nosignatures:

    LegendLine
    LegendMarker
    LegendPatch

Utility functions
-----------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    get_color
    get_colors
    get_default_style
    get_styles
    set_default_style

Tools
-----
.. autosummary::
    :toctree: generated/
    :template: class
    :nosignatures:

    MathematicalObject

Errors
------
Every exception GraphingLib raises derives from :class:`GraphingLibError` (also available
under its alias ``GraphingException``), so ``except graphinglib.GraphingException`` catches
anything the library raises. Most also derive from the matching built-in exception
(``ValueError``, ``TypeError``, ...), so existing ``except ValueError`` / ``except TypeError``
code keeps working too.

.. autoexception:: GraphingLibError
.. autoexception:: InvalidParameterError
.. autoexception:: InvalidParameterTypeError
.. autoexception:: IncompatibleArgumentsError
.. autoexception:: InvalidOperationError
.. autoexception:: LayoutError
.. autoexception:: StyleNotFoundError
.. autoexception:: StyleFileError
.. autoexception:: MissingOptionalDependencyError
.. autoexception:: UnsupportedFeatureError
.. autoexception:: PlottingError
