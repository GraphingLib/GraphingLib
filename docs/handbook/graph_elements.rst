=====================
Miscellaneous Objects
=====================

.. _point:
The :class:`~graphinglib.graph_elements.Point` Object
-----------------------------------------------------

The Point object is usefull for highlighting a specific point in a graph. It allows you to attach a label to it in addition to its coordinates. Here is how to declare a Point object and add its coordinates: ::

    point = gl.Point(1, 2, label="Something interesting here!")
    point.add_coordinates()

.. image:: ../images/point.png

There are many more parameters to be customized for the Point object, but those are all included in the figure style files and can therefore be left out most of the time. For the details on the other parameters, visit the :py:class:`Reference section on Point objects <graphinglib.graph_elements.Point>`.

.. seealso::

    The Point object is returned by methods of the :class:`~graphinglib.data_plotting_1d.Curve` objects like :py:meth:`~graphinglib.data_plotting_1d.Curve.get_point_at_x`, :py:meth:`~graphinglib.data_plotting_1d.Curve.get_points_at_y` and :py:meth:`~graphinglib.data_plotting_1d.Curve.intersection`.

The :class:`~graphinglib.graph_elements.Hlines` and :class:`~graphinglib.graph_elements.Vlines` Object
------------------------------------------------------------------------------------------------------

The Hlines and Vlines objects serve a similar purpose to the Point object, which is as markers for specific values in `x` or `y`. Here is an example of the use of Hlines and Vlines: ::

    import numpy as np
    import graphinglib as gl


    curve = gl.Curve.from_function(
        lambda x: 0.1 * x**2 + np.sin(3 * x) - np.cos(2 * x) + 1, 0, 5
    )
    hlines = gl.Vlines(
        [curve.x_data[20], curve.x_data[250]],
        [0, 0],
        [curve.y_data[20], curve.y_data[250]],
        line_styles="--",
        colors="gray",
    )
    vlines = gl.Hlines(
        [curve.y_data[20], curve.y_data[250]],
        [0, 0],
        [curve.x_data[20], curve.x_data[250]],
        line_styles="--",
        colors="gray",
    )

    figure = gl.Figure(x_lim=(0, 5), y_lim=(0, 6))
    figure.add_element(curve, hlines, vlines)
    figure.display()

.. image:: ../images/lines.png

For both the Hlines and Vlines it is possible to specify as many colors and line styles as there are lines instead of applying the same for all lines as is the case in the example above.

.. _text:
The :class:`~graphinglib.graph_elements.Text` Object
----------------------------------------------------

The Text object is used to display text on a figure. It also allows you to point from the text to a specified point using an arrow. Here is how to declare a text object and attach an arrow to it: ::

    text = gl.Text(4, 1, "There is nothing here!")
    text.attach_arrow((0.5, 1))

.. image:: ../images/text.png

There are many more parameters to be customized for the Text object and its arrow, but those are all included in the figure style files and can therefore be left out most of the time. For the details on the other parameters, visit the :py:class:`Reference section on Text objects <graphinglib.graph_elements.Text>`.