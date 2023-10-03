=====================
Miscellaneous Objects
=====================

The :class:`~graphinglib.graph_elements.Point` Object
-----------------------------------------------------

The Point object is usefull for highlighting a specific point in a graph. It allows you to attach a label to it in addition to its coordinates. Here is how to declare a Point object: ::

    point = gl.Point(1, 2, label="Something interesting here!")
    point.add_coordinates()

.. image:: ../images/point.png

There are many more parameters to be customized for the Point object, but those are all included in the figure style files and can therefore be left out most of the time. For the details on the other parameters, visit the :py:class:`Reference section on Point objects <graphinglib.graph_elements.Point>`.

.. seealso::

    The Point object is returned by methods of the :class:`~graphinglib.data_plotting_1d.Curve` objects like :py:meth:`~graphinglib.data_plotting_1d.Curve.get_point_at_x`, :py:meth:`~graphinglib.data_plotting_1d.Curve.get_points_at_y` and :py:meth:`~graphinglib.data_plotting_1d.Curve.intersection`.

The :class:`~graphinglib.graph_elements.Hlines` Object
------------------------------------------------------




The :class:`~graphinglib.graph_elements.Vlines` Object
------------------------------------------------------


The :class:`~graphinglib.graph_elements.Text` Object
----------------------------------------------------