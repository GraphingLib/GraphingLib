=====================================================
Creating a simple :class:`~graphinglib.figure.Figure`
=====================================================

Creating a basic figure using the :class:`~graphinglib.figure.Figure` object is easy. Here is an example of what plotting a sine function can look like. ::

    import numpy as np
    import graphinglib as gl
    

    figure = gl.Figure()

    sine = gl.Curve.from_function(lambda x: np.sin(x), 0, 2 * np.pi)

    figure.add_element(sine)
    figure.display()

.. image:: images/sine.png

The :py:meth:`~graphinglib.figure.Figure.display` method is used to show the figure on screen. It is also possible to use the :py:meth:`~graphinglib.figure.Figure.save_figure` method to save the figure to a specified path.

.. seealso:: 
    
    For the documentation on the ``from_function`` method, see the :py:meth:`Reference section on the Curve object <graphinglib.data_plotting_1d.Curve.from_function>` or the :doc:`handbook section on curves </handbook/curve>`.

We can specify the axis labels by using the ``x_label`` and ``y_label`` parameters of the figure. ::

    figure = gl.Figure(x_label="Time (s)", y_label="Potential (V)")

For further informations on the available parameters, please refer to the :doc:`Reference section on Figure objects <../reference/figure>`.

Figure style configuration
--------------------------

The ``figure_style`` parameter of the :class:`~graphinglib.figure.Figure` class allows you to specify a predefined style to use to change the appearance of the figure and the elements plotted inside it. You can specify a predefined style as follows: ::

    figure = gl.Figure(x_label="Time (s)", y_label="Potential (V)", figure_style="plain")

It is important to note that the parameters controlled by the specified style can be overridden simply by specifying the desired options. **The explicitly specified options will always be prioritized.**

.. seealso:: For the instructions on how to write a figure style file and what parameters are controlled by the figure style files, see :doc:`/handbook/figure_style_file`.
