===========================================
Scatter plots and fitting experimental data
===========================================

The :class:`~graphinglib.data_plotting_1d.Scatter` Object
---------------------------------------------------------

In GraphingLib, there are two ways to create a :class:`~graphinglib.data_plotting_1d.Scatter` object. If you want to plot existing data, you can use the normal constructor by passing in the x and y data as lists or numpy arrays. If you want to plot a function, you can use the :meth:`~graphinglib.data_plotting_1d.Scatter.from_function` method. This method takes in a function and a range of x values to evaluate the function at. In the latter case, you can also specify the number of points to evaluate the function at. Both of these alternatives are shown below.

.. code-block:: python

    # TODO: Add example of plotting existing data

    # TODO: Add example of plotting a function

    # TODO: Add image of the resulting plot

image:: images/scatter_plot.png

You can also add error bars for `x` and/or `y` or  by calling the :meth:`~graphinglib.data_plotting_1d.Scatter.add_errorbars` method like so:

.. code-block:: python

    # TODO: Add example of adding error bars

    # TODO: Add image of the resulting plot

image:: images/scatter_plot_with_errorbars.png

Just like with the :class:`~graphinglib.data_plotting_1d.Curve` object, you can add, subtract, multiply, and divide two :class:`~graphinglib.data_plotting_1d.Scatter` objects. You can also add, subtract, multiply, and divide a :class:`~graphinglib.data_plotting_1d.Scatter` object by a float or int.

.. warning ::
    If you add, subtract, multiply, or divide two :class:`~graphinglib.data_plotting_1d.Scatter` objects, the two objects must have the same x values. If they do not, an exception will be raised.

.. code-block:: python

    # TODO: Add example of adding two Scatter objects

    # TODO: Add image of the resulting plot

image:: images/scatter_plot_addition.png

Interpolation between data points is possible by calling the :meth:`~graphinglib.data_plotting_1d.Scatter.get_point_at_x` method and the :meth:`~graphinglib.data_plotting_1d.Scatter.get_points_at_y` method. The first returns a :class:`~graphinglib.data_plotting_1d.Point` object that represents the point on the curve at the specified x value. The second returns a list of :class:`~graphinglib.data_plotting_1d.Point` objects that represent the points on the curve at the specified y value.

.. code-block:: python

    # TODO: Add example of interpolation

    # TODO: Add image of the resulting plot

image:: images/scatter_plot_interpolation.png


Curve fitting
-------------

There are a number of curve fit objects that can be used to fit data. The most versatile is the :class:`~graphinglib.fits.FitFromFunction` object. This object takes in a function and a :class:`~graphinglib.data_plotting_1d.Scatter` object and fits the data to the function. However, the most common functions have their own dedicated fit objects to accelerate the fitting process. The most powerful of these is the :class:`~graphinglib.fits.FitFromPolynomial` object. All you need to do is pass in a :class:`~graphinglib.data_plotting_1d.Scatter` object and the degree of the polynomial you want to fit to the data:

.. code-block:: python

    # TODO: Add example of fitting a polynomial

    # TODO: Add image of the resulting plot

image:: images/scatter_plot_polynomial_fit.png

Currently, the following fit objects are available:
- :class:`~graphinglib.fits.FitFromPolynomial`
- :class:`~graphinglib.fits.FitFromExponential`
- :class:`~graphinglib.fits.FitFromLog`
- :class:`~graphinglib.fits.FitFromSquareRoot`
- :class:`~graphinglib.fits.FitFromSine`
- :class:`~graphinglib.fits.FitFromGaussian`

The details of how to use each of these fit objects, as well as the specific variables that are fitted (and how to access them), are described in the documentation for each object. For some of these, it can be useful to specify initial guesses for the fitted variables with the `guesses` argument.

Here is an example of fitting a sine function to some data:

.. code-block:: python

    # TODO: Add example of fitting a sine function

    # TODO: Add image of the resulting plot

image:: images/scatter_plot_sine_fit.png

And here is an example of fitting a specific, user-defined function to some data:

.. code-block:: python

    # TODO: Add example of fitting a user-defined function

    # TODO: Add image of the resulting plot

image:: images/scatter_plot_user_defined_fit.png




