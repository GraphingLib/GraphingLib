===========================================
Scatter plots and fitting experimental data
===========================================

The :class:`~graphinglib.data_plotting_1d.Scatter` Object
---------------------------------------------------------

In GraphingLib, there are two ways to create a :class:`~graphinglib.data_plotting_1d.Scatter` object. If you want to plot existing data, you can use the normal constructor by passing in the x and y data as lists or numpy arrays. If you want to plot a function, you can use the :meth:`~graphinglib.data_plotting_1d.Scatter.from_function` method. This method takes in a function and a range of x values to evaluate the function at. In the latter case, you can also specify the number of points to evaluate the function at. Both of these alternatives are shown below.

.. code-block:: python

    import graphinglib as gl
    import numpy as np

    # Create data
    x_data = np.linspace(0, 10, 100)
    y_data = 3 * x_data**2 - 2 * x_data + np.random.normal(0, 10, 100)

    # Create scatter plot from data
    scatter_1 = gl.Scatter(x_data, y_data, label="Data")

    # Create scatter plot from function
    scatter_2 = gl.Scatter.from_function(
        lambda x: -3 * x**2 - 2 * x + 350,
        x_min=0,
        x_max=10,
        number_of_points=20,
        label="Function",
    )

    # Create figure and display
    fig = gl.Figure(x_label="x", y_label="y")
    fig.add_element(scatter_1, scatter_2)
    fig.display()


.. image:: images/scatter_plot.png

You can also add error bars for `x` and/or `y` by calling the :meth:`~graphinglib.data_plotting_1d.Scatter.add_errorbars` method like so:

.. code-block:: python

    # Create data
    x_data = np.linspace(0, 10, 10)
    y_data = 3 * x_data**2 - 2 * x_data

    # Add errorbars with float or array/list of floats
    scatter = gl.Scatter(x_data, y_data)
    scatter.add_errorbars(x_error=0.3, y_error=0.1 * y_data)

.. image:: images/scatter_plot_with_errorbars.png

Just like with the :class:`~graphinglib.data_plotting_1d.Curve` object, you can add, subtract, multiply, and divide two :class:`~graphinglib.data_plotting_1d.Scatter` objects. You can also add, subtract, multiply, and divide a :class:`~graphinglib.data_plotting_1d.Scatter` object by a float or int.

.. warning ::
    If you add, subtract, multiply, or divide two :class:`~graphinglib.data_plotting_1d.Scatter` objects, the two objects must have the same x values. If they do not, an exception will be raised.

.. code-block:: python

    scatter_sine = gl.Scatter.from_function(
        lambda x: np.sin(x), x_min=0, x_max=2 * np.pi, label="Sine"
    )

    scatter_line = gl.Scatter.from_function(
        lambda x: x, x_min=0, x_max=2 * np.pi, label="Line"
    )

    scatter_addition = scatter_sine + scatter_line
    scatter_addition.label = "Sine + Line"

    scatter_plus_constant = scatter_sine + 3
    scatter_plus_constant.label = "Sine + 3"

.. image:: images/scatter_plot_addition.png

Interpolation between data points is possible by calling the :meth:`~graphinglib.data_plotting_1d.Scatter.get_point_at_x` method and the :meth:`~graphinglib.data_plotting_1d.Scatter.get_points_at_y` method. The first returns a :class:`~graphinglib.data_plotting_1d.Point` object that represents the point on the curve at the specified x value. The second returns a list of :class:`~graphinglib.data_plotting_1d.Point` objects that represent the points on the curve at the specified y value.

.. code-block:: python

    scatter = gl.Scatter.from_function(
        lambda x: np.sin(3 * x) * np.cos(x) ** 2,
        x_min=0,
        x_max=2 * np.pi,
        number_of_points=70,
        label="$\sin(3x)\cos^2(x)$",
    )

    point_at_4 = scatter.get_point_at_x(4, color="red")
    points_at_y_one_half = scatter.get_points_at_y(0.5, color="orange")

    fig = gl.Figure()
    # Use the * operator to unpack the list of points
    fig.add_element(scatter, point_at_4, *points_at_y_one_half)
    fig.display()

.. image:: images/scatter_plot_interpolation.png


Curve fitting
-------------

There are a number of curve fit objects that can be used to fit data. The most versatile is the :class:`~graphinglib.fits.FitFromFunction` object. This object takes in a function and a :class:`~graphinglib.data_plotting_1d.Scatter` object and fits the data to the function. However, the most common functions have their own dedicated fit objects to accelerate the fitting process. The most powerful of these is the :class:`~graphinglib.fits.FitFromPolynomial` object. All you need to do is pass in a :class:`~graphinglib.data_plotting_1d.Scatter` object and the degree of the polynomial you want to fit to the data:

.. code-block:: python

    # Create noisy data
    x = np.linspace(0, 10, 100)
    y = x**2 - 3 * x + 3 + np.random.normal(0, 7, 100)

    scatter = gl.Scatter(x, y, "Data")
    fit = gl.FitFromPolynomial(scatter, 2, "Fit")

    # Print the coefficients of the fit
    coefficients = fit.coeffs
    for i, c in enumerate(coefficients):
        print(f"coefficient of x^{i}: {c}")

    # Use the fit to predict value of y at x = 5
    print(f"Value of fit at x = 5 is y = {fit.function(5)}")
    predicted_point = fit.get_point_at_x(5, color="red")

    fig = gl.Figure()
    fig.add_element(scatter, fit, predicted_point)
    fig.display()

.. image:: images/scatter_plot_polynomial_fit.png

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




