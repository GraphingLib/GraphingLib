==========================================================
Using the :class:`~graphinglib.data_plotting_1d.Histogram`
==========================================================

Lets start by creating a simple Histogram of a normal distribution. ::

    import numpy as np
    import graphinglib as gl

    values = np.random.normal(loc=2, scale=5, size=500)

    histogram = gl.Histogram(values, number_of_bins=30, label="Distribution of values")

    figure = gl.Figure(x_label="Values", y_label="Counts")
    figure.add_element(histogram)
    figure.display()

.. image:: images/simplehistogram.png

In this example we can see that the legend includes the values of the distribution's mean and standard deviation. These values are also available through the :class:`~graphinglib.data_plotting_1d.Histogram` object using dot notation::

    mu = histogram.mean
    sigma = histogram.standard_deviation

It is also possible to overlay a normal fit of the distribution simply by setting the ``show_pdf`` parameter to ``"normal"``: ::

    histogram = gl.Histogram(
        values, number_of_bins=30, label="Distribution of values", show_pdf="normal"
    )

.. image:: images/histogrampdf.png

Plotting fit residuals
----------------------

You can create a Histogram from a previously created fit to display the residuals of said fit. Here is an example of how to create such a Histogram using the :py:meth:`~graphinglib.data_plotting_1d.Histogram.plot_residuals_from_fit` method: ::

    import numpy as np
    import graphinglib as gl

    x_data = np.linspace(0, 10, 100)
    y_data = 3 * x_data**2 - 2 * x_data + np.random.normal(0, 10, 100)
    scatter = gl.Scatter(x_data, y_data, label="Data")
    fit = gl.FitFromPolynomial(scatter, degree=2, label="Fit", color="red")

    multifigure = gl.MultiFigure(1, 2, size=(10, 5), reference_labels=False)
    subfigure1 = multifigure.add_SubFigure(0, 0, 1, 1)
    subfigure1.add_element(scatter, fit)

    residuals = gl.Histogram.plot_residuals_from_fit(
        fit, 15, label="Residual", show_pdf="normal"
    )
    subfigure2 = multifigure.add_SubFigure(0, 1, 1, 1, y_lim=(0, 0.06))
    subfigure2.add_element(residuals)

    multifigure.display()

.. image:: images/residuals.png
