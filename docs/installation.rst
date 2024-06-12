==========
Quickstart
==========

GraphingLib is based on two libraries that you are surely already familiar with: Matplotlib and SciPy. These two libraries are very powerful and versatile, but this extreme versatility often comes at a cost: readability, conciseness, and user friendliness. GraphingLib aims to simplify the process of creating figures for the overwhelming majority of use cases. To do this, we focused on three main points:

#. Object-oriented code
    Matplotlib’s API is very confusing and complicated for the average user. We have hidden away all of this complexity underneath a much simpler API based on objects. You create a :doc:`Figure <handbook/figure>` object to which you can add any number of other objects: a :doc:`Curve <handbook/curve>`, a :doc:`Fit <handbook/scatter_fitting>`, a :doc:`Histogram <handbook/histogram>`, a :ref:`Point <point>`, :ref:`Text <text>`, etc. Each object has properties which you can set during creation, and you can also access or change these properties at any time. This use of objects maintains independence between figures: you can create multiple figure and save and/or display only those you choose. It also conveniently regroups many different variables, functions, and properties which are often separate in standard code such as a figure and its axes, or a fit function, its fitted parameters and its covariance matrix. Overall, it leads to much cleaner, more readable code.
#. User defined figure styles
    The most tedious part of plotting with Matplotlib is having to redesign your figure from scratch every time. GraphingLib uses figure styles, which are prepackaged instructions for how a figure and its contents should look. You can either use GraphingLib’s styles or define your own using our GUI :doc:`GL Style Editor <handbook/figure_style_file>`. Either way, choosing a figure style is as simple as changing a keyword. The best part is that once you have set a figure style, any parameter you set manually has precedence over the figure style you set; the figure style is only there to control the parameters which you haven’t explicitely specified. You can thereby accelerate the process of creating beautiful plots while still retaining full control over the figure’s appearance.
#. Implementing plot-specific operations
    Some data analysis operations are so inextricably linked to plotting that we felt it could be useful to implement them directly in GraphingLib. We have therefore included :doc:`curve fits <handbook/scatter_fitting>`, arithmetic between two curves or between curves and constants, simple derivation and integration, arc length calculation, mean and standard deviation, and more. Under the hood, these are mostly standard NumPy and SciPy commands integrated into our object-oriented wrapper.


Installation
------------

Install Graphinglib with :command:`pip`: ::
    
    pip install graphinglib

Install from `GitHub source code <https://github.com/GraphingLib/GraphingLib>`_: ::

    pip install git+https://github.com/GraphingLib/GraphingLib.git

Install with **Poetry**: ::

    poetry add graphinglib

Usage
-----

To use Graphinglib in one of your projects, you need to import it with the following command: ::

    import graphinglib as gl

What next?
----------

Once this is done you can visit one of the links below to learn how to create your first Figure and then how to work with Curves, Scatters and Fits.

.. grid:: 1 1 3 3

    .. grid-item-card::
        :img-top: _static/icons/Figure.svg

        Creating a simple figure
        ^^^^^^^^^^^^^^^^^^^^^^^^

        Everything about simple Figures.
        ++++

        .. button-ref:: handbook/figure
            :expand:
            :color: primary
            :click-parent:

            Visit this section
    
    .. grid-item-card::
        :img-top: _static/icons/Curve.svg

        The Curve and its operations
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        The Curve, Curve arithmetics and Curve calculus.
        ++++

        .. button-ref:: handbook/curve
            :expand:
            :color: primary
            :click-parent:

            Visit this section
    
    .. grid-item-card::
        :img-top: _static/icons/Scatter.svg

        Scatter plots and fitting experimental data
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        The Scatter plots and data fitting methods.
        ++++

        .. button-ref:: handbook/scatter_fitting
            :expand:
            :color: primary
            :click-parent:

            Visit this section

