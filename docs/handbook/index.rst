:html_theme.sidebar_secondary.remove:

.. _handbook:
========
Handbook
========

Welcome to the Handbook section of the documentation! In this section, you will find examples and tutorials for the various objects and methods of the GraphingLib library. 

.. note:: 

    In every code snippets you will encounter, the following imports precede the snippet's code: ::

        import graphinglib as gl
        import numpy as np


Sections in this handbook
-------------------------

.. grid:: 1 1 3 3

    .. grid-item-card::
        :img-top: ../_static/icons/Figure.svg

        Creating a simple figure
        ^^^^^^^^^^^^^^^^^^^^^^^^

        Everything about simple Figures.
        ++++

        .. button-ref:: ./figure
            :expand:
            :color: primary
            :click-parent:

            Visit this section

    .. grid-item-card::
        :img-top: ../_static/icons/Curve.svg

        The Curve and its operations
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        The Curve, Curve arithmetics and Curve calculus.
        ++++

        .. button-ref:: ./curve
            :expand:
            :color: primary
            :click-parent:

            Visit this section
    
    .. grid-item-card::
        :img-top: ../_static/icons/Scatter.svg

        Scatter plots and fitting experimental data
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        The Scatter plots and data fitting methods.
        ++++

        .. button-ref:: ./scatter_fitting
            :expand:
            :color: primary
            :click-parent:

            Visit this section
    
    .. grid-item-card::
        :img-top: ../_static/icons/Histogram.svg

        Using the histogram
        ^^^^^^^^^^^^^^^^^^^

        The Histogram and plotting fit residuals.
        ++++

        .. button-ref:: ./histogram
            :expand:
            :color: primary
            :click-parent:

            Visit this section

    .. grid-item-card::
        :img-top: ../_static/icons/Arrow_point.svg

        Miscellaneous objects
        ^^^^^^^^^^^^^^^^^^^^^

        Point, Hlines/Vlines, Text and Table.
        ++++

        .. button-ref:: ./graph_elements
            :expand:
            :color: primary
            :click-parent:

            Visit this section

    .. grid-item-card::
        :img-top: ../_static/icons/Shapes.svg

        Creating shapes
        ^^^^^^^^^^^^^^^

        Rectangle, Circle, Arrow and Line.
        ++++

        .. button-ref:: ./shapes
            :expand:
            :color: primary
            :click-parent:

            Visit this section

    .. grid-item-card::
        :img-top: ../_static/icons/Multifigure.svg

        Creating a MultiFigure
        ^^^^^^^^^^^^^^^^^^^^^^

        Everything about MultiFigures.
        ++++

        .. button-ref:: ./multifigure
            :expand:
            :color: primary
            :click-parent:

            Visit this section

    .. grid-item-card::
        :img-top: ../_static/icons/2D_plotting.svg

        Plotting in 2D
        ^^^^^^^^^^^^^^

        Heatmap, Contour, VectorField and Stream.
        ++++

        .. button-ref:: ./2D_plotting
            :expand:
            :color: primary
            :click-parent:

            Visit this section

    .. grid-item-card::
        :img-top: ../_static/icons/Figure_style.svg

        Writing your own figure style file
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        GraphingLib's Style Editor and GraphingLib's Styles Showcase.
        ++++

        .. button-ref:: ./figure_style_file
            :expand:
            :color: primary
            :click-parent:

            Visit this section

.. toctree::
   :maxdepth: 2
   :hidden:

   figure
   curve
   scatter_fitting
   histogram
   graph_elements
   shapes
   multifigure
   2D_plotting
   figure_style_file
