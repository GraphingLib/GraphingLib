========================================================
Creating a :class:`~graphinglib.multifigure.MultiFigure`
========================================================

To create a :class:`~graphinglib.multifigure.MultiFigure`, you first have to decide what size you want the canvas' grid to be. You control this by setting the values of ``num_rows`` and ``num_cols`` as shown in the figure below. This grid is then used to place each set of axes in the MultiFigure.

.. image:: images/Canvas.png
   :scale: 30%

To create a MultiFigure, simply use the following line of code: ::

    multifigure = gl.MultiFigure(2, 2)

Then, to add a set of axes to the MultiFigure, we create a new :class:`~graphinglib.multifigure.SubFigure` with the following line of code::

    # (row start, column start, rows spanned, columns spanned)
    # This will create a 1x1 SubFigure in the top left corner of the grid
    subfigure1 = multifigure.add_subfigure(0, 0, 1, 1)

Once a SubFigure has been created, elements can be added to it by using the :py:meth:`~graphinglib.multifigure.SubFigure.add_element` method just like with a normal :class:`~graphinglib.figure.Figure`. It is important to note that **a single set of axes is not confined to a single square on the grid; it can span multiple squares.** This means it is possible to align the individual sets of axes however you want. For example, here is how you could insert 3 SubFigures in 2 rows with the one on the second row being centered: ::

    import numpy as np
    import graphinglib as gl

    sine = gl.Curve.from_function(lambda x: np.sin(x), 0, 2 * np.pi, label="sine")
    cosine = gl.Curve.from_function(lambda x: np.cos(x), 0, 2 * np.pi, label="cosine")
    tangent = gl.Curve.from_function(
        lambda x: np.tan(x), 0, 2 * np.pi, label="tangent", number_of_points=1000
    )

    multifigure = gl.MultiFigure(2, 4, size=(11, 8))

    # This SubFigure will span the two left columns of the first row
    subfigure1 = multifigure.add_SubFigure(0, 0, 1, 2)
    subfigure1.add_element(sine)
    # This SubFigure will span the two right columns of the first row
    subfigure2 = multifigure.add_SubFigure(0, 2, 1, 2)
    subfigure2.add_element(cosine)
    # This SubFigure will span the two middle columns of the second row
    subfigure3 = multifigure.add_SubFigure(1, 1, 1, 2)
    subfigure3.add_element(tangent)

    multifigure.display()

.. image:: images/multifigure.png

As you can see in the above figure, there are labels (a), b), c)) next to each SubFigure. These are the reference labels used to refer to each SubFigure in the caption of the figure when inserting it in a document. The boolean parameter ``reference_labels`` (in the :class:`~graphinglib.multifigure.MultiFigure` constructor) can turn these on or off.

Legends in MultiFigures
-----------------------

The legends in a MultiFigure can be added separately for every SubFigure or as a single legend combining every plot. This option is controlled by the ``general_legend`` parameter in the :py:meth:`~graphinglib.multifigure.MultiFigure.display` and :py:meth:`~graphinglib.multifigure.MultiFigure.save_figure` methods. By default, it is set to ``False`` so that each SubFigure controls its own legend. The two images below illustrate the different legend options.

.. image:: images/individuallegend.png
.. image:: images/generallegend.png

