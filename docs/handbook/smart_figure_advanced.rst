=========================================================================
Advanced :class:`~graphinglib.SmartFigure` Usage and Complete Reference
=========================================================================

This comprehensive guide details **every feature and method** of the :class:`~graphinglib.SmartFigure` class. For a quick introduction to basic usage, see :doc:`/handbook/smart_figure_simple`.

.. contents:: Table of Contents
   :local:
   :depth: 3

.. note::
    As a reminder, all parameters in the :class:`~graphinglib.SmartFigure` constructor can also be set or modified later using the corresponding properties or methods.

Elements Management
====================

The :class:`~graphinglib.SmartFigure` provides multiple ways to add and manage plot elements. Understanding the distinction between different methods is crucial for effective use.

Adding Elements: All the Ways
------------------------------

There are **four distinct ways** to add elements to a :class:`~graphinglib.SmartFigure`:

1. Through the constructor with the ``elements`` parameter
2. Using the :py:meth:`~graphinglib.SmartFigure.add_elements` method
3. Setting the :py:attr:`~graphinglib.SmartFigure.elements` property
4. Using the :py:meth:`~graphinglib.SmartFigure.__setitem__` method (indexing with ``[]``)

Constructor: ``elements`` parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When creating a :class:`~graphinglib.SmartFigure`, you can pass elements directly:

.. plot::
    :context: close-figs

    # For a 1x1 figure, all elements go to the single plot
    curve1 = gl.Curve.from_function(lambda x: np.sin(x), 0, 2*np.pi, label="sin")
    curve2 = gl.Curve.from_function(lambda x: np.cos(x), 0, 2*np.pi, label="cos")

    fig = gl.SmartFigure(elements=[curve1, curve2])
    fig.show()

.. plot::
    :context: close-figs

    # For multi-subplot figures, each element goes to a different subplot
    fig = gl.SmartFigure(2, 2, elements=[curve1, curve2, curve1, curve2])
    fig.show()

.. plot::
    :context: close-figs

    # You can also provide nested lists to control which elements go where
    fig = gl.SmartFigure(
        2, 2,
        elements=[
            [curve1, curve2],  # Both in first subplot
            curve1,            # Second subplot
            None,              # Third subplot (not drawn - blank space)
            [],                # Fourth subplot (drawn but empty)
        ]
    )
    fig.show()

Method: ``add_elements()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:meth:`~graphinglib.SmartFigure.add_elements` method **adds** elements without replacing existing ones:

.. plot::
    :context: close-figs

    fig = gl.SmartFigure(elements=[curve1])
    fig.add_elements(curve2)  # Adds curve2 to the same subplot
    fig.show()

.. plot::
    :context: close-figs

    # For multi-subplot figures
    fig = gl.SmartFigure(2, 1, elements=[curve1])
    fig.add_elements(None, curve2)  # curve2 goes to the second subplot
    fig.show()

Similarly to the constructor, you can use nested lists to specify multiple elements per subplot.

Property: ``elements``
^^^^^^^^^^^^^^^^^^^^^^

Setting the :py:attr:`~graphinglib.SmartFigure.elements` property **replaces all** existing elements:

.. plot::
    :context: close-figs

    fig = gl.SmartFigure(elements=[curve1])
    # This REPLACES curve1 with curve2
    fig.elements = [curve2]
    fig.show()

.. note::
   The ``elements`` property setter internally calls :py:meth:`~graphinglib.SmartFigure.add_elements`, but only after clearing all existing elements first. This can be used to reset the elements before adding new ones.

Indexing: ``__setitem__`` and ``__getitem__``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most powerful method for element management uses indexing, which allows elements to **span multiple subplots**:

.. plot::
    :context: close-figs

    fig = gl.SmartFigure(2, 3)

    # Add multiple elements to a subplot
    fig[0, 0] = [curve1, curve2]

    # Add element spanning multiple columns
    fig[0, 1:] = curve2  # Spans columns 1 and 2 of row 0

    # Add element spanning entire row
    fig[1, :] = gl.Curve.from_function(lambda x: x**2, -2, 2)

    fig.show()

.. plot::
    :context: close-figs

    # The += operator adds to existing elements
    fig = gl.SmartFigure(2, 2)
    fig[0, 0] = curve1
    fig[0, 0] += [curve2]  # Adds curve2 without removing curve1
    fig.show()

Retrieve elements using indexing:

.. code-block:: python

    # Retrieve elements using indexing
    fig = gl.SmartFigure(elements=[curve1, curve2])
    retrieved = fig[0, 0]  # Returns list: [curve1, curve2]
    print(f"Number of elements: {len(retrieved)}")

Output:

.. code-block:: none

    Number of elements: 2

Understanding ``None`` and ``[]``
-----------------------------------------------

The distinction between these values is crucial for controlling subplot appearance:

.. plot::
    :context: close-figs

    # None: Subplot is NOT drawn - leaves blank space
    # [], [None] or any list with None: Subplot IS drawn but empty

    fig = gl.SmartFigure(2, 3, elements=[
        curve1,   # Normal subplot with element
        None,     # NOT drawn - blank space
        [None],   # Drawn but empty
        [],       # Drawn but empty (same as [None])
        [curve1, None, curve2],  # curve1 and curve2 plotted, None ignored
        curve2    # Normal subplot with element
    ])
    fig.show()

Removing Elements
-----------------

To remove elements, set them to ``None``:

.. plot::
    :context: close-figs

    fig = gl.SmartFigure(2, 2, elements=[curve1, curve2])
    fig[1, :] = curve1  # Add spanning element

    # Remove element from specific subplot
    fig[0, 1] = None

    # Remove element spanning multiple subplots (must use exact slice used to add it)
    fig[1, :] = None    # Remove it (using same slice)

    fig.show()

.. warning::
   To remove a spanning element, you **must** use the exact slice that was used to add it. Using ``fig[0, :] = None`` will not remove single-subplot elements added in the first row.


Layout and Structure Control
=============================

Grid System
-----------

The :class:`~graphinglib.SmartFigure` uses a grid system defined by ``num_rows`` and ``num_cols``:

.. plot::
    :context: close-figs

    # 2x3 grid creates 6 possible subplot positions
    fig = gl.SmartFigure(num_rows=2, num_cols=3)

    # Can also use the shape property
    fig.shape = (3, 2)  # Now 3x2 grid
    fig.elements = [[]]*6
    fig.show()

.. note::
   Changing the ``num_rows`` or ``num_cols`` of a preexisting :class:`~graphinglib.SmartFigure` is only allowed if there are no elements in the rows/columns being removed.

This grid system can be exploited to create complex layouts by adding elements that span multiple rows or columns:

.. plot::
    :context: close-figs

    # 4 column layout that appears like a 2 column layout
    fig = gl.SmartFigure(num_rows=2, num_cols=4)
    fig[0, :2] = curve1
    fig[0, 2:] = curve1
    fig[1, 1:3] = curve2  # places in the middle of the bottom row
    fig.show()

.. plot::
    :context: close-figs

    # 6 column layout that appears like a 3 and 2 column layout
    fig = gl.SmartFigure(num_rows=2, num_cols=6)
    fig[0, :2] = curve1
    fig[0, 2:4] = curve1
    fig[0, 4:] = curve1
    fig[1, :3] = curve2
    fig[1, 3:] = curve2
    fig.show()

Figure Size
-----------

Control the overall figure size with the ``size`` parameter:

.. plot::
    :context: close-figs

    # Size in inches (width, height)
    fig = gl.SmartFigure(size=(10, 6), elements=[curve1])
    fig.show()

.. plot::
    :context: close-figs

    # Use "default" to let the style file determine the size
    fig = gl.SmartFigure(size="default", elements=[curve1])
    fig.show()

.. note::
   When a :class:`~graphinglib.SmartFigure` is nested inside another, its ``size`` parameter is ignored and determined by the parent figure's size and layout.

Padding Between Subplots
-------------------------

Control spacing between subplots with ``width_padding`` and ``height_padding``:

.. plot::
    :context: close-figs

    # More padding
    fig = gl.SmartFigure(
        2, 2,
        width_padding=0.2,
        height_padding=0.1,
        elements=[curve1, curve2, curve1, curve2]
    )
    fig.show()

.. plot::
    :context: close-figs

    # No padding (compact layout)
    fig = gl.SmartFigure(
        2, 2,
        width_padding=0,
        height_padding=0,
        remove_x_ticks=True,
        remove_y_ticks=True,
        reference_labels=False,
        elements=[curve1, curve2, curve1, curve2]
    )
    fig.show()

Subplot Size Ratios
-------------------

Control relative sizes of rows and columns with ``width_ratios`` and ``height_ratios``:

.. plot::
    :context: close-figs

    # Make middle column twice as wide as others
    fig = gl.SmartFigure(
        2, 3,
        width_ratios=[1, 2, 1],
        height_ratios=[1, 1],
        elements=[curve1]*6
    )
    fig.show()

.. plot::
    :context: close-figs

    # Make first row taller
    fig = gl.SmartFigure(
        2, 2,
        height_ratios=[2, 1],
        elements=[curve1]*4
    )
    fig.show()


Axes Configuration
==================

Labels and Titles
-----------------

Main Figure Labels
^^^^^^^^^^^^^^^^^^

Set labels for the entire figure:

.. plot::
    :context: close-figs

    fig = gl.SmartFigure(
        x_label="Time (s)",
        y_label="Voltage (V)",
        title="Signal Analysis",
        elements=[curve1]
    )
    fig.show()

.. plot::
    :context: close-figs

    # For multi-subplot figures, labels apply to all subplots
    fig = gl.SmartFigure(
        2, 2,
        x_label="Time (s)",
        y_label="Voltage (V)",
        title="Multi-Channel Analysis",
        elements=[curve1]*4
    )
    fig.show()

Individual Subplot Labels
^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``sub_x_labels``, ``sub_y_labels``, and ``subtitles`` for per-subplot customization:

.. plot::
    :context: close-figs

    fig = gl.SmartFigure(
        2, 2,
        sub_x_labels=["Time (ms)", "Time (μs)", "Frequency (Hz)", "Frequency (kHz)"],
        sub_y_labels=["Amplitude", "Power", "Phase", "Energy"],
        subtitles=["Channel 1", "Channel 2", "Channel 3", "Channel 4"],
        elements=[curve1]*4
    )
    fig.show()

.. note::
   These parameters require exactly as many labels as there are subplots (this feature is subject to change).

Axes Limits
-----------

Set axis limits for all subplots:

.. plot::
    :context: close-figs

    fig = gl.SmartFigure(
        x_lim=(0, 5),
        y_lim=(-1.5, 1.5),
        elements=[curve1]
    )
    fig.show()

Logarithmic Scales
------------------

Enable logarithmic scales:

.. plot::
    :context: close-figs

    exp_curve = gl.Curve.from_function(lambda x: np.exp(x), 0, 5)

    fig = gl.SmartFigure(
        log_scale_x=False,
        log_scale_y=True,
        elements=[exp_curve]
    )
    fig.show()

Aspect Ratios
-------------

Control aspect ratio of the data and the plot box:

.. plot::
    :context: close-figs

    # aspect_ratio controls data aspect ratio
    circle = gl.Circle(0, 0, 1)

    fig = gl.SmartFigure(
        aspect_ratio="equal",  # Makes circle truly circular
        elements=[circle]
    )
    fig.show()

.. plot::
    :context: close-figs

    # box_aspect_ratio controls the plot box shape
    fig = gl.SmartFigure(
        box_aspect_ratio=2.0,  # Box is twice as tall as wide
        elements=[curve1]
    )
    fig.show()

.. warning::
   Do not confuse ``aspect_ratio`` (data aspect) with ``box_aspect_ratio`` (box shape). ``aspect_ratio`` affects how data is displayed within the axes (e.g. the aspect ratio of the pixels if plotting a :class:`~graphinglib.Heatmap`), while ``box_aspect_ratio`` changes the physical size of the axes box.

Inverting Axes
--------------

Invert axes direction:

.. plot::
    :context: close-figs

    fig = gl.SmartFigure(
        invert_x_axis=True,
        invert_y_axis=False,
        elements=[curve1]
    )
    fig.show()

.. note::
    This can also be done by inverting directly the axes' limits with ``x_lim`` and ``y_lim`` parameters (e.g. ``x_lim=(max, min)``).

Removing Axes and Ticks
-----------------------

.. plot::
    :context: close-figs

    # Remove entire axes
    fig = gl.SmartFigure(
        remove_axes=True,
        elements=[curve1]
    )
    fig.show()

.. plot::
    :context: close-figs

    # Remove only ticks
    fig = gl.SmartFigure(
        remove_x_ticks=True,
        remove_y_ticks=False,
        elements=[curve1]
    )
    fig.show()

Sharing Axes
------------

Share axes between subplots for synchronized zooming and consistent limits:

.. plot::
    :context: close-figs

    fig = gl.SmartFigure(
        2, 2,
        share_x=True,
        share_y=True,
        elements=[curve1, curve2, curve1, curve2]
    )
    fig.show()

.. note::
   Axis sharing only affects direct subplots of the :class:`~graphinglib.SmartFigure`. Nested :class:`~graphinglib.SmartFigure` objects control their own axis sharing independently.

.. warning::
    The :class:`~graphinglib.SmartFigure` is currently testing an experimental method for perfectly aligning horizontally shared subplots when ``share_x=True``. This may not work in all cases and could be subject to change in future versions. If you encounter issues, please report them on the `GraphingLib GitHub issue tracker <https://github.com/GraphingLib/GraphingLib/issues>`_.


Advanced Customization Methods
==============================

Custom Ticks
------------

The :py:meth:`~graphinglib.SmartFigure.set_ticks` method provides fine-grained control over tick positions and labels:

.. plot::
    :context: close-figs

    fig = gl.SmartFigure(elements=[curve1])

    # Set custom tick positions
    fig.set_ticks(
        x_ticks=[0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
        x_tick_labels=["0", "π/2", "π", "3π/2", "2π"]
    )
    fig.show()

.. plot::
    :context: close-figs

    # Use automatic spacing
    fig = gl.SmartFigure(elements=[curve1])
    fig.set_ticks(
        x_tick_spacing=np.pi/4,
        y_tick_spacing=0.5
    )
    fig.show()

.. plot::
    :context: close-figs

    # Add minor ticks
    fig = gl.SmartFigure(elements=[curve1])
    fig.set_ticks(
        x_tick_spacing=np.pi/2,
        minor_x_tick_spacing=np.pi/8,
        minor_y_tick_spacing=0.05
    )
    fig.show()

.. plot::
    :context: close-figs

    # Reset to defaults
    fig = gl.SmartFigure(elements=[curve1])
    fig.set_ticks(x_tick_spacing=1.0)  # Set custom ticks
    fig.set_ticks(reset=True)  # Reset to default behavior
    fig.show()

.. note::
    Since tick position and tick spacing are conflicting parameters, trying to set both at the same time for the same axis will raise an error. Similarly, giving tick labels without tick positions will also raise an error. For more details on the possible combinations of parameters, see the :py:meth:`~graphinglib.SmartFigure.set_ticks` method docstring.

Tick Appearance
---------------

The :py:meth:`~graphinglib.SmartFigure.set_tick_params` method controls tick appearance. You can also give parameters separately for major and minor ticks as well as for each axis:

.. plot::
    :context: close-figs

    fig = gl.SmartFigure(elements=[curve1])

    # Customize major ticks
    fig.set_tick_params(
        axis="both",
        which="major",
        direction="in",
        length=10,
        width=2,
        color="red",
        label_size=12,
        label_color="blue"
    )

    # Customize minor ticks
    fig.set_tick_params(
        axis="both",
        which="minor",
        direction="inout",
        length=5,
        width=0.5
    )
    fig.set_ticks(minor_x_tick_spacing=np.pi/8, minor_y_tick_spacing=0.2)

    fig.show()

You can also control the presence of ticks and labels on specific sides of the plot:

.. plot::
    :context: close-figs

    fig = gl.SmartFigure(elements=[curve1])
    fig.set_tick_params(
        axis="x",
        draw_top_ticks=True,
        draw_top_labels=True,
        draw_bottom_labels=False
    )
    fig.set_tick_params(
        axis="y",
        draw_right_ticks=True,
        draw_right_labels=True,
        draw_left_ticks=False,
    )
    fig.show()

Grid Customization
------------------

The :py:meth:`~graphinglib.SmartFigure.set_grid` method enables and customizes grid lines:

.. plot::
    :context: close-figs

    fig = gl.SmartFigure(elements=[curve1])

    # Basic grid
    fig.set_grid()
    fig.show()

The same parameters are applied to both axes and major and minor grid lines, but you can control their visibility separately:

.. plot::
    :context: close-figs

    # Customize grid appearance
    fig = gl.SmartFigure(elements=[curve1])
    fig.set_grid(
        visible_x=True,
        visible_y=True,
        which_x="major",  # Only major gridlines on x
        which_y="both",   # Both major and minor on y
        color="lime",
        alpha=0.5,
        line_style=":",
        line_width=2
    )
    # Need minor ticks for minor gridlines
    fig.set_ticks(minor_y_tick_spacing=0.125)
    fig.show()

.. note::
    Once you have enabled the grid with :py:meth:`~graphinglib.SmartFigure.set_grid`, you can toggle its visibility using the :py:attr:`~graphinglib.SmartFigure.show_grid` property.

Text Padding
------------

The :py:meth:`~graphinglib.SmartFigure.set_text_padding_params` method controls spacing around text elements:

.. plot::
    :context: close-figs

    fig = gl.SmartFigure(
        2, 2,
        x_label="X Axis",
        y_label="Y Axis",
        title="Title",
        elements=[curve1]*4
    )

    # Increase padding
    fig.set_text_padding_params(
        x_label_pad=20,
        y_label_pad=10,
        title_pad=30
    )
    fig.show()

If you are using `sub_x_labels`, `sub_y_labels`, or `subtitles`, you can also control their padding independently:

.. plot::
    :context: close-figs

    fig = gl.SmartFigure(
        2, 2,
        y_label="Y Axis",
        sub_x_labels=["A", "B", "C", "D"],
        elements=[curve1]*4
    )
    fig.set_text_padding_params(
        sub_x_labels_pad=[5, 10, 15, 20],  # Different padding for each
        y_label_pad=15
    )
    fig.show()

Annotations
-----------

Other than adding :class:`~graphinglib.Text` elements directly to subplots, you can also add annotations to the figure as a whole using the ``annotations`` parameter:

.. plot::
    :context: close-figs

    # Create annotations (coordinates in range [0, 1])
    note1 = gl.Text(0.5, 0.95, "Top Center Note", "red", h_align="center")
    note2 = gl.Text(0.05, 0.05, "Bottom Left Note", "green")

    fig = gl.SmartFigure(
        elements=[curve1],
        annotations=[note1, note2]
    )
    fig.show()

.. note::
   Annotations use figure-relative coordinates, not data coordinates. This makes them useful for notes that should appear in fixed positions regardless of data scales. This also allows them to be placed outside the axes area.


Visual Styling
==============

Figure Styles
-------------

The ``figure_style`` parameter applies predefined visual themes:

.. plot::
    :context: close-figs

    # GraphingLib built-in styles
    fig = gl.SmartFigure(
        figure_style="dim",
        elements=[curve1]
    )
    fig.show()

.. plot::
    :context: close-figs

    # Matplotlib styles
    fig = gl.SmartFigure(
        figure_style="seaborn-v0_8",
        elements=[curve1]
    )
    fig.show()

.. seealso::
   See :doc:`/handbook/figure_style_file` for creating custom styles.

RC Parameters Method
--------------------

The :py:meth:`~graphinglib.SmartFigure.set_rc_params` method provides direct access to `matplotlib's rcParams <https://matplotlib.org/stable/users/explain/customizing.html>`_:

.. plot::
    :context: close-figs

    fig = gl.SmartFigure(elements=[curve1])

    fig.set_rc_params({
        "axes.facecolor": "#f0f0f0",
        "axes.edgecolor": "red",
        "axes.linewidth": 2,
        "xtick.color": "blue",
        "ytick.color": "green",
        "grid.color": "orange",
        "grid.linestyle": "--",
        "grid.linewidth": 1.5
    })
    fig.set_grid()

    fig.show()

.. plot::
    :context: close-figs

    # Reset parameters
    fig.set_rc_params(reset=True)
    fig.show()

Visual Parameters Method
------------------------

The :py:meth:`~graphinglib.SmartFigure.set_visual_params` method provides a convenient interface for common styling:

.. plot::
    :context: close-figs

    fig = gl.SmartFigure(
        x_label="X",
        y_label="Y",
        elements=[curve1]
    )

    fig.set_visual_params(
        figure_face_color="#dd834bff",
        axes_face_color="grey",
        axes_edge_color="navy",
        axes_line_width=2,
        legend_font_size=30,
        use_latex=True
    )

    fig.show()

.. plot::
    :nofigs:
    :include-source: false

    # This code is hidden and is used to properly reset matplotlib's rcparams, which seem to have trouble being properly reset between plots in some doc build environments.
    curve1 = gl.Curve.from_function(lambda x: np.sin(x), 0, 2*np.pi, label="sin")
    curve2 = gl.Curve.from_function(lambda x: np.cos(x), 0, 2*np.pi, label="cos")

.. plot::
    :context: close-figs

    # Color cycle for multiple elements
    fig = gl.SmartFigure(elements=[curve1, curve2])
    fig.set_visual_params(
        color_cycle=["#FF6B6B", "#4ECDC4", "#45B7D1"]
    )
    fig.show()

.. plot::
    :context: close-figs

    # Hide individual spines
    fig = gl.SmartFigure(elements=[curve1])
    fig.set_visual_params(
        hidden_spines=["top", "right"]
    )
    fig.show()

Legend Control
==============

Legend customization has many options:

Basic Legend Control
--------------------

.. plot::
    :context: close-figs

    # Control legend visibility
    fig = gl.SmartFigure(
        show_legend=False,  # Hide legend
        elements=[curve1, curve2]
    )
    fig.show()

.. plot::
    :context: close-figs

    # Legend location
    fig = gl.SmartFigure(
        legend_loc="lower left",
        elements=[curve1, curve2]
    )
    fig.show()

.. plot::
    :context: close-figs

    # Multi-column legend
    fig = gl.SmartFigure(
        legend_cols=2,
        elements=[curve1, curve2]
    )
    fig.show()

General Legend
--------------

For multi-subplot figures, create a single legend for all subplots:

.. plot::
    :context: close-figs

    fig = gl.SmartFigure(
        2, 2,
        general_legend=True,
        legend_loc=(0.4, 0.5),
        elements=[curve1, curve2, curve1, curve2]
    )
    fig.show()

.. code-block:: python

    # Outside legend positions
    fig = gl.SmartFigure(
        2, 2,
        general_legend=True,
        legend_loc="outside lower center",
        legend_cols=4,
        elements=[curve1, curve2, curve1, curve2]
    )
    fig.save("output.png")  # Or use inline Jupyter display

.. figure:: /handbook/images/outside_legend_example.png
    :align: center
    :width: 70%

.. warning::
    General legends with a location beginning with ``"outside"`` may not be displayed correctly since matplotlib does not change the window size to accomodate for the legend position. In such cases, use inline Jupyter display or save the figure to a file to see the legend on the figure.

If you have a nested :class:`~graphinglib.SmartFigure`, you can create a general legend for the entire figure including all sub-figures:

.. plot::
    :context: close-figs

    # Create individual figures
    fig1 = gl.SmartFigure(2, 1, elements=[curve1, curve1])
    fig2 = gl.SmartFigure(
        num_cols=2,
        elements=[curve2, fig1],
        general_legend=True,
        legend_loc="upper center"
    )

You can alternatively choose that each nested figure creates its own general legend, in which case you need to specify ``general_legend=True`` for that nested :class:`~graphinglib.SmartFigure`:

.. plot::
    :context: close-figs

    # Create individual figures
    fig1 = gl.SmartFigure(
        2, 1,
        general_legend=True,
        legend_loc="upper right",
        elements=[curve1, curve1]
    )
    fig2 = gl.SmartFigure(
        num_cols=2,
        legend_loc="center left",
        elements=[curve2, fig1]
    )

    fig2.show()

Custom Legend Elements
----------------------

The :py:meth:`~graphinglib.SmartFigure.set_custom_legend` method adds custom legend entries:

.. plot::
    :context: close-figs

    # Create custom legend elements
    custom_elem1 = gl.LegendPatch("Custom 1", "red", "cyan", hatch="///")
    custom_elem2 = gl.LegendLine("Custom 2", "blue", line_style="dashed")

    fig = gl.SmartFigure(elements=[curve1, curve2])
    fig.set_custom_legend(elements=[custom_elem1, custom_elem2])
    fig.show()

.. plot::
    :context: close-figs

    # Hide default elements, show only custom
    fig.hide_default_legend_elements = True
    fig.show()

.. plot::
    :context: close-figs

    # Hide custom elements, show only default
    fig.hide_default_legend_elements = False
    fig.hide_custom_legend_elements = True
    fig.show()


Reference Labels
================

Reference labels (a), b), c), etc.) help identify subplots in publications:

Basic Reference Labels
----------------------

Reference labels  are enabled by default:

.. plot::
    :context: close-figs

    fig = gl.SmartFigure(2, 2, elements=[curve1]*4)
    fig.show()

.. plot::
    :context: close-figs

    # Disable reference labels
    fig = gl.SmartFigure(
        2, 2,
        reference_labels=False,
        elements=[curve1]*4
    )
    fig.show()

Reference Label Position
------------------------

You can control where reference labels are placed using the ``reference_labels_loc`` parameter:

.. plot::
    :context: close-figs

    # Inside the plot
    fig = gl.SmartFigure(
        2, 2,
        reference_labels_loc="inside",
        elements=[curve1]*4
    )
    fig.show()

.. plot::
    :context: close-figs

    # Custom position (relative to top-left corner)
    fig = gl.SmartFigure(
        2, 2,
        reference_labels_loc=(0.02, -0.01),  # (x, y) offset in inches
        elements=[curve1]*4
    )
    fig.show()

Global Reference Label
----------------------

Label the entire figure instead of individual subplots:

.. plot::
    :context: close-figs

    fig = gl.SmartFigure(
        2, 2,
        global_reference_label=True,
        reference_labels=False,
        elements=[curve1]*4
    )
    fig.show()

Advanced Reference Label Customization
---------------------------------------

The :py:meth:`~graphinglib.SmartFigure.set_reference_labels_params` method provides detailed control over the reference labels' appearance:

.. plot::
    :context: close-figs

    fig = gl.SmartFigure(2, 2, elements=[curve1]*4)

    fig.set_reference_labels_params(
        color="red",
        font_size=16,
        font_weight="bold",
        start_index=2,  # Start from "c)"
        format=lambda letter: f"({letter.upper()})"  # (A), (B), etc.
    )
    fig.show()

Here we used the ``format`` parameter to customize the label format, which can be any function that takes a label letter string as input and returns a formatted string. This function is completely defined by the user, allowing for any desired format.

.. note::
    If you want to have different reference label styles for different subplots, consider using nested :class:`~graphinglib.SmartFigure` objects where each nested figure has its own reference label settings.


Nested SmartFigures
===================

One of the most powerful features is the ability to nest :class:`~graphinglib.SmartFigure` objects within each other.

Basic Nesting
-------------

.. plot::
    :context: close-figs

    # Create individual figures
    fig1 = gl.SmartFigure(elements=[curve1], reference_labels=False)
    fig2 = gl.SmartFigure(elements=[curve2], reference_labels=True)

    # Combine them
    parent = gl.SmartFigure(num_cols=2, elements=[fig1, fig2])
    parent.show()

.. plot::
    :context: close-figs

    # Each nested figure can have its own parameters
    fig1 = gl.SmartFigure(
        x_label="Time (s)",
        y_label="Signal 1",
        elements=[curve1]
    )
    fig2 = gl.SmartFigure(
        x_label="Frequency (Hz)",
        y_label="Signal 2",
        elements=[curve2]
    )

    parent = gl.SmartFigure(
        num_cols=2,
        title="Combined Analysis"
    )
    parent.elements = [fig1, fig2]
    parent.show()

Multi-level Nesting
-------------------

You can nest :class:`~graphinglib.SmartFigure` objects any number of levels deep:

.. plot::
    :context: close-figs

    # Level 1: Basic figures
    fig_a = gl.SmartFigure(elements=[curve1]).set_tick_params(color="red", width=10)
    fig_b = gl.SmartFigure(elements=[curve2]).set_tick_params(color="blue", width=10)

    # Level 2: Combine into two other figures
    row = gl.SmartFigure(num_cols=2, elements=[fig_a, fig_b], global_reference_label=True)
    col = gl.SmartFigure(num_rows=2, elements=[fig_a, fig_b], global_reference_label=True)

    # Level 3: Combine rows
    final = gl.SmartFigure(num_rows=2, elements=[row, col])
    final.show()

Style Inheritance
-----------------

The ``figure_style`` and visual parameters are inherited by nested figures:

.. plot::
    :context: close-figs

    # Parent style applies to all nested figures
    fig1 = gl.SmartFigure(elements=[curve1])
    fig2 = gl.SmartFigure(elements=[curve2])

    parent = gl.SmartFigure(
        num_cols=2,
        figure_style="dark",  # Applies to fig1 and fig2
        elements=[fig1, fig2]
    )
    parent.show()

However, if nested :class:`~graphinglib.SmartFigure` objects specify their visual parameters, those take precedence:

.. plot::
    :context: close-figs

    fig1.set_visual_params(axes_edge_color="red")

    parent = gl.SmartFigure(num_cols=2, elements=[fig1, fig2])
    parent.set_visual_params(axes_edge_color="green")  # fig1 stays red
    parent.show()

Working with Nested Figures
----------------------------

The :py:meth:`~graphinglib.SmartFigure.copy_with` method is especially useful when working with nested figures:

.. plot::
    :context: close-figs

    base_fig = gl.SmartFigure(
        x_label="Time",
        y_label="Amplitude",
        elements=[curve1]
    )

    # Create variations
    fig1 = base_fig.copy_with(title="Sine Wave")
    fig2 = base_fig.copy_with(title="Cosine Wave", elements=[curve2])

    parent = gl.SmartFigure(num_cols=2, elements=[fig1, fig2])
    parent.show()

It can also be used to combine figures into a parent figure by changing seamlessly their parameters:

.. plot::
    :context: close-figs

    parent = gl.SmartFigure(
        num_cols=2,
        x_label="Time",
        y_label="Amplitude",
        elements=[
            fig1.copy_with(x_label=None, y_label=None),
            fig2.copy_with(x_label=None, y_label=None)
        ]
    )
    parent.show()

Twin Axes
=========

The :class:`~graphinglib.SmartTwinAxis` class is very similar to :class:`~graphinglib.SmartFigure` but represents a single secondary axis. This object needs to be associated with a :class:`~graphinglib.SmartFigure` to be displayed and allows you to plot data with different scales on the same subplot. Similar to the :class:`~graphinglib.SmartFigure` you can customize its appearance with :py:meth:`~graphinglib.SmartTwinAxis.set_visual_params`, :py:meth:`~graphinglib.SmartTwinAxis.set_tick_params`, etc. and manage its elements with :py:meth:`~graphinglib.SmartTwinAxis.add_elements` and other methods.

Creating Twin Axes
------------------

.. plot::
    :context: close-figs

    # Create main figure with one curve
    temp_curve = gl.Curve.from_function(lambda x: 20 + 5*np.sin(x),
                                        0, 2*np.pi, label="Temperature (°C)")
    fig = gl.SmartFigure(
        x_label="Time (h)",
        y_label="Temperature (°C)",
        elements=[temp_curve]
    )

    # Add twin y-axis for humidity data
    humidity_curve = gl.Curve.from_function(lambda x: 60 + 20*np.cos(x), 0, 2*np.pi,
                                            label="Humidity (%)", color="blue")

    twin_y = fig.create_twin_axis(
        is_y=True,
        label="Humidity (%)",
        elements=[humidity_curve]
    )

    fig.show()

.. plot::
    :context: close-figs

    # Twin x-axis
    time_curve = gl.Curve.from_function(lambda x: np.sin(x), 0, 2*np.pi, label="Signal")
    fig = gl.SmartFigure(
        x_label="Time (s)",
        y_label="Amplitude",
        elements=[time_curve]
    )

    freq_curve = gl.Curve.from_function(lambda x: np.cos(x), 0, 2*np.pi,
                                        label="Frequency Response", color="red")

    twin_x = fig.create_twin_axis(
        is_y=False,
        label="Frequency (Hz)",
        elements=[freq_curve]
    )

    fig.show()

Direct Twin Axis Assignment
----------------------------

Even if the :class:`~graphinglib.SmartTwinAxis` can not be plotted alone, you can create it directly and assign it afterwards to a :class:`~graphinglib.SmartFigure`:

.. plot::
    :context: close-figs

    # Create twin axis directly
    twin = gl.SmartTwinAxis(
        label="Secondary Axis",
        axis_lim=(0, 100),
        log_scale=False,
        elements=[humidity_curve]
    )

    fig = gl.SmartFigure(
        y_label="Primary Axis",
        twin_y_axis=twin,
        elements=[temp_curve]
    )
    fig.show()

Twin Axis Customization
-----------------------

.. plot::
    :context: close-figs

    fig = gl.SmartFigure(elements=[temp_curve])

    twin_y = fig.create_twin_axis(
        is_y=True,
        label="Humidity (%)",
        axis_lim=(0, 100),
        elements=[humidity_curve]
    )

    # Customize twin axis appearance
    twin_y.set_visual_params(
        edge_color="blue",
        label_color="blue",
        line_width=2
    )

    # Customize twin axis ticks
    twin_y.set_tick_params(
        which="major",
        color="blue",
        label_color="blue"
    )

    fig.show()

.. plot::
    :context: close-figs

    fig = gl.SmartFigure(elements=[temp_curve])

    # Managing twin axis elements
    twin_y = fig.create_twin_axis(is_y=True, elements=[humidity_curve])

    # Add more elements
    extra_curve = gl.Curve.from_function(lambda x: 50 + 10*np.sin(2*x), 0, 2*np.pi,
                                         label="Extra", color="green")
    twin_y.add_elements(extra_curve)

    # Access elements
    first_element = twin_y[0]
    print(f"Twin axis has {len(twin_y)} elements")

Output:

.. code-block:: none

    Twin axis has 2 elements


.. note::
   Twin axes can only be created for single-subplot :class:`~graphinglib.SmartFigure` objects (1x1 grid).


Projections
===========

The :class:`~graphinglib.SmartFigure` supports various coordinate system projections through the ``projection`` parameter. This allows you to display data in non-Cartesian coordinate systems such as polar coordinates or astronomical World Coordinate Systems (WCS).

Supported Projections
---------------------

The ``projection`` parameter accepts:

- **Matplotlib projection strings**: `Any projection name supported by matplotlib <https://matplotlib.org/stable/api/projections_api.html>`_ (e.g., ``"polar"``, ``"aitoff"``, ``"hammer"``, ``"lambert"``, ``"mollweide"``, ``"rectilinear"``). You can get a list of available projections using ``matplotlib.projections.get_projection_names()``.
- **Projection objects**: Objects capable of creating a projection. For `astropy.wcs.WCS <https://docs.astropy.org/en/stable/wcs/index.html>`_ objects used for plotting astronomical data, use the specialized :class:`~graphinglib.SmartFigureWCS` class instead. You can read more about it in the :doc:`/handbook/smart_figure_wcs` documentation.

.. warning::
    3D projections (``"3d"``) are **not supported** at this time. These will be added in a future release, probably as a separate class `SmartFigure3D`.

Basic Projection Usage
----------------------

Polar Projection
^^^^^^^^^^^^^^^^

The most common projection is the polar coordinate system:

.. plot::
    :context: close-figs

    # Create data in polar coordinates
    theta = np.linspace(0, 2*np.pi, 100)
    r = 1 + np.sin(3*theta)

    # Create curve using polar projection
    polar_curve = gl.Curve(theta, r, label="Rose curve")

    fig = gl.SmartFigure(
        projection="polar",
        aspect_ratio="equal",
        elements=[polar_curve]
    )
    fig.show()

.. plot::
    :context: close-figs

    # Multiple elements in polar coordinates
    spiral = gl.Curve(
        theta,
        theta / (2*np.pi),
        label="Spiral",
        color="blue"
    )
    circle = gl.Curve(
        theta,
        np.ones_like(theta),
        label="Circle",
        color="red"
    )

    fig = gl.SmartFigure(
        projection="polar",
        title="Polar Coordinate System",
        elements=[spiral, circle]
    )
    fig.show()

Other Matplotlib Projections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Matplotlib provides several map-like projections useful for displaying data on spherical surfaces:

.. plot::
    :context: close-figs

    # Mollweide projection (useful for world maps)
    lon = np.linspace(-np.pi, np.pi, 100)
    lat = np.sin(3*lon) * np.pi/4

    curve = gl.Curve(lon, lat, label="Data")

    fig = gl.SmartFigure(
        projection="mollweide",
        title="Mollweide Projection",
        elements=[curve]
    )
    fig.show()

.. seealso::
   See the `Matplotlib Projections Documentation <https://matplotlib.org/stable/api/projections_api.html>`_ for more details on available projections and their usage.

Projections with Multiple Subplots
-----------------------------------

When using projections with multi-subplot figures, **all subplots share the same projection**:

.. plot::
    :context: close-figs

    # All subplots will use polar projection
    theta = np.linspace(0, 2*np.pi, 100)
    r1 = 1 + 0.5*np.cos(5*theta)
    r2 = 0.5 + 0.5*np.sin(4*theta)

    curve1 = gl.Curve(theta, r1, label="Curve 1")
    curve2 = gl.Curve(theta, r2, label="Curve 2")

    fig = gl.SmartFigure(
        1, 2,
        aspect_ratio="equal",
        projection="polar",
        elements=[curve1, curve2]
    )
    fig.show()

.. note::
    If you need different projections for different subplots, use nested :class:`~graphinglib.SmartFigure` objects where each can have its own projection.

Projections with Nested Figures
--------------------------------

Nested figures can have different projections:

.. plot::
    :context: close-figs

    # Create a Cartesian curve
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x)
    cartesian_curve = gl.Curve(x, y, label="Cartesian")

    # Create a polar figure
    polar_fig = gl.SmartFigure(
        projection="polar",
        aspect_ratio="equal",
        title="Polar",
        elements=[polar_curve]
    )

    # Combine them
    parent = gl.SmartFigure(
        num_cols=2,
        elements=[cartesian_curve, polar_fig]
    )
    parent.show()


Astronomical Data with SmartFigureWCS
-------------------------------------

For plotting astronomical data with World Coordinate System (WCS) projections, GraphingLib provides the specialized :class:`~graphinglib.SmartFigureWCS` class. This class extends :class:`~graphinglib.SmartFigure` with features specifically designed for astronomical coordinate systems.

Key features of :class:`~graphinglib.SmartFigureWCS`:

- Automatic celestial coordinate formatting (RA/Dec, Galactic, etc.)
- Support for FITS WCS metadata
- Curved coordinate grid lines following great circles
- All standard :class:`~graphinglib.SmartFigure` features (nesting, twin axes, styling, etc.)

.. seealso::
   See :doc:`/handbook/smart_figure_wcs` for comprehensive documentation on plotting astronomical data with :class:`~graphinglib.SmartFigureWCS`.


File Operations
===============

Displaying Figures
------------------

The :py:meth:`~graphinglib.SmartFigure.show` method displays a :class:`~graphinglib.SmartFigure`:

.. code-block:: python

    fig = gl.SmartFigure(elements=[curve1])

    # Basic display
    fig.show()

    # Fullscreen display
    fig.show(fullscreen=True)

.. warning::
   For figures with general legends in "outside" positions, use inline Jupyter display or save to a file for correct rendering.

Saving Figures
--------------

The :py:meth:`~graphinglib.SmartFigure.save` method is used to save the figure to various formats:

.. code-block:: python

    fig = gl.SmartFigure(elements=[curve1])

    # Save as PNG
    fig.save("output.png", dpi=300)

    # Save as PDF
    fig.save("output.pdf")

    # Save with transparent background
    fig.save("output.png", transparent=True)

Split PDF Saving
----------------

The ``split_pdf`` parameter allows the :class:`~graphinglib.SmartFigure` to save each of its subplot on a separate PDF page:

.. code-block:: python

    fig = gl.SmartFigure(2, 2, elements=[curve1]*4)

    # Each subplot becomes a separate page
    fig.save("multi_page.pdf", split_pdf=True)

If you want however to save multiple :class:`~graphinglib.SmartFigure` objects into a single multi-page PDF, you can use ``PdfPages`` from `matplotlib.backends.backend_pdf <https://matplotlib.org/stable/api/backend_pdf_api.html>`_:

.. code-block:: python

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages("output.pdf") as pdf:
        fig1.save(pdf)
        fig2.save(pdf)
        fig3.save(pdf)


Utility Methods and Properties
===============================

Copying Figures
---------------

Both :py:meth:`~graphinglib.SmartFigure.copy` and :py:meth:`~graphinglib.SmartFigure.copy_with` methods create copies of a :class:`~graphinglib.SmartFigure`. The former creates a deep copy, while the latter also deep copies but allows you to modify specific parameters at the same time:

.. plot::
    :context: close-figs

    original = gl.SmartFigure(
        x_label="Original",
        elements=[curve1]
    )

    # Deep copy
    copy1 = original.copy()
    copy1.x_label = "Copy 1"

    # Copy with modifications
    copy2 = original.copy_with(
        x_label="Copy 2",
        y_label="Modified",
        elements=[curve2]
    )

    parent = gl.SmartFigure(num_cols=3, size=(10, 5), elements=[original, copy1, copy2])
    parent.show()

Inspecting Figures
------------------

.. code-block:: python

    fig = gl.SmartFigure(2, 3, elements=[curve1]*6)

    # Get number of subplots (elements)
    print(f"Number of elements: {len(fig)}")  # 6

    # Get shape
    print(f"Shape: {fig.shape}")  # (2, 3)

    # Check if the figure has a single subplot (1x1)
    print(f"Is single subplot: {fig.is_single_subplot}")  # False

    single_fig = gl.SmartFigure(elements=[curve1])
    print(f"Is single subplot: {single_fig.is_single_subplot}")  # True

    # Access specific elements
    element_list = fig[0, 0]  # Returns list of Plottable objects
    print(f"Elements in (0,0): {len(element_list)}")


Method Chaining
---------------

Most methods return ``self`` to enable method chaining:

.. plot::
    :context: close-figs

    fig = (gl.SmartFigure(elements=[curve1])
           .set_visual_params(axes_edge_color="red")
           .set_ticks(x_tick_spacing=1.0)
           .set_tick_params(direction="out")
           .set_grid(color="gray", alpha=0.3))

    fig.show()


Complete Configuration Example
==============================

Here's a comprehensive example using many advanced features:

.. plot::
    :context: close-figs

    # Create base data
    x = np.linspace(0, 2*np.pi, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.sin(2*x)
    y4 = np.cos(2*x)

    # Create curves
    curve1 = gl.Curve(x, y1, label="sin(x)", color="blue")
    curve2 = gl.Curve(x, y2, label="cos(x)", color="red")
    curve3 = gl.Curve(x, y3, label="sin(2x)", color="green")
    curve4 = gl.Curve(x, y4, label="cos(2x)", color="orange")

    # Create nested figures
    fig_top_left = gl.SmartFigure(
        title="Fundamental Frequencies",
        elements=[curve1, curve2]
    )

    fig_top_right = gl.SmartFigure(
        title="Harmonics",
        elements=[curve3, curve4]
    )

    # Create bottom spanning figure
    fig_bottom = gl.SmartFigure(
        title="Combined View",
        elements=[curve1, curve2, curve3, curve4]
    ).set_visual_params(
        axes_edge_color="purple",
        hidden_spines=["top", "right"]
    )
    fig_bottom.hide_default_legend_elements = True  # prevent duplicate legend entries

    # Combine into parent figure
    parent = gl.SmartFigure(
        num_rows=2,
        num_cols=2,
        size=(10, 6),
        figure_style="plain",
        title="Comprehensive Signal Analysis",
        general_legend=True,
        legend_loc=(0.3, 0.46),
        legend_cols=4,
        width_padding=0.02,
        height_padding=0.07
    )

    # Add elements with spanning
    parent.elements = [fig_top_left, fig_top_right]
    parent[1, :] = fig_bottom  # Spans both columns

    # Global customization
    parent.set_visual_params(
        figure_face_color="#c7c4c4",
        font_size=11,
        title_font_size=14,
        title_font_weight="bold"
    )

    # Add global annotation
    note = gl.Text(1, 1, "Generated with GraphingLib",
                h_align="right",
                v_align="top",
                font_size=8,
                color="gray")
    parent.annotations = [note]

    parent.show()


Best Practices and Tips
=======================

Element Management
------------------

1. **Use the right method for the job:**

   - Use ``__init__`` or ``elements`` property for initial setup
   - Use ``add_elements()`` for incremental additions
   - Use ``__setitem__`` (indexing) for spanning elements or precise control

2. **Understand None behavior:**

   - ``None`` → subplot not drawn (blank space)
   - ``[None]`` or ``[]`` → subplot drawn but empty
   - Useful for creating asymmetric layouts

3. **Element removal:**

   - Use exact slice for removing spanning elements
   - ``fig[row, col] = None`` only removes elements at that exact position

Layout Design
-------------

1. **Use :py:meth:`~graphinglib.SmartFigure.copy_with` for variations:**

   Create a base figure and make copies with modifications instead of recreating from scratch.

2. **Plan your grid:**

   Sketch your layout before coding. Use ``width_ratios`` and ``height_ratios`` for non-uniform sizes.

3. **Leverage nesting:**

   Break complex figures into smaller :class:`~graphinglib.SmartFigure` objects and combine them.

Styling
-------

1. **Start with a style file:**

   Define your base appearance in a style file, then customize specific figures.

2. **Layer your customizations:**

   - Figure style (broadest)
   - ``set_visual_params()`` or ``set_rc_params()`` (figure-specific)
   - Individual element properties (finest control)

3. **Nested figure styling:**

   Remember that parent ``figure_style`` applies to all children, but child customizations override parent customizations.

Legends
-------

1. **Choose legend type wisely:**

   - Individual legends: Simple figures, different data types per subplot
   - General legend: Related data across subplots, cleaner appearance

2. **Custom legend elements:**

   Use for theoretical lines, annotations, or elements not directly plotted.

3. **Control visibility:**

   Use ``hide_default_legend_elements`` and ``hide_custom_legend_elements`` for precise control.

Troubleshooting
===============

**Figure too crowded:**
   Increase ``width_padding``/``height_padding`` or adjust ``size``.

**Labels overlapping:**
   Use :py:meth:`~graphinglib.SmartFigure.set_text_padding_params` to add spacing. You can even set negative padding values to reduce space.

**Subplots slightly misaligned:**
    If suplots in the same row/column are not perfectly aligned, make sure that no tick label is overlapping into the margins between subplots, as this causes matplotlib to force different subplot sizes to fit the tick label in the available box.

**Can't remove spanning element:**
   Ensure you're using the exact slice used to add it.

**Legend not showing:**
   Verify that the elements have labels, that ``show_legend=True``, and that ``hide_default_legend_elements`` or ``hide_custom_legend_elements`` are ``False``.

**Grid not visible:**
   Call :py:meth:`~graphinglib.SmartFigure.set_grid` first, then verify that ``show_grid`` is ``True``.

**Twin axis error:**
   Verify figure is single-subplot (``is_single_subplot == True``). You may need to create a nested :class:`~graphinglib.SmartFigure` for the subplot with twin axis.

**GraphingLib issues:**
    https://github.com/GraphingLib/GraphingLib/issues

See Also
========

- :doc:`/handbook/smart_figure_simple` - Quick start guide
- :doc:`/handbook/smart_figure_wcs` - Plotting astronomical data with WCS
- :doc:`/handbook/figure_style_file` - Creating custom style files
- :doc:`/generated/graphinglib.SmartFigure` - SmartFigure API reference
- :doc:`/generated/graphinglib.SmartTwinAxis` - Twin axis API reference
- :doc:`/generated/graphinglib.SmartFigureWCS` - WCS figure API reference
- `Matplotlib projections <https://matplotlib.org/stable/api/projections_api.html>`_ - Available projection types
