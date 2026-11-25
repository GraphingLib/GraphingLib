Animating Figures
=================

Getting started
---------------

Create a figure normally, then pass a callback to
:meth:`~graphinglib.SmartFigure.show_animation` or
:meth:`~graphinglib.SmartFigure.save_animation`. The callback receives a
zero-based frame number. Set the complete data for that frame because playback
can request the same index again. The callback can return ``None``.

.. code-block:: python

    import graphinglib as gl
    import numpy as np

    x = np.linspace(0, 2 * np.pi, 12)
    trace = gl.Curve(x, np.sin(x))
    trace.add_errorbars(y_error=0.15, cap_width=3)
    figure = gl.SmartFigure(elements=[trace], x_lim=(0, 7), y_lim=(-1.5, 1.5))

    def update(frame):
        phase = frame * 0.12
        trace.set_data(x, np.sin(x + phase), y_error=0.15 + 0.1 * np.sin(x + phase)**2)

    figure.show_animation(update, number_of_frames=30, interval=100)
    figure.save_animation("signal.gif", update, number_of_frames=30, interval=100)

Use :meth:`~graphinglib.Curve.set_data` and
:meth:`~graphinglib.Scatter.set_data` to assign coordinates and uncertainties
together. Errors can be scalars, per-point arrays or two-row arrays of lower
and upper errors. Omitted errors retain their current values and must match
the new data. Individual setters such as ``curve.y_data`` are also available.
Use assignments through the public API so the animation can update each
appearance of the same object, including appearances in nested or spanning
plots. Editing an array in place does not schedule an artist update.

Updating several plot types
---------------------------

The same callback can update geometry, vectors and table cells.

.. code-block:: python

    import graphinglib as gl
    import numpy as np

    point = gl.Point(0, 0.5, label="sample")
    ellipse = gl.Ellipse(1, 0.5, 0.3, 0.2)
    panel = gl.SmartFigure(elements=[point, ellipse], x_lim=(-0.5, 2), y_lim=(0, 1))
    grid = np.linspace(0, 1, 5)
    x_grid, y_grid = np.meshgrid(grid, grid)
    field = gl.VectorField(x_grid, y_grid, np.ones_like(x_grid), np.zeros_like(y_grid))
    table = gl.Table([["0"]], col_labels=["Frame"])
    figure = gl.SmartFigure(1, 3, elements=[panel, field, table])

    def update(frame):
        point.coordinates = (frame / 10, 0.5)
        ellipse.angle = 6 * frame
        field.set_data(np.cos(frame / 10), np.sin(frame / 10))
        table.cell_text = [[str(frame)]]

    figure.save_animation("adapters.gif", update, number_of_frames=16)

The :download:`nested animation example
<../../examples/animations/nested_live.py>` exports a GIF with a Curve,
Scatter, Heatmap, and Text. Run it with ``--play`` to open a playback window.

Live update support
-------------------

.. list-table::
   :header-rows: 1
   :widths: 21 40 39

   * - Element
     - Live assignments
     - Set up before animation
   * - :class:`~graphinglib.Curve`
     - Coordinates, errors, line styles, uncertainty bands, and area fills.
       Call ``add_error_curves(...)`` in a frame to introduce a band. Pass
       ``error_curves_fill_between=True`` to shade it, or assign
       ``show_error_curves=False`` to hide it. Call
       ``get_area_between(..., fill_between=True)`` to introduce a fill.
       Assign ``fill_between_bounds`` to change its interval, or ``None``
       to hide it. Assign ``fill_between_other_curve`` to change the
       boundary, or ``None`` to fill to zero. The fill uses the final
       coordinates of both curves in the frame. When switching to an
       unplotted boundary, assign it before updating its data so the
       session can roll back both changes if the frame fails.
       On Cartesian and polar plots, call ``add_errorbars(...)`` to introduce bars.
       Pass an error of ``None`` to ``set_data(...)`` to remove that direction.
       Assign ``show_errorbars`` to toggle both directions. Omitted live
       styles keep their current values. Set ``cap_width=0`` to hide caps.
     - Set up a nonempty Cartesian plot to animate caps on an existing
       error-bar direction. Both curves must continue to cover the configured
       fill interval.
   * - :class:`~graphinglib.Scatter`
     - Coordinates, configured errors, marker shape, size, edge width,
       opacity, face and edge colors, colormap, and color range. Set per-point
       colors for the new point count in the same callback. Existing colorbars
       can be shown or hidden and continue to track their intensity channels.
       Assign ``color_map_range = None`` to restore their individual initial
       scales. Configured error bars can change color,
       line width, cap width and cap thickness. Cartesian and polar error
       bars use the same live addition, removal and visibility rules as Curve.
     - Legend entry availability and colorbar placement.
       Set up a nonempty Cartesian plot to animate existing caps. Colorbars need an
       intensity array at setup. An intensity channel introduced during
       playback without an explicit range starts with a 0–1 scale.
   * - :class:`~graphinglib.Heatmap`
     - Image values, pixel dimensions, opacity and aspect. Supply both
       coordinate arrays in ``set_data(image, x_mesh=x, y_mesh=y)`` to
       switch to a mesh or move its corners. Clear both ``x_mesh`` and
       ``y_mesh`` in one callback to return to a raster. Rasters can change
       interpolation, origin and extents. Scalar data keeps a shared color
       scale and any existing colorbar across mode switches. Assign
       ``None`` to restore baseline raster extents or scalar color limits.
     - Scalar or RGB(A) pixel kind, normalization type, and colorbar
       placement. Baseline raster extents use the first raster appearance,
       and implicit scalar color limits use the initial frame.
   * - :class:`~graphinglib.Text`
     - Content, position, font size, color, opacity, alignment, rotation,
       and highlight color, opacity and padding, including figure annotations.
       Set ``highlight_color = None`` to hide the box. On a subplot,
       ``add_arrow(...)`` can show or restyle an arrow, and setting
       ``arrow_pointing_to = None`` hides it.
     - Placement on a subplot or the figure. Arrows use subplot coordinates.
   * - :class:`~graphinglib.Hlines`, :class:`~graphinglib.Vlines`
     - Line positions, colors, widths, styles, and opacity. Counts can
       grow, shrink, or reach zero. Existing legend samples follow these
       changes and show up to three horizontal or four vertical lines.
       Supply matching positions, per-line styles, and any per-line bounds
       in the same callback. Set both bounds to draw bounded lines or clear
       both with ``None`` to span the axes. Spanning lines follow the viewport
       even as its limits change.
     - Line orientation.
   * - :class:`~graphinglib.Point`
     - Position, marker shape, size, face and edge colors, edge width and
       opacity. The caption can change its label, coordinate display, font size,
       color and alignment, even if the point starts without a label.
     - The point and its caption use subplot data coordinates.
   * - :class:`~graphinglib.Arrow`, :class:`~graphinglib.Line`
     - Endpoints, width, color and opacity. Arrows can change head style,
       head size, number of heads and shrink. Lines can show or hide caps
       and change their width.
     - Two-sided arrows require a compatible head style.
   * - :class:`~graphinglib.Polygon`, :class:`~graphinglib.Circle`,
       :class:`~graphinglib.Ellipse`, :class:`~graphinglib.Rectangle`
     - Vertices, transformations, position and dimensions. Assign ``fill``
       to show or hide the fill, and ``edge_color = None`` to hide the edge.
       Fill color and opacity, and edge color, width and style, can change.
       Circles and ellipses can change their sampling resolution.
     - Live sampling resolution uses the outline produced by a circle or
       ellipse, including affine transforms of that outline.
   * - :class:`~graphinglib.Table`
     - Body and header text, colors and alignment, row and column count,
       cell borders, column widths, absolute scaling, and placement. To remove
       a header, set both its labels and its colors to ``None``.
     - Supply a nonempty rectangular body. Explicit labels, color arrays and
       widths must match the final dimensions in the same callback.
   * - :class:`~graphinglib.VectorField`
     - Positions and components via
       ``set_data(u, v, x_data=x, y_data=y)``. Frames can change arrow count.
       Arrow color, shaft width, head size, length scale and angle convention
       can also change.
     - The axes retain their initial limits unless changed separately. Automatic
       scaling uses the first frame with a nonzero, unmasked vector. Assign
       ``scale = None`` to request a new automatic scale.
   * - :class:`~graphinglib.Histogram`
     - Sample data, bin count or explicit edges, orientation, normalization,
       face and edge colors, opacity, line width and summary statistics.
       Call ``add_pdf(...)`` to show or restyle the normal density overlay
       and its guides, and ``remove_pdf()`` to hide them. Bar, step and
       filled-step modes can change during playback. A zero standard
       deviation hides the overlay.
     - Integer bins recalculate edges from each frame's data. Explicit edges
       stay fixed until reassigned.
   * - :class:`~graphinglib.Contour`
     - Scalar data and mesh via ``set_data(z, x_mesh=x, y_mesh=y)``, levels,
       filled or line mode, opacity, colormap, color limits, and line widths.
       An existing colorbar follows these updates and can be hidden and shown.
     - Colorbar placement and initial availability. Integer levels use the
       first finite data range and stay fixed on data-only updates. Reassign
       ``levels`` to calculate them again. Filled contours need at least two
       levels. On Matplotlib 3.7, updating the level count reuses separate
       collections for each level.
   * - Fit classes
     - Call ``fit.refit()`` after updating the source, or ``fit.refit(source)``
       to use new data. Call ``show_residual_curves(...)`` to add residual
       curves or change their multiplier and styling, and
       ``hide_residual_curves()`` to hide them. Residuals use the final fit
       and multiplier in each frame. Fitted lines can change style and
       create or retarget area fills using the same methods as Curve.
     - Model type and solver options. The generated equation label refreshes
       in the model, on the plot line, and in an existing legend entry.
   * - :class:`~graphinglib.Stream`
     - Grid and velocity field via ``set_data(u, v, x_data=x, y_data=y)``,
       density, color and line width (uniform or grid-mapped), colormap,
       and arrow size. Streamlines and arrowheads follow the new field,
       including frames with no trajectories.
     - Axis scale and limits follow the same viewport rules as other plots.
       Intensity colors use the initial range, or 0–1 if added during playback.
   * - :class:`~graphinglib.PlottableAxMethod`
     - Call ``set_artist_properties(...)`` with reversible Matplotlib
       properties. For a bar plot, ``set_artist_properties(index=0, height=2)``
       sets the first rectangle's height in each appearance.
     - The initial Axes call establishes the artist count and order.
       Use a matching getter and setter for each property. Changes apply to the
       current artists. Initial method arguments and axis limits are set separately.

Axes and subplot settings
-------------------------

Axes keep their initial viewport until you assign ``x_lim`` or ``y_lim`` on
the SmartFigure that draws the axes. For a layout, a tuple applies to its
direct subplots. A list follows subplot order, including slots occupied by
nested SmartFigures. Leave those slots as ``None`` and change each nested
figure directly. List entries of ``None`` leave individual subplots alone.
Assigning ``None`` to the property restores the initial limits. Within a
tuple, ``None`` keeps that endpoint. Set ``invert_x_axis`` or
``invert_y_axis`` to change direction, using a bool for all direct plots or
a list with ``None`` for untouched slots. Limits and inversion assigned in
the same callback use the final direction. Use ``log_scale_x`` and
``log_scale_y`` to switch between linear and logarithmic scales, using the
same list rules. The final limits must be positive on a logarithmic axis,
and explicitly configured tick positions and labels are preserved. Shared
axes use one set of limits, direction and scale. Twin axes accept
``axis_lim``, ``invert_axis`` and ``log_scale``.

Assign ``aspect_ratio`` or ``box_aspect_ratio`` to change the data scale
ratio or the physical shape of direct subplots. Lists follow subplot order
and use the usual defaults for omitted entries. Assign nested figures
directly. Use ``"auto"`` for the data aspect or ``None`` for the box
aspect to restore automatic sizing. A fixed box combined with a fixed data
aspect can move the limits of a free axis. If both axes are shared, the
requested ratios must agree with the shared limits.

Use ``set_ticks(...)`` in a callback to change tick positions, spacing, and
major tick labels on standard plots and twin axes. Tick updates use the
frame's final scale and keep the view limits fixed. ``reset=True`` restores
the defaults from the active scale before applying new specifications.
On WCS plots, update world-coordinate tick positions or spacing with Astropy
quantities. You can also change the approximate tick count, coordinate
formatter, and minor tick frequency. Reset restores the projection's
initial tick defaults.

Use ``set_tick_params(...)`` in a callback to change tick and label
appearance on direct subplots, twin axes, or WCS coordinates. Select major
and minor ticks independently on standard plots and twins. WCS supports a
separate minor tick length. Tick styling composes with visibility changes
in the same frame. ``reset=True`` restores the figure style defaults for
the selected ticks before applying any new arguments.

Use ``remove_axes``, ``remove_x_ticks`` or ``remove_y_ticks`` to hide and reveal
decorations on direct subplots. These flags follow the same list rules and
apply to each subplot individually, including when the viewport is shared.
Tick flags hide both marks and labels. Switching them off reveals the
current side and label settings. Use ``remove_axes`` and ``remove_ticks`` on
a twin axis to control its decorations separately.

Assign ``show_grid`` to toggle grid lines in the same way. Call
``set_grid(...)`` during a frame to change directions, tick groups, or
appearance. It turns on the new grid for direct subplots. Assign
``show_grid`` afterward in the same callback to choose where to show it.
Use ``reset=True`` to restore the grid style the figure inherited at setup
before applying new settings. Without an explicit grid setup, showing the grid
uses the figure style's grid axes and tick groups. WCS grids use major ticks.

Use ``set_visual_params(...)`` to change figure and axes face colors, axes
edge color and line width, or ``hidden_spines`` for the direct subplots.
Pass ``hidden_spines=[]`` to restore the spine visibility from the figure
style. On a twin axis, use ``edge_color``, ``line_width``, and ``hide_spine``.

Assign ``title``,
``x_label``, or ``y_label`` to change a SmartFigure's text, or assign
``label`` on a twin axis. Use ``None`` to clear the text. Keep the initial
figure's margins wide enough for the longest labels in the animation.

Assign ``subtitles``, ``sub_x_labels``, or ``sub_y_labels`` to update text
in subplot order. A spanning plot occupies one entry. An omitted entry or
``None`` clears the corresponding subplot text. A nested figure's own
title or label takes precedence over the parent entry. Set the nested
figure's property to ``None`` to use the parent's current text.

Call ``set_text_padding_params(...)`` to change title and label padding.
Subplot padding follows subplot order. A nested figure can override the
parent slot padding with its own settings. Use ``reset=True`` to clear
that figure's overrides and fall back to inherited or figure style padding.

Configure subplot membership, axis sharing, and projections before playback.
Supported plottables inside an inset can update in place. Configure inset
bounds, limits, zoom indicators, and membership before playback.
A callback that tries to change one of these settings fails the frame and
restores its pending edits.

Legends and colorbars
---------------------

The initial figure establishes legend candidates and colorbar placement.
For a scalar Heatmap, Contour, or Scatter with an intensity array, call ``set_color_bar_params(reserve=True)`` before playback to reserve
space for a colorbar while starting with ``show_color_bar=False``. Assign
``show_color_bar=True`` in a callback to reveal it. Use
``set_color_bar_params(label=..., ticks=...)`` to change the label or
major tick positions on an existing colorbar. Use an empty string to clear
the label, an empty list to hide the ticks, or ``ticks=None`` to restore the
tick rule established at setup.

Assign ``show_legend`` to show or hide a legend, even if it
starts hidden. Pass a boolean for all direct subplots or use a list with
``None`` for untouched slots. Control nested or shared legends on the
SmartFigure that owns them. A general legend takes one boolean. Outside
legends reserve space before playback so toggling them preserves the layout.

Assign ``legend_loc`` or ``legend_cols`` in a callback to move or repack a
legend. Use a scalar for a general legend. For local legends, lists follow
subplot order. Assign settings on the nested figure that owns the legend.
Moving a legend to or from an outside position, or changing its column
count while it is outside, recalculates the layout margin once for that
frame. Showing or hiding a legend keeps its current margin.

Use ``hide_default_legend_elements`` and ``hide_custom_legend_elements``
to toggle groups of entries during playback. A general legend respects the
filters on its owner and on nested figures. Hidden rows collapse within the
legend. When every entry is hidden, the legend disappears until at least one
entry becomes visible again.

Assign ``label`` to rename existing entries for curves, fits,
scatter plots, histograms, and reference lines. Assigning ``None`` clears the
text when there is no generated label, and keeps the entry's swatch.
Pass ``reserve_legend_entries=True`` to ``show_animation`` or
``save_animation`` to reserve entries for supported plots that start with
``label=None``, including plots in nested figures and on twin axes. A
reserved entry appears when assigned a nonempty label. Assign ``None`` or
an empty string to hide it again. For an outside legend, leave enough
space for the longest label you plan to display.
Existing swatches for plain fitted lines, curves, scatter plots, and
reference lines follow supported visual styles. A Curve swatch also follows
error-bar directions and caps introduced during Cartesian or polar playback.
Custom entries and other swatches retain their initial styling. Use explicit
style values in callbacks.

Playback and export
-------------------

Successful frames leave their assignments in the plottables. If the callback
or application of its artist edits fails, pending changes roll back to the
last successful frame. A failed export preserves any existing destination.
GIF uses Pillow, and MP4 uses FFmpeg when available. By default, export
requests ``1000 / interval`` frames per second. Pass ``fps`` to override it.
GIF delays are quantized to hundredths of a second, and Pillow may combine
visually identical frames.
