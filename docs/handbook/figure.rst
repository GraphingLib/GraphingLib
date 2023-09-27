==============================================
The :class:`~graphinglib.figure.Figure` Object
==============================================

The :class:`~graphinglib.figure.Figure` object controls the base of every data visualization in Graphinglib. First, lets declare a figure object and specify every possible parameters. ::

    import graphinglib as gl

    figure = gl.Figure(
        x_label="x axis",
        y_label="y axis",
        x_lim=None,
        y_lim=None,
        size=(6.4, 4.8),
        log_scale_x=False,
        log_scale_y=False,
        show_grid=False,
        legend_is_boxed=False,
        tick_are_in=True,
        figure_style="plain",
        use_latex=False,
        font_size=12,
    )



