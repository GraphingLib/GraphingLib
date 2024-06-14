==================================
Writing your own figure style file
==================================

In GraphingLib, all objects have default values for most parameters. For example, a curve object has a default line width, and a figure object has a default background color. These defaults are governed by the style you choose to give your figure when creating it:

.. plot::

    fig = gl.Figure(figure_style='plain')
    fig2 = gl.Figure(figure_style='dark')

GraphingLib has a number of built-in styles which are showcased at the bottom of this page. The parameters controlled by styles are called "defaults" for a reason; they can always be overridden by explicitely specifying a parameter when creating an object:

.. plot::

    fig = gl.Figure(figure_style="dark") # uses the "dark" style

    fig2 = gl.Figure(figure_style="dark")
    fig2.set_visual_params(axes_edge_color="red") # "dark" style, but axes color is overridden

If no style is specified, the user's default style is used. This default style can be set by the user using the `set_default_style` function or by using GraphingLib's Style Editor. Once set, the default style will be saved and used for all figures created without a specified style. The default style can also be retrieved using the `get_default_style` function.

.. plot::

    print(gl.get_default_style()) # prints "plain"
    gl.set_default_style("dark")
    print(gl.get_default_style()) # prints "dark"

.. plot::
    :include-source: false
    
    gl.set_default_style("plain")

When you install GraphingLib for the first time, the default style is the "plain" style. You can also create your own styles or modify existing ones. To do this, you can use GraphingLib's Style Editor as described below.

GraphingLib's Style Editor
---------------------------

If you have GraphingLib installed, you can run the following command in the terminal:

.. code-block:: bash

    glse

You will be greeted with a GUI which allows you to create, edit, and delete styles. The Style Editor interface consists of a left panel and a right panel, which let you customize and previewing styles respectively. In the left panel, you can adjust all the style settings for the currently opened style. These changes are immediately reflected in the right panel, which displays a figure with the applied customizations. You can see how the style appears in different contexts by browsing through the example figures provided within the interface. If Auto Switch is checked, the example figure will automatically switch to the most appropriate example depending on what customization tab you are currently on. Additionally, there is an option to upload a custom Python script that creates a GraphingLib figure, allowing users to apply and preview the style on their own figures.

Create a new style
~~~~~~~~~~~~~~~~~~

To create a new style, go to File → New. You will be asked to choose an existing style to use as a starting point.

You can now customize the style by going through the different tabs in the left panel. Remember, you are choosing the default values for each object type: these are the values that will be used if you do not specify a parameter when creating an object. You will always be able to override these defaults by explicitely specifying a parameter.

Once you're happy with your style, you can go to File → Save and you will be prompted to enter a name for your new style. Any styles you create will be saved to a platform-specific user configuration directory created by GraphingLib when your first custom style is generated (you don't have to worry about choosing where to save your styles). This means that if you uninstall GraphingLib or update it, your styles will not be deleted. There is also a built-in mechanism which updates your custom styles automatically when new objects or parameters are added to GraphingLib with an update. You can therefore safely update GraphingLib without worrying about your custom styles breaking. Any new objects or parameters will be set to the same value as the "plain" style, but you can always edit your custom styles later.

Edit an existing style
~~~~~~~~~~~~~~~~~~~~~~

You can open an existing style by going to File → Open. You will be presented with a list of existing styles to choose from. You can then proceed as usual, editing the style to your liking. Once you are done, you can either save your changes to the existing style by going to File → Save, or you can save your changes to a new style by going to File → Save As.

You will notice that you can also edit GraphingLib's built-in styles (GraphingLib will prioritize your edited versions if they exist). Don't worry, this will not break anything. If you want to revert to the original style, you can always delete your custom style and GraphingLib will fall back to the built-in style again. For example, if you want to edit the "dark" style, you can open it, make your changes, and save it. GraphingLib will now use your edited version of the "dark" style instead of the built-in one. If you want to revert to the original "dark" style, you can delete your edited version of it and GraphingLib will use the built-in "dark" style again.

At any time, you can click on "View unsaved changes" to see what changes you have made to the style since you last saved it. This can be useful if you want to make sure you haven't accidentally changed something you didn't mean to.

Manage styles
~~~~~~~~~~~~~

If you want to view and manage all your styles, you can go to File → Manage styles. Here you can see a list of all your custom styles, as well as the built-in styles. There is a color legend to help you distinguish between built-in and custom styles. You can also identify which built-in styles are currently overridden by a custom style of the same name by the small "2" icon next to the style name.

Once you have selected a style in the list, you are presented with four options:

- Rename: Change the name of the style. You can only rename custom styles. Built-in styles cannot be renamed, but you can duplicate them and rename the duplicate.
- Duplicate: Create a copy of the style with a new name.
- Delete: Delete the style. You can only delete custom styles. Built-in styles cannot be deleted. Deleting a custom style which overrides a built-in style will revert GraphingLib to using the built-in style.
- Set as default: Set the style as the default style. This means that any figures created without a specified style will use this style. You can see the current default style in the top left corner of the "Manage styles" window.

GraphingLib Styles Showcase
---------------------------
Here are the currently available built-in styles in GraphingLib:

.. plot::
    :include-source: false

        def create_fig(style):
            colors = gl.get_colors(style)

            # Figure 1
            curves = gl.Hlines(
                [i / 8 + 0.1 for i in range(8)], 0, 1, line_widths=9, colors=colors[0:8]
            )
            fig1 = gl.Figure(y_lim=(-0.1, 1.1))
            fig1.add_elements(curves)
            #######################
            # Figure 2
            data = np.random.normal(0, 1, 1000)
            hist = gl.Histogram(data, 20, normalize=True)
            hist._label = None
            hist.add_pdf()
            fig2 = gl.Figure(y_lim=(0, 0.5))
            fig2.add_elements(hist)
            #######################
            # Figure 3
            curve = gl.Curve.from_function(
                lambda x: (-np.sin(x * 2 * np.pi) + x) * x, 0, 1, "A curve", colors[1]
            )
            curve.get_area_between(0, 1, fill_between=True, fill_color=colors[1])
            fig3 = gl.Figure()
            fig3.add_elements(curve)
            #######################
            # Figure 4: Stream plot
            field = gl.Stream.from_function(
                lambda x, y: (np.sin(x) + np.cos(y), (y + x) ** 2),
                (-2, 2),
                (-2, 2),
                color=colors[2],
            )
            fig4 = gl.Figure()
            fig4.add_elements(field)
            #######################
            # Figure 5: Curve fit
            x_data = np.linspace(0, 2 * np.pi, 100)
            y_data = np.sin(x_data) + np.random.normal(0, 0.1, 100)
            scatter = gl.Scatter(x_data, y_data, face_color=colors[3], edge_color=colors[3])
            fit = gl.FitFromSine(scatter, color=colors[5])
            fit.label = None
            fig5 = gl.Figure()
            fig5.add_elements(scatter, fit)
            #######################
            # Create MultiFigure and display
            canvas = gl.MultiFigure(
                5, 4, reference_labels=False, title=style, size=(8, 8), figure_style=style
            )
            canvas.add_figure(fig1, 0, 0, 1, 4)
            canvas.add_figure(fig2, 1, 0, 2, 2)
            canvas.add_figure(fig3, 1, 2, 2, 2)
            canvas.add_figure(fig4, 3, 0, 2, 2)
            canvas.add_figure(fig5, 3, 2, 2, 2)
            canvas.set_visual_params()
            canvas.show()
            # canvas.save_figure(f"docs/handbook/images/{style}_showcase.png")


        for style in [
            "plain",
            "dark",
            "dim",
            "horrible",
        ]:
            create_fig(style)

Here is the code used to generate the above figures if you want to try it out with your own styles:

.. code-block:: python

    import numpy as np
    import graphing_lib as gl

    style = "plain" # Change this to the style you want to showcase
    colors = gl.get_colors(style)

    ######################
    # Figure 1: Show main colors
    curves = gl.Hlines(
        [i / 8 + 0.1 for i in range(8)], 0, 1, line_widths=9, colors=colors[0:8]
    )
    fig1 = gl.Figure(y_lim=(-0.1, 1.1))
    fig1.add_elements(curves)
    #######################
    # Figure 2: Histogram
    data = np.random.normal(0, 1, 1000)
    hist = gl.Histogram(data, 20, normalize=True)
    hist._label = None
    hist.add_pdf()
    fig2 = gl.Figure(y_lim=(0, 0.5))
    fig2.add_elements(hist)
    #######################
    # Figure 3: Curve and filled area
    curve = gl.Curve.from_function(
        lambda x: (-np.sin(x * 2 * np.pi) + x) * x, 0, 1, "A curve", colors[1]
    )
    curve.get_area_between(0, 1, fill_between=True, fill_color=colors[1])
    fig3 = gl.Figure()
    fig3.add_elements(curve)
    #######################
    # Figure 4: Stream plot
    field = gl.Stream.from_function(
        lambda x, y: (np.sin(x) + np.cos(y), (y + x) ** 2),
        (-2, 2),
        (-2, 2),
        color=colors[2],
    )
    fig4 = gl.Figure()
    fig4.add_elements(field)
    #######################
    # Figure 5: Curve fit
    x_data = np.linspace(0, 2 * np.pi, 100)
    y_data = np.sin(x_data) + np.random.normal(0, 0.1, 100)
    scatter = gl.Scatter(x_data, y_data, face_color=colors[3], edge_color=colors[3])
    fit = gl.FitFromSine(scatter, color=colors[5])
    fit.label = None
    fig5 = gl.Figure()
    fig5.add_elements(scatter, fit)
    #######################
    # Create MultiFigure from all figures
    canvas = gl.MultiFigure(
        5, 4, reference_labels=False, title=style, size=(8, 8), figure_style=style
    )

    canvas.add_figure(fig1, 0, 0, 1, 4)
    canvas.add_figure(fig2, 1, 0, 2, 2)
    canvas.add_figure(fig3, 1, 2, 2, 2)
    canvas.add_figure(fig4, 3, 0, 2, 2)
    canvas.add_figure(fig5, 3, 2, 2, 2)
    canvas.set_visual_params()
    canvas.show()