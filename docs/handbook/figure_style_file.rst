==================================
Writing your own figure style file
==================================

In GraphingLib, all objects have default values for most parameters. For example, a curve object has a default line width, and a figure object has a default background color. These defaults are governed by the style you choose to give your figure when creating it:

.. code-block:: python

    fig = Figure(style='plain')
    fig2 = Figure(style='dark')

GraphingLib has a number of built-in styles which are showcased at the bottom of this page. The parameters controlled by styles are called "defaults" for a reason; they can always be overridden by explicitely specifying a parameter when creating an object:

.. code-block:: python

    fig = Figure(figure_style="dark") # uses the "dark" style

    fig2 = Figure(figure_style="dark")
    fig2.set_visual_params(axes_edge_color="red") # "dark" style, but axes color is overridden

If no style is specified, the user's default style is used. This default style can be set by the user using the `set_default_style` function. Once set, the default style will be saved and used for all figures created without a specified style. The default style can also be retrieved using the `get_default_style` function.

.. code-block:: python

    print(gl.get_default_style()) # prints "plain"
    gl.set_default_style("dark")
    print(gl.get_default_style()) # prints "dark"

When you install GraphingLib for the first time, the default style is the "plain" style. You can also create your own styles or modify existing ones. To do this, you can use GraphingLib's style editor through the terminal.

GraphingLib's Style Editor
---------------------------

If you have GraphingLib installed, you can run the following command in the terminal:

.. code-block:: bash

    graphinglib

You will be greeted with a menu that looks like this:

.. code-block:: text

    ======================================
    Welcome to GraphingLib's style editor!
    ======================================

    What would you like to do?
    1. Create a new style
    2. Edit an existing style
    3. Delete an existing style
    4. Exit


    Enter a number:

Let's go through each option.

Create a new style
~~~~~~~~~~~~~~~~~~

Write 1 and press enter. You will be asked to enter a name for your new style. You can then choose an existing style to base your new style on. This accelerates the process of creating a new style, as you can skip the steps of defining parameters that you want to be the same as the style you are basing your new style on. For each GraphingLib object type, you will be asked whether you want to customize the defaults for that object type.

.. code-block:: text

    Do you want to customize Curve settings? (y/n):

Pressing enter will default to "no". If you choose "no", the defaults for that object will be copied from the style you are basing your new style on. If you choose "yes", you will be asked to enter a value for each parameter (again, enter will default to the value of the base style which is displayed in parentheses). Once all objects have been customized or copied, the style will be saved and the style editor will exit. You can now use your new style by using the name you gave it.

.. note::

    Any styles you create will be saved to a platform-specific user configuration directory created by GraphingLib when your first custom style is generated. This means that if you uninstall GraphingLib or update it, your styles will not be deleted. There is also a built-in mechanism which updates your custom styles when new objects or parameters are added to GraphingLib with an update. This means that you can safely update GraphingLib without worrying about your custom styles breaking. Any new objects or parameters will be set to the same value as the "plain" style, but you can always edit your custom styles later.

Edit an existing style
~~~~~~~~~~~~~~~~~~~~~~

Write 2 and press enter. You will be presented with a list of existing styles. Enter the number of the style you want to edit. You will then be asked whether you want to customize the default settings for each object type. Press enter to copy the defaults from the style you are editing, or enter a value to override the default. Once all objects have been customized or copied, the style will be saved and the style editor will exit.

You will notice that you can also edit GraphingLib's built-in styles (GraphingLib will prioritize your edited versions if they exist). Don't worry, this will not break anything. If you want to revert to the original style, you can always delete your custom style and GraphingLib will use the built-in style again. It can be especially useful to edit the "plain" style, as this is the default style that is used when no style is specified.

Delete an existing style
~~~~~~~~~~~~~~~~~~~~~~~~

Write 3 and press enter. You will be presented with a list of existing styles. Enter the number of the style you want to delete. You will be asked to confirm that you want to delete the style. Once you confirm, the style will be deleted and the style editor will exit. You will not be able to delete GraphingLib's built-in styles (but you can delete your edited versions of them).


GraphingLib Styles Showcase
---------------------------
Here are the currently available styles in GraphingLib.

Plain style:

.. image:: images/plain_showcase.png

Dim style:

.. image:: images/dim_showcase.png

Dark style:

.. image:: images/dark_showcase.png

Horrible style:

.. image:: images/horrible_showcase.png