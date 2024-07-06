============================================
New contributors and development environment
============================================

GraphingLib's Philosophy
------------------------

The primary goal of GraphingLib is straightforward: to simplify the creation of beautiful graphs for users. Contributors should strive to keep the API intuitive and user-friendly while maintaining many customization options. We've carefully designed the existing API with these principles in mind. If you're ever uncertain about how to implement your idea, don't hesitate to reach out by submitting an issue on GitHub or commenting on a relevant issue.

Development Process
---------------------

Here are the steps to make your **first contributions** :

1. Go to `GraphingLib's GitHub <https://github.com/GraphingLib/GraphingLib/>`_ and create your own copy of the project by clicking the "Fork" button.

2. Clone the fork on your computer. ::

    git clone https://github.com/your-username/GraphingLib.git

3. Move into the newly created folder. ::

    cd GraphingLib

4. Set the remote for your fork. ::

    git remote add upstream https://github.com/GraphingLib/GraphingLib.git

   This step will allow you to pull the latest updates made on the main repository into your fork.

   .. note::

        For setting up your development environment, you can refer to our :ref:`recommended development environment <devenv>`.

5. Create a new branch to develop your feature and then switch to it.

   * You can create a branch on GitHub by going in the list of branches and clicking "New Branch". You can then enter the following lines in your terminal ::

        git fetch origin
        git checkout your-branch-name
    
   * You can also create a branch directly from the terminal with this command ::

        git checkout -b your-branch-name

   .. attention::

        The branch name will appear in your pull request so please choose a **sensible** name.

6. Make your changes. See our :ref:`coding guidelines <codeguidelines>` to learn how to write code for GraphingLib.

7. Commit the changes regularly as you make them. Try to use meaningful but short commit messages. ::

    git add .
    git commit -m "a short commit message"

8. Push your changes to your forked repository. ::

    git push origin your-branch-name

9. Go to the github repository where your fork of GraphingLib resides. You should see a banner telling you that your branch is behind the ``GraphingLib/main`` branch with a button to create a pull request. Click the button and write your pull request by following the :ref:`guidelines for PR submission <prguidelines>`.

If you are a **returning contributor** update your GraphingLib fork from the main repository :

1. Get the latest changes from GraphingLib's repository ::

    git fetch upstream

2. Make sure you're on the ``main`` branch in your repository ::

    git checkout main

3. Merge the changes from the ``upstream`` remote into your ``main`` branch ::

    git merge upstream/main

4. Push the changes to your fork's remote ::

    git push origin main

You can find a more detailed introduction to Git and GitHub on `GitHub's documentation <https://docs.github.com/en/get-started>`_.

.. _devenv:

Recommended Development Environment
-----------------------------------

The minimal Python version required for GraphingLib is 3.10. We recommend setting up a dedicated virtual environment to develop your contributions. Although it is not mandatory, we suggest using Poetry to manage your virtual environment. For information on how to install and get started with Poetry, visit `their website <https://python-poetry.org/docs/#installing-with-the-official-installer>`_. We also recommend configuring Poetry to create the virtual environment inside the project directory. To do this, run the following command ::

    poetry config virtualenvs.in-project true

Once you have Poetry installed and your fork is cloned on your computer, run this command from within the main folder to create a virtual environment and install GraphingLib's dependencies ::

    poetry install

You can also use any other virtual environment manager or none at all if you prefer. If you don't use Poetry, you will need to install GraphingLib from source to build the documentation ::

    pip install git+https://github.com/GraphingLib/GraphingLib.git

.. _codeguidelines:

Coding Guidelines
-----------------

When contributing to GraphingLib, please follow these simple guidelines to ensure that your code is consistent with the rest of the library:

* Adhere to the `PEP 8 <https://peps.python.org/pep-0008/>`_ Style Guide.
* Use the following import convention ::

    import graphinglib as gl

* Use descriptive variable names, even if they are longer.
* Use CamelCase for classes (ex : ``MyClass``) and snake_case for functions, methods and variables (ex : ``my_function_or_variable``).
* Add unit tests for any features you introduce.

Code Documentation
^^^^^^^^^^^^^^^^^^

Thorough documentation is essential for making GraphingLib easy to use. Please follow these guidelines to document your changes:

* Use Numpy style docstrings to document all classes, methods, and functions that will be available to users. Simple one-line docstrings are acceptable for hidden methods and functions.
* Clear variable and function names should reduce the need for extensive comments throughout the code. However, for long functions, it is good practice to add comments to help other contributors understand the code.
* To improve code clarity, add type hints for all functions and classes you create. This helps other contributors understand the code better.

.. _prguidelines:

Guidelines for Submitting a Pull Request
----------------------------------------

In order to submit your pull request, here are a few things you should do :

1. If you modified any part of the code (excluding documentation), run the unit tests to ensure everything is in order.

2. If you modified the documentation pages, try to :ref:`build the documentation <builddoc>` locally to ensure there are no problems.

3. If your changes modify the API or you have added or modified a function, create a short release note in the ``docs/release_notes/upcoming_changes`` directory. This should be added to your branch after the creation of your PR (you need a PR number to do this step). Your release note should be a reST file named ``<PR-NUMBER>.<TAG>.rst``, where ``<PR-NUMBER>`` is the number of your pull request and ``<TAG>`` is one of the following :

   * ``new_feature`` : For new features added to GraphingLib
   * ``improvement`` : For changes improving the efficiency and/or functioning of a feature
   * ``compatibility`` : For changes affecting backwards compatibility (not for removal of deprecated features)
   * ``deprecation`` : For setting a feature as deprecated (not yet removed but emitting a ``DeprecationWarning``)
   * ``expired`` : For removed deprecated features
   * ``change`` : For any other changes that don't fit in the previous categories

The file should have the following format : ::

    Title for your changes
    ----------------------
    A short description of how the changes will affect users.

.. note::

    It is possible to create two separate files (with the same ``<PR-NUMBER>`` but different ``<TAG>``) if your changes fall into two categories.

.. _doccontrib:

Contributing to the Documentation
---------------------------------

We encourage contributors to update the documentation website (especially the :ref:`Handbook section <handbook>`) whenever new features are added to GraphingLib. Adding new examples to the :ref:`Gallery section <example_gallery>` is also highly appreciated. The documentation pages are written in reStructuredText format for which you can find a syntax guide `here <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_. The documentation is then built as HTML files via Sphinx. 

.. _builddoc:

Building the Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

To build the website locally, use these commands in the terminal and then open the ``docs/_build/html/index.html`` file in your browser ::

    cd docs
    make html

.. note:: 
    
    If you are using Windows, the make command most likely won't be recognized. You can use the following commands instead: ::

        cd docs
        ./make html

Documentation Structure
^^^^^^^^^^^^^^^^^^^^^^^

The documentation is located in the ``docs`` folder at the root directory of the project. Here is a scheme illustrating the structure of this folder : ::

    docs
    ├── _static ................................... (Static directory)
    │   ├── icons ................................. (Icons)
    │   ├── graphinglib.css ....................... (Custom style)
    │   └── switcher.json ......................... (Version switcher configuration)
    ├── _templates ................................ (Page templates)
    │   └── autosummary
    │       └── class.rst ......................... (Class API page template)
    ├── example_thumbs ............................ (Example gallery thumbnails)
    ├── handbook .................................. (Handbook section)
    │   └── images ................................ (Images for the Handbook)
    ├── release_notes ............................. (Release notes section)
    ├── sphinxext ................................. (Sphinx extensions)
    │   └── gallery_generator.py .................. (Sphinx extension for generating the examples gallery)
    ├── api.rst ................................... (API section home page)
    ├── conf.py ................................... (Sphinx configuration file)
    ├── contributing.rst .......................... (Contributing page)
    ├── index.rst ................................. (Home page)
    ├── installation.rst .......................... (Quickstart page)
    ├── make.bat
    ├── Makefile
    └── requirements.txt .......................... (Required extensions and packages for build on RTD)

**Precisions on some files/folders :**

* The ``_static`` directory is copied as is into the build directory. It contains icons, style customization files (CSS), and the version switcher configuration file.
* The ``switcher.json`` file populates the version switcher dropdown menu at the top right of the website, linking documentation versions to their URLs.
* The ``_templates`` folder contains RST templates used for the automatic generation of certain pages, such as the API pages.
* The ``example_thumbs`` folder must be present when building the docs which is why it is kept in the repository even when empty. Its contents are automatically generated when building the documentation.
* The ``gallery_generator.py`` Sphinx extension generates the examples gallery and each example page.
* The ``api.rst`` file is the homepage of the API section. The individual pages are generated by the ``sphinx.ext.autosummary`` extension.
* The ``conf.py`` file specifies the configuration for building the documentation with Sphinx.
* The ``requirements.txt`` file lists the dependencies required for building the documentation website on Read The Docs.
* Once the documentation is built, three additional directories are created in the ``docs`` folder : ``_build``, ``examples`` and ``generated``. These directories are ignored by Git and are not present in the remote repository. You can keep them or delete them as needed since they are automatically generated by Sphinx. If you ever encounter issues with changes not appearing after a build, try deleting these directories and rebuilding the documentation.

Examples Gallery
^^^^^^^^^^^^^^^^

The Gallery page is automatically generated from the examples located in the ``examples`` folder in the root directory of the project. Each example is a Python (.py) file with a header to specify the title ::

    """
    Example's title
    ===============

    _thumb: .4, .4
    """

The code generating the example must be able to run as a standalone file for the example page to be generated. This means that you should be able to run the example file on your computer and the plot should be displayed.

Repository Structure
--------------------

There are three types of branches in GraphingLib's repository :

* The ``main`` branch serves as the primary development branch. Most pull requests are merged to this branch.
* The ``maintenance/A.B.x`` branches are created to prepare for the release of the ``A.B.0`` version and serve as maintenance branches for correcting bugs in older versions. 
* The ``doc/A.B.C`` branches are created immediately after the release of the ``A.B.C`` version and provide a stable branch for the documentation.

``maintenance`` branches
^^^^^^^^^^^^^^^^^^^^^^^^

Maintenance branches are created to prepare for upcoming releases. Once most of the changes planned for a major or minor release are completed, a ``maintenance`` branch is created from the ``main`` branch for final preparations. Release tags are created from this branch, not from the ``main`` branch. After the version is released, the branch is only used for bug fixes and patch releases for the related minor version.

For example, when the milestone for version ``1.5.0`` is nearly completed, the ``maintenance/1.5.x`` branch is created. Any remaining unresolved issues for this milestone will be merged to this branch instead of the ``main`` branch. The ``v1.5.0`` tag is created upon the release. If a bug is found, it can be fixed in this branch and the ``v1.5.1`` tag can be created and released as a patch.

``doc`` branches
^^^^^^^^^^^^^^^^

``doc`` branches provide stable documentation between versions releases. This allows us to modify the documentation in the ``maintenance`` branches as we correct bugs and only update the website when a patch is released. Pull requests to ``doc`` branches are rare and typically only for correcting typos or misleading passages. A ``doc`` branch is created immediately after every release, whether major, minor or patch. It is merged into its corresponding ``maintenance`` branch just before the next release and is then replaced by a new ``doc`` branch. Every minor version always has one associated ``doc`` branch for its latest patch release.

For example, after the release of ``v1.5.0``, a ``doc/1.5.0`` branch will be created from the ``maintenance/1.5.x`` branch. If a bug is detected and fixed in the ``maintenance`` branch, the ``doc/1.5.0`` branch is merged back into ``maintenance/1.5.x`` and then deleted. The ``v1.5.1`` tag is created, along with a new ``doc/1.5.1`` branch.
