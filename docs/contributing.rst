===========================
Contributing to GraphingLib
===========================

GraphingLib welcomes the contribution of the coding community to add new features, correct bugs and generally improve the library. Wether you are a skilled coder or just beginning, you can surely bring something to this project. Here are the broad categories of contributions you can make to GraphingLib listed alphabetically :

* Adding examples to our documentation site
* Adding unit tests for existing features
* Developping new features
* Maintaining the code
* Writing documentation (see :ref:`the section on the documentation <doccontrib>`)

This page will provide you with the information necessary to contribute to this project. If you have additionnal questions, feel free to connect with us by way of GitHub issues. Simply visit our `GitHub page <https://github.com/GraphingLib/GraphingLib/>`_ and create a new issue in the "Issues" tab at the top of the page or comment on an existing and related issue.

Developpement process
---------------------

Here are the steps to make your **first contributions** :

1. Go to `GraphingLib's GitHub <https://github.com/GraphingLib/GraphingLib/>`_ and create your own copy of the project by clicking the "Fork" button.

2. Clone the project's copy on your computer. ::

    git clone https://github.com/your-username/GraphingLib.git

3. Move into the newly created folder. ::

    cd GraphingLib

4. Set the remote for your fork. ::

    git remote add updtream https://github.com/GraphingLib/GraphingLib.git

   This step will allow you to pull the latest updates made on the main repository into your fork.

   .. note::

        For setting up your developpement environment, you can refer to our :ref:`recommended developpement environment <devenv>`.

5. Create a new branch to develop your feature and then switch to it.

   * You can do this directly on GitHub by going in the branches list and clicking "New Branch". You then enter the following lines in your terminal ::

        git fetch origin
        git checkout your-branch-name
    
   * You can also create a branch directly from the terminal with this command ::

        git checkout -b your-branch-name

   .. attention::

        The branch name will appear in the pull request message so please choose a **sensible** name.

6. Make your changes. See our :ref:`coding guidelines <codeguidelines>` to learn how to write code for GraphingLib.

7. Commit the changes while you make them. Try to use meanignful but short commit messages. ::

    git add .
    git commit -m "a short commit message"

8. Push your changes to your forked repository. ::

    git push origin your-branch-name

9. Go to the github repository where your fork of GraphingLib resides. You should see a banner telling you that your branch is behind the ``GraphingLib/main`` branch with a button to create a pull request. Create a pull request following the :ref:`guidelines for PR submission <prguidelines>`.

If you are a **returning contributor** update your GraphingLib fork from the main repository :

1. Get the latest changes from GraphingLib's repository ::

    git fetch upstream

2. Switch to the ``main`` branch in your repository ::

    git checkout main

3. Merge the changes from the ``upstream`` remote into your ``main`` branch ::

    git merge upstream/main

4. Push the changes to your fork's remote ::

    git push origin main

You can find a more detailed introduction to Git and GitHub on `GitHub's documentation <https://docs.github.com/en/get-started>`_.

.. _devenv:

Recommended developpement environment
-------------------------------------

The minimal Python version necessary for GraphingLib is version 3.10 and we recommand you setup a dedicated virtual environment to develop your contributions. GraphingLib is developped using Poetry as dependency and virtual environment manager. For information on how to install and get started with Poetry, visit `their website <https://python-poetry.org/docs/#installing-with-the-official-installer>`_. We recommand setting Poetry so that it creates the virtual environment inside the project directory. To do so, run this command ::

    poetry config virtualenvs.in-project true

One you have Poetry installed and the repository cloned on your computer, run this command to create a virtual environment and install GraphingLib and its dependencies ::

    poetry install

GraphingLib's philosophy
------------------------



GraphingLib's repository structure
----------------------------------



Code structure
--------------



.. _codeguidelines:

Coding guidelines
-----------------



Code documentation
^^^^^^^^^^^^^^^^^^



.. _prguidelines:

Guideline for submitting a pull request
---------------------------------------

Before submitting your pull request, here are a few things you should do :

1. If you modified any part of the code (not applicable to the documentation), run the unit tests to make sure that everything is in order.

2. If your changes bring modifications to the API or if you've added or modified a function, please create a short release note in the ``docs/release_notes/upcoming_changes`` directory. Your release note should be a reST file named as ``<PR-NUMBER>.<TAG>.rst``, where ``<PR-NUMBER>`` is the number of your pull request and ``<TAG>`` is one of the following :

   * ``new_feature`` : For new features added to GraphingLib

   * ``improvement`` : For changes improving the efficiency of the code

   * ``compatibility`` : For changes affecting backwards compatibility (not for removal of deprecated features)

   * ``deprecation`` : For setting a feature as deprecated (not yet removed but emitting a ``DeprecationWarning``)

   * ``expired`` : For removed deprecated features

   * ``change`` : For other changes

The file should have the following format : ::

    Title for your changes
    ----------------------
    A short description of how the changes will affect users.

.. _doccontrib:

Contributing to the documentation
---------------------------------

Making changes to this documentation website is encouraged when new features are added to GraphingLib. The addition of new examples in the :ref:`Gallery section <example_gallery>` is also welcomed. The documentation pages are written in reStructuredText format for which you can find a syntax guide `here <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_. The documentation is then built as HTML files via Sphinx. 

Building the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

To build the website locally, use these commands in the terminal ::

    cd docs
    make html

.. note:: 
    
    For Windows, it is possible that the command ``make`` won't be recongnized. In such case use those commands instead ::

        cd docs
        ./make html

Documentation structure
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

* The ``_static`` directory is copied as is into the build directory. It is used to save icons, style customization files (CSS files) and, in our case, the version switcher configuration file.

* The ``switcher.json`` file is used to populate the version switcher dropdown menu at the top right of the website. In it, documentation versions are linked to their URLs.

* The ``_templates`` folder contains RST templates used for automatic generation of some pages like the API pages.

* The ``example_thumbs`` folder must be present when buidling the docs which is why it is kept in the repository even though it is empty.

* The ``gallery_generator.py`` Sphinx extension is the script used to generate the examples gallery and each example page.

* The ``api.rst`` is simply the homepage of the section as the individual pages are generated by the ``sphinx.ext.autosummary`` extention.

* The ``conf.py`` file specifies the configuration used to build the documentation with Sphinx.

* The ``requirements.txt`` file contains the list of dependencies used when building the documentation website on Read The Docs.

Examples gallery
^^^^^^^^^^^^^^^^

The Gallery page is generated automatically from the examples located in the ``examples`` folder in the root directory of the project. The examples themselves are Python (.py) files with a specific header to specify the title ::

    """
    Example's title
    ===============

    _thumb: .4, .4
    """

The code generating the example must run as a standalone file for the example page to be generated. This means that you should be able to run the code on your computer and the plot should be displayed.
