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

GraphingLib's philosophy
------------------------

GraphingLib's main goal is simple : make the creation of beautiful graphs simple for its users. This implies that contributors should always try to make the API as simple as possible for the user, all while maintaining a great level of customization. We've put a great deal of thought into the API and if you're ever not sure about the way to implement your idea, don't hesitate to reach out by submitting an issue on GitHub or commenting on a relevant issue.

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

The minimal Python version necessary for GraphingLib is version 3.10 and we recommand you setup a dedicated virtual environment to develop your contributions. Though it is not necessary, we recommend the use of Poetry to setup your virtual environment. For information on how to install and get started with Poetry, visit `their website <https://python-poetry.org/docs/#installing-with-the-official-installer>`_. We recommand setting Poetry so that it creates the virtual environment inside the project directory. To do so, run this command ::

    poetry config virtualenvs.in-project true

One you have Poetry installed and the repository cloned on your computer, run this command to create a virtual environment and install GraphingLib and its dependencies ::

    poetry install

You can also use any other virtual environment manager or none at all if you'd prefer. In the case you don't use Poetry, you will need to install GraphingLib from source to build the documentation ::

    pip install git+https://github.com/GraphingLib/GraphingLib.git

.. _codeguidelines:

Coding guidelines
-----------------

Here are the simple coding guidelines we ask you to follow :

* Please follow the `PEP 8 <https://peps.python.org/pep-0008/>`_ Style Guide.
* Use the following import convention ::

    import graphinglib as gl

* Use decriptive variable names even though it makes them longer.
* Use camel case for classes (ex : ``MyClass``) and snake case for functions, methods and variables (ex : ``my_function_or_variable``).
* Please add unit tests for the features you add.

Code documentation
^^^^^^^^^^^^^^^^^^

A torough documentation of the code is one of the way we use to make GraphingLib easy to use. Here are the guidelines you should follow to document your changes :

* We use the Numpy style docstrings to document every class, methods and functions which will be available to the users. Simple docstrings are accepted for hidden methods and functions.
* If you use clear enough variable and function names, you shouldn't need to add that many comments troughout the code. Nevertheless, if a function is very long, it is a good practice to add some comments to help other contributors understand what it is doing.
* To make the code clearer, we also ask that you add type hints for every functions and classes you create. This allows other contributors to better understand the code.

.. _prguidelines:

Guideline for submitting a pull request
---------------------------------------

For submitting your pull request, here are a few things you should do :

1. If you modified any part of the code (not applicable to the documentation), run the unit tests to make sure that everything is in order.

2. If you've modified the documentation pages, try to :ref:`build the documentation <builddoc>` localy to make sure there are no problem.

3. If your changes bring modifications to the API or if you've added or modified a function, please create a short release note in the ``docs/release_notes/upcoming_changes`` directory. This should be added to your branch after the creation of your PR so that you have a PR number created. Your release note should be a reST file named as ``<PR-NUMBER>.<TAG>.rst``, where ``<PR-NUMBER>`` is the number of your pull request and ``<TAG>`` is one of the following :

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

.. note::

    It is possible to create two separate files (with the same ``<PR-NUMBER>`` but different ``<TAG>``) if your changes fall into two categories.

.. _doccontrib:

Contributing to the documentation
---------------------------------

Making changes to this documentation website is encouraged when new features are added to GraphingLib. The addition of new examples in the :ref:`Gallery section <example_gallery>` is also welcomed. The documentation pages are written in reStructuredText format for which you can find a syntax guide `here <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_. The documentation is then built as HTML files via Sphinx. 

.. _builddoc:

Building the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

To build the website locally, use these commands in the terminal and then open the ``docs/_build/html/index.html`` file in your browser ::

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

Repository structure
--------------------

There are three types of branches in GraphingLib's repository :

* The ``main`` branch serves as the primary development branch. Most pull requests are merged to this branch.
* The ``maintenance/A.B.x`` branches are created to prepare the release of the ``A.B.0`` version and serve as maintenance branches for correcting bugs on older versions. 
* The ``doc/A.B.C`` branches are created right after the release of the ``A.B.C`` version and server as a stable branch for the documentation.

``maintenance`` branches
^^^^^^^^^^^^^^^^^^^^^^^^

As said before, those branches are created to prepare an upcoming release. Once most of the changes planned for a major or minor release are done, this branch is created from the ``main`` branch and the final preparations are made onto it. The release tags are created from this branch, not from the ``main`` branch. Once the version is released, the branch is only used for correcting bugs and releasing patches for the related minor version. For example, when the milestone for version ``1.5.0`` is nearly completed, the branch ``maintenance/1.5.x`` is created and the remaining unresolved issues will be merged to this branch instead of the ``main`` branch. The ``v1.5.0`` tag is created upon the release and if a bug is found, it can be fixed in this branch and the ``v1.5.1`` tag can be created and released as a patch.

``doc`` branches
^^^^^^^^^^^^^^^^

The purpose of these branches is to provide a stable documentation between versions releases. This lets us modify the documentation in the maintenance branches as we correct bugs and only update the website when we actually release the patch. We will therefore rarely merge pull requests to the ``doc`` branches, unless it is to correct a typo or a misleading passage. A ``doc`` branch is created right after every release, whether it is a major, minor or patch release. It will be merged into its corresponding ``maintenance`` branch just before the next release and will be replaced by the next ``doc`` branch. Every minor version will always have one associated ``doc`` branch for its latest patch release. For example, just after the release of ``v1.5.0``, a ``doc/1.5.0`` branch will be created from the ``maintenance/1.5.x`` branch. After a bug is detected and fixed in the ``maintenance`` branch, the ``doc/1.5.0`` branch will be merged to ``maintenance/1.5.x`` and then deleted. The ``v1.5.1`` tag can then be created along with a new ``doc/1.5.1`` branch.

Reviewers' workflow
-------------------

.. important::

    The reviewing process is a collaboration between the contributor and the reviewer and in such, it must take place with respect. All contributions are generous and the reviewers should always explain their reasons for asking for changes or when refusing a pull request.

Once a pull request is submitted, reviewers will verify that the proposed changes fit the guidelines we have described here. As a reviewer, you will need to test the code in order to assert that everything is in order and request changes if necessary. To do so, create a local branch from the pull request ::

    git fetch origin pull/<PULL NUMBER>/head:<BRANCH NAME>

where ``<PULL NUMBER>`` is the number of the pull request and ``<BRANCH NAME>`` is the name you would like to give your local branch. Here is a checklist of what you should look for :

* Unit tests have been added and/or modified.
* All unit tests pass succesfully.
* The documentation is complete.
* The documentation builds with no errors or warnings.
* Docstrings are clear and complete.
* The pull request correctly links to the related issue, if applicable.

Once the review is comlete and if it has not already been done, you can ask for the PR issuer to add a short release note to the ``docs/release_notes/upcoming_changes`` directory. If everything is complete, the PR can be merged.
