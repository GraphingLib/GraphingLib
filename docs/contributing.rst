===========================
Contributing to GraphingLib
===========================

GraphingLib welcomes the contribution of the coding community to add new features, correct bugs and generally improve the library. Wether you are a skilled coder or just beginning, you can surely bring something to this project. Here are the broad categories of contributions you can make to GraphingLib listed alphabetically :

* Adding examples to our documentation site
* Adding unit tests for existing features
* Developping new features
* Maintaining the code
* Writing documentation

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



.. _prguidelines:
Guideline for submitting a pull request
---------------------------------------



Contributing to the documentation
---------------------------------


