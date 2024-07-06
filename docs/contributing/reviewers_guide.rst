================
Reviewers' guide
================

Review workflow
---------------

.. important::

    The reviewing process is a collaboration between the contributor and the reviewer and should always be conducted with respect. All contributions are valuable, and reviewers should always explain their reasons for requesting changes or rejecting a pull request.

Once a pull request is submitted, reviewers will ensures that the proposed changes adhere to the guidelines provided. As a reviewer, you will need to test the code to verify that everything is in order and request changes if necessary. To do so, create a local branch from the pull request ::

    git fetch origin pull/<PULL NUMBER>/head:<BRANCH NAME>

where ``<PULL NUMBER>`` is the number of the pull request and ``<BRANCH NAME>`` is the name you assign to the local branch you are creating. Here is a checklist to guide your review :

* Unit tests have been added and/or modified.
* All unit tests pass succesfully.
* The documentation builds with no errors or warnings.
* The relevant sections of the documentation are complete and accurate.
* Docstrings are clear and comprehensive.
* The pull request correctly links to the related issue, if applicable.

Once the review is complete, if it has not already been done, you can ask for the PR issuer to add a short release note to the ``docs/release_notes/upcoming_changes`` directory. If everything is satisfactory, give your approval and tag one of the maintainers to merge the pull request.

Release workflow
----------------

.. note::

    We use the following versioning notation:

    ``X.Y.Z`` where

    - ``X`` is a **major** version
    - ``Y`` is a **minor** version
    - ``Z`` is a **patch** version

    Note also:

    - ``X.Y+1`` is the futur minor version
    - ``X.Y.Z-1`` is the previous patch version

Major/minor version release workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here are the step to make a major or minor release:

1. Once a milestone is almost completed, a new branch ``maintenance/X.Y.x`` is created and the final preparations before the release will be merged into this branch and the ``vX.Y.0`` tag will be created from this branch.
2. Update the documentation.

   * Merge the previous ``doc/X.Y-1.Z`` branch into the branch you are going to tag on **without deleting** the branch.
   * Remove ``.dev`` from the ``__version__`` in ``_version.py``.
   * If relevant, add a ``highlights.rst`` file with a bulleted list of highlights for the release.
   * Build the documentation site locally (:ref:`see how to build the documentation <builddoc>`)
   * Empty the ``doc/release_notes/upcoming_changes`` directory.
   * Update the ``switcher.json`` file to add a new documentation version with the following values: ::
     
       {
           "name": "X.Y",
           "version": "X.Y.0",
           "url": "https://www.graphinglib.org/doc-X.Y.0/",
           "preferred": true
       }
    
     Remove the ``"preferred": true`` for the previous version.
   * Rebuild the documentation to make sure it builds without problems.

3. Draft a release on GitHub. Copy the release notes from the documentation and adapt the syntax for Markdown.
4. Create the ``vX.Y.0`` tag.
5. Merge the ``maintenance/X.Y.x`` (branch without deleting it) to the ``main`` branch. Bump version to ``X.Y+1.0`` (or ``X+1.0.0`` for major version) in ``pyproject.toml`` and ``setup.py``. Bump ``__version__`` to ``X.Y+1.0.dev`` in ``_version.py``.
6. Create a new ``doc/X.Y.0`` branch from the ``maintenance/X.Y.0`` branch. Change the GraphingLib's source URL in ``requirements.txt`` to ``git+https://github.com/GraphingLib/GraphingLib@doc/X.Y.0``.
7. Manually trigger a build of the "latest" version on *Read the Docs* to update the project. Activate the version ``doc/X.Y.0`` and make it the default version in the admin settings.
8. Bump ``__version__`` to ``X.Y.1`` in ``_version.py``. 

Patch version release workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. attention::

    For a patch release, no updates are made on the ``main`` branch. If a bug correction has to be also applied to the next major/minor release, add an issue so that it is modified in the ``main`` branch. Unfortunately we don't have an automatic backporting method as of now.  

Here are the step to make a patch release:

1. Update the documentation.

   * Merge the previous ``doc/X.Y.Z-1`` branch into the ``maintenance/X.Y.x`` branch and **delete** the old documentation branch
   * Bump ``__version__`` from ``X.Y.Z-1`` to ``X.Y.Z`` in ``_version.py``.
   * Build the documentation site locally (:ref:`see how to build the documentation <builddoc>`)
   * If relevant, add a ``highlights.rst`` file with a bulleted list of highlights for the release.
   * Empty the ``doc/release_notes/upcoming_changes`` directory.
   * Update the ``switcher.json`` file to bump the ``"version"`` key of the ``X.Y.Z-1`` version to ``X.Y.Z`` and the ``"url"`` from ``/doc-X.Y.Z-1/`` to ``/doc-X.Y.Z/``. The ``"preferred"`` configuration is **left untouched**.
   * Rebuild the documentation to make sure it builds without problems.

2. Draft a release on GitHub. Copy the release notes from the documentation and adapt the syntax for Markdown.
3. Create the ``vX.Y.Z`` tag.
4. Create a new ``doc/X.Y.Z`` branch from the ``maintenance/X.Y.Z`` branch.
5. Bump version to ``X.Y.Z+1`` in ``pyproject.toml`` and ``setup.py``
6. Manually trigger a build of the "latest" version on *Read the Docs* to update the project. If the version ``doc-X.Y.Z-1`` is still active, deactivate it. Activate the ``doc-X.Y.Z`` version. If this patch release is on the **latest major/minor version**, set this new version as default in admin settings, else no changes necessary.
