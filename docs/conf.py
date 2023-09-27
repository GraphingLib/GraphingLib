# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Graphinglib"
copyright = "2023, Gustave Coulombe, Yannick Lapointe"
author = "Gustave Coulombe and Yannick Lapointe"
release = "v1.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_logo = "../images/GraphingLib-Logo-Bolder.svg"
html_static_path = ["_static"]

# -- Extension options -------------------------------------------------------

intersphinx_mapping = {
    "Matplotlib": (
        "https://matplotlib.org/stable/",
        None,
    ),
    "Numpy": ("https://numpy.org/doc/stable/", None),
    "Scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "Python": ("https://docs.python.org/", None),
    "Numpy typing": ("https://numpy.org/devdocs/", None),
}
pygments_dark_style = "github-dark"
