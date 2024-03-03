# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

from graphinglib import __version__

sys.path.insert(0, os.path.abspath("sphinxext"))

project = "GraphingLib"
copyright = "2024, Gustave Coulombe, Yannick Lapointe"
author = "Gustave Coulombe and Yannick Lapointe"
release = __version__

json_url = "https://graphinglib.readthedocs.io/en/latest/_static/switcher.json"

if "dev" in release:
    version_match = "dev"
else:
    version_match = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
    "sphinx_favicon",
    "sphinx_design",
    "gallery_generator",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["graphinglib.css"]
html_theme_options = {
    "github_url": "https://github.com/GraphingLib/GraphingLib",
    "logo": {
        "text": "GraphingLib",
        "image_dark": "../images/GraphingLib-Logo-Bolder.svg",
        "image_light": "../images/GraphingLib-Logo-Bolder.svg",
    },
    "pygment_light_style": "tango",
    "pygment_dark_style": "github-dark",
    "show_toc_level": 2,
    "show_prev_next": False,
    "switcher": {
        "json_url": json_url,
        "version_match": version_match,
    },
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    "show_version_warning_banner": True,
}
html_context = {"default_mode": "dark"}
html_show_sourcelink = False
favicons = ["icons/GraphingLib-favicon_250x250.png"]
html_sidebars = {
    "handbook/index": [],
    "installation": [],
    "examples/index": [],
}

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

autodoc_type_aliases = {
    "Iterable": "Iterable",
    "ArrayLike": "ArrayLike",
}
