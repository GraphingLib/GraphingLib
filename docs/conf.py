# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import time

from graphinglib import __version__

sys.path.insert(0, os.path.abspath("sphinxext"))

project = "GraphingLib"
copyright = f"2023-{time.strftime('%Y')}, Gustave Coulombe, Yannick Lapointe"
author = "Gustave Coulombe and Yannick Lapointe"
release = __version__

json_url = "https://www.graphinglib.org/latest/_static/switcher.json"

if "dev" in release:
    version_match = "dev"
else:
    version_match = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
    "sphinx_favicon",
    "sphinx_design",
    "gallery_generator",
    "matplotlib.sphinxext.plot_directive",
    "release_notes_generator",
    "sphinxext.opengraph",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static", "example_thumbs"]
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
    "header_links_before_dropdown": 10,
}
html_context = {"default_mode": "dark"}
html_show_sourcelink = False
favicons = ["icons/GraphingLib-favicon_250x250.png"]
html_sidebars = {
    "handbook/index": [],
    "installation": [],
    "examples/index": [],
    "contributing/index": [],
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

autosummary_generate = True

numpydoc_show_class_members = False

plot_include_source = True
plot_html_show_source_link = False
plot_pre_code = "import graphinglib as gl\nimport numpy as np"

opg_site_url = "https://www.graphinglib.org/
ogp_image = "https://raw.githubusercontent.com/GraphingLib/GraphingLib/main/images/opengraph_GL.PNG"
opg_description = "GraphingLib : A Python library for creating publication-quality figures with ease."
