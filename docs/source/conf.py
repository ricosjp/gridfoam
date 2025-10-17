# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path("../../src").resolve()))

# -- pyvista configuration ---------------------------------------------------
import pyvista

pyvista.start_xvfb()
pyvista.BUILDING_GALLERY = True
pyvista.OFF_SCREEN = True
# Preferred plotting style for documentation
pyvista.set_plot_theme("document")
pyvista.global_theme.window_size = [1024, 768]
pyvista.global_theme.font.size = 22
pyvista.global_theme.font.label_size = 22
pyvista.global_theme.font.title_size = 22
pyvista.global_theme.return_cpos = False
pyvista.set_jupyter_backend(None)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import gridfoam

project = "gridfoam"
copyright = "2025, RICOS"
author = "RICOS"
version = gridfoam.__version__
release = gridfoam.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "numpydoc",
    "sphinx.ext.githubpages",
    "sphinxcontrib.mermaid",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
    "pyvista.ext.plot_directive",
    "pyvista.ext.viewer_directive",
    "sphinx_design",
]

sphinx_gallery_conf = {
    "examples_dirs": "../../tutorials",
    "gallery_dirs": "tutorials",
    "within_subsection_order": "FileNameSortKey",
    "filename_pattern": r"/*\.py",
    "image_scrapers": (
        "matplotlib",
        "pyvista",
    ),
}


# -- Options for HTML output -------------------------------------------------
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    # "logo": {
    #     "image_light": "logo.png",
    #     "image_dark": "logo_dark.png",
    # },
    # https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/header-links.html#fontawesome-icons
    # "icon_links": [
    #     {
    #         "name": "GitHub",
    #         "url": "https://github.com/arviz-devs/arviz",
    #         "icon": "fa-brands fa-github",
    #     },
    # ],
    "navbar_start": ["navbar-logo", "navbar-version"],
    "navbar_align": "content",
    "header_links_before_dropdown": 5,
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
    # "use_edit_page_button": True,
    # "analytics": {"google_analytics_id": "G-W1G68W77YV"},
    # "external_links": [
    #     {"name": "The ArviZ project", "url": "https://www.arviz.org"},
    # ],
}
html_context = {
    # "github_user": "arviz-devs",
    # "github_repo": "arviz",
    # "github_version": "main",
    "doc_path": "doc/source/",
    "default_mode": "light",
}
html_sidebars: dict[str, list] = {"index": []}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# use additional pages to add a 404 page
html_additional_pages = {
    "404": "404.html",
}

html_favicon = "_static/favicon.ico"
html_show_sphinx = False

# -- Extension configuration -------------------------------------------------
# Generate API documentation when building
autosummary_generate = True
autodoc_typehints = "none"

# numpydoc configuration
# numpydoc_show_class_members = False
# numpydoc_xref_param_type = True
# numpydoc_xref_ignore = {
#     "of",
#     "or",
#     "optional",
#     "default",
#     "1D",
#     "2D",
#     "3D",
#     "n-dimensional",
#     "K",
#     "M",
#     "N",
#     "S",
# }
# numpydoc_xref_aliases = {
#     "DataArray": ":class:`~xarray.DataArray`",
#     "Dataset": ":class:`~xarray.Dataset`",
#     "DataTree": ":class:`~xarray.DataTree`",
#     "Labeller": ":ref:`Labeller <labeller_api>`",
#     "ndarray": ":class:`~numpy.ndarray`",
#     "InferenceData": ":class:`~arviz.InferenceData`",
#     "matplotlib_axes": ":class:`matplotlib Axes <matplotlib.axes.Axes>`",
#     "bokeh_figure": ":class:`Bokeh Figure <bokeh.plotting.figure>`",
# }

# Add any paths that contain templates here, relative to this directory.
templates_path = ["../_templates"]

# codeautolink
codeautolink_autodoc_inject = False
codeautolink_search_css_classes = ["highlight-default"]
codeautolink_concat_default = True
