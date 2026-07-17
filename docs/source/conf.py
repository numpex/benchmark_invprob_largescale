# Configuration file for the Sphinx documentation builder.

# -- Path setup ------------------------------------------------------
import sys
from pathlib import Path

# Add project root to path so Sphinx can import modules.
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# -- Project information --------------------------------------------

project = "Benchmark on Inference for Large-Scale Inverse Problems"
copyright = "2026, authors"
author = "authors"

version = "0.1.0"
release = "0.1.0"

# -- General configuration ------------------------------------------

extensions = [
    # Registers Sphinx-Gallery directives used in pre-generated pages.
    "sphinx_gallery.load_style",
    # Responsive grids and cards used by the landing page.
    "sphinx_design",
    # Adds a copy button to code blocks.
    "sphinx_copybutton",
]

exclude_patterns = []
language = "en"

# -- Options for HTML output ----------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "Large-Scale Inverse Problems Benchmark"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# The standalone operational guides are reached directly from the home page.
html_sidebars = {
    "getting_started/run_on_cluster": [],
    "getting_started/config_guide": [],
    "**": ["sidebar-nav-bs.html"],
}

html_theme_options = {
    "logo": {
        "image_light": "_static/logo_numpex.png",
        "image_dark": "_static/logo_numpex.png",
        "alt_text": "NumPEx",
        "text": "Benchmarking Large-Scale Inverse Problems",
    },
    "show_nav_level": 2,
    "navigation_depth": 3,
    "collapse_navigation": False,
    "show_toc_level": 2,
    "navbar_align": "left",
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/numpex/benchmark_invprob_largescale",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
}
