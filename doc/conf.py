import os
import sys

sys.path.insert(0, os.path.abspath('../sonify'))

project = 'sonify'

author = 'Liam Toney'

html_show_copyright = False

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'recommonmark',
    'sphinx.ext.viewcode',
]

html_theme = 'sphinx_rtd_theme'

templates_path = ['_templates']

napoleon_numpy_docstring = False

master_doc = 'index'

autodoc_mock_imports = ['colorcet', 'matplotlib', 'numpy', 'obspy', 'scipy']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'obspy': ('https://docs.obspy.org/', None),
}

html_theme_options = {'prev_next_buttons_location': None}
