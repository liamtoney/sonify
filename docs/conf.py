import sys
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

version = version('sonify')

html_theme = 'sphinx_rtd_theme'

templates_path = ['_templates']

napoleon_numpy_docstring = False

master_doc = 'index'

autodoc_mock_imports = ['matplotlib', 'numpy', 'obspy', 'scipy', 'tqdm']

intersphinx_mapping = {
    'obspy': ('https://docs.obspy.org/', None),
    'python': ('https://docs.python.org/3/', None),
}

html_theme_options = {'prev_next_buttons_location': None}
