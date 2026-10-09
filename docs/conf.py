from importlib import metadata

project = 'sonify'

author = 'Liam Toney'

html_show_copyright = False

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
]

version = metadata.version('sonify')

html_theme = 'sphinx_rtd_theme'

templates_path = ['_templates']

intersphinx_mapping = {
    'obspy': ('https://docs.obspy.org/', None),
    'python': ('https://docs.python.org/3/', None),
}

html_theme_options = {'prev_next_buttons_location': None}
