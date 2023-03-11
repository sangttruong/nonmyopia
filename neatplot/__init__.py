"""
Neat plotting utilities for Python.
"""

from pathlib import Path
import matplotlib.pyplot as plt


def set_style(style_str='default'):
    if style_str == 'default':
        plt.style.use((Path(__file__).parent / 'matplotlibrc').resolve())
    elif style_str == 'fonts':
        plt.style.use((Path(__file__).parent / 'matplotlibrc_fonts').resolve())


def save_figure(file_name='figure', ext_list=None, dir_name='.', white_background=True):
    """Save figure for all extensions in ext_list."""

    # Default ext_list
    if ext_list is None:
        ext_list = ['pdf', 'png']

    # If ext_list is a single str
    if isinstance(ext_list, str):
        ext_list = [ext_list]

    # Set facecolor and edgecolor
    (fc, ec) = ('w', 'w') if white_background else ('none', 'none')

    # Save each type in ext_list
    for ext in ext_list:
        save_str = str(Path(dir_name) / file_name) + '.' + ext
        plt.savefig(save_str, bbox_inches='tight', facecolor=fc, edgecolor=ec)
        print(f'Saved figure {save_str}')


def update_rc(key_str, value):
    """Update matplotlib rc params."""
    plt.rcParams.update({key_str: value})
