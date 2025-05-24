"""
Extension to skip cells in Jupyter Notebooks.

Created on Mon Apr 24 00:07:39 2017
@author: Robbe Sneyders
Source: https://github.com/RobbeSneyders/Jupyter-skip-extension/blob/master/skip_kernel_extension.py
"""

from ast import literal_eval
from typing import cast

from IPython.core.getipython import get_ipython
from IPython.core.interactiveshell import InteractiveShell


def skip(line, cell=None) -> None:
    """Skips execution of the current line/cell."""
    if literal_eval(line):
        return

    shell: InteractiveShell = cast(InteractiveShell, get_ipython())
    shell.run_cell(cell)


def load_ipython_extension(shell) -> None:
    """Registers the skip magic when the extension loads."""
    shell.register_magic_function(skip, "line_cell")


def unload_ipython_extension(shell) -> None:
    """Unregisters the skip magic when the extension unloads."""
    del shell.magics_manager.magics["cell"]["skip"]
