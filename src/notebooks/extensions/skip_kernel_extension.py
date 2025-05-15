# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 00:07:39 2017
@author: Robbe Sneyders
Source: https://github.com/RobbeSneyders/Jupyter-skip-extension/blob/master/skip_kernel_extension.py
"""
from IPython import get_ipython


def skip(line, cell=None):
    '''Skips execution of the current line/cell.'''
    if eval(line):
        return

    get_ipython().run_cell(cell)


def load_ipython_extension(shell):
    '''Registers the skip magic when the extension loads.'''
    shell.register_magic_function(skip, 'line_cell')


def unload_ipython_extension(shell):
    '''Unregisters the skip magic when the extension unloads.'''
    del shell.magics_manager.magics['cell']['skip']