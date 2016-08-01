"""Miscellenious utility functions
"""
import sys


def ipython_info():
    ip = False
    if 'ipykernel' in sys.modules:
        ip = 'notebook'
    elif 'IPython' in sys.modules:
        ip = 'terminal'
    return ip


def progress_bar(loopRange):
    try:
        if ipython_info() == 'notebook':
            from tqdm import tqdm_notebook
            return tqdm_notebook(loopRange)
        else:
            from tqdm import tqdm
            return tqdm(loopRange)
    except ImportError:
        return loopRange
