"""Miscellenious utility functions
"""
import sys
try:
    from tqdm import tqdm_notebook, tqdm
    __has_tqdm = True
except ImportError:
    __has_tqdm = False


def ipython_info():
    ip = False
    if 'ipykernel' in sys.modules:
        ip = 'notebook'
    elif 'IPython' in sys.modules:
        ip = 'terminal'
    return ip


def progress_bar(loopRange):
    if __has_tqdm:
        if ipython_info() == 'notebook':
            return tqdm_notebook(loopRange)
        else:
            return tqdm(loopRange)
    else:
        return loopRange
