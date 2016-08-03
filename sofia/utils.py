"""Miscellenious utility functions
"""
import sys
from itertools import cycle
spinner = cycle(['-', '/', '|', '\\'])


def env_info():
    ip = False
    if 'ipykernel' in sys.modules:
        ip = 'jupyter_notebook'
    elif 'IPython' in sys.modules:
        ip = 'ipython_terminal'
    else:
        ip = 'terminal'
    return ip
__env = env_info()


def progress_bar(curIDX, maxIDX=None):
    """ Display a spinner or a progress bar

    Parameters
    ----------
    curIDX : int
       Current position in the loop
    maxIDX : int, optional
       Number of iterations. Will force a spinner if set to None. [Default: None]
    """
    if maxIDX is None:
        print('\r' + next(spinner), end='', flush=True)
    else:
        amount_done = curIDX / (int(maxIDX) - 1)
        print('\rProgress: [{0:50s}] {1:.1f}%'.format('#' * int(amount_done * 50), amount_done * 100), end="", flush=True)
        if amount_done >= 1:
            print('\n')
