"""Miscellenious utility functions
"""
import sys
from itertools import cycle
spinner = cycle(['-', '/', '|', '\\'])


def env_info():
    """ Guess environment based on sys.modules.

    Returns
    -------
    env : string{'jupyter_notebook', 'ipython_terminal', 'terminal'}
       Guesed environment
    """
    if 'ipykernel' in sys.modules:
        return 'jupyter_notebook'
    elif 'IPython' in sys.modules:
        return 'ipython_terminal'
    else:
        return 'terminal'
    return ip


__env = env_info()


def progress_bar(curIDX, maxIDX=None, description='Progress'):
    """ Display a spinner or a progress bar

    Parameters
    ----------
    curIDX : int
       Current position in the loop
    maxIDX : int, optional
       Number of iterations. Will force a spinner if set to None. [Default: None]
    description : string, optional
       Clarify what's taking time
    """
    if maxIDX is None:
        print('\r' + description + ': ' + next(spinner), end='', flush=True)
    else:
        amount_done = curIDX / (int(maxIDX) - 1)
        print('\r' + description + ': [{0:50s}] {1:.1f}%'.format('#' * int(amount_done * 50), amount_done * 100), end="", flush=True)
        if amount_done >= 1:
            print('\n')
