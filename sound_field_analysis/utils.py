"""Miscellenious utility functions
"""
import sys
from itertools import cycle
from numpy import log10, abs, repeat, atleast_1d, broadcast_to, pi
from scipy.signal import resample
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
        maxIDX = (int(maxIDX) - 1)
        if maxIDX == 0:
            amount_done = 1
        else:
            amount_done = curIDX / maxIDX
        print('\r' + description + ': [{0:50s}] {1:.1f}%'.format('#' * int(amount_done * 50), amount_done * 100), end="", flush=True)
        if amount_done >= 1:
            print('\n')


def db(data, power=False):
    '''Convenience function to calculate the 20*log10(abs(x))

    Parameters
    ----------
    data : array_like
       signals to be converted to db
    power : boolean
       data is a power signal and only needs factor 10

    Returns
    -------
    db : array_like
       20 * log10(abs(data))
    '''
    if power:
        factor = 10
    else:
        factor = 20
    return factor * log10(abs(data))


def deg2rad(deg):
    """Converts from degree [0 ... 360] to radiant [0 ... 2 pi]
    """
    return deg % 360 / 180 * pi


def rad2deg(rad):
    """Converts from radiant [0 ... 2 pi] to degree [0 ... 360]
    """
    return rad / pi * 180 % 360


def nearest_to_value(array, value):
    """Returns nearest value inside an array
    """
    return array[(abs(array - value)).argmin()]


def logical_IDX_of_nearest(array, value):
    """Returns logical indices of nearest values inside array
    """
    return array == nearest_to_value(array, value)


def interleave_channels(left_channel, right_channel, style=None):
    """Interleave left and right channels. Style == 'SSR' checks if we total 360 channels
    """
    if not left_channel.shape == right_channel.shape:
        raise ValueError('left_channel and right_channel have to be of same dimensions!')

    if style == 'SSR':
        if not (left_channel.shape[0] == 360):
            raise ValueError('Provided arrays to have 360 channels (Nchannel x Nsamples).')

    output_data = repeat(left_channel, 2, axis=0)
    output_data[1::2, :] = right_channel

    return output_data


def simple_resample(data, original_fs, target_fs):
    """Wrap scipy.signal.resample with a simpler API
    """
    return resample(data, num=int(data.shape[1] * target_fs / original_fs), axis=1)


def scalar_broadcast_match(a, b):
    """ Returns arguments as np.array, if one is a scalar it will broadcast the other one's shape.
    """
    a, b = atleast_1d(a, b)
    if a.size == 1 and b.size != 1:
        a = broadcast_to(a, b.shape)
    elif b.size == 1 and a.size != 1:
        b = broadcast_to(b, a.shape)
    return a, b
