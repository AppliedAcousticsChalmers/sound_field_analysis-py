"""
Miscellaneous utility functions"""

import sys
from datetime import datetime
from itertools import cycle

import numpy as np
from scipy.signal import resample

spinner = cycle(['-', '/', '|', '\\'])


def env_info():
    """ Guess environment based on sys.modules.

    Returns
    -------
    env : string{'jupyter_notebook', 'ipython_terminal', 'terminal'}
       Guessed environment
    """
    if 'ipykernel' in sys.modules:
        return 'jupyter_notebook'
    elif 'IPython' in sys.modules:
        return 'ipython_terminal'
    else:
        return 'terminal'


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
        print('\r' + description + ': [{0:50s}] {1:.1f}%'.format(
            '#' * int(amount_done * 50), amount_done * 100), end="", flush=True)
        if amount_done >= 1:
            print('\n')


def db(data, power=False):
    """Convenience function to calculate the 20*log10(abs(x))

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
    """
    if power:
        factor = 10
    else:
        factor = 20
    return factor * np.log10(np.abs(data))


def deg2rad(deg):
    """Converts from degree [0 ... 360] to radiant [0 ... 2pi]
    """
    return deg % 360 / 180 * np.pi


def rad2deg(rad):
    """Converts from radiant [0 ... 2pi] to degree [0 ... 360]
    """
    return rad / np.pi * 180 % 360

def cart2sph(cartesian_coords, is_deg=False):
    """
    Parameters
    ----------
    cartesian_coords : numpy.ndarray
        cartesian coordinates (x, y, z) of size [3; number of coordinates]
    is_deg : bool, optional
        if values should be calculated in degrees (radians otherwise)

    Returns
    -------
    numpy.ndarray
        calculated spherical coordinates (azimuth [0 ... 2pi], colatitude [0 ... pi], radius [meter]) of size [3; number of coordinates]
    """
    x = cartesian_coords[0]
    y = cartesian_coords[1]
    z = cartesian_coords[2]

    az = np.arctan2(y, x)  # return values -pi ... pi
    r = np.sqrt(np.power(x, 2) + np.power(y, 2) + np.power(z, 2))
    col = np.arccos(z/r)

    az %= (2 * np.pi)  # converting to 0 ... 2pi
    if is_deg:
        az = rad2deg(az)
        col = rad2deg(col)

    return np.vstack((az, col, r))

def sph2cart(spherical_coords, is_deg=False):
    """
    Parameters
    ----------
    spherical_coords : numpy.ndarray
        spherical coordinates (azimuth, colatitude, radius) of size [3; number of coordinates]
    is_deg : bool, optional
        True if values are given in degrees (radians otherwise), [default: False]

    Returns
    -------
    numpy.ndarray
        cartesian coordinates (x, y, z) of size [3; number of coordinates]
    """
    az = spherical_coords[0]
    col = spherical_coords[1]
    r = spherical_coords[2]

    if is_deg:
        az = deg2rad(az)
        col = deg2rad(col)

    x = r * np.sin(col) * np.cos(az)
    y = r * np.sin(col) * np.sin(az)
    z = r * np.cos(col)

    return np.vstack((x, y, z))

def nearest_to_value_IDX(array, target_val):
    """Returns nearest value inside an array
    """
    return (np.abs(array - target_val)).argmin()


def nearest_to_value(array, target_val):
    """Returns nearest value inside an array
    """
    return array[nearest_to_value_IDX(array, target_val)]


def nearest_to_value_logical_IDX(array, target_val):
    """Returns logical indices of nearest values inside array
    """
    return array == nearest_to_value(array, target_val)


def interleave_channels(left_channel, right_channel, style=None):
    """Interleave left and right channels. Style == 'SSR' checks if we total 360 channels
    """
    if not left_channel.shape == right_channel.shape:
        raise ValueError('left_channel and right_channel have to be of same dimensions!')

    if style == 'SSR':
        if not (left_channel.shape[0] == 360):
            raise ValueError('Provided arrays to have 360 channels (Nchannel x Nsamples).')

    output_data = np.repeat(left_channel, 2, axis=0)
    output_data[1::2, :] = right_channel

    return output_data


def simple_resample(data, original_fs, target_fs):
    """Wrap scipy.signal.resample with a simpler API
    """
    return resample(data, num=int(data.shape[1] * target_fs / original_fs), axis=1)


def scalar_broadcast_match(a, b):
    """ Returns arguments as np.array, if one is a scalar it will broadcast the other one's shape.
    """
    a, b = np.atleast_1d(a, b)
    if a.size == 1 and b.size != 1:
        a = np.broadcast_to(a, b.shape)
    elif b.size == 1 and a.size != 1:
        b = np.broadcast_to(b, a.shape)
    return a, b


def frq2kr(target_frequency, freq_vector):
    """Returns the kr bin closest  to the target frequency

    Parameters
    ----------
    target_frequency : float
       Target frequency
    freq_vector : array_like
       Array containing the available frequencys

    Returns
    -------
    krTarget : int
       kr bin closest to target frequency
    """

    return (np.abs(freq_vector - target_frequency)).argmin()


def stack(vector_1, vector_2):
    """Stacks two 2D vectors along the same-sized dimension or the smaller one"""
    vector_1, vector_2 = np.atleast_2d(vector_1, vector_2)
    M1, N1 = vector_1.shape
    M2, N2 = vector_2.shape

    if M1 == M2 and (M1 < N1 or M2 < N2):
        out = np.vstack([vector_1, vector_2])
    elif N1 == N2 and (N1 < M1 or N2 < M2):
        out = np.hstack([vector_1, vector_2])
    else:
        raise ValueError('vector_1 and vector_2 dont have a common dimension.')
    return np.squeeze(out)


def current_time():
    return datetime.now()
