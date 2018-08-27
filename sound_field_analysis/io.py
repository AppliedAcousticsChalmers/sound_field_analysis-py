'''Input-Output functions
'''

from scipy import io as sio
import numpy as _np
from collections import namedtuple
from . import utils


class ArrayConfiguration(namedtuple('ArrayConfiguration', 'array_radius array_type transducer_type scatter_radius dual_radius')):
    """ Tuple of type ArrayConfiguration

    Parameters
    ----------
    array_radius : float
       Radius of array
    array_type : {'open', 'rigid'}
       Type array
    transducer_type: {'omni', 'cardioid'}
       Type of transducer,
    scatter_radius : float, optional
       Radius of scatterer, required for `array_type` == 'rigid'. (Default: equal to array_radius)
    dual_radius : float, optional
       Radius of second array, required for `array_type` == 'dual'
    """
    __slots__ = ()

    def __new__(cls, array_radius, array_type, transducer_type, scatter_radius=None, dual_radius=None):
        if array_type not in {'open', 'rigid', 'dual'}:
            raise ValueError('Sphere configuration has to be either open, rigid, or dual.')
        if transducer_type not in {'omni', 'cardioid'}:
            raise ValueError('Transducer type has to be either omni or cardioid.')
        if array_type == 'rigid' and not scatter_radius:
            scatter_radius = array_radius
        if array_type == 'dual' and not dual_radius:
            raise ValueError('For a dual array configuration, dual_radius must be provided.')
        if array_type == 'dual' and transducer_type == 'cardioid':
            raise ValueError('For a dual array configuration, cardioid transducers are not supported.')

        self = super(ArrayConfiguration, cls).__new__(cls, array_radius, array_type, transducer_type, scatter_radius, dual_radius)
        return self

    def __repr__(self):
        return 'ArrayConfiguration(\n' + ',\n'.join(
            '    {0} = {1}'.format(name, repr(data).replace('\n', '\n      '))
            for name, data in zip(['array_radius', 'array_type', 'transducer_type', 'scatter_radius', 'dual_radius'], self)) + ')'


class TimeSignal(namedtuple('TimeSignal', 'signal fs delay')):
    """ Tuple of type TimeSignal

    Parameters
    ----------
    signal : array_like
       Array of signals of shape [nSignals x nSamples]
    fs : int
       Sampling frequency
    delay : float

    """
    __slots__ = ()

    def __new__(cls, signal, fs, delay=None):
        signal = _np.atleast_2d(signal)
        no_of_signals = signal.shape[1]
        fs = _np.asarray(fs)
        delay = _np.asarray(delay)

        if (fs.size != 1) and (fs.size != no_of_signals):
            raise ValueError('fs can either be a scalar or an array with one element per signal.')
        if (delay.size != 1) and (delay.size != no_of_signals):
            raise ValueError('delay can either be a scalar or an array with one element per signal.')

        self = super(TimeSignal, cls).__new__(cls, signal, fs, delay)
        return self

    def __repr__(self):
        return 'TimeSignal(\n' + ',\n'.join(
            '    {0} = {1}'.format(name, repr(data).replace('\n', '\n      '))
            for name, data in zip(['signal', 'fs', 'delay'], self)) + ')'


class SphericalGrid(namedtuple('SphericalGrid', 'azimuth colatitude radius weight')):
    """ Tuple of type SphericalGrid

    Parameters
    ----------
    Azimuth, Colatitude : float
    Radius, Weights : float, optional
    """
    __slots__ = ()

    def __new__(cls, azimuth, colatitude, radius=None, weight=None):
        azimuth = _np.asarray(azimuth)
        colatitude = _np.asarray(colatitude)
        if radius is not None:
            radius = _np.asarray(radius)
        if weight is not None:
            weight = _np.asarray(weight)
        if azimuth.size != colatitude.size:
            raise ValueError('Azimuth and colatitude have to contain the same number of elements.')
        if (radius is not None) and (radius.size != 1) and (radius.size != azimuth.size):
            raise ValueError('Radius can either be a scalar or an array of same size as azimuth/colatitude.')
        if (weight is not None) and (weight.size != 1) and (weight.size != azimuth.size):
            raise ValueError('Weight can either be a scalar or an array of same size as azimuth/colatitude.')

        self = super(SphericalGrid, cls).__new__(cls, azimuth, colatitude, radius, weight)
        return self

    def __repr__(self):
        return 'SphericalGrid(\n' + ',\n'.join(
            '    {0} = {1}'.format(name, repr(data).replace('\n', '\n      '))
            for name, data in zip(['azimuth', 'colatitude', 'radius', 'weight'], self)) + ')'


class ArraySignal(namedtuple('ArraySignal', 'signal grid configuration temperature')):
    """ Tuple of type ArraySignal

    Parameters
    ----------
    signals : TimeSignal
       Holds time domain signals and sampling frequency fs
    grid : SphericalGrid
       Location grid of all time domain signals
    configuration : ArrayConfiguration
       Information on array configuration
    temperature : array_like, optional
       Temperature in room or at each sampling position
    """
    __slots__ = ()

    def __new__(cls, signal, grid, configuration=None, temperature=None):
        signal = TimeSignal(*signal)
        grid = SphericalGrid(*grid)
        if configuration is not None:
            configuration = ArrayConfiguration(*configuration)
        self = super(ArraySignal, cls).__new__(cls, signal, grid, configuration, temperature)
        return self

    def __repr__(self):
        return 'ArraySignal(\n' + ',\n'.join(
            '    {0} = {1}'.format(name, repr(data).replace('\n', '\n      '))
            for name, data in zip(['signal', 'grid', 'configuration', 'temperature'], self)) + ')'


def read_miro_struct(file_name, channel='irChOne', transducer_type='omni', scatter_radius=None):
    """ Reads miro matlab files.

    Parameters
    ----------
    matFile : filepath
       Path to file that has been exported as a struct
    channel : string, optional
       Channel that holds required signals. Default: 'irChOne'
    transducer_type : {omni, cardoid}, optional
       Sets the type of transducer used in the recording. Default: omni
    scatter_radius : float, option
       Radius of the scatterer. Default: None
       
    Note: This function expects a slightly modified miro file in that it expects a field `colatitude` instead of `elevation`. This is for avoiding confusion as may miro file contain colatitude data in the elevation field.

    Returns
    -------
    array_signal : ArraySignal
       Tuple containing a TimeSignal `signal`, SphericalGrid `grid`, ArrayConfiguration `configuration` and the air temperature
    """
    current_data = sio.loadmat(file_name)

    time_signal = TimeSignal(signal=_np.squeeze(current_data[channel]).T,
                             fs=_np.squeeze(current_data['fs']))

    mic_grid = SphericalGrid(azimuth=_np.squeeze(current_data['azimuth']),
                             colatitude=_np.squeeze(current_data['colatitude']),
                             radius=_np.squeeze(current_data['radius']),
                             weight=_np.squeeze(current_data['quadWeight']))
    
    if (mic_grid.colatitude < 0).any():    
        print('WARNING: The \'colatitude\' data contains negative values, which is an indication that it is actually elevation.')
    
    if _np.squeeze(current_data['scatterer']):
        sphere_config = 'rigid'
    else:
        sphere_config = 'open'
    array_config = ArrayConfiguration(mic_grid.radius, sphere_config, transducer_type, scatter_radius)

    return ArraySignal(time_signal, mic_grid, array_config, _np.squeeze(current_data['avgAirTemp']))


def empty_time_signal(no_of_signals, signal_length):
    """Returns an empty np rec array that has the proper data structure

    Parameters
    ----------
    no_of_signals : int
       Number of signals to be stored in the recarray
    signal_length : int
       Length of the signals to be stored in the recarray

    Returns
    -------
    time_data : recarray
       Structured array  with following fields:
    ::
       .signal           [Channels X Samples]
       .fs               Sampling frequency in [Hz]
       .azimuth          Azimuth of sampling points
       .colatitude       Colatitude of sampling points
       .radius           Array radius in [m]
       .grid_weights     Weights of quadrature
       .air_temperature  Average temperature in [C]
    """
    return _np.rec.array(_np.zeros(no_of_signals,
                         dtype=[('signal', str(signal_length) + 'f8'),
                                ('fs', 'f8'),
                                ('azimuth', 'f8'),
                                ('colatitude', 'f8'),
                                ('radius', 'f8'),
                                ('grid_weights', 'f8'),
                                ('air_temperature', 'f8')]))


def load_array_signal(filename):
    """Convenience function to load ArraySignal saved into np data structures

    Parameters
    ----------
    filename : string
       File to load

    Returns
    -------
    Y : ArraySignal
       See io.ArraySignal
    """
    return ArraySignal(*_np.load(filename))


def read_wavefile(filename):
    """ Reads in wavefiles and returns data [Nsig x Nsamples] and fs
    Parameters
    ----------
    filename, string
       Filename of wave file to be read

    Returns
    -------
    data, array_like
       Data of dim [Nsig x Nsamples]

    fs, int
       Sampling frequency of read data
    """
    fs, data = sio.wavfile.read(filename)
    return data.T, fs


def write_SSR_IRs(filename, time_data_l, time_data_r, wavformat="float"):
    """Takes two time signals and writes out the horizontal plane as HRIRs for the SoundScapeRenderer.
    Ideally, both hold 360 IRs but smaller sets are tried to be scaled up using repeat.

    Parameters
    ----------
    filename : string
       filename to write to
    time_data_l, time_data_l : io.ArraySignal
       ArraySignals for left/right ear
    fs : sampling frequency
    wavformat : string
       wav file format to write. Either "float" or "int16"
    """
    #equator_IDX_left = utils.nearest_to_value_logical_IDX(time_data_l.grid.colatitude, _np.pi / 2)
    #equator_IDX_right = utils.nearest_to_value_logical_IDX(time_data_r.grid.colatitude, _np.pi / 2)

    #IRs_left = time_data_l.signal.signal[equator_IDX_left]
    #IRs_right = time_data_r.signal.signal[equator_IDX_right]
    IRs_left = time_data_l.signal.signal
    IRs_right = time_data_r.signal.signal

    #if _np.mod(360 / IRs_left.shape[0], 1) == 0:
    #    IRs_left = _np.repeat(IRs_left, 360 / IRs_left.shape[0], axis=0)
    #else:
    #    raise ValueError('Number of channels for left ear cannot be fit into 360.')
    #if _np.mod(360 / IRs_right.shape[0], 1) == 0:
    #    IRs_right = _np.repeat(IRs_right, 360 / IRs_right.shape[0], axis=0)
    #else:
    #    raise ValueError('Number of channels for left ear cannot be fit into 360.')

    IRs_to_write = utils.interleave_channels(IRs_left, IRs_right, style="SSR")
    #data_to_write = utils.simple_resample(IRs_to_write, original_fs=time_data_l.signal.fs, target_fs=44100)
    data_to_write = IRs_to_write

    # Fix SSR IR alignment stuff: left<>right flipped and 90 degree rotation
    #data_to_write = _np.flipud(data_to_write)
    #data_to_write = _np.roll(data_to_write, -90, axis=0)

    if wavformat == "float":
        sio.wavfile.write(filename, int(time_data_l.signal.fs), data_to_write.astype(_np.float32).T)
    elif wavformat == "int16":
        sio.wavfile.write(filename, int(time_data_l.signal.fs), (data_to_write * 32767).astype(_np.int16).T)
    else:
        raise TypeError("Format " + wavformat + "not known. Should be either 'float' or 'int16'.")
