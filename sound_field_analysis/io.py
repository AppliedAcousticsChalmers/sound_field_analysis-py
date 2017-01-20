'''Input-Output functions
'''

from scipy import io as sio
import numpy as _np
from . import utils


def read_miro_struct(file_name, channel='irChOne'):
    """ Reads miro matlab files.

    Parameters
    ----------
    matFile : filepath
       Path to file that has been exported as a struct like so
       ::
         load SOFiA_A1;
         SOFiA_A1_struct = struct(SOFiA_A1);
         save('SOFiA_A1_struct.mat', , '-struct', 'SOFiA_A1_struct');
    channel : string, optional
       Channel that holds required signals. Default: 'irChOne'

    Returns
    -------
    td : recarray
    `time_data` array with fields from empty_time_signal()
    ::
       .signal           [Channels X Samples]
       .fs               Sampling frequency in [Hz]
       .azimuth          Azimuth of sampling points
       .colatitude       Colatitude of sampling points
       .radius           Array radius in [m]
       .grid_weights     Weights of quadrature
       .air_temperature  Average temperature in [C]
    """
    current_data = sio.loadmat(file_name)
    no_of_signals = int(_np.squeeze(current_data['nIr']))
    signal_length = int(_np.squeeze(current_data['taps']))

    td = empty_time_signal(no_of_signals, signal_length)

    td.azimuth = _np.squeeze(current_data['azimuth'])
    td.colatitude = _np.pi / 2 - _np.squeeze(current_data['elevation'])
    td.airtemperature = _np.squeeze(current_data['avgAirTemp'])
    td.grid_weights = _np.squeeze(current_data['quadWeight'])
    td.radius = _np.squeeze(current_data['radius'])
    td.fs = _np.squeeze(current_data['fs'])
    td.signal = _np.squeeze(current_data[channel]).T

    return td


def empty_time_signal(no_of_signals, signal_length):
    """Returns an empty np rec array that has the proper data structure

    Parameter
    ---------
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


def load_time_signal(filename):
    """Convenience function to load saved np data structures

    Parameter
    ---------
    filename : string
       File to load

    Returns
    -------
    time_data : recarray
       Structured array  with following fields:
    ::
       .IR               [Channels X Samples]
       .fs               Sampling frequency in [Hz]
       .azimuth          Azimuth of sampling points
       .colatitude       Colatitude of sampling points
       .radius           Array radius in [m]
       .grid_weights     Weights of quadrature
       .air_temperature  Average temperature in [C]
    """
    return _np.rec.array(_np.load(filename))


def read_wavefile(filename):
    """ Reads in wavefiles and returns data [Nsig x Nsamples] and fs
    Parameter
    ---------
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


def write_SSR_IRs(filename, time_data_l, time_data_r):
    """Takes two time signals and interprets them as left/right HRIRs for the SoundScapeRenderer

    Parameters
    ----------
    filename : string
       filename to write to
    time_data_l, time_data_l : time_data recarrays
       time_data arrays for left/right channel.
    """
    equator_IDX_left = utils.logical_IDX_of_nearest(time_data_l.colatitude, _np.pi / 2)
    equator_IDX_right = utils.logical_IDX_of_nearest(time_data_r.colatitude, _np.pi / 2)

    IRs_left = time_data_l.signal[equator_IDX_left]
    IRs_right = time_data_r.signal[equator_IDX_right]

    if IRs_left.shape[0] == 180:
        IRs_left = _np.repeat(IRs_left, 2, axis=0)
        IRs_right = _np.repeat(IRs_right, 2, axis=0)

    IRs_to_write = utils.interleave_channels(IRs_left, IRs_right, style="SSR")
    data_to_write = utils.simple_resample(IRs_to_write, original_fs=time_data_l.fs[0], target_fs=44100)

    # Fix SSR IR alignment stuff: left<>right flipped and 90 degree rotation
    data_to_write = _np.flipud(data_to_write)
    data_to_write = _np.roll(data_to_write, -90, axis=0)

    sio.wavfile.write(filename, 44100, data_to_write.T)  # wavfile.write expects [Nsamples x Nsignals]
