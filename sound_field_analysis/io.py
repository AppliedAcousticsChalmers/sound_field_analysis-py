'''Input-Output functions
'''

from scipy import io as sio
import numpy as _np


def read_miro_struct(file_name):
    """ Reads miro matlab files.

    Parameters
    ----------
    matFile : filepath
       .miro file that has been exported as a struct like so
       ::
         load SOFiA_A1;
         SOFiA_A1_struct = struct(SOFiA_A1);
         save('SOFiA_A1_struct.mat', , '-struct', 'SOFiA_A1_struct');

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
    td.signal = _np.squeeze(current_data['irChOne']).T

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
