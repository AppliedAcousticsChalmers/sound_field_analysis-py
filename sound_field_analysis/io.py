'''Input-Output functions
'''

from pathlib import Path
from scipy import io as sio
from collections import namedtuple
import numpy as _np


def readMiroStruct(matFile):
    """ Reads miro matlab files.

    Parameters
    ----------
    matFile : filepath
       .miro file that has been exported as a struct like so
       ::
         load SOFiA_A1;
         SOFiA_A1_struct = struct(SOFiA_A1);
         save('SOFiA_A1_struct.mat', 'SOFiA_A1_struct');

    Returns
    -------
    time_data : named tuple
    `time_data` tuple with following fields
    ::
       .signals [Channels X Samples]
       .fs               Sampling frequency in [Hz]
       .azimuth          Azimuth of sampling points
       .colatitude       Colatitude of sampling points
       .grid_weights     Weights of quadrature
       .air_temperature  Average temperature in [C]
       .radius           Array radius in [m]
       .centerIR         Impulse response of center mic (if available), zero otherwise
    """
    # Scipy import of matlab struct
    mat = sio.loadmat(matFile)
    filename = Path(matFile).stem
    data = mat[filename]

    time_data = namedtuple('time_data', 'signals, fs, azimuth, colatitude, grid_weights, air_temperature, radius, centerIR')
    time_data.signals = data['irChOne'][0][0].T
    time_data.fs = data['fs'][0][0][0][0]
    time_data.azimuth = data['azimuth'][0][0][0]
    time_data.colatitude = data['elevation'][0][0][0]
    time_data.grid_weights = data['quadWeight'][0][0][0]
    time_data.air_temperature = data['avgAirTemp'][0][0][0][0]
    time_data.radius = data['radius'][0][0][0][0]
    time_data.centerIR = _np.array(data['irCenter'][0][0]).flatten()  # Flatten nested array

    return time_data
