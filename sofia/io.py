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
    timeData : named tuple
    `timeData tuple with following fields
    ::
       .impulseResponses [Channels X Samples]
       .FS
       .radius           Array radius
       .averageAirTemp   Temperature in [C]
       (.centerIR        [1 x Samples] )
    """
    # Import matlab struct
    mat = sio.loadmat(matFile)
    filename = Path(matFile).stem
    data = mat[filename]

    # timeData tuple
    timeData = namedtuple('timeData', 'FS, radius, quadratureGrid, downSample, averageAirTemp, irOverlay, centerIR, impulseResponse')
    timeData.FS = data['fs'][0][0][0][0]
    timeData.radius = data['radius'][0][0][0][0]
    if data['resampleToFS'][0][0]:
        timeData.downSample = timeData.FS / data['resampleToFS'][0][0]
    else:
        timeData.downSample = 1
    timeData.quadratureGrid = _np.array([data['azimuth'][0][0][0],
                                         data['elevation'][0][0][0],
                                         data['quadWeight'][0][0][0]]).T
    timeData.averageAirTemp = data['avgAirTemp'][0][0][0][0]

    # TODO: hcomp, resample
    timeData.centerIR = _np.array(data['irCenter'][0][0]).flatten()  # Flatten nested array
    timeData.impulseResponse = data['irChOne'][0][0].T

    timeData.irOverlay = _np.abs(_np.mean(timeData.impulseResponse, 0))
    timeData.irOverlay /= _np.max(timeData.irOverlay)

    return timeData
