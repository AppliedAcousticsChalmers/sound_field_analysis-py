# SOFiA example 3: A measured plane wave from AZ180°, EL90°
#                  in the anechoic chamber using a cardioid mic.

import numpy as np
from sofia import gen, process, plot
from vispy import scene
from scipy import io
from collections import namedtuple

dataFile = 'data/SOFiA_A4_struct.mat'
Nsft = 5       # Spatial Fourier Transform Order
Nrf = Nsft     # Radial Filter Order
amp_maxdB = 10 # Maximum modal amplification [dB]
ac = 1         # Array configuration: Open sphere with cardioid mic
Nmtx = Nsft    # Plot Order
krIndex = 70   # kr-bin (frequency) to plot

# Import matlab struct
mat = io.loadmat(dataFile)
data = mat['SOFiA_A4_struct']

# timeData tuple
timeData = namedtuple('timeData', 'FS, radius, quadratureGrid, downSample, averageAirTemp, irOverlay, centerIR, impulseResponse')
timeData.FS = data['fs'][0][0][0][0]
timeData.radius = data['radius'][0][0][0][0]
if data['resampleToFS'][0][0]:
    timeData.downSample = timeData.FS / data['resampleToFS'][0][0]
else:
    timeData.downSample = 1
timeData.quadratureGrid = np.array([data['azimuth'][0][0][0],
                                    data['elevation'][0][0][0],
                                    data['quadWeight'][0][0][0]]).T
timeData.averageAirTemp = data['avgAirTemp'][0][0][0][0]

# TODO: hcomp, resample
timeData.centerIR = np.array(data['irCenter'][0][0]).flatten()  # Flatten nested array
timeData.impulseResponse = data['irChOne'][0][0].T

timeData.irOverlay = np.abs(np.mean(timeData.impulseResponse, 0))
timeData.irOverlay /= np.max(timeData.irOverlay)

# Transform time domain data to frequency domain and generate kr-vector
fftData, kr, f, _ = process.fdt(timeData)

# Spatial Fourier transform
Pnm = process.stc(Nsft, fftData, timeData.quadratureGrid)

# Radial filters
dn, _ = gen.mf(Nrf, kr, ac, amp_maxdB=amp_maxdB, )
dn = process.rfi(dn)  # TODO: Radial Filter improvement

# Plot
mtxData = plot.makeMTX(Pnm, dn, Nviz=Nmtx, krIndex=krIndex)
# TODO: display
