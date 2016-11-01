# SFA example 5: Spatiotemporal resolution

import numpy as np
import sys
sys.path.insert(0, '../')
from sound_field_analysis import io, gen, process, plot

matFile = 'data/SOFiA_A3_struct.mat'

Nsft = 5        # Spatial Fourier Transform Order
Nrf = Nsft      # Radial Filter Order
amp_maxdB = 10  # Maximum modal amplification [dB]
ac = 2          # Array configuration: Rigid sphere
Nviz = Nsft     # Plot Order
krIndex = 80    # kr-bin (frequency) to plot

FFToversize = 2
startSample = [50, 760, 1460, 2170]  # Examplary values (Enable area plot)
K = len(startSample)
blockSize = 256

# Read in .mat struct
timeData = io.readMiroStruct(matFile)

vizMtx = np.empty((K, 181, 360))
for k in range(0, K):
    # Transform time domain data to frequency domain and generate kr-vector
    fftData, kr, f, _ = process.FFT(timeData, FFToversize=FFToversize, firstSample=startSample[k], lastSample=startSample[k] + blockSize)

    # Spatial Fourier transform
    Pnm = process.spatFT(Nsft, fftData, timeData.quadratureGrid)

    # Radial filters
    dn, _ = gen.radFilter(Nrf, kr, ac, amp_maxdB=amp_maxdB)

    # Generate data to visualize
    mtxData = plot.makeMTX(Pnm, dn, krIndex, Nviz=Nviz)
    vizMtx[k] = np.abs(mtxData)

plot.plot3Dgrid(2, 2, vizMtx, style='shape', normalize=True)

print("3D visualization opened in browser window, exiting.")
