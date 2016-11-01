# SFA example 2: Sampled unity plane wave simulation for different kr
# Generate a full audio spectrum plane wave using sampledWave

import numpy as np
import sys
sys.path.insert(0, '../')
from sound_field_analysis import gen, process, plot

pi = np.pi
r = 0.1      # Array radius
ac = 0       # Rigid Sphere
FS = 48000   # Sampling Frequency
NFFT = 128   # FFT-Bin
AZ = 0       # Azimuth angle
EL = pi / 2  # Elevation angle
Nsft = 5     # Transform order
Nrf = Nsft   # Radial filter order
Nviz = Nsft  # Visualization order
krIDX = [15, 23, 29, 39]  # kr-bin for subfigures

quadrature_grid, _ = gen.lebedev(110)

fftData, kr = gen.sampledWave(r=r, gridData=quadrature_grid, ac=ac, FS=FS, NFFT=NFFT, AZ=AZ, EL=EL)

# Spatial Fourier Transform
Pnm = process.spatFT(Nsft, fftData, quadrature_grid)

# Make radial filters
dn, _ = gen.radFilter(Nrf, kr, ac)

# Generate data to visualize
mtxDataLOW = plot.makeMTX(Pnm, dn, krIDX[0], Nviz=Nviz)
mtxDataMID = plot.makeMTX(Pnm, dn, krIDX[1], Nviz=Nviz)
mtxDataHIGH = plot.makeMTX(Pnm, dn, krIDX[2], Nviz=Nviz)
mtxDataVHIGH = plot.makeMTX(Pnm, dn, krIDX[3], Nviz=Nviz)

vizMtx = [np.abs(mtxDataLOW),
          np.abs(mtxDataMID),
          np.abs(mtxDataHIGH),
          np.abs(mtxDataVHIGH)]

plot.plot3Dgrid(2, 2, vizMtx, style='shape', normalize=True)

print("3D visualization opened in browser window, exiting.")
