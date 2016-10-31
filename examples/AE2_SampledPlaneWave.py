# SOFiA example 2: Sampled unity plane wave simulation for different kr
# Generate a full audio spectrum plane wave using S/W/G
# Additionally requires vispy, see http://vispy.org

import numpy as np
from sofia import gen, process, plot

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
mtxDataLOW = plot.makeMTX(Pnm, dn, Nviz, krIDX[0])
mtxDataMID = plot.makeMTX(Pnm, dn, Nviz, krIDX[1])
mtxDataHIGH = plot.makeMTX(Pnm, dn, Nviz, krIDX[2])
mtxDataVHIGH = plot.makeMTX(Pnm, dn, Nviz, krIDX[3])

vizMtx = [np.abs(mtxDataLOW),
          np.abs(mtxDataMID),
          np.abs(mtxDataHIGH),
          np.abs(mtxDataVHIGH)]

plot.plotGrid(2, 2, vizMtx, style='shape', bgcolor='white', colorize=False, normalize=True)

input("3D visualization opened in new window.\nUse mouse to look around, scroll to zoom and shift + drag do move around.\nPress any key in the console to exit.")

# To export to png:
# >> from vispy import io
# >> img = canvas.render()
# >> io.write_png("img/AE2_grid.png", img)
