# SOFiA example 1: Ideal unity plane wave simulation
# Additionally requires vispy, see http://vispy.org

import numpy as np
from sofia import gen, plot

pi = np.pi
N = 9        # Order
Nrf = N      # Radial filter order
Nviz = N     # Visualization order
krViz = 30   # Select kr bin for vizualisation

r = 0.5      # Array radius
ac = 2       # Array configuration, 2: Rigid sphere array
FS = 48000   # Sampling Frequency
NFFT = 128   # FFT-Bins
AZ = pi / 3  # Azimuth angle
EL = pi / 3  # Elevation angle

# Generate an ideal plane wave using W/G/C (Wave Generator Core)
Pnm, kr = gen.wgc(N, r, ac, FS, NFFT, AZ, EL)

# Make radial filters for the rigid sphere array
dn, _ = gen.mf(Nrf, kr, ac)

# Generate visualization data
vizMTX = plot.makeMTX(Pnm, dn, Nviz, krViz)

# Visualize
layout = {'title': 'Ideal unity plane wave',
          'height': 800,
          'width': 800}

plot.plot3D(vizMTX, style='shape', layout=layout)

print("3D visualization opened in browser window, exiting.")
