# SFA example 1: Ideal unity plane wave simulation

import numpy as np
import sys
sys.path.insert(0, '../')
from sound_field_analysis import gen, plot

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

# Generate an ideal plane wave
Pnm, kr = gen.idealWave(N, r, ac, FS, NFFT, AZ, EL)

# Generate radial filters for the rigid sphere array
dn, _ = gen.radFilter(Nrf, kr, ac)

# Generate visualization data
vizMTX = plot.makeMTX(Pnm, dn, krIndex=krViz, Nviz=Nviz)

# Visualize
layout = {'title': 'Ideal unity plane wave',
          'height': 800,
          'width': 800}

plot.plot3D(vizMTX, style='shape', layout=layout)

print("3D visualization opened in browser window, exiting.")
