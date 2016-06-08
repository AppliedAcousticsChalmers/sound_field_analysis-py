# SOFiA example 7: Impulse response reconstruction on a simulated
#                  sampled unity plane wave.
# SOFiA Version  : R13-0306

import numpy as np
from sofia import gen, process
import matplotlib.pyplot as plt

# Generate a full audio spectrum plane wave using S/W/G
r = 0.2         # Array radius
ac = 2          # Rigid Sphere Array
FS = 24000      # Sampling Frequency
NFFT = 1024     # FFT-Bins
AZ = 0          # Azimuth angle
EL = np.pi / 2  # Elevation angle

quadrature_grid = gen.lebedev(110, 0)  # EXAMPLE GRID LEB110, No Plot

fftData, kr = gen.swg(r, quadrature_grid, ac, FS, NFFT, AZ, EL)

# Spatial Fourier Transform
Nsft = 5
Pnm = process.stc(Nsft, fftData, quadrature_grid)

# Make radial filters
Nrf = Nsft      # radial filter order
limit = 150     # Amplification Limit (Keep the result numerical stable at low frequencies)

dn = gen.mf(Nrf, kr, ac, a_max=limit)

# Plane wave decomposition for different look directions
Npdc = Nsft     # Decomposition order
OmegaL = [0, np.pi / 2, np.pi / 2, np.pi / 2]  # Looking towards the wave and to one side

Y = process.pdc(Npdc, OmegaL, Pnm, dn)

# Reconstruct impulse responses
impulseResponses = process.tdt(Y, 0)

# Make IR causal:
# impulseResponses = [impulseResponses(:, end / 2 + 1:end), impulseResponses(:, 1:end / 2)]

# Plot results (Impulse Responses)
plt.subplot(1, 2, 1)
plt.plot(impulseResponses.T)
plt.title('Impulse response')
plt.xlabel('Samples')
plt.ylabel('Amplitude')

plt.axis([-100, NFFT, -0.2, 1.2])
plt.grid()

# Plot results (Spectra)

spectrum = np.abs(np.fft.fft(impulseResponses).T)
fscale = np.fft.fftfreq(impulseResponses.shape[1], 1 / FS)

plt.subplot(1, 2, 2)
plt.semilogx(fscale, 20 * np.log10(spectrum))
plt.title('Spectrum')
plt.xlabel('Frequency in Hz')
plt.ylabel('Magnitude')
plt.axis([50, FS / 2, -60, 30])
plt.grid()
