# SOFiA example 7: Impulse response reconstruction on a simulated
#                  sampled unity plane wave.
# Generate a full audio spectrum plane wave using S/W/G

import numpy as np
from sofia import gen, process
import matplotlib.pyplot as plt

pi = np.pi
Nsft = 5        # Spatial fourier transform order
Nrf = Nsft      # Radial filter order
Npdc = Nsft     # Decomposition order
OmegaL = np.array([[0, pi / 2],  # Looking directions for plane wave decomposition
                   [pi / 2, pi / 2]])
limit = 150     # Amplification limit of radial filters
r = 0.2         # Array radius
ac = 2          # Rigid Sphere Array
FS = 24000      # Sampling Frequency
NFFT = 1024     # FFT-Bins
AZ = 0          # Azimuth angle
EL = pi / 2  # Elevation angle

# Generate Lebedev grid of order 110
quadrature_grid, _ = gen.lebedev(110)

# Retrieve FFT of plane wave sampled by grid using the Sampled Wave Generator wrapper (SWG)
fftData, kr = gen.swg(r=r, gridData=quadrature_grid, ac=ac, FS=FS, NFFT=NFFT, AZ=AZ, EL=EL)

# Spatial Fourier Transform (STC)
Pnm = process.stc(Nsft, fftData, quadrature_grid)

# Generate modal radial filters (MF)
dn, _ = gen.mf(Nrf, kr, ac, a_max=limit)

# Plane wave decomposition (PDC) for supplied look directions
Y = process.pdc(Npdc, OmegaL, Pnm, dn)

# Reconstruct time domain signal (TDT)
impulseResponses = process.tdt(Y)

# Make IR causal (flip first & second half):
impulseResponses = np.hstack(np.array_split(impulseResponses, 2, axis = 1)[::-1])

# TODO: fix scaling (currently constant factor 19)
impulseResponses = impulseResponses / 19

# Plot results (Impulse Responses)
plt.subplot(1, 2, 1)
plt.plot(impulseResponses.T)
plt.title('Impulse response')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.axis([0, NFFT, -0.2, 1.2])
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

plt.show()
