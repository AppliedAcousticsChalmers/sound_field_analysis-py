# SOFiA example 6: Impulse response reconstruction on a simulated
#                  ideal unity plane wave.
# Generate a full audio spectrum plane wave using I/W/G

import numpy as np
from sofia import gen, process
import matplotlib.pyplot as plt

pi = np.pi
Nwave = 5     # Wave order
Nrf = Nwave   # Radial filter order
Npdc = Nwave  # Decomposition order
r = 1         # Array radius
ac = 2        # Array configuration: 2-Rigid Sphere
FS = 48000    # Sampling Frequency
NFFT = 1024   # FFT-Bins
AZ = 0        # Azimuth angle
EL = pi / 2   # Elevation Angle
OmegaL = np.array([[0, pi / 2],  # Looking directions for plane wave decomposition
                   [pi / 2, pi / 2]])

Pnm, kr = gen.wgc(Nwave, r, ac, FS, NFFT, AZ, EL)

# Generate modal radial filters (MF)
dn, _ = gen.mf(Nrf, kr, ac)

# Running a plane wave decomposition (PDC) for supplied look directions
Y = process.pdc(Npdc, OmegaL, Pnm, dn)

# Reconstruct time domain signal (TDT)
impulseResponses = process.tdt(Y)

# Plot results (Impulse Responses)
print('Plotting...')

plt.subplot(1, 2, 1)
plt.plot(impulseResponses.T)
plt.title('Impulse response')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.axis([-200, NFFT, -0.2, 1.2])
plt.grid()


# Plot results (Spectra)
spectrum = np.abs(np.fft.fft(impulseResponses).T)
fscale = np.fft.fftfreq(impulseResponses.shape[1], 1 / FS)

plt.subplot(1, 2, 2)
plt.semilogx(fscale, 20 * np.log10(spectrum))
plt.title('Spectrum')
plt.xlabel('Frequency in Hz')
plt.ylabel('Magnitude')
plt.axis([50, 20000, -60, 30])
plt.grid()

plt.show()
