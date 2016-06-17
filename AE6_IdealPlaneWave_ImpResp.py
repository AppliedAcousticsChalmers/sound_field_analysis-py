# SOFiA example 6: Impulse response reconstruction on a simulated
#                  ideal unity plane wave.
# SOFiA Version  : R13-0306
# Generate a full audio spectrum plane wave using I/W/G

import numpy as np
from sofia import gen, process
import matplotlib.pyplot as plt

Nwave = 5    # Wave order
r = 1        # Array radius
ac = 2       # Array configuration: 2-Rigid Sphere
FS = 48000   # Sampling Frequency
NFFT = 1024  # FFT-Bins
AZ = 0       # Azimuth angle
EL = np.pi / 2  # Elevation Angle

Pnm, kr = gen.wgc(Nwave, r, ac, FS, NFFT, AZ, EL)

# Make radial filters
Nrf = Nwave                  # radial filter order
dn, beam = gen.mf(Nrf, kr, ac)

# Running a plane wave decomposition for different look directions

Npdc = Nwave                # Decomposition order
OmegaL = np.array([[0, np.pi / 2], [np.pi / 2, np.pi / 2]])  # Looking towards the wave and to one side

Y = process.pdc(Npdc, OmegaL, Pnm, dn)

# Reconstruct impulse responses
impulseResponses = process.tdt(Y)

#%% Plot results (Impulse Responses)
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
