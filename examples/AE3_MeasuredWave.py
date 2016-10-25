# SOFiA example 3
# Plotting a measured plane wave
# The recording was done in an anechoic chamber using a cardioid mic, the plane wave was approximated by a speaker at `AZ=180°`, `EL=90°`.

import sys
sys.path.insert(0, '../')
from sound_field_analysis import io, gen, process, plot

Nsft = 5        # Spatial Fourier Transform order
Nrf = Nsft      # Radial Filter order
amp_maxdB = 10  # Maximum modal amplification [dB]
ac = 1          # Array configuration: Open sphere with cardioid mic
Nmtx = Nsft     # Plot order

# Read in data
# `timeData` is a named tuple containing the following fields:
# - `impulseResponses` - Recorded impulse responses [Channels X Samples]
# - `FS` - Sampling frequency in [Hz]
# - `radius` - Array radius in [m]
# - `quadratureGrid` - Az, EL, W of the quadrature
# - `averageAirTemp` - Temperature in [C]
# - `centerIR` - Impulse response of center mic (if available), zero otherwise
# The provided `data/SOFiA_A4_struct.mat` file is a miro configuration that has been exported as a struct from Matlab.

timeData = io.readMiroStruct('data/SOFiA_A4_struct.mat')

# Frequency domain transform
# Transform impulse responses from the time domain to the frequency domain with the corresponding kr and frequency vectors.
fftData, kr, f, _ = process.fdt(timeData)

# Spatial Fourier transform
# Apply Spatial Fourier transform of the supplied order `Nsft` to the transfer functions, given their respective positions on the quadrature.
# `Pnm` then holds the `N^2 - 1` rows (corresponding to `n0m0`, `n1m-1`, `n1m0`, `n1m1`, `n2m-2`, ... ,`nNmN`) with the coefficients.
Pnm = process.stc(Nsft, fftData, timeData.quadratureGrid)

# Radial filters
# Generate radial filters based on the array configuration `ac`, order `Nrf`, soft-limited at the `amp_maxdB`.
dn, _ = gen.mf(Nrf, kr, ac, amp_maxdB=amp_maxdB)

# Plot
# To plot a specific frequency, you can use the utility function frqToKr to convert to the closest kr bin.
# `vizMTX` holds 180 x 360 complex values of 65160 plane waves for a 1 degree resultion plot.
fDraw = 1000
krDraw = plot.frqToKr(fDraw, f)
vizMTX = plot.makeMTX(Pnm, dn, Nviz=Nmtx, krIndex=krDraw)
plot.plot3D(vizMTX, style='shape')  # Other styles: 'sphere', 'flat'
