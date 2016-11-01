# SFA example 4: Level/Space Resolution
import sys
sys.path.insert(0, '../')
from sound_field_analysis import io, gen, process, plot

matFile = 'data/SOFiA_A2_struct.mat'
Nsft = 5        # Spatial Fourier Transform Order
Nrf = Nsft      # Radial Filter Order
amp_maxdB = 10  # Maximum modal amplification [dB]
ac = 2          # Array configuration: Rigid sphere
Nmtx = Nsft     # Plot Order
krIndex = 128   # kr-bin (frequency) to plot

# Read in .mat struct
timeData = io.readMiroStruct(matFile)

# Transform time domain data to frequency domain and generate kr-vector
fftData, kr, f, _ = process.FFT(timeData)

# Spatial Fourier transform
Pnm = process.spatFT(Nsft, fftData, timeData.quadratureGrid)

# Radial filters
dn, _ = gen.radFilter(Nrf, kr, ac, amp_maxdB=amp_maxdB)

# Plot
vizMTX = plot.makeMTX(Pnm, dn, Nviz=Nmtx, krIndex=krIndex)
fig = plot.plot3D(vizMTX, style='shape', colorize=False)

print("3D visualization opened in browser window, exiting.")
