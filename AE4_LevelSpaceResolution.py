# SOFiA example 4: Level/Space Resolution

from sofia import io, gen, process, plot

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
fftData, kr, f, _ = process.fdt(timeData)

# Spatial Fourier transform
Pnm = process.stc(Nsft, fftData, timeData.quadratureGrid)

# Radial filters
dn, _ = gen.mf(Nrf, kr, ac, amp_maxdB=amp_maxdB)
dn = process.rfi(dn)  # TODO: Radial Filter improvement

# Plot
vizMTX = plot.makeMTX(Pnm, dn, Nviz=Nmtx, krIndex=krIndex)
canvas = plot.visualize3D(vizMTX, style='shape', colorize=False)

input("3D visualization opened in new window.\nUse mouse to look around, scroll to zoom and shift + drag do move around.\nPress any key in the console to exit.")
