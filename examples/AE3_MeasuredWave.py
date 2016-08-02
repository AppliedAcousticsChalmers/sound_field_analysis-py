# SOFiA example 3: A measured plane wave from AZ180°, EL90°
#                  in the anechoic chamber using a cardioid mic.

from sofia import io, gen, process, plot

matFile = 'data/SOFiA_A4_struct.mat'
Nsft = 5        # Spatial Fourier Transform Order
Nrf = Nsft      # Radial Filter Order
amp_maxdB = 10  # Maximum modal amplification [dB]
ac = 1          # Array configuration: Open sphere with cardioid mic
Nmtx = Nsft     # Plot Order
krIndex = 70    # kr-bin (frequency) to plot

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

# To export to png:
# >> from vispy import io
# >> img = canvas.render()
# >> io.write_png("img/AE2_grid.png", img)
