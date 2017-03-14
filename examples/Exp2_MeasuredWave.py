
# coding: utf-8

# # sound_field_analysis example 2 - Visualizing measured data
# The recording was done in an anechoic chamber using a cardioid mic, the plane wave is approximated by a speaker at `Azimuth=180°`, `Colatitude=90°` at a distance of 3m.

# In[1]:

from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

import sys
sys.path.insert(0, '../')
from sound_field_analysis import io, gen, process, plot, utils


# ## Read in data
# `array_data` is a tuple as defined in io.ArraySignal, which simply bundles three more basic data types:
# 
# - `.signal` - is a io.TimeSignal containing:
#     - `.signal.signal`
#     - `.signal.fs`
# - `.grid`   - is a io.SphericalGrid containing:
#     - `.grid.azimuth`
#     - `.grid.colatitude`
#     - `.grid.radius`
#     - `.grid.weight`
# - `.configuration` - is a io.ArrayConfiguration containing:
#     - `.configuration.array_radius`
#     - `.configuration.array_type`
#     - `.configuration.transducer_type`
#     - `.configuration.scatter_radius`
# - `.temperature`   - Temperature in [C]
# 
# The function io.read_miro_struct handles this, only the transducer type has to be set to `cardioid` manually (default: `omni`), as it cannot be extraced from the miro file.

# In[2]:

array_data = io.read_miro_struct('data/SOFiA_A4_struct.mat', transducer_type='cardioid')


# ## Time > Frequency > Spatial domain
# The impulse responses are now read into `array_data.signal`. To bring the time-domain signals to the spatial fourier domain, first the normal Fourier transform (`process.FFT`) has to be applied.
# 
# ## Spatial domain & radial filters
# Now the data is ready for the spatial fourier transform (`process.spatFT`); the resulting coefficients hold `order^2 - 1` rows. The radial filters are also calculated, based on the array configuration, the sampling frequency and the number of FFT bins (NFFT) used.
# 
# Both the spatial transform and then radial filters are also order-limited to the same order.

# In[ ]:

fftData, f = process.FFT(array_data.signal)
NFFT = fftData.shape[1]*2-1

order = 5
spatial_coefficients = process.spatFT(fftData, array_data.grid, order_max=order)
radial_filter = gen.radial_filter_fullspec(order, NFFT=NFFT, fs=array_data.signal.fs, array_configuration=array_data.configuration)


# ## Plot
# To visualized the recorded data, the spherical fourier coefficients and the radial filters are passed to plot.makeMTX() along with a specific kr bin. The function returns the sound pressure at a resolution of 1 degree, which then can be visualized using plot.plot3D(). Several styles ('shape', 'sphere', 'flat') exist.
# 
# To plot a specific frequency, we use utils.nearest_to_value_IDX() find the index closest to the desired frequency.

# In[ ]:

vizMTX1 = plot.makeMTX(spatial_coefficients, radial_filter, kr_IDX=utils.nearest_to_value_IDX(f, 100))
vizMTX2 = plot.makeMTX(spatial_coefficients, radial_filter, kr_IDX=utils.nearest_to_value_IDX(f, 1000))
vizMTX3 = plot.makeMTX(spatial_coefficients, radial_filter, kr_IDX=utils.nearest_to_value_IDX(f, 5000))
vizMTX4 = plot.makeMTX(spatial_coefficients, radial_filter, kr_IDX=utils.nearest_to_value_IDX(f, 10000))

plot.plot3Dgrid(2, 2, [vizMTX1, vizMTX2, vizMTX3, vizMTX4], 'shape')

