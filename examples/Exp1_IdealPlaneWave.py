
# coding: utf-8

# # sound_field_analysis example 1 - Ideal plane wave simulation

# In the following notebook, the sound field of a plane wave impinging on a spherical microphone array is generated and visualized, both as an idealized plane wave as well as a "real" wave sampled by a microphone array.
# 
# Then, the impulse responses at two different looking direction are extraced using plane wave decomposition and compared. I can be shown that simulating the microphone introduces strong spatial aliasing in higher frequencies.
# 
# ### Setup

# In[1]:

import sys
sys.path.insert(0, '../')
from sound_field_analysis import gen, plot, utils, process

from plotly.offline import init_notebook_mode
init_notebook_mode()


# ## Configuration
# First, the overall limit for the order in spherical domain is set. A lower order will allow for faster calculation time at a loss of fidelity in the radial spatial domain.
# 
# Then, a virtual microphone array that is to be simulated is configured by using three parameters:
# - radius of the array
# - sphere configuration (open, rigid or dual)
# - transducers type (omni or cardioid)
# 
# For the spatial sampling, a quadrature grid is be generated using the gen.lebedev() module, which returnes a object holding the positions and weights of a specific Lebedev configuration.
# 
# Lastly, the wavetype ('plane') and the direction of the wave are defined. NFFT denotes the number of the FFT bins and fs the sampling frequency used for the simulation.

# In[2]:

order = 10

array_configuration = [0.5, 'rigid', 'cardioid']
quadrature_grid = gen.lebedev(max_order=order)

wavetype = 'plane'
azimuth = utils.deg2rad(0)
colatitude = utils.deg2rad(90)

NFFT = 128
fs = 48000


# ## Simulated array data and radial filters
# `gen.ideal_wave()` returns the spherical fourier coefficiens of a ideal plane wave as captured by an ideal array of the provided configuration.
# `gen.sampled_wave()` returns the spherical fourier coefficiens as captured by an simulated real array.
# 
# `gen.radial_filter()` returns the radial filters for the current array configuration, which is identical for both cases.

# In[3]:

ideal_array_data = gen.ideal_wave(order, fs, azimuth, colatitude, array_configuration, NFFT=NFFT)
simulated_array_data = gen.sampled_wave(order, fs, NFFT, array_configuration, quadrature_grid, azimuth, colatitude)

radial_filter = gen.radial_filter_fullspec(order, NFFT, fs, array_configuration)


# ## Visualization
# 
# To visualize the plane waves, the spherical fourier coefficients and the generated radial filters are passed to plot.makeMTX() along with a kr bin. The function returns the sound pressure at a resolution of 1 degree, which then can be visualized using plot.plot3D(). Several styles ('shape', 'sphere', 'flat') are available.

# In[4]:

kr_IDX = 64
vizMTX_ideal = plot.makeMTX(ideal_array_data, radial_filter, kr_IDX)
vizMTX_simulated = plot.makeMTX(simulated_array_data, radial_filter, kr_IDX)


# In[5]:

fig = plot.plot3Dgrid(rows=1, cols=2, viz_data=[vizMTX_ideal, vizMTX_simulated], style='shape')


# ## Extracting impulse responses
# In order to extract the impulse response, a plane wave decomposition (PWD) is performed using `process.plane_wave_decomp()`. The output in the frequency domain can then simply be transformed to the time domain using `process.iFFT()`. We can calculate several looking directions at once by supplying a vector of azimuth / colatitude pairs.
# 
# 
# For the ideal plane wave, the impulse response will be almost dirac impulse-like in the direction of the plane wave (azimuth = 0°, colatitude = 90°) and almost zero at a 90° angle (like azimuth = 90°, colatitude = 90°). The spatial resampling exhibit strong aliasing above a certain cutoff frequency (see spectrum), which results in temporal smearing in the time domain.

# In[6]:

looking_directions = [[0, utils.deg2rad(90)],
                      [utils.deg2rad(90), utils.deg2rad(90)]]
Y_ideal = process.plane_wave_decomp(order, looking_directions, ideal_array_data, radial_filter)
Y_sampled = process.plane_wave_decomp(order, looking_directions, simulated_array_data, radial_filter)

impulseResponses = process.iFFT(utils.stack(Y_ideal, Y_sampled))
spectrum, f = process.FFT([impulseResponses, fs])


# ## Plot time signal and frequency response

# In[7]:

plot.plot2D(impulseResponses, viz_type='time', fs=fs)
plot.plot2D(utils.db(spectrum), viz_type='logFFT', fs=fs)

