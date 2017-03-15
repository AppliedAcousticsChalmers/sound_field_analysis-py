
# coding: utf-8

# # Read .sofa HRIR
# 
# SOFA (Spatially Oriented Format for Acoustics) files (https://www.sofaconventions.org) are HDF5 containers that hold data like Head-Related Impulse Responses (HRIRs), Binaural Room Impulse Responses (BRIRs) or Directional Impulse Responses (DRIRs, array impulse responses).
# 
# Additionally, they hold important meta information and are a great way to archive such recording data. Unfortunately, there currently is no Python API available (You may find at Matlab/Octave and C++ implementation here: https://www.sofaconventions.org/mediawiki/index.php/Software_and_APIs).
# 
# Luckily, sofa files can be read using the netCDF4 package, the extracted impulse responses can then be saved into a format that is (currently) easier to work with like numpy's .npy. Note that this is much less optimized for filesize!
# 
# ## Dependencies
# This example mainly relies on the **netCDF4** package to read in the sofa format. sound_field_analysis is only used for some format definitions .

# In[1]:

from netCDF4 import Dataset
import numpy as np
import sys
sys.path.insert(0, '../../sound_field_analysis-py')
from sound_field_analysis import io, plot, process


# ## Load .sofa file
# Plenty of sofa files are listed at https://www.sofaconventions.org/mediawiki/index.php/Files
# In this example, the "mit_kemar_large_pinna.sofa" was used.

# In[2]:

filename = 'mit_kemar_large_pinna'
path = 'sofa/'
sofa_file = Dataset(path + filename + '.sofa', 'r', format='NETCDF4')
sofa_file


# ## SOFA content
# If everything went correctly, you should see a description of the selection .sofa file above.
# 
# Generally, HRIR / BRIR sets will have R=2 receiver positions (left and right ear) at I=1 listener position (usually 0,0,0), with impulse responses of M source positions (with E=1 emitter position, usually 0,0,0) of length N.
# 
# 
# Other data such as array recordings can have several receiver positions for a single source position and the script below needs to be adjusted accordingly.
# 
# As an example, the ear distance is calculated as the difference between the y coordinates of the left and the right ear.

# In[3]:

print('SourcePosition: ' + str(sofa_file['SourcePosition']))
print('Data.IR: ' + str(sofa_file['Data.IR']))
print('Ear distance: ' + str( sofa_file['ReceiverPosition'][1, 1] - sofa_file['ReceiverPosition'][0, 1] ) + ' m')


# In[4]:

# extract IRs
HRIRs_l = np.squeeze(sofa_file['Data.IR'][:,0,:])
HRIRs_r = np.squeeze(sofa_file['Data.IR'][:,1,:])
Az, El, R = np.squeeze(np.hsplit(sofa_file['SourcePosition'][:], 3))
fs = sofa_file['Data.SamplingRate'][:][0]


# In[5]:

sofa_file.close()


# ## Save as npy file
# To now write the data into the handy binary .npy format, we could simply call np.save().
# 
# Specifically for the sound_field_analysis toolbox, we convert the angles into radiants (and elevation to colatitude), create an io.ArraySignal and finally save that.

# In[6]:

azimuth = Az / 180*np.pi
colatitude = np.pi / 2 - El / 180 * np.pi

hrir_full_l = io.ArraySignal(io.TimeSignal(HRIRs_l, fs), io.SphericalGrid(azimuth, colatitude, R))
hrir_full_r = io.ArraySignal(io.TimeSignal(HRIRs_r, fs), io.SphericalGrid(azimuth, colatitude, R))


# In[7]:

np.save(filename + '_L', hrir_full_l)
np.save(filename + '_R', hrir_full_r)

