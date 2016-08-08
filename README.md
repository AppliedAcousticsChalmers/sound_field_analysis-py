# Sound Field Analysis toolbox for Python
The *sound_field_analysis* toolbox (short: *sfa*) is a Python port of the [Sound Field Analysis Toolbox (SOFiA) toolbox](http://audiogroup.web.th-koeln.de/SOFiA_wiki/WELCOME.html), originally by Benjamin Bernschütz[<sup>[1]</sup>](#references). The main goal of the *sfa* toolbox is to analyze, visualize and process sound field data recorded by spherical microphone arrays. Furthermore, various types of testdata may be generated to evaluate the implemented functions.

The package is pure python and PEP8 compliant (except line-length). Please expect things to be slow for now and for the API to break, as the development is still very much ongoing.

## Requirements
We use Python 3.5 for development. Chances are that earlier version will work too but this is untested.

The following external libraries are required:
- [NumPy](http://www.numpy.org)
- [SciPy](http://www.scipy.org)
- [Plotly](https://plot.ly/python/) (for plotting)

## Installation
We highly recommend the [Anaconda](https://www.continuum.io/downloads) python environment. Once installed, you can use the following steps to create a new environment with the *sfa* toolbox.

1. Add the [conda-forge](https://conda-forge.github.io) channel:  
  `conda config --add channels conda-forge`
2. Create a new environment:
  `create --name sfa numpy scipy plotly sound_field_analysis`
3. Activate this environment:
  `source activate sfa`
 
Alternatively, you can simply *sfa* install it through pip (`pip install sound_field_analysis`).

## Documentation
Please find the full documentation at https://qulab.github.io/sound_field_analysis-py/!

## Examples
The following examples are available as Jupyter notebooks, either statically on [github](examples/) or interactivally on [nbviewer](http://nbviewer.jupyter.org/github/QULab/sound_field_analysis-py/tree/master/examples/). You can of course also simply download the examples and run them locally!

### AE1: Ideal plane wave
Ideal unity plane wave simulation and 3D plot.
[View on nbviewer
![AE1_IdealPlaneWave shape](examples/img/AE1_shape.png?raw=true "AE1_IdealPlaneWave shape")](http://nbviewer.jupyter.org/github/QULab/sound_field_analysis-py/blob/master/examples/AE1_IdealPlaneWave.ipynb)

### AE3: Measured plane wave
A measured plane wave from AZ=180°, EL=90° in the anechoic chamber using a cardioid mic.

[View on nbviewer
![AE3_MeasuredPlaneWave shape](examples/img/AE3_shape.png?raw=true "AE3_MeasuredPlaneWave shape")](http://nbviewer.jupyter.org/github/QULab/sound_field_analysis-py/blob/master/examples/AE3_MeasuredWave.ipynb)

#### AE6: Impulse response of ideal plane wave
Impulse Response reconstruction on a simulated ideal unity plane wave
[View on nbviewer
![AE6_IdealPlaneWave_ImpResp](examples/img/AE6_IdealPlaneWave_ImpResp.png?raw=true "AE6_IdealPlaneWave_ImpResp")](http://nbviewer.jupyter.org/github/QULab/sound_field_analysis-py/blob/master/examples/AE6_IdealPlaneWave_ImpResp.ipynb)


#### AE7: Impulse response of sampled plane wave
Impulse response reconstruction on a simulated sampled unity plane wave

[View on nbviewer
![AE7_SampledPlaneWave_ImpResp](examples/img/AE7_SampledPlaneWave_ImpResp.png?raw=true "AE7_SampledPlaneWave_ImpResp")](http://nbviewer.jupyter.org/github/QULab/sound_field_analysis-py/blob/master/examples/AE7_SampledPlaneWave_ImpResp.ipynb)

## Contact
SOFiA-py is under development by Christoph Hohnerlein (`christoph.hohnerlein[at]qu.tu-berlin.de`) as part of the [Artificial Reverberation for Sound Field Synthesis](https://www.qu.tu-berlin.de/menue/forschung/laufende_projekte/artificial_reverberation_for_sound_field_synthesis_dfg/) project at the [Quality and Usability Lab](https://www.qu.tu-berlin.de) of the TU Berlin.

## References
SOFiA-py is based on the Matlab/C++ toolbox [SOFiA](https://github.com/fietew/sofia-toolbox) by Benjamin Bernschütz. For more information you may refer to the original publication:

[1] [Bernschütz, B., Pörschmann, C., Spors, S., and Weinzierl, S. (2011). SOFiA Sound Field Analysis Toolbox. Proceedings of the ICSA International Conference on Spatial Audio](http://spatialaudio.net/sofia-sound-field-analysis-toolbox-2/)

The Lebedev grid generation was adapted from an implementaion by [Richard P. Muller](https://github.com/gabrielelanaro/pyquante/blob/master/Data/lebedev_write.py).
