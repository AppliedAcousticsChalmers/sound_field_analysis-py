# SOFiA-py
SOFiA-py is a Python port of the [Sound Field Analysis Toolbox (SOFiA) toolbox](https://github.com/fietew/sofia-toolbox), originally by Benjamin Bernschütz[<sup>[1]</sup>](#references). The main goal of the SOFiA toolbox is to analyze, visualize and process soundfield data recorded by spherical microphone arrays. Furthermore, various types of testdata may be generated to evaluate the implemented functions.

The package is pure python and PEP8 compliant (except line-length). As the port is ongoing, no serious attempts at speed optimization have been made, please expect things to be slow for now.

## Requirements
The following external libraries are required to use most of the supplied functions:
- [NumPy](http://www.numpy.org)
- [SciPy](http://www.scipy.org)
- [matplotlib](http://matplotlib.org) (for 2D plotting)
- [vispy](http://vispy.org) (for 3D plotting)

## Overview
- `gen` contains functions to generate data:
    - `wgc`: Wave Generator
    - `mf`: Modal Radial Filter Generator
    - `lebedev`: Lebedev Quadrature
    - `swg`: Sampled Wave Generator
-  `process` contains processing functions:
    - `pdc`: Plane Wave Decomposition
    - `tdt`: Time Domain Reconstruction
    - `stc`: Fast Spatial Fourier Transform
    - `itc`: Fast Inverse Spatial Fourier Transform
- `plot` contains function for visualization:
    -  `makeMTX`: Generate 3D-matrix data
    -  `visualize3D`: Draw matrix data in 3D
-  `sph` contains helper functions for dealing with spherical harmonics:
    - `sph_harm`: More robust spherical harmonic coefficients
    -  `spbessel` / `dspbessel`: Spherical Bessel and derivative
    - `spneumann` / `dspneumann`: Spherical Neumann (Bessel 2nd kind) and derivative
    - `sphankel` / `dsphankel`: Spherical Hankel and derivative
    - `cart2sph` / `sph2cart`: Convert cartesion to spherical coordinates and vice versa


## Working examples
## AE1: Ideal Plane Wave
Ideal unity plane wave simulation and 3D plot.
#### Colorized 3D scatter:
![AE1_IdealPlaneWave colored scatter](img/AE1_cscatter.png?raw=true "AE1_IdealPlaneWave colored scatter")
#### Shape-based visualization:
![AE1_IdealPlaneWave shape](img/AE1_shape.png?raw=true "AE1_IdealPlaneWave shape")


### AE6: Impulse response of ideal plane wave
Impulse Response reconstruction on a simulated ideal unity plane wave

![AE6_IdealPlaneWave_ImpResp result](img/AE6_IdealPlaneWave_ImpResp.png?raw=true "AE6_IdealPlaneWave_ImpResp result")

### AE7: Impulse response of sampled plane wave
Impulse response reconstruction on a simulated sampled unity plane wave

![AE7_SampledPlaneWave_ImpResp result](img/AE7_SampledPlaneWave_ImpResp.png?raw=true "AE7_SampledPlaneWave_ImpResp result")

## Contact
SOFiA-py is under development by Christoph Hohnerlein (`firstname.lastname[at]qu.tu-berlin.de`) as part of the [Artificial Reverberation for Sound Field Synthesis](https://www.qu.tu-berlin.de/menue/forschung/laufende_projekte/artificial_reverberation_for_sound_field_synthesis_dfg/) project at the [Quality and Usability Lab](https://www.qu.tu-berlin.de) of the TU Berlin.

## References
SOFiA-py is based on the Matlab/C++ toolbox [SOFiA](https://github.com/fietew/sofia-toolbox) by Benjamin Bernschütz. For more information you may refer to the original publication:
[1] Bernschütz, B., Pörschmann, C., Spors, S., and Weinzierl, S. (2011). SOFiA Sound Field Analysis Toolbox. Proceedings of the ICSA International Conference on Spatial Audio

The Lebedev grid generation was adapted from an implementaion by [Richard P. Muller](https://github.com/gabrielelanaro/pyquante/blob/master/Data/lebedev_write.py).
