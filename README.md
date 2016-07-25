# SOFiA-py
SOFiA-py is a Python port of the [Sound Field Analysis Toolbox (SOFiA) toolbox](http://audiogroup.web.th-koeln.de/SOFiA_wiki/WELCOME.html), originally by Benjamin Bernschütz[<sup>[1]</sup>](#references). The main goal of the SOFiA toolbox is to analyze, visualize and process soundfield data recorded by spherical microphone arrays. Furthermore, various types of testdata may be generated to evaluate the implemented functions.

The package is pure python and PEP8 compliant (except line-length). As the port is ongoing, no serious attempts at speed optimization have been made, please expect things to be slow for now.

## Requirements
The following external libraries are required to use most of the supplied functions:
- [NumPy](http://www.numpy.org)
- [SciPy](http://www.scipy.org)
- [matplotlib](http://matplotlib.org) (for 2D plotting)
- [vispy](http://vispy.org) (for 3D plotting)

## Documentation
Please find the full documentation at https://qulab.github.io/sofia-py/!

## Working examples
## AE1: Ideal Plane Wave
Ideal unity plane wave simulation and 3D plot.
#### Colorized 3D scatter:
![AE1_IdealPlaneWave colored scatter](img/AE1_cscatter.png?raw=true "AE1_IdealPlaneWave colored scatter")
#### Shape-based visualization:
![AE1_IdealPlaneWave shape](img/AE1_shape.png?raw=true "AE1_IdealPlaneWave shape")

## AE2: Sampled Plane Wave
Sampled unity plane wave simulation for different kr

![AE2_SampledPlaneWave](img/AE2_grid.png?raw=true "AE2_SampledPlaneWave")

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

[1] [Bernschütz, B., Pörschmann, C., Spors, S., and Weinzierl, S. (2011). SOFiA Sound Field Analysis Toolbox. Proceedings of the ICSA International Conference on Spatial Audio](http://spatialaudio.net/sofia-sound-field-analysis-toolbox-2/)

The Lebedev grid generation was adapted from an implementaion by [Richard P. Muller](https://github.com/gabrielelanaro/pyquante/blob/master/Data/lebedev_write.py).
