# SOFiA-toolbox-py
Python port of the [SOFiA toolbox](https://github.com/fietew/sofia-toolbox) by Benjamin Bernschütz. This is currently work in progress and will most probably not include all original functions. For more information you may refer to the original publication: [Bernschütz, B., Pörschmann, C., Spors, S., and Weinzierl, S. (2011). SOFiA Sound Field Analysis Toolbox. Proceedings of the ICSA International Conference on Spatial Audio](http://spatialaudio.net/sofia-sound-field-analysis-toolbox-2/)

## Requirements
The following external libraries are required to use most of the supplied functions:
- [NumPy](http://www.numpy.org)
- [SciPy](http://www.scipy.org)
- [matplotlib](http://matplotlib.org) (for 2D plotting)
- [vispy](http://vispy.org) (for 3D plotting)

Lebedev grid generation adapted from [Richard P. Muller](https://github.com/gabrielelanaro/pyquante/blob/master/Data/lebedev_write.py)

## Working examples
##AE1: Ideal Plane Wave
Ideal unity plane wave simulation and 3D plot

![AE1_IdealPlaneWave colored scatter](img/AE1_cscatter.png?raw=true "AE1_IdealPlaneWave colored scatter")
![AE1_IdealPlaneWave shape](img/AE1_shape.png?raw=true "AE1_IdealPlaneWave shape")

### AE6: Impulse response of ideal plane wave
Impulse Response reconstruction on a simulated ideal unity plane wave

![AE6_IdealPlaneWave_ImpResp result](img/AE6_IdealPlaneWave_ImpResp.png?raw=true "AE6_IdealPlaneWave_ImpResp result")

### AE7: Impulse response of sampled plane wave
Impulse response reconstruction on a simulated sampled unity plane wave

![AE7_SampledPlaneWave_ImpResp result](img/AE7_SampledPlaneWave_ImpResp.png?raw=true "AE7_SampledPlaneWave_ImpResp result")

## Contact
This repository is currently maintained by Christoph Hohnerlein (firstname.lastname[at]qu.tu-berlin.de).
