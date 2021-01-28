Sound Field Analysis toolbox for Python
=======================================
.. image:: https://api.travis-ci.org/QULab/sound_field_analysis-py.svg
.. image:: https://ci.appveyor.com/api/projects/status/u0koxo5vcitmbghc?svg=true

The *sound\_field\_analysis* toolbox (short: *sfa*) is a Python port of the `Sound Field Analysis Toolbox (SOFiA) toolbox`_, originally by Benjamin Bernschütz `[1]`_. The main goal of the *sfa* toolbox is to analyze, visualize and process sound field data recorded by spherical microphone arrays. Furthermore, various types of test-data may be generated to evaluate the implemented functions. It is an essential building block of `ReTiSAR`_, an implementation  of real time binaural rendering of spherical microphone array data.


Requirements
------------

We use Python 3.9 for development. Chances are that earlier version will work too but this is currently untested.

The following external libraries are required:

-  `NumPy`_
-  `SciPy`_
-  `Pysofaconventions`_
-  `Jupyter`_ (for running *Notebooks* locally)
-  `Plotly`_ (for plotting)


Installation
------------

For performance and convenience reasons we highly recommend to use `Conda`_ (miniconda for simplicity) to manage your Python installation. Once installed, you can use the following steps to receive and use *sfa*, depending on your use case:

A.  From PyPI:

    Install into an existing environment (without example *Jupyter Notebooks*):
    ``pip install sound_field_analysis``

B.  By cloning (or downloading) the repository and setting up a new environment:

    ``git clone https://github.com/AppliedAcousticsChalmers/sound_field_analysis-py.git``

    ``cd sound_field_analysis-py/``

    Create a new `Conda`_ environment from the specified dependencies:
    ``conda env create --file environment.yml --force``

    Activate the environment:
    ``source activate sfa``

    **Optional:** Install additional dependencies for development purposes (locally run *Jupyter Notebooks* with example, run tests, generate documentation):
    ``conda env update --file environment_dev.yml``

.. C.  From `conda-forge`_ channel: **[outdated]**

    Install into an existing environment:
    ``conda install -c conda-forge sound_field_analysis``


Documentation
-------------

Find the full documentation at https://appliedacousticschalmers.github.io/sound_field_analysis-py/.


Examples
--------

The following examples are available as Jupyter notebooks, either statically on `GitHub`_ or interactively on `nbviewer`_. You can of course also simply download the examples and run them locally!


Exp1: Ideal plane wave
^^^^^^^^^^^^^^^^^^^^^^

Ideal unity plane wave simulation and 3D plot.

`View interactively on nbviewer <https://nbviewer.jupyter.org/github/AppliedAcousticsChalmers/sound_field_analysis-py/blob/master/examples/Exp1_IdealPlaneWave.ipynb>`__

|AE1_img|_

.. |AE1_img| image:: examples/img/AE1_shape.png?raw=true
.. _AE1_img: https://nbviewer.jupyter.org/github/AppliedAcousticsChalmers/sound_field_analysis-py/blob/master/examples/Exp1_IdealPlaneWave.ipynb


Exp2: Measured plane wave
^^^^^^^^^^^^^^^^^^^^^^^^^

A measured plane wave from AZ=180°, EL=90° in the anechoic chamber using a cardioid mic.

`View interactively on nbviewer <https://nbviewer.jupyter.org/github/AppliedAcousticsChalmers/sound_field_analysis-py/blob/master/examples/Exp2_MeasuredWave.ipynb>`__

|AE2_img|_

.. |AE2_img| image:: examples/img/AE2_shape.png?raw=true
.. _AE2_img: https://nbviewer.jupyter.org/github/AppliedAcousticsChalmers/sound_field_analysis-py/blob/master/examples/Exp2_MeasuredWave.ipynb


Exp4: Binaural rendering
^^^^^^^^^^^^^^^^^^^^^^^^

Render a spherical microphone array impulse response measurement binaurally. The example shows examples for loading miro or `SOFA`_ files.

`View interactively on nbviewer <https://nbviewer.jupyter.org/github/AppliedAcousticsChalmers/sound_field_analysis-py/blob/master/examples/Exp4_BinauralRendering.ipynb>`__


Version history
---------------

*v2021.1.12*
    * Update MIRO struct loading (quadrature weights are now optional)
    * Fix to prevent Python 3.8 syntax warnings
    * Improve Exp4 (general code structure and utilizing Spherical Head Filter and Spherical Harmonics Tapering)

*v2020.1.30*
    * Update README and PyPI package

*v2019.11.6*
    * Update internal documentation and string formatting

*v2019.8.15*
    * Change version number scheme to CalVer
    * Improve Exp4
    * Update `read_SOFA_file()`
    * Update 2D plotting functions
    * Improve `write_SSR_IRs()`
    * Improve `Conda`_ environment setup for `Jupyter`_ Notebooks
    * Update `miro_to_struct()`

*2019-07-30 v0.9*
    * Implement `SOFA`_ import
    * Update Exp4 to contain `SOFA`_ import
    * Delete obsolete Exp3
    * Add named tuple `HRIRSignal`
    * Implement `cart2sph()` and `sph2cart()` utility functions
    * Add `Conda`_ environment file for convenient installation of required packages

*2019-07-11 v0.8*
    * Implement Spherical Harmonics coefficients tapering
    * Update Spherical Head Filter to consider tapering

*2019-06-17 v0.7*
    * Implement Bandwidth Extension for Microphone Arrays (BEMA)
    * Edit `read_miro_struct()`, named tuple `ArraySignal` and `miro_to_struct.m` to load center measurements

*2019-06-11 v0.6*
    * Implement Radial Filter Improvement from `Sound Field Analysis Toolbox (SOFiA) toolbox`_

*2019-05-23 v0.5*
    * Implement Spherical Head Filter
    * Implement Spherical Fourier Transform using pseudo-inverse
    * Extract real time capable spatial Fourier transform
    * Extract reversed m index function (Update Exp4)


References
----------

The *sound_field_analysis* toolbox is based on the Matlab/C++ `Sound Field Analysis Toolbox (SOFiA) toolbox`_ by
Benjamin Bernschütz. For more information you may refer to the original publication:

[1] `Bernschütz, B., Pörschmann, C., Spors, S., and Weinzierl, S. (2011). SOFiA Sound Field Analysis Toolbox.
Proceedings of the ICSA International Conference on Spatial Audio <http://spatialaudio
.net/sofia-sound-field-analysis-toolbox-2/>`_

The Lebedev grid generation was adapted from an implementation by `Richard P. Muller <https://github
.com/gabrielelanaro/pyquante/blob/master/Data/lebedev_write.py>`_.

.. _Sound Field Analysis Toolbox (SOFiA) toolbox: http://audiogroup.web.th-koeln.de/SOFiA_wiki/WELCOME.html
.. _ReTiSAR: https://github.com/AppliedAcousticsChalmers/ReTiSAR
.. _[1]: #references
.. _NumPy: http://www.numpy.org
.. _SciPy: http://www.scipy.org
.. _Pysofaconventions: https://github.com/andresperezlopez/pysofaconventions
.. _Jupyter: https://jupyter.org/
.. _Plotly: https://plot.ly/python/
.. _Conda: https://www.continuum.io/downloads
.. _conda-forge: https://conda-forge.github.io
.. _GitHub: examples/
.. _nbviewer: http://nbviewer.jupyter.org/github/AppliedAcousticsChalmers/sound_field_analysis-py/tree/master/examples/
.. _SOFA: https://www.sofaconventions.org/mediawiki/index.php/SOFA_(Spatially_Oriented_Format_for_Acoustics)
