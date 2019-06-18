Sound Field Analysis toolbox for Python
=======================================
.. image:: https://api.travis-ci.org/QULab/sound_field_analysis-py.svg
.. image:: https://ci.appveyor.com/api/projects/status/u0koxo5vcitmbghc?svg=true

The *sound\_field\_analysis* toolbox (short: *sfa*) is a Python port of
the `Sound Field Analysis Toolbox (SOFiA) toolbox`_, originally by
Benjamin Bernschütz\ `[1]`_. The main goal of the *sfa* toolbox is to
analyze, visualize and process sound field data recorded by spherical
microphone arrays. Furthermore, various types of test-data may be
generated to evaluate the implemented functions.

The package is pure python and PEP8 compliant (except line-length).
Please expect things to be slow for now and for the API to break, as the
development is still very much ongoing.

Requirements
------------

We use Python 3.5 for development. Chances are that earlier version will
work too but this is currently untested.

The following external libraries are required:

-  `NumPy`_
-  `SciPy`_
-  `Plotly`_ (for plotting)
-  `netCDF4`_ (for sofa_import in Exp3)

Installation
------------
You can simply install *sfa* through pip (``pip install sound_field_analysis``).

We highly recommend the `Anaconda`_ python environment. Once installed,
you can use the following steps to create a new environment with the
*sfa* toolbox.

#. Create a new environment:
   ``conda create --name sfa numpy scipy plotly``
#. Activate this environment:
   ``source activate sfa``
#. Install from pypi:
   ``pip install sound_field_analysis``

Soon, you may also install directly from the `conda-forge`_ channel using
``conda install -c conda-forge sound_field_analysis``.

Documentation
-------------

Please find the full documentation over at
https://appliedacousticschalmers.github.io/sound_field_analysis-py/!

Examples
--------

The following examples are available as Jupyter notebooks, either
statically on `github`_ or interactively on `nbviewer`_. You can of
course also simply download the examples and run them locally!

Exp1 Ideal plane wave
~~~~~~~~~~~~~~~~~~~~~

Ideal unity plane wave simulation and 3D plot.

`View interactively on nbviewer <https://nbviewer.jupyter.org/github/AppliedAcousticsChalmers/sound_field_analysis-py/blob/master/examples/Exp1_IdealPlaneWave.ipynb>`__

|AE1_img|_

.. |AE1_img| image:: examples/img/AE1_shape.png?raw=true
.. _AE1_img: https://nbviewer.jupyter.org/github/AppliedAcousticsChalmers/sound_field_analysis-py/blob/master/examples/Exp1_IdealPlaneWave.ipynb


Exp2: Measured plane wave
~~~~~~~~~~~~~~~~~~~~~~~~~

A measured plane wave from AZ=180°, EL=90° in the anechoic chamber using
a cardioid mic.

`View interactively on nbviewer <https://nbviewer.jupyter.org/github/AppliedAcousticsChalmers/sound_field_analysis-py/blob/master/examples/Exp2_MeasuredWave.ipynb>`__

|AE3_img|_

.. |AE3_img| image:: examples/img/AE3_shape.png?raw=true
.. _AE3_img: https://nbviewer.jupyter.org/github/AppliedAcousticsChalmers/sound_field_analysis-py/blob/master/examples/Exp2_MeasuredWave.ipynb


Exp3: Import data in SOFA format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Install dependencies for `netCDF4`_ (see example documentation) (this only worked on Mac OS X):
   ``brew install hdf5``
   ``brew install netcdf``
#. Install additional dependency from pypi:
   ``pip install sound_field_analysis[sofa_import]``

`SOFA format`_

`View interactively on nbviewer <https://nbviewer.jupyter.org/github/AppliedAcousticsChalmers/sound_field_analysis-py/blob/master/examples/Exp3_Import_SOFA.ipynb>`__


Exp4: Binaural rendering
^^^^^^^^^^^^^^^^^^^^^^^^

Render a spherical microphone array measurement binaurally

`View interactively on nbviewer <https://nbviewer.jupyter.org/github/AppliedAcousticsChalmers/sound_field_analysis-py/blob/master/examples/Exp4_BinauralRendering.ipynb>`__


Version history
^^^^^^^^^^^^^^^

*2019-06-17 V0.7*
    * Implement Bandwidth Extension for Microphone Arrays (BEMA)
    * Edit read_miro_struct, named tuple ArraySignal and miro_to_struct.m to load center measurements

*2019-06-11 V0.6*
    * Port of Radial Filter Improvement from SOFiA

*2019-05-23 V0.5*
    * Implement Spherical Head Filter
    * Implement Spherical Fourier Transform using pseudoinverse
    * Extract real time capable Spatial Fourier Transform
    * Outsource reversed m index function (Exp. 4)


References
^^^^^^^^^^
The *sound_field_analysis* toolbox is based on the Matlab/C++ `Sound Field Analysis Toolbox (SOFiA) toolbox`_ by Benjamin Bernschütz. For more information you may refer to the original publication:

[1] `Bernschütz, B., Pörschmann, C., Spors, S., and Weinzierl, S. (2011). SOFiA Sound Field Analysis Toolbox. Proceedings of the ICSA International Conference on Spatial Audio <http://spatialaudio.net/sofia-sound-field-analysis-toolbox-2/>`_

The Lebedev grid generation was adapted from an implementation by `Richard P. Muller <https://github.com/gabrielelanaro/pyquante/blob/master/Data/lebedev_write.py>`_.


.. _Sound Field Analysis Toolbox (SOFiA) toolbox: http://audiogroup.web.th-koeln.de/SOFiA_wiki/WELCOME.html
.. _[1]: #references
.. _NumPy: http://www.numpy.org
.. _SciPy: http://www.scipy.org
.. _Plotly: https://plot.ly/python/
.. _netCDF4: http://github.com/Unidata/netcdf4-python
.. _Anaconda: https://www.continuum.io/downloads
.. _conda-forge: https://conda-forge.github.io
.. _github: examples/
.. _nbviewer: http://nbviewer.jupyter.org/github/AppliedAcousticsChalmers/sound_field_analysis-py/tree/master/examples/
.. _SOFA format: https://www.sofaconventions.org/mediawiki/index.php/SOFA_(Spatially_Oriented_Format_for_Acoustics)
