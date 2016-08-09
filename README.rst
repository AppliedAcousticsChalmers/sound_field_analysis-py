Sound Field Analysis toolbox for Python
=======================================

The *sound\_field\_analysis* toolbox (short: *sfa*) is a Python port of
the `Sound Field Analysis Toolbox (SOFiA) toolbox`_, originally by
Benjamin Bernschütz\ `[1]`_. The main goal of the *sfa* toolbox is to
analyze, visualize and process sound field data recorded by spherical
microphone arrays. Furthermore, various types of testdata may be
generated to evaluate the implemented functions.

The package is pure python and PEP8 compliant (except line-length).
Please expect things to be slow for now and for the API to break, as the
development is still very much ongoing.

Requirements
------------

We use Python 3.5 for development. Chances are that earlier version will
work too but this is untested.

The following external libraries are required:

-  `NumPy`_
-  `SciPy`_
-  `Plotly`_ (for plotting)

Installation
------------

We highly recommend the `Anaconda`_ python environment. Once installed,
you can use the following steps to create a new environment with the
*sfa* toolbox.

#. Add the `conda-forge`_ channel:
   ``conda config --add channels conda-forge``
#. Create a new environment:
   ``create --name sfa numpy scipy plotly sound_field_analysis``
#. Activate this environment:
   ``source activate sfa``

Alternatively, you can simply install through pip
(``pip install sound_field_analysis``).

Documentation
-------------

Please find the full documentation over at
https://qulab.github.io/sound_field_analysis-py/!

Examples
--------

The following examples are available as Jupyter notebooks, either
statically on `github`_ or interactivally on `nbviewer`_. You can of
course also simply download the examples and run them locally!

AE1: Ideal plane wave
~~~~~~~~~~~~~~~~~~~~~

Ideal unity plane wave simulation and 3D plot.

`View interactively on nbviewer <http://nbviewer.jupyter.org/github/QULab/sound_field_analysis-py/blob/master/examples/AE1_IdealPlaneWave.ipynb>`__

|AE1_img|_

.. |AE1_img| image:: examples/img/AE1_shape.png?raw=true
.. _AE1_img: http://nbviewer.jupyter.org/github/QULab/sound_field_analysis-py/blob/master/examples/AE1_IdealPlaneWave.ipynb


AE3: Measured plane wave
~~~~~~~~~~~~~~~~~~~~~~~~

A measured plane wave from AZ=180°, EL=90° in the anechoic chamber using
a cardioid mic.

`View interactively on nbviewer <http://nbviewer.jupyter.org/github/QULab/sound_field_analysis-py/blob/master/examples/AE3_MeasuredWave.ipynb>`__

|AE3_img|_

.. |AE3_img| image:: examples/img/AE3_shape.png?raw=true
.. _AE3_img: http://nbviewer.jupyter.org/github/QULab/sound_field_analysis-py/blob/master/examples/AE3_MeasuredWave.ipynb

AE6: Impulse response of ideal plane wave
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Impulse Response reconstruction on a simulated ideal unity plane wave

`View interactively on nbviewer <http://nbviewer.jupyter.org/github/QULab/sound_field_analysis-py/blob/master/examples/AE6_IdealPlaneWave_ImpResp.ipynb>`__

|AE6_img|_

.. |AE6_img| image:: examples/img/AE6_IdealPlaneWave_ImpResp.png?raw=true
.. _AE6_img: http://nbviewer.jupyter.org/github/QULab/sound_field_analysis-py/blob/master/examples/AE6_IdealPlaneWave_ImpResp.ipynb



AE7: Impulse response of sampled plane wave
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Impulse response reconstruction on a simulated sampled unity plane wave

`View interactively on nbviewer <http://nbviewer.jupyter.org/github/QULab/sound_field_analysis-py/blob/master/examples/AE7_SampledPlaneWave_ImpResp.ipynb>`__

|AE7_img|_

.. |AE7_img| image:: examples/img/AE7_SampledPlaneWave_ImpResp.png?raw=true
.. _AE7_img: http://nbviewer.jupyter.org/github/QULab/sound_field_analysis-py/blob/master/examples/AE7_SampledPlaneWave_ImpResp.ipynb

References
^^^^^^^^^^
SOFiA-py is based on the Matlab/C++ toolbox [SOFiA](https://github.com/fietew/sofia-toolbox) by Benjamin Bernschütz. For more information you may refer to the original publication:

[1] `Bernschütz, B., Pörschmann, C., Spors, S., and Weinzierl, S. (2011). SOFiA Sound Field Analysis Toolbox. Proceedings of the ICSA International Conference on Spatial Audio <http://spatialaudio.net/sofia-sound-field-analysis-toolbox-2/>`_

The Lebedev grid generation was adapted from an implementaion by `Richard P. Muller <https://github.com/gabrielelanaro/pyquante/blob/master/Data/lebedev_write.py>`_.


.. _Sound Field Analysis Toolbox (SOFiA) toolbox: http://audiogroup.web.th-koeln.de/SOFiA_wiki/WELCOME.html
.. _[1]: #references
.. _NumPy: http://www.numpy.org
.. _SciPy: http://www.scipy.org
.. _Plotly: https://plot.ly/python/
.. _Anaconda: https://www.continuum.io/downloads
.. _conda-forge: https://conda-forge.github.io
.. _github: examples/
.. _nbviewer: http://nbviewer.jupyter.org/github/QULab/sound_field_analysis-py/tree/master/examples/
