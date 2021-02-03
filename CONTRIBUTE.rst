Contribute
==========

Development
-----------

Given you have `Conda`_ installed, run the following commands to clone the repository into a new folder ``sound_field_analysis-py``, install necessary tools into a new `Conda`_ environment and activate it:

  |  ``git clone https://github.com/AppliedAcousticsChalmers/sound_field_analysis-py.git``
  |  ``cd sound_field_analysis-py/``
  |  ``conda env create --file environment.yml --force``
  |  ``source activate sfa``

You can now work on the *sfa* toolbox inside the ``sound_field_analysis-py`` folder. Using ``ipython``, you may use the following magic commands to ensure reload on any changes inside the package:

  |  ``%load_ext autoreload``
  |  ``%autoreload 2``


Documentation
-------------

If you want to compile the documentation (PDF and/or HTML), you need to additionally install `Sphinx <https://www.sphinx-doc.org/en/master/>`_ and `sphinx_rtd_theme <https://github.com/readthedocs/sphinx_rtd_theme>`_. Furthermore, you want to individually clone the ``_gh-pages_`` branch into a separate directory (in this case next to the development directory):

  |  ``conda env update --file environment_dev.yml``
  |  ``git clone --single-branch --branch gh-pages https://github.com/AppliedAcousticsChalmers/sound_field_analysis-py.git ../sound_field_analysis-docs``
  |  ``cd sound_field_analysis-docs/``

You need to install a compact `LaTeX`_ backend as well as the required packages, in case you don't have an extensive distribution like `MacTeX <https://www.tug.org/mactex/morepackages.html>`_ already:

  |  ``brew install basictex``
  |  ``eval "$(/usr/libexec/path_helper)"``
  |  ``texhash``
  |  ``sudo tlmgr update --self``
  |  ``sudo tlmgr install latexmk fncychap titlesec tabulary varwidth framed wrapfig capt-of needspace collection-fontsrecommended``

Now you can recompile the HTML pages and PDF readme (given you have `LaTeX`_ installed):

  |  ``make html -C ../sound_field_analysis-py/doc``
  |  ``make latexpdf -C ../sound_field_analysis-py/doc``

If you decide on a different folder structure, you may edit the following line in ``doc/Makefile`` to decide on where to move the HTML documentation:

  |  ``HTMLDIR       = ../../sound_field_analysis-docs``

.. _Conda: https://conda.io/en/master/miniconda.html
.. _LaTeX: https://www.latex-project.org/
