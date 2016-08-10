# Contribute
## Dev environment
Given you have [anaconda](https://www.continuum.io/downloads) installed, run the following commands to clone the repository into a new folder `sound_field_analysis-py`, install necessary tools into a new conda environment and activate it:
```
git clone https://github.com/QULab/sound_field_analysis-py.git
conda create -n sfa_dev scipy numpy plotly pip ipython
source activate sfa_dev
pip install -e sound_field_analysis-py
```
You can now work on the *sfa* toolbox inside the  `sound_field_analysis-py` folder. Using `ipython`, you may use the following magic commands to ensure reload on any changes inside the package:
```
%load_ext autoreload
%autoreload 2
```

## Documentation
If you want to compile the documentation (pdf and/or html), you need to additionally install sphinx and sphinx_rtd_theme and clone the gh-pages branch:

```
conda install sphinx sphinx_rtd_theme
git clone --single-branch --branch gh-pages https://github.com/QULab/sound_field_analysis-py.git sound_field_analysis-docs
```

Now you can compile the pdf readme (given you have latex installed) and html pages by running `make latexpdf` or `make html` from the `sound_field_analysis-py\doc` directory.

If you decide on a different folder structure, you may edit the following line in `doc/Makefile` to decide on where to move the html documentation:

```
HTMLDIR       = ../../sound_field_analysis-docs
```
