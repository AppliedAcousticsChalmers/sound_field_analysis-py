from setuptools import setup

setup(
    name='sofia-py',
    version='0.0.2',
    description='Python port of the SOFiA toolbox for Matlab',
    url='https://github.com/QULab/sofia-py',
    author='QU Lab / Christoph Hohnerlein',
    author_email='christoph.hohnerlein@qu.tu-berlin.de',
    license='GPLv3',
    install_requires=[
        'Numpy',
        'Scipy',
        'matplotlib',
        'Vispy'
    ],
)
