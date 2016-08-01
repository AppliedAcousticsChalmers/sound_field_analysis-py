from setuptools import setup

setup(
    name='sofia-py',
    version='0.0.3',
    description='Python port of the SOFiA toolbox for Matlab',
    url='https://qulab.github.io/sofia-py/',
    author='QU Lab / Christoph Hohnerlein',
    author_email='christoph.hohnerlein@qu.tu-berlin.de',
    license='GPLv3',
    packages=['gen', 'io', 'lebedev', 'plot', 'process', 'sph', 'utils'],
    install_requires=[
        'Numpy',
        'Scipy',
        'plotly'
    ],
)
