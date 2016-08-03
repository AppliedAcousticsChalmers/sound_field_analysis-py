from setuptools import setup

setup(
    name='sofia-py',
    version='0.0.3',
    description='Python port of the SOFiA toolbox for Matlab',
    url='https://qulab.github.io/sofia-py/',
    author='QU Lab / Christoph Hohnerlein',
    author_email='christoph.hohnerlein@qu.tu-berlin.de',
    license='GPLv3',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Multimedia :: Sound/Audio',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.5',
    ],
    keywords='spherical microphone array analysis',
    install_requires=[
        'numpy',
        'scipy'
    ],
    package_data={
        'examples': ['examples'],
    },
    extras_require={
        'plotting': ["plotly"],
        'progress_bars': ["tqdm"]
    },
    packages=['gen', 'io', 'lebedev', 'plot', 'process', 'sph', 'utils']
)
