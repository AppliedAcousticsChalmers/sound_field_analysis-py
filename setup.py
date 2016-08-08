from setuptools import setup

setup(
    name='sound_field_analysis',
    version='0.1dev',
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
    keywords='sound field analysis spherical microphone array',
    install_requires=[
        'scipy',
        'numpy'
    ],
    package_data={
        'examples': ['examples'],
    },
    extras_require={
        'plotting': ["plotly"]
    },
    packages=['sound_field_analysis']
)
