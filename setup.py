from setuptools import setup

version = open("sound_field_analysis/_version.py").readlines()[-1].split()[-1].strip("\"'")

setup(
    name='sound_field_analysis',
    version=version,
    description='Analyze, visualize and process sound field data recorded by spherical microphone arrays.',
    url='https://github.com/AppliedAcousticsChalmers/sound_field_analysis-py/',
    author='QU Lab / Christoph Hohnerlein',
    author_email='christoph.hohnerlein@qu.tu-berlin.de',
    license='GPLv3',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Multimedia :: Sound/Audio',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
    ],
    keywords='sound field analysis spherical microphone array',
    install_requires=[
        'scipy',
        'numpy',
    ],

    package_data={
        'examples': ['examples'],
    },
    extras_require={
        'plotting': ["plotly"],
        'sofa_import': ["netCDF4"],  # see Exp3_Import_SOFA.py description for installation
    },
    packages=['sound_field_analysis'],
)
