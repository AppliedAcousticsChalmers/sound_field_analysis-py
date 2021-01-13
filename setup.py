from setuptools import find_packages, setup

setup(
    name='sound_field_analysis',
    url='https://github.com/AppliedAcousticsChalmers/sound_field_analysis-py/',
    version=open('sound_field_analysis/_version.py', mode='r',
                 encoding='utf-8').readlines()[-1].split()[-1].strip('"\''),
    license='GPLv3',

    description='Analyze, visualize and process sound field data recorded by '
                'spherical microphone arrays.',
    long_description=open('README.rst', mode='r', encoding='utf-8').read(),
    keywords='sound field analysis spherical microphone array',

    author='Chalmers University of Technology / Jens Ahrens',
    author_email='jens.ahrens@chalmers.se',

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Multimedia :: Sound/Audio',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Operating System :: OS Independent',
    ],

    python_requires='>=3.7',
    install_requires=[
        'scipy',
        'numpy',
        'pysofaconventions',
    ],

    package_data={
        'examples': ['examples'],
    },

    extras_require={
        'examples': ['jupyter'],
        'plotting': ['plotly'],
        'testing':  ['pytest'],
    },

    packages=find_packages(),
)
