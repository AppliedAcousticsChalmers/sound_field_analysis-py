"""Numerically verify generator outputs
"""

import numpy as np
import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')
from sound_field_analysis import gen


def test_ideal_wave_open_pressure():
    results_open_pressure = np.array([[3.5449077018, -0.0642580673, 0.0065998911, 0.0205155302, -0.0064606444],
                                      [0.0000000000, -0.0047384232 + 0.0030425066j, -0.0274719035 + 0.0176395064j, 0.0055649133 - 0.0035731897j, 0.0128614005 - 0.0082582103j],
                                      [0.0000000000, 0.0000000000 + 0.0051133718j, 0.0000000000 + 0.0296457389j, 0.0000000000 - 0.0060052616j, 0.0000000000 - 0.0138791155j],
                                      [0.0000000000, 0.0047384232 + 0.0030425066j, 0.0274719035 + 0.0176395064j, -0.0055649133 - 0.0035731897j, -0.0128614005 - 0.0082582103j]])
    np.testing.assert_allclose(gen.ideal_wave(order=1, azimuth=1, colatitude=1, fs=48000, F_NFFT=8, array_radius=0.5, scatter_radius=None, array_configuration='open', transducer_type='pressure', wavetype='plane', delay=0, c=343, distance=1), results_open_pressure)


def test_ideal_wave_open_velocity():
    results_open_velocity = np.array([[1.7724538509, -0.0321290337 + 0.0027319952j, 0.0032999455 + 0.0158392581j, 0.0102577651 - 0.0032085181j, -0.0032303222 - 0.0074153959j],
                                      [0.3289843794 + 0.5123628138j, -0.0203150233 - 0.0264276926j, -0.0120591719 + 0.011431183j, 0.0085159795 + 0.0071428379j, 0.0046695870 - 0.0068718764j],
                                      [0.5529057800, -0.0301605294 + 0.0025566859j, 0.0028180708 + 0.0148228695j, 0.0096360135 - 0.0030026308j, -0.0029598052 - 0.0069395577j],
                                      [0.3289843794 - 0.5123628138j, -0.0155766001 + 0.0294701991j, 0.0154127316 + 0.0062083233j, 0.0029510662 - 0.0107160276j, -0.0081918134 - 0.0013863338j]])
    np.testing.assert_allclose(gen.ideal_wave(order=1, azimuth=1, colatitude=1, fs=48000, F_NFFT=8, array_radius=0.5, scatter_radius=None, array_configuration='open', transducer_type='velocity', wavetype='plane', delay=0, c=343, distance=1), results_open_velocity)


def test_ideal_wave_rigid_pressure():
    results_rigid_pressure = np.array([[-0.0643576059 + 0.0054621765j, -0.0643576059 + 0.0054621765j, +0.0063112724 + 0.0316758864j, 0.0205545085 - 0.0064167993j, -0.0063930795 - 0.0148304841j],
                                       [-0.0396107140 - 0.0535098563j, -0.0396107140 - 0.0535098563j, -0.0241659405 + 0.0228929274j, 0.0169234688 + 0.0143553364j, +0.0093641662 - 0.0137598001j],
                                       [-0.0603210321 + 0.0040133970j, -0.0603210321 + 0.0040133970j, +0.0056361414 + 0.0296971018j, 0.0192720269 - 0.0058881867j, -0.0059196103 - 0.0139060852j],
                                       [-0.0321725013 + 0.0582858777j, -0.0321725013 + 0.0582858777j, +0.0308730596 + 0.0124472082j, 0.0060106225 - 0.0213623945j, -0.0164086190 - 0.0027887150j]])
    np.testing.assert_allclose(gen.ideal_wave(order=1, azimuth=1, colatitude=1, fs=48000, F_NFFT=8, array_radius=0.5, scatter_radius=None, array_configuration='rigid', transducer_type='pressure', wavetype='plane', delay=0, c=343, distance=1), results_rigid_pressure)


def test_ideal_wave_rigid_velocity():
    results_rigid_velocity = np.array([[-0.0643576059 + 0.0054621765j, -0.0643576059 + 0.0054621765j, +0.0063112724 + 0.0316758864j, 0.0205545085 - 0.0064167993j, -0.0063930795 - 0.0148304841j],
                                       [-0.0396107140 - 0.0535098563j, -0.0396107140 - 0.0535098563j, -0.0241659405 + 0.0228929274j, 0.0169234688 + 0.0143553364j, +0.0093641662 - 0.0137598001j],
                                       [-0.0603210321 + 0.0040133970j, -0.0603210321 + 0.0040133970j, +0.0056361414 + 0.0296971018j, 0.0192720269 - 0.0058881867j, -0.0059196103 - 0.0139060852j],
                                       [-0.0321725013 + 0.0582858777j, -0.0321725013 + 0.0582858777j, +0.0308730596 + 0.0124472082j, 0.0060106225 - 0.0213623945j, -0.0164086190 - 0.0027887150j]])
    np.testing.assert_allclose(gen.ideal_wave(order=1, azimuth=1, colatitude=1, fs=48000, F_NFFT=8, array_radius=0.5, scatter_radius=None, array_configuration='rigid', transducer_type='velocity', wavetype='plane', delay=0, c=343, distance=1), results_rigid_velocity)
