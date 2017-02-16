"""Numerically verify generator outputs
"""

import numpy as np
import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')
from sound_field_analysis import gen, io


def test_ideal_wave_open_omni():
    array_configuration = io.ArrayConfiguration(array_radius=0.5, array_type='open', transducer_type='omni')
    results_open_omni = np.array([[3.5449077018, -0.0642580673, 0.0065998911, 0.0205155302, -0.0064606444],
                                  [0.0000000000, -0.0047384232 + 0.0030425066j, -0.0274719035 + 0.0176395064j, 0.0055649133 - 0.0035731897j, 0.0128614005 - 0.0082582103j],
                                  [0.0000000000, 0.0000000000 + 0.0051133718j, 0.0000000000 + 0.0296457389j, 0.0000000000 - 0.0060052616j, 0.0000000000 - 0.0138791155j],
                                  [0.0000000000, 0.0047384232 + 0.0030425066j, 0.0274719035 + 0.0176395064j, -0.0055649133 - 0.0035731897j, -0.0128614005 - 0.0082582103j]])
    np.testing.assert_allclose(gen.ideal_wave(order=1, azimuth=1, colatitude=1, fs=48000, NFFT=8, array_configuration=array_configuration, wavetype='plane', delay=0, c=343, distance=1), results_open_omni)


def test_ideal_wave_open_cardioid():
    array_configuration = io.ArrayConfiguration(array_radius=0.5, array_type='open', transducer_type='cardioid')
    results_open_cardioid = np.array([[1.7724538509, -0.0321290337 + 0.0027319952j, 0.0032999455 + 0.0158392581j, 0.0102577651 - 0.0032085181j, -0.0032303222 - 0.0074153959j],
                                      [0.3289843794 + 0.5123628138j, -0.0203150233 - 0.0264276926j, -0.0120591719 + 0.011431183j, 0.0085159795 + 0.0071428379j, 0.0046695870 - 0.0068718764j],
                                      [0.5529057800, -0.0301605294 + 0.0025566859j, 0.0028180708 + 0.0148228695j, 0.0096360135 - 0.0030026308j, -0.0029598052 - 0.0069395577j],
                                      [0.3289843794 - 0.5123628138j, -0.0155766001 + 0.0294701991j, 0.0154127316 + 0.0062083233j, 0.0029510662 - 0.0107160276j, -0.0081918134 - 0.0013863338j]])
    np.testing.assert_allclose(gen.ideal_wave(order=1, azimuth=1, colatitude=1, fs=48000, NFFT=8, array_configuration=array_configuration, wavetype='plane', delay=0, c=343, distance=1), results_open_cardioid)


def test_ideal_wave_rigid_omni():
    array_configuration = io.ArrayConfiguration(array_radius=0.5, array_type='rigid', transducer_type='omni', scatter_radius=0.5)
    results_rigid_omni = np.array([[-0.0643576059 + 0.0054621765j, -0.0643576059 + 0.0054621765j, +0.0063112724 + 0.0316758864j, 0.0205545085 - 0.0064167993j, -0.0063930795 - 0.0148304841j],
                                   [-0.0396107140 - 0.0535098563j, -0.0396107140 - 0.0535098563j, -0.0241659405 + 0.0228929274j, 0.0169234688 + 0.0143553364j, +0.0093641662 - 0.0137598001j],
                                   [-0.0603210321 + 0.0040133970j, -0.0603210321 + 0.0040133970j, +0.0056361414 + 0.0296971018j, 0.0192720269 - 0.0058881867j, -0.0059196103 - 0.0139060852j],
                                   [-0.0321725013 + 0.0582858777j, -0.0321725013 + 0.0582858777j, +0.0308730596 + 0.0124472082j, 0.0060106225 - 0.0213623945j, -0.0164086190 - 0.0027887150j]])
    np.testing.assert_allclose(gen.ideal_wave(order=1, azimuth=1, colatitude=1, fs=48000, NFFT=8, array_configuration=array_configuration, wavetype='plane', delay=0, c=343, distance=1), results_rigid_omni)


def test_ideal_wave_rigid_cardioid():
    array_configuration = io.ArrayConfiguration(array_radius=0.5, array_type='rigid', transducer_type='cardioid', scatter_radius=0.5)
    results_rigid_cardioid = np.array([[-0.0643576059 + 0.0054621765j, -0.0643576059 + 0.0054621765j, +0.0063112724 + 0.0316758864j, 0.0205545085 - 0.0064167993j, -0.0063930795 - 0.0148304841j],
                                       [-0.0396107140 - 0.0535098563j, -0.0396107140 - 0.0535098563j, -0.0241659405 + 0.0228929274j, 0.0169234688 + 0.0143553364j, +0.0093641662 - 0.0137598001j],
                                       [-0.0603210321 + 0.0040133970j, -0.0603210321 + 0.0040133970j, +0.0056361414 + 0.0296971018j, 0.0192720269 - 0.0058881867j, -0.0059196103 - 0.0139060852j],
                                       [-0.0321725013 + 0.0582858777j, -0.0321725013 + 0.0582858777j, +0.0308730596 + 0.0124472082j, 0.0060106225 - 0.0213623945j, -0.0164086190 - 0.0027887150j]])
    np.testing.assert_allclose(gen.ideal_wave(order=1, azimuth=1, colatitude=1, fs=48000, NFFT=8, array_configuration=array_configuration, wavetype='plane', delay=0, c=343, distance=1), results_rigid_cardioid)


def test_sampled_wave_open_omni():
    array_configuration = io.ArrayConfiguration(array_radius=0.1, array_type='open', transducer_type='omni')
    results_open_omni = np.array([[3.5449077018110, 1.4287027823479, 2.6599580310768, 2.8285359206193, 1.3306204041198],
                                  [0, -8.8558806782570e-01j, 1.4008429690376e+00j, -1.3302966027074e+00j, 7.0344988487660e-01j],
                                  [0, -4.2327000000000e-09j, -8.4654000000000e-09j, -1.2698100000000e-08j, -1.6930900000000e-08j],
                                  [0, -8.8558806782570e-01j, 1.4008429690376e+00j, -1.3302966027074e+00j, 7.0344988487660e-01j]])
    np.testing.assert_allclose(gen.sampled_wave(order=1, fs=44100, NFFT=8, array_configuration=array_configuration, gridData=gen.lebedev(1), wave_azimuth=0, wave_colatitude=1.570796327), results_open_omni, atol=1e-12)


def test_sampled_wave_open_cardioid():
    array_configuration = io.ArrayConfiguration(array_radius=0.01, array_type='open', transducer_type='cardioid')
    results_open_cardioid = np.array([[1.7724538509055e+00, 1.4967049769369e+00 + 4.9979728197650e-01j, 9.2685600388710e-01 + 5.3305766228400e-01j, 5.9485301572270e-01 + 6.8667687465800e-02j, 8.1088542184460e-01 - 4.6048110907780e-01j],
                                      [7.2360125455830e-01, 3.8587923539310e-01 + 6.1212416373860e-01j, -3.1204034948190e-01 + 6.5286244731520e-01j, -7.1865788302760e-01 + 8.4195921629000e-02j, -4.5404623356140e-01 - 5.6290300056530e-01j],
                                      [-2.0990000000000e-10, -2.0990000000000e-10 - 2.1160000000000e-10j, -2.0990000000000e-10 - 4.2330000000000e-10j, -2.0990000000000e-10 - 6.3490000000000e-10j, -2.0950000000000e-10 - 8.4640000000000e-10j],
                                      [7.2360125455830e-01, 3.8587923539310e-01 + 6.1212416373860e-01j, -3.1204034948190e-01 + 6.5286244731520e-01j, -7.1865788302760e-01 + 8.4195921629000e-02j, -4.5404623356140e-01 - 5.6290300056530e-01j]])
    np.testing.assert_allclose(gen.sampled_wave(order=1, fs=44100, NFFT=8, array_configuration=array_configuration, gridData=gen.lebedev(1), wave_azimuth=0, wave_colatitude=1.570796327), results_open_cardioid, atol=1e-12)


def test_sampled_wave_rigid_omni():
    array_configuration = io.ArrayConfiguration(array_radius=0.02, array_type='rigid', transducer_type='omni', scatter_radius=0.02)
    results_rigid_omni = np.array([[0.9472960355177 + 1.239732059493j, 0.9472960355177 + 1.239732059493j, -0.7902248094539 + 0.4226175305586j, 0.8286396985424 - 0.9508074768159j, 1.7529564648366 + 0.0437757668065j],
                                   [0.3549788246869 + 1.3371731078613j, 0.3549788246869 + 1.3371731078613j, -0.3947603076913 - 0.8852152143603j, 1.1641716344906 - 1.0195528191828j, 2.6296097316976 + 0.9851546185098j],
                                   [0.0655088575950 + 2.1549809558741j, 0.0655088575950 + 2.1549809558741j, -0.4997132416047 + 1.5918051665416j, -0.9420263643131 - 0.9580586241078j, -0.4901197584785 - 2.4850108102247j],
                                   [-0.2830619780170 + 1.3565859759891j, -0.2830619780170 + 1.3565859759891j, -1.6824181809209 - 0.8893273205157j, -0.7414617411157 - 1.0455863226985j, 0.1632355365807 + 0.9376936227764j]])

    np.testing.assert_allclose(gen.sampled_wave(order=1, fs=44100, NFFT=8, array_configuration=array_configuration, gridData=gen.lebedev(1), wave_azimuth=-0.1, wave_colatitude=1), results_rigid_omni, atol=1e-12)


def test_sampled_wave_rigid_cardioid():
    array_configuration = io.ArrayConfiguration(array_radius=0.09, array_type='rigid', transducer_type='cardioid', scatter_radius=0.03)
    results_rigid_cardioid = np.array([[-0.6028519679960 + 0.7669321885739j, -0.6028519679960 + 0.7669321885739j, -0.6309703729264 - 0.118525523167j, 0.5051010853392 - 0.9241602374611j, -1.9734139441889 + 1.3279590179038j],
                                       [-1.4798350447341 + 1.4311552084547j, -1.4798350447341 + 1.4311552084547j, -2.4363413945739 - 0.9481818498745j, 0.2425901521620 - 0.7852102296214j, 0.4098504468155 + 1.3042452522818j],
                                       [0.6945400983242 + 1.2385466804706j, 0.6945400983242 + 1.2385466804706j, -0.2614760811439 - 1.9436454222383j, -0.2776623725191 + 1.9303607155363j, 0.7109138718829 - 1.1523060702016j],
                                       [0.6345515953532 + 1.2573361420525j, 0.6345515953532 + 1.2573361420525j, 0.4413875163426 - 0.9350494243153j, 2.0392805712585 - 0.5951475909762j, -0.0200635317571 + 1.5561501103273j]])

    np.testing.assert_allclose(gen.sampled_wave(order=1, fs=44100, NFFT=8, array_configuration=array_configuration, gridData=gen.lebedev(1), wave_azimuth=0.1, wave_colatitude=2), results_rigid_cardioid, atol=1e-12)
