"""This should test all sph functions numerically compared to ground truth from Matlab.

- For bessel and related functions, all permutations of arguments n, k = [-1.5, -1, 0, 1, 1.5] are tested both
individually and as a matrix.

- For spherical harmonics, a random set of large and small orders with large/small & positive/negative arguments are checked.

"""

import numpy as np
from py.test import approx
import sys
sys.path.insert(0, '../')
from sound_field_analysis import sph


def test_sph_harm():
    assert sph.sph_harm(0, 0, 0, 0, type='complex') == approx(0.282094791773878)
    assert sph.sph_harm(0, 1, 0.1, 0.1, type='complex') == approx(0.486161534508712)
    assert sph.sph_harm(-2, 2, -0.1, -0.1, type='complex') == approx(0.003773142018527 + 0.000764853752555j)
    assert sph.sph_harm(2, 2, -0.1, 0.1, type='complex') == approx(0.003773142018527 - 0.000764853752555j)
    assert sph.sph_harm(17, 17, 7, 6, type='complex') == approx(2.202722537106286e-10 - 8.811259608537731e-11j)
    assert sph.sph_harm(-17, 17, 6, -7, type='complex') == approx(4.945682459644879e-05 - 4.843297071701738e-04j)


def test_besselj():
    n, k = generate_n_k()
    n_grid, k_grid = generate_n_k_grid()

    results = np.array([[-0.680560185349146j, 0.557936507910100, 0.511827671735918, -0.557936507910100, -0.387142217276067j],
                       [-1.102495575160179j, 0.440050585744934, 0.765197686557967, -0.440050585744934, -0.240297839123427j],
                       [np.nan, 0, 1, 0, 0],
                       [-1.102495575160179, -0.440050585744934, 0.765197686557967, 0.440050585744934, 0.240297839123427],
                       [-0.680560185349146, -0.557936507910100, 0.511827671735918, 0.557936507910100, 0.387142217276067]]).T
    for i, cur_n in enumerate(n):
        for j, cur_k in enumerate(k):
            np.testing.assert_allclose(sph.besselj(cur_n, cur_k), results[i, j], err_msg=('n: ' + str(cur_n) + ', k: ' + str(cur_k)))

    np.testing.assert_allclose(sph.besselj(n_grid, k_grid), results.T, err_msg=('Matrix computation failed!'))


def test_hankel():
    n, k = generate_n_k()
    n_grid, k_grid = generate_n_k_grid()

    results = np.array([[-0.387142217276068 - 0.680560185349146j, -0.557936507910100 - 0.412308626973911j, -0.511827671735918 + 0.382448923797759j, 0.557936507910100 + 0.412308626973911j, 0.680560185349146 - 0.387142217276067j],
                        [-0.240297839123427 - 1.102495575160179j, -0.440050585744933 - 0.781212821300289j, -0.765197686557966 + 0.088256964215677j, 0.440050585744933 + 0.781212821300289j, 1.102495575160179 - 0.240297839123427j],
                        [np.nan, np.nan, np.nan, np.nan, np.nan],
                        [-1.102495575160179 - 0.240297839123427j, -0.440050585744934 + 0.781212821300289j, 0.765197686557966 + 0.088256964215677j, 0.440050585744934 - 0.781212821300289j, 0.240297839123427 - 1.102495575160179j],
                        [-0.680560185349146 - 0.387142217276067j, -0.557936507910100 + 0.412308626973911j, 0.511827671735918 + 0.382448923797759j, 0.557936507910100 - 0.412308626973911j, 0.387142217276068 - 0.680560185349146j]]).T

    for i, cur_n in enumerate(n):
        for j, cur_k in enumerate(k):
            np.testing.assert_allclose(sph.hankel(cur_n, cur_k), results[i, j], err_msg=('n: ' + str(cur_n) + ', k: ' + str(cur_k)))

    np.testing.assert_allclose(sph.hankel(n_grid, k_grid), results.T, err_msg=('Matrix computation failed!'))


def test_spbessel():
    n, k = generate_n_k()
    n_grid, k_grid = generate_n_k_grid()

    results = np.array([[0.570951329882802j, 0.047158134445135, -0.664996657736036, 0.396172970712222, 0.237501513490303j],
                       [0.551521620248092j, 0.540302305868140, -0.841470984807896, 0.301168678939757, 0.144010162091969j],
                       [0, 0, 1, 0, 0],
                       [-0.551521620248092, 0.540302305868140, 0.841470984807896, 0.301168678939757, 0.144010162091969],
                       [-0.570951329882802, 0.047158134445135, 0.664996657736036, 0.396172970712222, 0.237501513490303]]).T

    for i, cur_n in enumerate(n):
        for j, cur_k in enumerate(k):
            np.testing.assert_allclose(sph.spbessel(cur_n, cur_k), results[i, j], err_msg=('n: ' + str(cur_n) + ', k: ' + str(cur_k)))

    np.testing.assert_allclose(sph.spbessel(n_grid, k_grid), results.T, err_msg=('Matrix computation failed!'))


def test_dspbessel():
    n, k = generate_n_k()
    n_grid, k_grid = generate_n_k_grid()

    results = np.array([[0.047184403529369j, 0.696435414032793, -0.396172970712222, -0.136766030119740, -0.175115474065630j],
                        [-0.131750648032077j, 1.381773290676036, -0.301168678939757, -0.239133626928383, -0.191496215018168j],
                        [0, 0, 0, 0.333333333333333, 0],
                        [-0.131750648032077, -1.381773290676036, -0.301168678939757, 0.239133626928383, 0.191496215018168],
                        [0.047184403529369, -0.696435414032793, -0.396172970712222, 0.136766030119740, 0.175115474065630]]).T

    for i, cur_n in enumerate(n):
        for j, cur_k in enumerate(k):
            np.testing.assert_allclose(sph.dspbessel(cur_n, cur_k), results[i, j], err_msg='n: ' + str(cur_n) + ', k: ' + str(cur_k))

    np.testing.assert_allclose(sph.dspbessel(n_grid, k_grid), results.T, err_msg=('Matrix computation failed!'))


def generate_n_k():
    return np.array([-1.5, -1, 0, 1, 1.5]), np.array([-1.5, -1, 0, 1, 1.5])


def generate_n_k_grid():
    return np.meshgrid([-1.5, -1, 0, 1, 1.5], [-1.5, -1, 0, 1, 1.5])
