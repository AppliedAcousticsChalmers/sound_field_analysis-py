"""This should test all sph functions numerically compared to ground truth from
Matlab.

- For bessel and related functions, all permutations of arguments
n, k = [-1.5, -1, 0, 1, 1.5] are tested both individually and as a matrix.

- For spherical harmonics, a random set of large and small orders with
large/small & positive/negative arguments are checked.
"""

import numpy as np
from pytest import approx

from sound_field_analysis import sph


def test_sph_harm_complex_standard():
    """Implementation verified against AKtools [1]_.

    Notes
    -----
    `AKsh` uses the inverted order of the parameters n and m. The inclination
    angle is referenced as elevation although colatitude is used. The angles
    are specified in degrees.

    References
    ----------
    .. [1] Brinkmann, F., and Weinzierl, S. (2017). “AKtools - An Open Software
        Toolbox for Signal Acquisition, Processing, and Inspection in
        Acoustics,” AES Conv. 142, Audio Engineering Society, Berlin,
        Germany, 1–6.
    """
    for kind in ["complex", "complex_SHT", "complex_Spaudiopy", "complex_AKT"]:
        print(f"{kind = }")

        # AKsh(0, 0, 0, 0, 'complex')
        assert sph.sph_harm(n=0, m=0, az=0, co=0, kind=kind) == approx(
            0.282094791773878
        )

        # AKsh(1, 0, rad2deg(0.1), rad2deg(0.1), 'complex')
        assert sph.sph_harm(n=1, m=0, az=0.1, co=0.1, kind=kind) == approx(
            0.486161534508712
        )

        # AKsh(2, -2, rad2deg(-0.1), rad2deg(-0.1), 'complex')
        assert sph.sph_harm(n=2, m=-2, az=-0.1, co=-0.1, kind=kind) == approx(
            0.003773142018527 + 0.000764853752555j
        )

        # AKsh(2, 2, rad2deg(-0.1), rad2deg(0.1), 'complex')
        assert sph.sph_harm(n=2, m=2, az=-0.1, co=0.1, kind=kind) == approx(
            0.003773142018527 - 0.000764853752555j
        )

        # AKsh(3, 1, rad2deg(4), rad2deg(-3), 'complex')
        assert sph.sph_harm(n=3, m=1, az=4, co=-3, kind=kind) == approx(
            0.116275001805827 + 0.134625671696022j
        )  # different for kind == "complex_SFS"

        # AKsh(17, 17, rad2deg(7), rad2deg(6), 'complex')
        assert sph.sph_harm(n=17, m=17, az=7, co=6, kind=kind) == approx(
            -2.202722537106273e-10 + 8.811259608538045e-11j
        )  # different for kind == "complex_SFS"

        # AKsh(17, -17, rad2deg(6), rad2deg(-7), 'complex')
        assert sph.sph_harm(n=17, m=-17, az=6, co=-7, kind=kind) == approx(
            4.945682459644794e-05 - 4.843297071701655e-04j
        )


def test_sph_harm_complex_alternative():
    """Implementation verified against soundfieldsynthesis [2]_ [3]_.

    Notes
    -----
    `sphharm` uses the inverted order of the parameters n and m as well as the
    inverted order of the directions azimuth and colatitude. The angles are
    specified in radians.

    References
    ----------
    .. [2] J. Ahrens, Analytic Methods of Sound Field Synthesis, 1. Auflage.
        Berlin, Heidelberg: Springer Berlin Heidelberg, 2012.
    .. [3] https://github.com/JensAhrens/soundfieldsynthesis/blob/master/Common/spherical/sphharm.m
    """
    for kind in ["complex_GumDur", "complex_SFS"]:
        print(f"{kind = }")

        # sphharm(0, 0, 0, 0, 'complex')
        assert sph.sph_harm(n=0, m=0, co=0, az=0, kind=kind) == approx(
            0.282094791773878
        )

        # sphharm(1, 0, 0.1, 0.1, 'complex')
        assert sph.sph_harm(n=1, m=0, co=0.1, az=0.1, kind=kind) == approx(
            0.486161534508712
        )

        # sphharm(2, -2, -0.1, -0.1, 'complex')
        assert sph.sph_harm(n=2, m=-2, co=-0.1, az=-0.1, kind=kind) == approx(
            0.003773142018527 + 0.000764853752555j
        )

        # sphharm(2, 2, 0.1, -0.1, 'complex')
        assert sph.sph_harm(n=2, m=2, co=0.1, az=-0.1, kind=kind) == approx(
            0.003773142018527 - 0.000764853752555j
        )

        # sphharm(2, 2, -3, 4, 'complex')
        assert sph.sph_harm(n=2, m=2, co=0.1, az=-0.1, kind=kind) == approx(
            0.003773142018527 - 0.000764853752555j
        )

        # sphharm(3, 1, -3, 4, 'complex')
        assert sph.sph_harm(n=3, m=1, co=-3, az=4, kind=kind) == approx(
            -0.116275001805827 - 0.134625671696022j
        )  # different for kind == "complex_AKT"

        # sphharm(17, 17, 6, 7, 'complex')
        assert sph.sph_harm(n=17, m=17, co=6, az=7, kind=kind) == approx(
            2.202722537106286e-10 - 8.811259608537731e-11j
        )  # different for kind == "complex_AKT"

        # sphharm(17, -17, -7, 6, 'complex')
        assert sph.sph_harm(n=17, m=-17, co=-7, az=6, kind=kind) == approx(
            4.945682459644879e-05 - 4.843297071701738e-04j
        )


def test_sph_harm_real_standard():
    """Implementation verified against Spherical-Harmonic-Transform [4]_.

    Notes
    -----
    `getSH` generates the full set of SH coefficients for the given maximum
    order n. Therefore, the degree m needs to be extracted afterwards. The
    directions azimuth and colatitude are provided in a combined matrix with
    the angles specified in radians.

    References
    ----------
    .. [4] https://github.com/polarch/Spherical-Harmonic-Transform/blob/master/inverseSHT.m
    """
    for kind in ["real", "real_SHT", "real_Spaudiopy"]:
        print(f"{kind = }")

        # getSH(0, [0, 0], 'real')
        assert sph.sph_harm(n=0, m=0, az=0, co=0, kind=kind) == approx(
            0.282094791773878
        )

        # n = 1; r = getSH(n, [0.1, 0.1], 'real'); r((n+1)^2 - n - 0)
        assert sph.sph_harm(n=1, m=0, az=0.1, co=0.1, kind=kind) == approx(
            0.486161534508712
        )

        # n = 2; r = getSH(n, [-0.1, -0.1], 'real'); r((n+1)^2 - n - 2)
        assert sph.sph_harm(n=2, m=-2, az=-0.1, co=-0.1, kind=kind) == approx(
            -0.001081666550095
        )

        # n = 2; r = getSH(n, [-0.1, 0.1], 'real'); r((n+1)^2 - n + 2)
        assert sph.sph_harm(n=2, m=2, az=-0.1, co=0.1, kind=kind) == approx(
            0.005336028615361
        )

        # n = 3; r = getSH(3, [4, -3], 'real'); r((n+1)^2 - n + 1)
        assert sph.sph_harm(n=3, m=1, az=4, co=-3, kind=kind) == approx(
            -0.164437684518757
        )

        # n = 17; r = getSH(17, [7, 6], 'real'); r((n+1)^2 - n + 17)
        assert sph.sph_harm(n=17, m=17, az=7, co=6, kind=kind) == approx(
            3.115120086120583e-10
        )

        # n = 17; r = getSH(17, [6, -7], 'real'); r((n+1)^2 - n - 17)
        assert sph.sph_harm(n=17, m=-17, az=6, co=-7, kind=kind) == approx(
            6.849456405402495e-04
        )


def test_sph_harm_real_alternative():
    """Implementation verified against AKtools [1]_.

    Notes
    -----
    `AKsh` uses the inverted order of the parameters n and m. The inclination
    angle is referenced as elevation although colatitude is used. The angles
    are specified in degrees.

    References
    ----------
    .. [1] Brinkmann, F., and Weinzierl, S. (2017). “AKtools - An Open Software
        Toolbox for Signal Acquisition, Processing, and Inspection in
        Acoustics,” AES Conv. 142, Audio Engineering Society, Berlin,
        Germany, 1–6.
    """
    for kind in ["real_Zotter", "real_AKT", "real_SFS"]:
        print(f"{kind = }")

        # AKsh(0, 0, 0, 0, 'real')
        assert sph.sph_harm(n=0, m=0, az=0, co=0, kind=kind) == approx(
            0.282094791773878
        )

        # AKsh(1, 0, rad2deg(0.1), rad2deg(0.1), 'real')
        assert sph.sph_harm(n=1, m=0, az=0.1, co=0.1, kind=kind) == approx(
            0.486161534508712
        )

        # AKsh(2, -2, rad2deg(-0.1), rad2deg(-0.1), 'real')
        assert sph.sph_harm(n=2, m=-2, az=-0.1, co=-0.1, kind=kind) == approx(
            0.001081666550095
        )

        # AKsh(2, 2, rad2deg(-0.1), rad2deg(0.1), 'real')
        assert sph.sph_harm(n=2, m=2, az=-0.1, co=0.1, kind=kind) == approx(
            0.005336028615361
        )

        # AKsh(3, 1, rad2deg(4), rad2deg(-3), 'real')
        assert sph.sph_harm(n=3, m=1, az=4, co=-3, kind=kind) == approx(
            -0.164437684518757
        )

        # AKsh(17, 17, rad2deg(7), rad2deg(6), 'real')
        assert sph.sph_harm(n=17, m=17, az=7, co=6, kind=kind) == approx(
            3.115120086120565e-10
        )

        # AKsh(17, -17, rad2deg(6), rad2deg(-7), 'real')
        assert sph.sph_harm(n=17, m=-17, az=6, co=-7, kind=kind) == approx(
            -6.849456405402378e-04
        )


def test_besselj():
    n, k = generate_n_k()
    results = np.array(
        [
            [
                -0.680560185349146j,
                0.557936507910100,
                0.511827671735918,
                -0.557936507910100,
                -0.387142217276067j,
            ],
            [
                -1.102495575160179j,
                0.440050585744934,
                0.765197686557967,
                -0.440050585744934,
                -0.240297839123427j,
            ],
            [np.nan, 0, 1, 0, 0],
            [
                -1.102495575160179,
                -0.440050585744934,
                0.765197686557967,
                0.440050585744934,
                0.240297839123427,
            ],
            [
                -0.680560185349146,
                -0.557936507910100,
                0.511827671735918,
                0.557936507910100,
                0.387142217276067,
            ],
        ]
    ).T
    run_numerical_test(sph.besselj, n, k, results)


def test_hankel1():
    n, k = generate_n_k()
    results = np.array(
        [
            [
                -0.387142217276068 - 0.680560185349146j,
                -0.557936507910100 - 0.412308626973911j,
                -0.511827671735918 + 0.382448923797759j,
                0.557936507910100 + 0.412308626973911j,
                0.680560185349146 - 0.387142217276067j,
            ],
            [
                -0.240297839123427 - 1.102495575160179j,
                -0.440050585744933 - 0.781212821300289j,
                -0.765197686557966 + 0.088256964215677j,
                0.440050585744933 + 0.781212821300289j,
                1.102495575160179 - 0.240297839123427j,
            ],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [
                -1.102495575160179 - 0.240297839123427j,
                -0.440050585744934 + 0.781212821300289j,
                0.765197686557966 + 0.088256964215677j,
                0.440050585744934 - 0.781212821300289j,
                0.240297839123427 - 1.102495575160179j,
            ],
            [
                -0.680560185349146 - 0.387142217276067j,
                -0.557936507910100 + 0.412308626973911j,
                0.511827671735918 + 0.382448923797759j,
                0.557936507910100 - 0.412308626973911j,
                0.387142217276068 - 0.680560185349146j,
            ],
        ]
    ).T
    run_numerical_test(sph.hankel1, n, k, results)


def test_hankel2():
    n, k = generate_n_k()
    results = np.array(
        [
            [
                0.387142217276067 - 0.680560185349146j,
                1.673809523730299 + 0.412308626973911j,
                1.535483015207755 - 0.382448923797759j,
                -1.673809523730299 - 0.412308626973911j,
                -0.680560185349146 - 0.387142217276067j,
            ],
            [
                0.240297839123427 - 1.102495575160179j,
                1.320151757234800 + 0.781212821300289j,
                2.295593059673900 - 0.088256964215677j,
                -1.320151757234800 - 0.781212821300289j,
                -1.102495575160179 - 0.240297839123427j,
            ],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [
                -1.102495575160179 + 0.240297839123427j,
                -0.440050585744934 - 0.781212821300289j,
                0.765197686557966 - 0.088256964215677j,
                0.440050585744934 + 0.781212821300289j,
                0.240297839123427 + 1.102495575160179j,
            ],
            [
                -0.680560185349146 + 0.387142217276067j,
                -0.557936507910100 - 0.412308626973911j,
                0.511827671735918 - 0.382448923797759j,
                0.557936507910100 + 0.412308626973911j,
                0.387142217276068 + 0.680560185349146j,
            ],
        ]
    ).T
    run_numerical_test(sph.hankel2, n, k, results)


def test_neumann():
    n, k = generate_n_k()
    results = np.array(
        [
            [
                0.387142217276067j,
                -0.412308626973911 + 1.115873015820199j,
                0.382448923797759 + 1.023655343471836j,
                0.412308626973911 - 1.115873015820199j,
                -0.680560185349146j,
            ],
            [
                0.240297839123427j,
                -0.781212821300289 + 0.880101171489867j,
                0.088256964215677 + 1.530395373115933j,
                0.781212821300289 - 0.880101171489867j,
                -1.102495575160179j,
            ],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [
                -0.240297839123427,
                0.781212821300289,
                0.088256964215677,
                -0.781212821300289,
                -1.102495575160179,
            ],
            [
                -0.387142217276067,
                0.412308626973911,
                0.382448923797759,
                -0.412308626973911,
                -0.680560185349146,
            ],
        ]
    ).T
    run_numerical_test(sph.neumann, n, k, results)


def test_spbessel():
    n, k = generate_n_k()
    results = np.array(
        [
            [
                0.570951329882802j,
                0.047158134445135,
                -0.664996657736036,
                0.396172970712222,
                0.237501513490303j,
            ],
            [
                0.551521620248092j,
                0.540302305868140,
                -0.841470984807896,
                0.301168678939757,
                0.144010162091969j,
            ],
            [0, 0, 1, 0, 0],
            [
                -0.551521620248092,
                0.540302305868140,
                0.841470984807896,
                0.301168678939757,
                0.144010162091969,
            ],
            [
                -0.570951329882802,
                0.047158134445135,
                0.664996657736036,
                0.396172970712222,
                0.237501513490303,
            ],
        ]
    ).T
    run_numerical_test(sph.spbessel, n, k, results)


def test_spneumann():
    n, k = generate_n_k()
    results = np.array(
        [
            [
                1.141902659765604 + 0.421926429899149j,
                0.664996657736036,
                0.047158134445135,
                -0.696435414032793,
                0.475003026980606 + 0.953938771346504j,
            ],
            [
                1.103043240496184 + 0.979105073187780j,
                0.841470984807896,
                0.540302305868140,
                -1.381773290676036,
                0.288020324183939 + 2.068823847343615j,
            ],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [
                0.979105073187780,
                0.841470984807896,
                -0.540302305868140,
                -1.381773290676036,
                -2.068823847343615,
            ],
            [
                0.421926429899149,
                0.664996657736036,
                -0.047158134445135,
                -0.696435414032793,
                -0.953938771346504,
            ],
        ]
    ).T
    run_numerical_test(sph.spneumann, n, k, results)


def test_dspbessel():
    n, k = generate_n_k()
    results = np.array(
        [
            [
                0.047184403529369j,
                0.696435414032793,
                -0.396172970712222,
                -0.136766030119740,
                -0.175115474065630j,
            ],
            [
                -0.131750648032077j,
                1.381773290676036,
                -0.301168678939757,
                -0.239133626928383,
                -0.191496215018168j,
            ],
            [0, 0, 0, 0.333333333333333, 0],
            [
                -0.131750648032077,
                -1.381773290676036,
                -0.301168678939757,
                0.239133626928383,
                0.191496215018168,
            ],
            [
                0.047184403529369,
                -0.696435414032793,
                -0.396172970712222,
                0.136766030119740,
                0.175115474065630,
            ],
        ]
    ).T
    run_numerical_test(sph.dspbessel, n, k, results)


def test_sphankel1():
    n, k = generate_n_k()
    results = np.array(
        [
            [
                -0.421926429899149 + 0.570951329882802j,
                -0.047158134445135 + 0.664996657736036j,
                0.664996657736036 + 0.047158134445135j,
                -0.396172970712222 - 0.696435414032793j,
                -0.953938771346504 + 0.237501513490303j,
            ],
            [
                -0.979105073187780 + 0.551521620248092j,
                -0.540302305868140 + 0.841470984807896j,
                0.841470984807896 + 0.540302305868140j,
                -0.301168678939757 - 1.381773290676036j,
                -2.068823847343615 + 0.144010162091970j,
            ],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [
                -0.551521620248092 + 0.979105073187780j,
                0.540302305868140 + 0.841470984807896j,
                0.841470984807896 - 0.540302305868140j,
                0.301168678939757 - 1.381773290676036j,
                0.144010162091969 - 2.068823847343615j,
            ],
            [
                -0.570951329882802 + 0.421926429899149j,
                0.047158134445135 + 0.664996657736036j,
                0.664996657736036 - 0.047158134445135j,
                0.396172970712222 - 0.696435414032793j,
                0.237501513490303 - 0.953938771346504j,
            ],
        ]
    ).T
    run_numerical_test(sph.sphankel1, n, k, results)


def test_sphankel2():
    n, k = generate_n_k()
    results = np.array(
        [
            [
                0.421926429899149 - 1.712853989648406j,
                -0.047158134445135 - 0.664996657736036j,
                0.664996657736036 - 0.047158134445135j,
                -0.396172970712222 + 0.696435414032793j,
                0.953938771346504 - 0.712504540470909j,
            ],
            [
                0.979105073187780 - 1.654564860744276j,
                -0.540302305868140 - 0.841470984807897j,
                0.841470984807897 - 0.540302305868140j,
                -0.301168678939757 + 1.381773290676036j,
                2.068823847343615 - 0.432030486275908j,
            ],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [
                -0.551521620248092 - 0.979105073187780j,
                0.540302305868140 - 0.841470984807896j,
                0.841470984807896 + 0.540302305868140j,
                0.301168678939757 + 1.381773290676036j,
                0.144010162091969 + 2.068823847343615j,
            ],
            [
                -0.570951329882802 - 0.421926429899149j,
                0.047158134445135 - 0.664996657736036j,
                0.664996657736036 + 0.047158134445135j,
                0.396172970712222 + 0.696435414032793j,
                0.237501513490303 + 0.953938771346504j,
            ],
        ]
    ).T
    run_numerical_test(sph.sphankel2, n, k, results)


def test_dsphankel1():
    n, k = generate_n_k()
    results = np.array(
        [
            [
                -0.813296628046787 + 0.047184403529369j,
                -0.696435414032793 + 0.396172970712222j,
                0.396172970712222 + 0.696435414032793j,
                0.136766030119740 - 0.881422417598589j,
                -1.167971522345024 - 0.175115474065630j,
            ],
            [
                -1.579271310749725 - 0.131750648032077j,
                -1.381773290676036 + 0.301168678939757j,
                0.301168678939757 + 1.381773290676036j,
                0.239133626928383 - 2.223244275483933j,
                -4.192954545171260 - 0.191496215018168j,
            ],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [
                -0.131750648032077 - 1.579271310749725j,
                -1.381773290676036 - 0.301168678939757j,
                -0.301168678939757 + 1.381773290676036j,
                0.239133626928383 + 2.223244275483933j,
                0.191496215018168 + 4.192954545171260j,
            ],
            [
                0.047184403529369 - 0.813296628046787j,
                -0.696435414032793 - 0.396172970712222j,
                -0.396172970712222 + 0.696435414032793j,
                0.136766030119740 + 0.881422417598589j,
                0.175115474065630 + 1.167971522345024j,
            ],
        ]
    ).T
    run_numerical_test(sph.dsphankel1, n, k, results)


def test_dsphankel2():
    n, k = generate_n_k()
    results = np.array(
        [
            [
                0.813296628046787 - 0.141553210588107j,
                -0.696435414032793 - 0.396172970712222j,
                0.396172970712222 - 0.696435414032793j,
                0.136766030119740 + 0.881422417598589j,
                1.167971522345024 + 0.525346422196891j,
            ],
            [
                1.579271310749725 + 0.395251944096230j,
                -1.381773290676036 - 0.301168678939756j,
                0.301168678939757 - 1.381773290676036j,
                0.239133626928383 + 2.223244275483933j,
                4.192954545171260 + 0.574488645054505j,
            ],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [
                -0.131750648032077 + 1.579271310749725j,
                -1.381773290676036 + 0.301168678939757j,
                -0.301168678939757 - 1.381773290676036j,
                0.239133626928383 - 2.223244275483933j,
                0.191496215018168 - 4.192954545171260j,
            ],
            [
                0.047184403529369 + 0.813296628046787j,
                -0.696435414032793 + 0.396172970712222j,
                -0.396172970712222 - 0.696435414032793j,
                0.136766030119740 - 0.881422417598589j,
                0.175115474065630 - 1.167971522345024j,
            ],
        ]
    ).T
    run_numerical_test(sph.dsphankel2, n, k, results)


def test_dspneumann():
    n, k = generate_n_k()
    results = np.array(
        [
            [
                0.094368807058738 + 0.813296628046787j,
                0.396172970712222,
                0.696435414032793,
                -0.881422417598589,
                -0.350230948131260 + 1.167971522345024j,
            ],
            [
                -0.263501296064153 + 1.579271310749725j,
                0.301168678939757,
                1.381773290676036,
                -2.223244275483933,
                -0.382992430036336 + 4.19295454517126j,
            ],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [
                -1.579271310749725,
                -0.301168678939757,
                1.381773290676036,
                2.223244275483933,
                4.192954545171260,
            ],
            [
                -0.813296628046787,
                -0.396172970712222,
                0.696435414032793,
                0.881422417598589,
                1.167971522345024,
            ],
        ]
    ).T
    run_numerical_test(sph.dspneumann, n, k, results)


def generate_n_k():
    return np.array([-1.5, -1, 0, 1, 1.5]), np.array([-1.5, -1, 0, 1, 1.5])


def generate_n_k_grid():
    return np.meshgrid([-1.5, -1, 0, 1, 1.5], [-1.5, -1, 0, 1, 1.5])


def run_numerical_test(function, n, k, results):
    n_grid, k_grid = np.meshgrid(n, k)

    if results.shape != n_grid.shape:
        raise ValueError("results passed do not match the shape of n/k")

    # Test both as scalars
    for i, cur_n in enumerate(n):
        for j, cur_k in enumerate(k):
            np.testing.assert_allclose(
                function(cur_n, cur_k),
                results[i, j],
                err_msg=f"n: {cur_n}, k: {cur_k}",
            )

    # Test first as scalar
    for i, cur_n in enumerate(n):
        np.testing.assert_allclose(
            function(cur_n, k_grid[:, i]),
            results[i, :],
            err_msg=f"n: {cur_n}, k: {k_grid[:, i]}",
        )

    # Test two as scalar
    for j, cur_k in enumerate(k):
        np.testing.assert_allclose(
            function(n_grid[j, :], cur_k),
            results[:, j],
            err_msg=f"n: {n_grid[j, :]}, k: {cur_k}",
        )

    # Test matrix
    np.testing.assert_allclose(
        function(n_grid, k_grid), results.T, err_msg="Matrix computation failed!"
    )
