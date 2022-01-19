"""This should test all sph functions numerically compared to ground truth from
Matlab.

- For spherical harmonics, a random set of large and small orders with
large/small & positive/negative arguments are checked.
"""

import pytest

from sound_field_analysis import sph


@pytest.mark.parametrize(
    "kind", ["complex", "complex_SHT", "complex_Spaudiopy", "complex_AKT"]
)
def test_sph_harm_complex_standard(kind):
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
    # AKsh(0, 0, 0, 0, 'complex')
    assert sph.sph_harm(n=0, m=0, az=0, co=0, kind=kind) == pytest.approx(
        0.282094791773878
    )

    # AKsh(1, 0, rad2deg(0.1), rad2deg(0.1), 'complex')
    assert sph.sph_harm(n=1, m=0, az=0.1, co=0.1, kind=kind) == pytest.approx(
        0.486161534508712
    )

    # AKsh(2, -2, rad2deg(-0.1), rad2deg(-0.1), 'complex')
    assert sph.sph_harm(n=2, m=-2, az=-0.1, co=-0.1, kind=kind) == pytest.approx(
        0.003773142018527 + 0.000764853752555j
    )

    # AKsh(2, 2, rad2deg(-0.1), rad2deg(0.1), 'complex')
    assert sph.sph_harm(n=2, m=2, az=-0.1, co=0.1, kind=kind) == pytest.approx(
        0.003773142018527 - 0.000764853752555j
    )

    # AKsh(3, 1, rad2deg(4), rad2deg(-3), 'complex')
    assert sph.sph_harm(n=3, m=1, az=4, co=-3, kind=kind) == pytest.approx(
        0.116275001805827 + 0.134625671696022j
    )  # different for kind == "complex_SFS"

    # AKsh(17, 17, rad2deg(7), rad2deg(6), 'complex')
    assert sph.sph_harm(n=17, m=17, az=7, co=6, kind=kind) == pytest.approx(
        -2.202722537106273e-10 + 8.811259608538045e-11j
    )  # different for kind == "complex_SFS"

    # AKsh(17, -17, rad2deg(6), rad2deg(-7), 'complex')
    assert sph.sph_harm(n=17, m=-17, az=6, co=-7, kind=kind) == pytest.approx(
        4.945682459644794e-05 - 4.843297071701655e-04j
    )


@pytest.mark.parametrize("kind", ["complex_GumDur", "complex_SFS"])
def test_sph_harm_complex_alternative(kind):
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
    # sphharm(0, 0, 0, 0, 'complex')
    assert sph.sph_harm(n=0, m=0, co=0, az=0, kind=kind) == pytest.approx(
        0.282094791773878
    )

    # sphharm(1, 0, 0.1, 0.1, 'complex')
    assert sph.sph_harm(n=1, m=0, co=0.1, az=0.1, kind=kind) == pytest.approx(
        0.486161534508712
    )

    # sphharm(2, -2, -0.1, -0.1, 'complex')
    assert sph.sph_harm(n=2, m=-2, co=-0.1, az=-0.1, kind=kind) == pytest.approx(
        0.003773142018527 + 0.000764853752555j
    )

    # sphharm(2, 2, 0.1, -0.1, 'complex')
    assert sph.sph_harm(n=2, m=2, co=0.1, az=-0.1, kind=kind) == pytest.approx(
        0.003773142018527 - 0.000764853752555j
    )

    # sphharm(2, 2, -3, 4, 'complex')
    assert sph.sph_harm(n=2, m=2, co=0.1, az=-0.1, kind=kind) == pytest.approx(
        0.003773142018527 - 0.000764853752555j
    )

    # sphharm(3, 1, -3, 4, 'complex')
    assert sph.sph_harm(n=3, m=1, co=-3, az=4, kind=kind) == pytest.approx(
        -0.116275001805827 - 0.134625671696022j
    )  # different for kind == "complex_AKT"

    # sphharm(17, 17, 6, 7, 'complex')
    assert sph.sph_harm(n=17, m=17, co=6, az=7, kind=kind) == pytest.approx(
        2.202722537106286e-10 - 8.811259608537731e-11j
    )  # different for kind == "complex_AKT"

    # sphharm(17, -17, -7, 6, 'complex')
    assert sph.sph_harm(n=17, m=-17, co=-7, az=6, kind=kind) == pytest.approx(
        4.945682459644879e-05 - 4.843297071701738e-04j
    )


@pytest.mark.parametrize("kind", ["real", "real_SHT", "real_Spaudiopy"])
def test_sph_harm_real_standard(kind):
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
    # getSH(0, [0, 0], 'real')
    assert sph.sph_harm(n=0, m=0, az=0, co=0, kind=kind) == pytest.approx(
        0.282094791773878
    )

    # n = 1; r = getSH(n, [0.1, 0.1], 'real'); r((n+1)^2 - n - 0)
    assert sph.sph_harm(n=1, m=0, az=0.1, co=0.1, kind=kind) == pytest.approx(
        0.486161534508712
    )

    # n = 2; r = getSH(n, [-0.1, -0.1], 'real'); r((n+1)^2 - n - 2)
    assert sph.sph_harm(n=2, m=-2, az=-0.1, co=-0.1, kind=kind) == pytest.approx(
        -0.001081666550095
    )

    # n = 2; r = getSH(n, [-0.1, 0.1], 'real'); r((n+1)^2 - n + 2)
    assert sph.sph_harm(n=2, m=2, az=-0.1, co=0.1, kind=kind) == pytest.approx(
        0.005336028615361
    )

    # n = 3; r = getSH(3, [4, -3], 'real'); r((n+1)^2 - n + 1)
    assert sph.sph_harm(n=3, m=1, az=4, co=-3, kind=kind) == pytest.approx(
        -0.164437684518757
    )

    # n = 17; r = getSH(17, [7, 6], 'real'); r((n+1)^2 - n + 17)
    assert sph.sph_harm(n=17, m=17, az=7, co=6, kind=kind) == pytest.approx(
        3.115120086120583e-10
    )

    # n = 17; r = getSH(17, [6, -7], 'real'); r((n+1)^2 - n - 17)
    assert sph.sph_harm(n=17, m=-17, az=6, co=-7, kind=kind) == pytest.approx(
        6.849456405402495e-04
    )


@pytest.mark.parametrize("kind", ["real_Zotter", "real_AKT", "real_SFS"])
def test_sph_harm_real_alternative(kind):
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
    # AKsh(0, 0, 0, 0, 'real')
    assert sph.sph_harm(n=0, m=0, az=0, co=0, kind=kind) == pytest.approx(
        0.282094791773878
    )

    # AKsh(1, 0, rad2deg(0.1), rad2deg(0.1), 'real')
    assert sph.sph_harm(n=1, m=0, az=0.1, co=0.1, kind=kind) == pytest.approx(
        0.486161534508712
    )

    # AKsh(2, -2, rad2deg(-0.1), rad2deg(-0.1), 'real')
    assert sph.sph_harm(n=2, m=-2, az=-0.1, co=-0.1, kind=kind) == pytest.approx(
        0.001081666550095
    )

    # AKsh(2, 2, rad2deg(-0.1), rad2deg(0.1), 'real')
    assert sph.sph_harm(n=2, m=2, az=-0.1, co=0.1, kind=kind) == pytest.approx(
        0.005336028615361
    )

    # AKsh(3, 1, rad2deg(4), rad2deg(-3), 'real')
    assert sph.sph_harm(n=3, m=1, az=4, co=-3, kind=kind) == pytest.approx(
        -0.164437684518757
    )

    # AKsh(17, 17, rad2deg(7), rad2deg(6), 'real')
    assert sph.sph_harm(n=17, m=17, az=7, co=6, kind=kind) == pytest.approx(
        3.115120086120565e-10
    )

    # AKsh(17, -17, rad2deg(6), rad2deg(-7), 'real')
    assert sph.sph_harm(n=17, m=-17, az=6, co=-7, kind=kind) == pytest.approx(
        -6.849456405402378e-04
    )
