"""
Module containing various spherical harmonics helper functions:

`besselj` / `neumann`
    Bessel function of first and second kind of order n at kr.
`hankel1` / `hankel2`
    Hankel function of first and second kind of order n at kr.
`spbessel` / `spneumann`
    Spherical Bessel function of first and second kind of order n at kr.
`sphankel1` / `sphankel2`
    Spherical Hankel of first and second kind of order n at kr.
`dspbessel` / `dspneumann`
    Derivative of spherical Bessel of first and second kind of order n at kr.
`dsphankel1` / `dsphankel2`
    Derivative spherical Hankel of first and second kind of order n at kr.
`spherical_extrapolation`
    Factor that relate signals recorded on a sphere to it's center.
`array_extrapolation`
    Factor that relate signals recorded on a sphere to it's center. In the
    rigid configuration, a scatter_radius that is different to the array radius
    may be set.
`sph_harm`
    Compute spherical harmonics.
`sph_harm_large`
    Compute spherical harmonics for large orders > 84.
`sph_harm_all`
    Compute all spherical harmonic coefficients up to degree nMax.
`mnArrays`
    Generate degrees n and orders m up to nMax.
`reverseMnIds`
    Generate reverse indexes according to stacked coefficients of orders m up
    to nMax.
`cart2sph` / `sph2cart`
    Convert cartesian to spherical coordinates and vice versa.
`kr`
    Generate kr vector for given f and array radius.
`kr_full_spec`
    Generate full spectrum kr.
"""

import numpy as _np
from scipy import special as scy

from .io import ArrayConfiguration
from .utils import scalar_broadcast_match


def besselj(n, z):
    """Bessel function of first kind of order n at kr. Wraps
    `scipy.special.jn(n, z)`.

    Parameters
    ----------
    n : array_like
        Order
    z: array_like
        Argument

    Returns
    -------
    J : array_like
        Values of Bessel function of order n at position z
    """
    return scy.jv(n, _np.complex_(z))


def neumann(n, z):
    """Bessel function of second kind (Neumann / Weber function) of order n at
    kr. Implemented as `(hankel1(n, z) - besselj(n, z)) / 1j`.

    Parameters
    ----------
    n : array_like
        Order
    z: array_like
        Argument

    Returns
    -------
    Y : array_like
        Values of Hankel function of order n at position z
    """
    return (hankel1(n, z) - besselj(n, z)) / 1j


def hankel1(n, z):
    """Hankel function of first kind of order n at kr. Wraps
    `scipy.special.hankel2(n, z)`.

    Parameters
    ----------
    n : array_like
        Order
    z: array_like
        Argument

    Returns
    -------
    H1 : array_like
        Values of Hankel function of order n at position z
    """
    return scy.hankel1(n, z)


def hankel2(n, z):
    """Hankel function of second kind of order n at kr. Wraps
    `scipy.special.hankel2(n, z)`.

    Parameters
    ----------
    n : array_like
        Order
    z: array_like
        Argument

    Returns
    -------
    H2 : array_like
        Values of Hankel function of order n at position z
    """
    return scy.hankel2(n, z)


def spbessel(n, kr):
    """Spherical Bessel function (first kind) of order n at kr.

    Parameters
    ----------
    n : array_like
        Order
    kr: array_like
        Argument

    Returns
    -------
    J : complex float
        Spherical Bessel
    """
    n, kr = scalar_broadcast_match(n, kr)

    if _np.any(n < 0) | _np.any(kr < 0) | _np.any(_np.mod(n, 1) != 0):
        J = _np.zeros(kr.shape, dtype=_np.complex_)

        kr_non_zero = kr != 0
        J[kr_non_zero] = _np.lib.scimath.sqrt(_np.pi / 2 / kr[kr_non_zero]) * besselj(
            n[kr_non_zero] + 0.5, kr[kr_non_zero]
        )
        J[_np.logical_and(kr == 0, n == 0)] = 1
    else:
        J = scy.spherical_jn(n.astype(_np.int_), kr)
    return _np.squeeze(J)


def spneumann(n, kr):
    """Spherical Neumann (Bessel second kind) of order n at kr.

    Parameters
    ----------
    n : array_like
        Order
    kr: array_like
        Argument

    Returns
    -------
    Yv : complex float
        Spherical Neumann (Bessel second kind)
    """
    n, kr = scalar_broadcast_match(n, kr)

    if _np.any(n < 0) | _np.any(_np.mod(n, 1) != 0):
        Yv = _np.full(kr.shape, _np.nan, dtype=_np.complex_)

        kr_non_zero = kr != 0
        Yv[kr_non_zero] = _np.lib.scimath.sqrt(_np.pi / 2 / kr[kr_non_zero]) * neumann(
            n[kr_non_zero] + 0.5, kr[kr_non_zero]
        )
        Yv[kr < 0] = -Yv[kr < 0]
    else:
        Yv = scy.spherical_yn(n.astype(_np.int_), kr)
        # return possible infs as nan to stay consistent
        Yv[_np.isinf(Yv)] = _np.nan
    return _np.squeeze(Yv)


def sphankel1(n, kr):
    """Spherical Hankel (first kind) of order n at kr.

    Parameters
    ----------
    n : array_like
        Order
    kr: array_like
        Argument

    Returns
    -------
    hn1 : complex float
        Spherical Hankel function hn (first kind)
    """
    n, kr = scalar_broadcast_match(n, kr)
    hn1 = _np.full(n.shape, _np.nan, dtype=_np.complex_)
    kr_nonzero = kr != 0
    hn1[kr_nonzero] = (
        _np.sqrt(_np.pi / 2)
        / _np.lib.scimath.sqrt(kr[kr_nonzero])
        * hankel1(n[kr_nonzero] + 0.5, kr[kr_nonzero])
    )
    return hn1


def sphankel2(n, kr):
    """Spherical Hankel (second kind) of order n at kr

    Parameters
    ----------
    n : array_like
       Order
    kr: array_like
       Argument

    Returns
    -------
    hn2 : complex float
       Spherical Hankel function hn (second kind)
    """
    n, kr = scalar_broadcast_match(n, kr)
    hn2 = _np.full(n.shape, _np.nan, dtype=_np.complex_)
    kr_nonzero = kr != 0
    hn2[kr_nonzero] = (
        _np.sqrt(_np.pi / 2)
        / _np.lib.scimath.sqrt(kr[kr_nonzero])
        * hankel2(n[kr_nonzero] + 0.5, kr[kr_nonzero])
    )
    return hn2


def dspbessel(n, kr):
    """Derivative of spherical Bessel (first kind) of order n at kr.

    Parameters
    ----------
    n : array_like
        Order
    kr: array_like
        Argument

    Returns
    -------
    J' : complex float
        Derivative of spherical Bessel
    """
    return _np.squeeze(
        (n * spbessel(n - 1, kr) - (n + 1) * spbessel(n + 1, kr)) / (2 * n + 1)
    )


def dspneumann(n, kr):
    """Derivative spherical Neumann (Bessel second kind) of order n at kr.

    Parameters
    ----------
    n : array_like
        Order
    kr: array_like
        Argument

    Returns
    -------
    Yv' : complex float
        Derivative of spherical Neumann (Bessel second kind)
    """
    n, kr = scalar_broadcast_match(n, kr)
    if _np.any(n < 0) | _np.any(_np.mod(n, 1) != 0) | _np.any(_np.mod(kr, 1) != 0):
        return spneumann(n, kr) * n / kr - spneumann(n + 1, kr)
    else:
        return scy.spherical_yn(
            n.astype(_np.int_), kr.astype(_np.complex_), derivative=True
        )


def dsphankel1(n, kr):
    """Derivative spherical Hankel (first kind) of order n at kr.

    Parameters
    ----------
    n : array_like
        Order
    kr: array_like
        Argument

    Returns
    -------
    dhn1 : complex float
        Derivative of spherical Hankel function hn' (second kind)
    """
    n, kr = scalar_broadcast_match(n, kr)
    dhn1 = _np.full(n.shape, _np.nan, dtype=_np.complex_)
    kr_nonzero = kr != 0
    dhn1[kr_nonzero] = 0.5 * (
        sphankel1(n[kr_nonzero] - 1, kr[kr_nonzero])
        - sphankel1(n[kr_nonzero] + 1, kr[kr_nonzero])
        - sphankel1(n[kr_nonzero], kr[kr_nonzero]) / kr[kr_nonzero]
    )
    return dhn1


def dsphankel2(n, kr):
    """Derivative spherical Hankel (second kind) of order n at kr.

    Parameters
    ----------
    n : array_like
        Order
    kr: array_like
        Argument

    Returns
    -------
    dhn2 : complex float
        Derivative of spherical Hankel function hn' (second kind)
    """
    n, kr = scalar_broadcast_match(n, kr)
    dhn2 = _np.full(n.shape, _np.nan, dtype=_np.complex_)
    kr_nonzero = kr != 0
    dhn2[kr_nonzero] = 0.5 * (
        sphankel2(n[kr_nonzero] - 1, kr[kr_nonzero])
        - sphankel2(n[kr_nonzero] + 1, kr[kr_nonzero])
        - sphankel2(n[kr_nonzero], kr[kr_nonzero]) / kr[kr_nonzero]
    )
    return dhn2


def spherical_extrapolation(
    order, array_configuration, k_mic, k_scatter=None, k_dual=None
):
    """Factor that relate signals recorded on a sphere to it's center.

    Parameters
    ----------
    order : int
        Order
    array_configuration : io.ArrayConfiguration
        List/Tuple/ArrayConfiguration, see io.ArrayConfiguration
    k_mic : array_like
        K vector for microphone array
    k_scatter: array_like, optional
        K vector for scatterer  [Default: same as k_mic]
    k_dual : float, optional
        Radius of second array, required for `array_type` == 'dual'

    Returns
    -------
    b : array, complex
    """
    array_configuration = ArrayConfiguration(*array_configuration)

    if array_configuration.array_type == "open":
        if array_configuration.transducer_type == "omni":
            return bn_open_omni(order, k_mic)
        elif array_configuration.transducer_type == "cardioid":
            return bn_open_cardioid(order, k_mic)
    elif array_configuration.array_type == "rigid":
        if array_configuration.transducer_type == "omni":
            return bn_rigid_omni(order, k_mic, k_scatter)
        elif array_configuration.transducer_type == "cardioid":
            return bn_rigid_cardioid(order, k_mic, k_scatter)
    elif array_configuration.array_type == "dual":
        return bn_dual_open_omni(order, k_mic, k_dual)


def array_extrapolation(order, freqs, array_configuration, normalize=True):
    """Factor that relate signals recorded on a sphere to it's center. In the
    rigid configuration, a scatter_radius that is different to the array radius
    may be set.

    Parameters
    ----------
    order : int
        Order
    freqs : array_like
        Frequencies
    array_configuration : io.ArrayConfiguration
        List/Tuple/ArrayConfiguration, see io.ArrayConfiguration
    normalize: bool, optional
        Normalize by 4 * pi * 1j ** order [Default: True]

    Returns
    -------
    b : array, complex
        Coefficients of shape [nOrder x nFreqs]
    """
    array_configuration = ArrayConfiguration(*array_configuration)

    freqs, order = _np.meshgrid(freqs, order)
    k_mic = kr(freqs, array_configuration.array_radius)
    k_scatter = None
    k_dual = None

    if normalize:
        scale_factor = _np.squeeze(4 * _np.pi * 1j ** order)
    else:
        scale_factor = 1

    if array_configuration.array_type == "open":
        k_scatter = None
    elif array_configuration.array_type == "rigid":
        if array_configuration.scatter_radius is None:
            scatter_radius = array_configuration.array_radius
        else:
            scatter_radius = array_configuration.scatter_radius
        k_scatter = kr(freqs, scatter_radius)

        # Replace leading k==0 with the next element to avoid nan
        if _np.any(k_mic[:, 0] == 0):
            k_mic[:, 0] = k_mic[:, 1]
        if _np.any(k_scatter[:, 0] == 0):
            k_scatter[:, 0] = k_scatter[:, 1]
    elif array_configuration.array_type == "dual":
        k_dual = kr(freqs, array_configuration.dual_radius)

    return scale_factor * spherical_extrapolation(
        order, array_configuration, k_mic, k_scatter, k_dual
    )


def bn_open_omni(n, krm):
    return spbessel(n, krm)


def bn_open_cardioid(n, krm):
    return 0.5 * (spbessel(n, krm) - 1j * dspbessel(n, krm))


def bn_rigid_omni(n, krm, krs):
    # krm: for mic radius, krs: for sphere radius
    krm, krs = scalar_broadcast_match(krm, krs)
    return spbessel(n, krm) - (
        (dspbessel(n, krs) / dsphankel2(n, krs)) * sphankel2(n, krm)
    )
    # # for krm == krs one could use
    # return 1j / (krm ** 2 * dsphankel2(n, krm))


def bn_rigid_cardioid(n, krm, krs):
    # Reference for Filter design for rigid sphere with cardioid microphones:
    # P. Plessas, F. Zotter: Microphone arrays around rigid spheres for spatial
    # recording and holography, DAGA 2010
    # krm: for mic radius, krs: for sphere radius
    krm, krs = scalar_broadcast_match(krm, krs)
    return (
        spbessel(n, krm)
        - 1j * dspbessel(n, krm)
        + (1j * dsphankel2(n, krm) - sphankel2(n, krm))
        * (dspbessel(n, krs) / dsphankel2(n, krs))
    )


def bn_dual_open_omni(n, kr1, kr2):
    # Reference: Rafaely et al, "High-resolution plane-wave decomposition in
    # an auditorium using a dual-radius scanning spherical microphone array"
    # JASA, 122(5), 2007
    # kr1, kr2 are the kr values of the two different microphone spheres
    # Implementation by Nils Peters, November 2011*/
    bn1 = bn_open_omni(n, kr1)
    bn2 = bn_open_omni(n, kr2)
    return _np.where(_np.abs(bn1) >= _np.abs(bn2), bn1, bn2)


def sph_harm(m, n, az, co, kind="complex"):
    """Compute spherical harmonics.

    Parameters
    ----------
    m : (int)
        Order of the spherical harmonic. abs(m) <= n
    n : (int)
        Degree of the harmonic, sometimes called l. n >= 0
    az : (float)
        Azimuthal (longitudinal) coordinate [0, 2pi], also called Theta.
    co : (float)
        Polar (colatitudinal) coordinate [0, pi], also called Phi.
    kind : {'complex', 'complex_SciPy', 'complex_SHT', 'complex_spaudiopy',
        'complex_AKT', 'complex_GumDur', 'complex_SFS', 'real', 'real_SHT',
        'real_spaudiopy', 'real_Zotter', 'real_AKT', 'real_SFS'}, optional
        Spherical harmonic coefficients' data type according to conventions /
        definitions referenced in the Note below. [Default: 'complex']

    Returns
    -------
    y_mn : (complex float) or (float)
        Spherical harmonic of order m and degree n, sampled at theta = az,
        phi = co

    Notes
    -----
    The different spherical harmonic conventions are used for example in:
    'complex' [5]_ == 'complex_SciPy' [10]_ == 'complex_SHT' [11]_ ==
    'complex_spaudiopy' [12]_ == 'complex_AKT' [13]_;
    vs. 'complex_GumDur' [7]_ == 'complex_SFS' [14]_;
    vs. 'real' [8]_ == 'real_SHT' [11]_ == 'real_spaudiopy' [12]_;
    vs. 'real_Zotter' [9]_ == 'real_AKT' [13]_ == 'real_SFS' [14]_.

    References
    ----------
    .. [7] Gumerov, N. A., and Duraiswami, R. (2005). Fast Multipole Methods for
        the Helmholtz Equation in Three Dimensions, Elsevier Science,
        Amsterdam, NL, 520 pages. doi:10.1016/B978-0-08-044371-3.X5000-5
    .. [8] Williams, E. G. (1999). Fourier Acoustics: Sound Radiation and
        Nearfield Acoustical Holography, (E. G. Williams, Ed.) Academic Press,
        London, UK, 1st ed., 1–306 pages. doi:10.1016/B978-012753960-7/50001-2
    .. [9] Zotter, F. (2009). Analysis and Synthesis of Sound-Radiation with
        Spherical Arrays University of Music and Performing Arts Graz, Austria,
        192 pages.
    .. [10] https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sph_harm.html
    .. [11] https://github.com/polarch/Spherical-Harmonic-Transform/blob/master/inverseSHT.m
    .. [12] https://github.com/chris-hld/spaudiopy/blob/master/spaudiopy/sph.py
    .. [13] Brinkmann, F., and Weinzierl, S. (2017). “AKtools - An Open Software
        Toolbox for Signal Acquisition, Processing, and Inspection in
        Acoustics,” AES Conv. 142, Audio Engineering Society, Berlin,
        Germany, 1–6.
    .. [14] https://github.com/JensAhrens/soundfieldsynthesis/blob/master/Common/spherical/sphharm.m
    """
    # SAFETY CHECKS
    kind = kind.lower()
    if kind not in [
        "complex",
        "complex_scipy",
        "complex_sht",
        "complex_spaudiopy",
        "complex_akt",
        "complex_gumdur",
        "complex_sfs",
        "real",
        "real_sht",
        "real_spaudiopy",
        "real_zotter",
        "real_akt",
        "real_sfs",
    ]:
        raise ValueError(f'Invalid kind "{kind}".')

    if "complex" in kind:
        Y = _np.asarray(scy.sph_harm(m, n, az, co))
        if kind in ["complex_gumdur", "complex_sfs"]:
            # apply Condon-Shortley phase also for positive m
            mg0 = m > 0
            _np.multiply(Y, _np.float_power(-1.0, m, where=mg0), out=Y, where=mg0)
        return Y

    else:  # "real"
        Y = _np.asarray(scy.sph_harm(abs(m), n, az, co))
        _np.multiply(Y, _np.sqrt(2), out=Y, where=m > 0)
        _np.multiply(Y.imag, _np.sqrt(2), out=Y, where=m < 0, dtype=Y.dtype)
        if kind in ["real_zotter", "real_akt", "real_sfs"]:
            _np.multiply(Y, -1, out=Y, where=m < 0)  # negate for negative m
        return _np.float_power(-1.0, m) * Y.real


def sph_harm_large(m, n, az, co, kind="complex", force_large=False):
    """Compute spherical harmonics for large orders > 84.

    Parameters
    ----------
    m : (int)
        Order of the spherical harmonic. abs(m) <= n
    n : (int)
        Degree of the harmonic, sometimes called l. n >= 0
    az : (float)
        Azimuthal (longitudinal) coordinate [0, 2pi], also called Theta.
    co : (float)
        Polar (colatitudinal) coordinate [0, pi], also called Phi.
    kind : {'complex', 'complex_GumDur', 'real', 'real_Zotter'}, optional
        Spherical harmonic coefficients' data type, see `sph.sph_harm()` for
        notes on the different conventions [Default: 'complex']
    force_large: bool, optional
        If the computation method for large orders should be forced even if a
        small order is given. [Default: False]

    Returns
    -------
    y_mn : (complex float) or (float)
        Spherical harmonic of order m and degree n, sampled at
        theta = az,
        phi = co

    Notes
    -----
    Y_n,m (theta, phi) = ((n - m)! * (2l + 1)) / (4pi * (l + m))^0.5
    * exp(i m phi) * P_n^m(cos(theta)) as per https://dlmf.nist.gov/14.30
    Pmn(z) are the associated Legendre functions of the first kind,
    like `scipy.special.lpmv()`, which calculates P(0...m 0...n) and its
    derivative but won't return +inf at high orders.
    """
    if not force_large and _np.all(_np.abs(m) < 84):
        return sph_harm(m=m, n=n, az=az, co=co, kind=kind)
    else:
        # SAFETY CHECKS
        kind = kind.lower()
        if kind not in [
            "complex",
            "complex_scipy",
            "complex_sht",
            "complex_spaudiopy",
            "complex_akt",
            "complex_gumdur",
            "complex_sfs",
            "real",
            "real_sht",
            "real_spaudiopy",
            "real_zotter",
            "real_akt",
            "real_sfs",
        ]:
            raise ValueError(f'Invalid kind "{kind}".')

        m = _np.atleast_1d(m)
        n = _np.atleast_1d(n)
        az = _np.atleast_1d(az)
        co = _np.atleast_1d(co)
        if not (m.shape == n.shape == az.shape == co.shape):
            raise ValueError(
                "This function only handles input data of identical shape."
            )

        abs_m = abs(m)
        P = _np.empty(co.shape)
        # noinspection PyTypeChecker
        for i in _np.ndindex(P.shape):
            P[i] = scy.lpmn(abs_m[i], n[i], _np.cos(co[i]))[0][-1][-1]
        # P = scy.lpmv(abs_m, n, _np.cos(co))

        preFactor1 = _np.sqrt((2 * n + 1) / (4 * _np.pi))
        try:
            preFactor2 = _np.sqrt(scy.factorial(n - abs_m) / scy.factorial(n + abs_m))
        except OverflowError:  # integer division for very large orders
            preFactor2 = _np.sqrt(scy.factorial(n - abs_m) // scy.factorial(n + abs_m))
        Y = preFactor1 * preFactor2 * P

        if "complex" in kind:
            _np.multiply(Y, _np.float_power(-1.0, m, where=m < 0), out=Y, where=m < 0)
            if kind in ["complex_gumdur", "complex_sfs"]:
                # apply Condon-Shortley phase also for positive m
                _np.multiply(
                    Y, _np.float_power(-1.0, m, where=m > 0), out=Y, where=m > 0
                )
            return Y * _np.exp(1j * m * az)

        else:  # "real"
            _np.multiply(Y, _np.sqrt(2) * _np.cos(m * az), out=Y, where=m > 0)
            _np.multiply(Y, _np.sqrt(2) * _np.sin(abs_m * az), out=Y, where=m < 0)
            if kind in ["real_zotter", "real_akt", "real_sfs"]:
                _np.multiply(Y, -1, out=Y, where=m < 0)  # negate for negative m
            return _np.float_power(-1.0, m) * Y.real


def sph_harm_all(nMax, az, co, kind="complex"):
    """Compute all spherical harmonic coefficients up to degree nMax.

    Parameters
    ----------
    nMax : (int)
        Maximum degree of coefficients to be returned. n >= 0
    az: (float), array_like
        Azimuthal (longitudinal) coordinate [0, 2pi], also called Theta.
    co : (float), array_like
        Polar (colatitudinal) coordinate [0, pi], also called Phi.
    kind : {'complex', 'real'}, optional
        Spherical harmonic coefficients' data type [Default: 'complex']

    Returns
    -------
    y_mn : (complex float) or (float), array_like
        Spherical harmonics of degrees n [0 ... nMax] and all corresponding
        orders m [-n ... n], sampled at [az, co]. dim1 corresponds to az/co
        pairs, dim2 to oder/degree (m, n) pairs like 0/0, -1/1, 0/1, 1/1,
        -2/2, -1/2 ...
    """
    m, n = mnArrays(nMax)
    mA, azA = _np.meshgrid(m, az)
    nA, coA = _np.meshgrid(n, co)
    return sph_harm(m=mA, n=nA, az=azA, co=coA, kind=kind)


def mnArrays(nMax):
    """Generate degrees n and orders m up to nMax.

    Parameters
    ----------
    nMax : (int)
        Maximum degree of coefficients to be returned. n >= 0

    Returns
    -------
    m : (int), array_like
        0, -1, 0, 1, -2, -1, 0, 1, 2, ... , -nMax ..., nMax
    n : (int), array_like
        0, 1, 1, 1, 2, 2, 2, 2, 2, ... nMax, nMax, nMax
    """
    # Degree n = 0, 1, 1, 1, 2, 2, 2, 2, 2, ...
    degs = _np.arange(nMax + 1)
    n = _np.repeat(degs, degs * 2 + 1)

    # Order m = 0, -1, 0, 1, -2, -1, 0, 1, 2, ...
    # http://oeis.org/A196199
    elementNumber = _np.arange((nMax + 1) ** 2) + 1
    t = _np.floor(_np.sqrt(elementNumber - 1)).astype(int)
    m = elementNumber - t * t - t - 1

    return m, n


def reverseMnIds(nMax):
    """Generate reverse indexes according to stacked coefficients of orders m up
    to nMax.

    Parameters
    ----------
    nMax : (int)
        Maximum degree of coefficients reverse indexes to be returned. n >= 0

    Returns
    -------
    rev_ids : (int), array_like
        0, 3, 2, 1, 8, 7, 6, 5, 4, ...
    """
    m_ids = list(range(nMax * (nMax + 2) + 1))
    for o in range(nMax + 1):
        id_start = o ** 2
        id_end = id_start + o * 2 + 1
        m_ids[id_start:id_end] = reversed(m_ids[id_start:id_end])
    return m_ids


def cart2sph(x, y, z):
    """Convert cartesian coordinates x, y, z to spherical coordinates
    az, el, r.
    """
    hxy = _np.hypot(x, y)
    r = _np.hypot(hxy, z)
    el = _np.arctan2(z, hxy)
    az = _np.arctan2(y, x)
    return az, el, r


def sph2cart(az, el, r):
    """Convert spherical coordinates az, el, r to cartesian coordinates
    x, y, z.
    """
    rcos_theta = r * _np.cos(el)
    x = rcos_theta * _np.cos(az)
    y = rcos_theta * _np.sin(az)
    z = r * _np.sin(el)
    return x, y, z


def kr(f, radius, temperature=20):
    """Generate kr vector for given f and array radius.

    Parameters
    ----------
    f : array_like
       Frequencies to calculate the kr for
    radius : float
       Radius of array
    temperature : float, optional
       Room temperature in degree Celsius [Default: 20]

    Returns
    -------
    kr : array_like
       2 * pi * f / c(temperature) * r
    """
    return 2 * _np.pi * f / (331.5 + 0.6 * temperature) * radius


def kr_full_spec(fs, radius, NFFT, temperature=20):
    """Generate full spectrum kr.

    Parameters
    ----------
    fs : int
       Sampling rate in Hertz
    radius : float
       Radius
    NFFT : int
       Number of frequency bins
    temperature : float, optional
       Temperature in degree Celsius [Default: 20 C]

    Returns
    -------
    kr : array_like
       kr vector of length NFFT/2 + 1 spanning the frequencies of 0:fs/2
    """
    freqs = _np.linspace(0, fs / 2, int(NFFT / 2 + 1))
    return kr(freqs, radius, temperature)


# DEBUG
def _print_bessel_functions(n, k):
    print("bessel:", besselj(n, k))
    print("hankel2:", hankel2(n, k))
    print("spbessel:", spbessel(n, k))
    print("dspbessel:", dspbessel(n, k))
    print("spneumann:", spneumann(n, k))
    print("dspneumann:", dspneumann(n, k))
    print("sphankel2:", sphankel2(n, k))
    print("dsphankel2:", dsphankel2(n, k))


def _print_mic_scaling(N, freqs, array_radius, scatter_radius=None, dual_radius=None):
    if not scatter_radius:
        scatter_radius = array_radius
    if not dual_radius:
        dual_radius = array_radius
    print(
        "bn_open_omni:", array_extrapolation(N, freqs, [array_radius, "open", "omni"])
    )
    print(
        "bn_open_cardioid:",
        array_extrapolation(N, freqs, [array_radius, "open", "cardioid"]),
    )
    print(
        "bn_rigid_omni:",
        array_extrapolation(N, freqs, [array_radius, "rigid", "omni", scatter_radius]),
    )
    print(
        "bn_rigid_cardioid:",
        array_extrapolation(
            N, freqs, [array_radius, "rigid", "cardioid", scatter_radius]
        ),
    )
    print(
        "bn_dual_open_omni:",
        array_extrapolation(
            N, freqs, [array_radius, "dual", "omni", None, dual_radius]
        ),
    )


def _print_bns(N, k_mic, k_scatter):
    print("bn_open_omni:", bn_open_omni(N, k_mic) * 4 * _np.pi * 1j ** N)
    print("bn_open_cardioid:", bn_open_cardioid(N, k_mic) * 4 * _np.pi * 1j ** N),
    print("bn_rigid_omni:", bn_rigid_omni(N, k_mic, k_scatter) * 4 * _np.pi * 1j ** N),
    print(
        "bn_rigid_cardioid:",
        bn_rigid_cardioid(N, k_mic, k_scatter) * 4 * _np.pi * 1j ** N,
    )
    print(
        "bn_dual_open_omni:",
        bn_dual_open_omni(N, k_mic, k_scatter) * 4 * _np.pi * 1j ** N,
    )
