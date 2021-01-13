# -*- coding: utf-8 -*-
"""
Collection of spherical helper functions:

`sph_harm`
   More robust spherical harmonic basis functions
`spbessel / dspbessel`
   Spherical Bessel and derivative
`spneumann / dspneumann`
   Spherical Neumann (Bessel 2nd kind) and derivative
`sphankel / dsphankel`
   Spherical Hankel (second kind) and derivative
`cart2sph / sph2cart`
   Convert cartesian to spherical coordinates and vice versa
"""

from math import factorial as fact

import numpy as _np
from scipy import special as scy

from .io import ArrayConfiguration
from .utils import scalar_broadcast_match


def besselj(n, z):
    """Bessel function of first kind of order n at kr.
    Wraps scipy.special.jn(n, z).

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
    """Bessel function of second kind (Neumann / Weber function) of order n at kr.
    Implemented as (hankel1(n, z) - besselj(n, z)) / 1j

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
    """Bessel function of third kind (Hankel function) of order n at kr.
    Wraps scipy.special.hankel1(n, z)

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
    """Bessel function of third kind (Hankel function) of order n at kr.
    Wraps scipy.special.hankel2(n, z)

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
    """Spherical Bessel function (first kind) of order n at kr

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
        J[kr_non_zero] = _np.lib.scimath.sqrt(_np.pi / 2 / kr[kr_non_zero]) * besselj(n[kr_non_zero] + 0.5,
                                                                                      kr[kr_non_zero])
        J[_np.logical_and(kr == 0, n == 0)] = 1
    else:
        J = scy.spherical_jn(n.astype(_np.int), kr)
    return _np.squeeze(J)


def spneumann(n, kr):
    """Spherical Neumann (Bessel second kind) of order n at kr

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
        Yv[kr_non_zero] = _np.lib.scimath.sqrt(_np.pi / 2 / kr[kr_non_zero]) * neumann(n[kr_non_zero] + 0.5,
                                                                                       kr[kr_non_zero])
        Yv[kr < 0] = -Yv[kr < 0]
    else:
        Yv = scy.spherical_yn(n.astype(_np.int), kr)
        Yv[_np.isinf(Yv)] = _np.nan  # return possible infs as nan to stay consistent
    return _np.squeeze(Yv)


def sphankel1(n, kr):
    """Spherical Hankel (first kind) of order n at kr

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
    hn1[kr_nonzero] = _np.sqrt(_np.pi / 2) / _np.lib.scimath.sqrt(kr[kr_nonzero]) * hankel1(n[kr_nonzero] + 0.5,
                                                                                            kr[kr_nonzero])
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
    hn2[kr_nonzero] = _np.sqrt(_np.pi / 2) / _np.lib.scimath.sqrt(kr[kr_nonzero]) * hankel2(n[kr_nonzero] + 0.5,
                                                                                            kr[kr_nonzero])
    return hn2


def dspbessel(n, kr):
    """Derivative of spherical Bessel (first kind) of order n at kr

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
    return _np.squeeze((n * spbessel(n - 1, kr) - (n + 1) * spbessel(n + 1, kr)) / (2 * n + 1))


def dspneumann(n, kr):
    """Derivative spherical Neumann (Bessel second kind) of order n at kr

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
        return scy.spherical_yn(n.astype(_np.int), kr.astype(_np.complex), derivative=True)


def dsphankel1(n, kr):
    """Derivative spherical Hankel (first kind) of order n at kr

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
            sphankel1(n[kr_nonzero] - 1, kr[kr_nonzero]) - sphankel1(n[kr_nonzero] + 1, kr[kr_nonzero])
            - sphankel1(n[kr_nonzero], kr[kr_nonzero]) / kr[kr_nonzero])
    return dhn1


def dsphankel2(n, kr):
    """Derivative spherical Hankel (second kind) of order n at kr

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
            sphankel2(n[kr_nonzero] - 1, kr[kr_nonzero]) - sphankel2(n[kr_nonzero] + 1, kr[kr_nonzero])
            - sphankel2(n[kr_nonzero], kr[kr_nonzero]) / kr[kr_nonzero])
    return dhn2


def spherical_extrapolation(order, array_configuration, k_mic, k_scatter=None,
                            k_dual=None):
    """ Factor that relate signals recorded on a sphere to it's center.

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

    if array_configuration.array_type == 'open':
        if array_configuration.transducer_type == 'omni':
            return bn_open_omni(order, k_mic)
        elif array_configuration.transducer_type == 'cardioid':
            return bn_open_cardioid(order, k_mic)
    elif array_configuration.array_type == 'rigid':
        if array_configuration.transducer_type == 'omni':
            return bn_rigid_omni(order, k_mic, k_scatter)
        elif array_configuration.transducer_type == 'cardioid':
            return bn_rigid_cardioid(order, k_mic, k_scatter)
    elif array_configuration.array_type == 'dual':
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
    normalize: Bool, optional
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

    if array_configuration.array_type == 'open':
        k_scatter = None
    elif array_configuration.array_type == 'rigid':
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
    elif array_configuration.array_type == 'dual':
        k_dual = kr(freqs, array_configuration.dual_radius)

    return scale_factor * spherical_extrapolation(order, array_configuration,
                                                  k_mic, k_scatter, k_dual)


def bn_open_omni(n, krm):
    return spbessel(n, krm)


def bn_open_cardioid(n, krm):
    return 0.5 * (spbessel(n, krm) - 1j * dspbessel(n, krm))


def bn_rigid_omni(n, krm, krs):
    krm, krs = scalar_broadcast_match(krm, krs)
    return spbessel(n, krm) - (dspbessel(n, krs) / dsphankel2(n, krs)) * sphankel2(n, krm)


def bn_rigid_cardioid(n, krm, krs):
    #  Rerence for Filter design for rigid sphere with cardioid microphones:
    #  P. Plessas, F. Zotter: Microphone arrays around rigid spheres for spatial recording and holography, DAGA 2010
    #  krm: for mic radius, krs: for sphere radius
    krm, krs = scalar_broadcast_match(krm, krs)
    return spbessel(n, krm) - 1j * dspbessel(n, krm) + (1j * dsphankel2(n, krm) - sphankel2(n, krm)) * (
            dspbessel(n, krs) / dsphankel2(n, krs))


def bn_dual_open_omni(n, kr1, kr2):
    # Reference: Rafaely et al,
    #  High-resolution plane-wave decomposition in an auditorium using a dual-radius scanning spherical microphone array
    #  JASA, 122(5), 2007
    # kr1, kr2 are the kr values of the two different microphone spheres
    # Implementation by Nils Peters, November 2011*/
    bn1 = bn_open_omni(n, kr1)
    bn2 = bn_open_omni(n, kr2)
    return _np.where(_np.abs(bn1) >= _np.abs(bn2), bn1, bn2)


def sph_harm(m, n, az, co, kind='complex'):
    """Compute spherical harmonics

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
    kind : {'complex', 'real'}, optional
        Spherical harmonic coefficients data type according to complex [1] or
        real definition [2] [Default: 'complex']

    Returns
    -------
    y_mn : (complex float) or (float)
        Spherical harmonic of order m and degree n, sampled at theta = az,
        phi = co

    References
    ----------
    [1] scipy.special.sph_harm
    [2] F. Zotter, “Analysis and synthesis of sound-radiation with spherical
        arrays,” University of Music and Performing Arts, 2009.
    """
    # SAFETY CHECKS
    kind = kind.lower()
    if kind not in {'complex', 'real'}:
        raise ValueError('Invalid kind: Choose either complex or real.')
    m = _np.atleast_1d(m)

    Y = scy.sph_harm(m, n, az, co)
    if kind == 'complex':
        return Y
    else:  # kind == 'real'
        Y[_np.where(m > 0)] = (_np.float_power(-1.0, m)[_np.where(m > 0)]
                               * _np.sqrt(2) * _np.real(Y[_np.where(m > 0)]))
        Y[_np.where(m == 0)] = _np.real(Y[_np.where(m == 0)])
        Y[_np.where(m < 0)] = _np.sqrt(2) * _np.imag(Y[_np.where(m < 0)])
        return _np.real(Y)


def sph_harm_large(m, n, az, co, kind='complex'):
    """Compute spherical harmonics for large orders > 84

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
    kind : {'complex', 'real'}, optional
        Spherical harmonic coefficients data type according to complex [1] or
        real definition [2] [Default: 'complex']

    Returns
    -------
    y_mn : (complex float) or (float)
        Spherical harmonic of order m and degree n, sampled at
        theta = az,
        phi = co

    Notes
    -----
    Y_n,m (theta, phi) = ((n - m)! * (2l + 1)) / (4pi * (l + m))^0.5
    * exp(i m phi) * P_n^m(cos(theta)) as per http://dlmf.nist.gov/14.30
    Pmn(z) are the associated Legendre functions of the first kind,
    like scipy.special.lpmv, which calculates P(0...m 0...n) and its derivative
    but won't return +inf at high orders.

    References
    ----------
    [1] scipy.special.sph_harm and scipy.special.lpmv
    [2] F. Zotter, “Analysis and synthesis of sound-radiation with spherical
        arrays,” University of Music and Performing Arts, 2009.
    """
    if _np.all(_np.abs(m) < 84):
        return sph_harm(m, n, az, co, kind=kind)
    else:
        # SAFETY CHECKS
        kind = kind.lower()
        if kind not in {'complex', 'real'}:
            raise ValueError('Invalid kind: Choose either complex or real.')
        m = _np.atleast_1d(m)

        # TODO: confirm that this uses the correct SH definition
        mAbs = _np.abs(m)
        if isinstance(co, _np.ndarray):
            P = _np.empty(co.size)
            for k in range(0, co.size):
                P[k] = scy.lpmn(mAbs, n, _np.cos(co[k]))[0][-1][-1]
        else:
            P = scy.lpmn(mAbs, n, _np.cos(co))[0][-1][-1]
        preFactor1 = _np.sqrt((2 * n + 1) / (4 * _np.pi))
        try:
            preFactor2 = _np.sqrt(fact(n - mAbs) / fact(n + mAbs))
        except OverflowError:  # integer division for very large orders
            preFactor2 = _np.sqrt(fact(n - mAbs) // fact(n + mAbs))

        Y = preFactor1 * preFactor2 * _np.exp(1j * m * az) * P
        if kind == 'complex':
            return Y if m >= 0 else _np.conj(Y)
        else:  # kind == 'real'
            Y[_np.where(m > 0)] = (_np.float_power(-1.0, m)[_np.where(m > 0)]
                                   * _np.sqrt(2)
                                   * _np.real(Y[_np.where(m > 0)]))
            Y[_np.where(m == 0)] = _np.real(Y[_np.where(m == 0)])
            Y[_np.where(m < 0)] = _np.sqrt(2) * _np.real(Y[_np.where(m < 0)])
            return _np.real(Y)


def sph_harm_all(nMax, az, co, kind='complex'):
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
        Spherical harmonic coefficients data type [Default: 'complex']

    Returns
    -------
    y_mn : (complex float) or (float), array_like
        Spherical harmonics of degrees n [0 ... nMax] and all orresponding
        orders m [-n ... n], sampled at [az, co]. dim1 corresponds to az/co
        pairs, dim2 to oder/degree (m, n) pairs like 0/0, -1/1, 0/1, 1/1,
        -2/2, -1/2 ...
    """
    m, n = mnArrays(nMax)
    mA, azA = _np.meshgrid(m, az)
    nA, coA = _np.meshgrid(n, co)
    return sph_harm(mA, nA, azA, coA, kind=kind)


def mnArrays(nMax):
    """Returns degrees n and orders m up to nMax.

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
    """Returns reverse indexes according to stacked coefficients of orders m up to nMax.

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
    """Converts cartesian coordinates x, y, z to spherical coordinates az, el, r."""
    hxy = _np.hypot(x, y)
    r = _np.hypot(hxy, z)
    el = _np.arctan2(z, hxy)
    az = _np.arctan2(y, x)
    return az, el, r


def sph2cart(az, el, r):
    """Converts spherical coordinates az, el, r to cartesian coordinates x, y, z."""
    rcos_theta = r * _np.cos(el)
    x = rcos_theta * _np.cos(az)
    y = rcos_theta * _np.sin(az)
    z = r * _np.sin(el)
    return x, y, z


def kr(f, radius, temperature=20):
    """Return kr vector for given f and array radius

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
    """Returns full spectrum kr

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
    print(' '.join(('bessel:', str(besselj(n, k)))))
    print(' '.join(('hankel2:', str(hankel2(n, k)))))
    print(' '.join(('spbessel:', str(spbessel(n, k)))))
    print(' '.join(('dspbessel:', str(dspbessel(n, k)))))
    print(' '.join(('spneumann:', str(spneumann(n, k)))))
    print(' '.join(('dspneumann:', str(dspneumann(n, k)))))
    print(' '.join(('sphankel2:', str(sphankel2(n, k)))))
    print(' '.join(('dsphankel2:', str(dsphankel2(n, k)))))


def _print_mic_scaling(N, freqs, array_radius, scatter_radius=None, dual_radius=None):
    if not scatter_radius:
        scatter_radius = array_radius
    if not dual_radius:
        dual_radius = array_radius
    print(' '.join(('bn_open_omni:',
                    str(array_extrapolation(N, freqs, [array_radius, 'open', 'omni'])))))
    print(' '.join(('bn_open_cardioid:',
                    str(array_extrapolation(N, freqs, [array_radius, 'open', 'cardioid'])))))
    print(' '.join(('bn_rigid_omni:',
                    str(array_extrapolation(N, freqs, [array_radius, 'rigid', 'omni', scatter_radius])))))
    print(' '.join(('bn_rigid_cardioid:',
                    str(array_extrapolation(N, freqs, [array_radius, 'rigid', 'cardioid', scatter_radius])))))
    print(' '.join(('bn_dual_open_omni:',
                    str(array_extrapolation(N, freqs, [array_radius, 'dual', 'omni', None, dual_radius])))))


def _print_bns(N, k_mic, k_scatter):
    print(' '.join(('bn_open_omni:', str(bn_open_omni(N, k_mic) * 4 * _np.pi * 1j ** N))))
    print(' '.join(('bn_open_cardioid:', str(bn_open_cardioid(N, k_mic) * 4 * _np.pi * 1j ** N))))
    print(' '.join(('bn_rigid_omni:', str(bn_rigid_omni(N, k_mic, k_scatter) * 4 * _np.pi * 1j ** N))))
    print(' '.join(('bn_rigid_cardioid:', str(bn_rigid_cardioid(N, k_mic, k_scatter) * 4 * _np.pi * 1j ** N))))
    print(' '.join(('bn_dual_open_omni:', str(bn_dual_open_omni(N, k_mic, k_scatter) * 4 * _np.pi * 1j ** N))))
