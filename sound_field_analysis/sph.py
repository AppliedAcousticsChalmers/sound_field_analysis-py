# -*- coding: utf-8 -*-
"""
Collection of spherical helper functions:


`sph_harm`
   More robust spherical harmonic coefficients
`spbessel / dspbessel`
   Spherical Bessel and derivative
`spneumann / dspneumann`
   Spherical Neumann (Bessel 2nd kind) and derivative
`sphankel / dsphankel`
   Spherical Hankel (second kind) and derivative
`cart2sph / sph2cart`
   Convert cartesion to spherical coordinates and vice versa
"""

import numpy as _np
from scipy import special as scy
from math import factorial as fact


def besselj(n, z):
    """Bessel function of first kind. Wraps scipy.special.jn(n, z)
    Parameters
    ----------
    n : array_like
       Order (float)
    z: array_like
       Argument (float or complex)

    Returns
    -------
    J : array_like
       Values of Bessel function of order n at position z
    """
    return scy.jn(n, z)


def besselh(n, z):
    """Bessel function of third kind (Hankel function). Wraps scipy.special.hankel1(n, z)
    Parameters
    ----------
    n : array_like
       Order (float)
    z: array_like
       Argument (float or complex)

    Returns
    -------
    H : array_like
       Values of Hankel function of order n at position z
    """
    return scy.hankel1(n, z)


def spbessel(n, kr):
    """Spherical Bessel function

    Parameters
    ----------
    n : int
       Order
    kr: int
       Degree

    Returns
    -------
    J : complex float
       Spherical Bessel
    """
    return scy.spherical_jn(n, kr, derivative=False)


def dspbessel(n, kr):
    """Derivative of spherical Bessel

    Parameters
    ----------
    n : int
       Order
    kr: int
       Degree

    Returns
    -------
    J' : complex float
       Derivative of spherical Bessel
    """
    return scy.spherical_jn(n, kr, derivative=True)


def spneumann(n, kr):
    """Spherical Neumann (Bessel second kind)

    Parameters
    ----------
    n : int
       Order
    kr: int
       Degree

    Returns
    -------
    Yv : complex float
       Spherical Neumann (Bessel second kind)
    """
    return scy.spherical_yn(n, kr, derivative=False)


def dspneumann(n, kr):
    """Derivative spherical Neumann (Bessel second kind) of order n

    Parameters
    ----------
    n : int
       Order
    kr: int
       Degree

    Returns
    -------
    Yv' : complex float
       Derivative of spherical Neumann (Bessel second kind)
    """
    return scy.spherical_yn(n, kr, derivative=True)


def sphankel(n, kr):
    """Spherical Hankel hn of the second kind

    Parameters
    ----------
    n : int
       Order
    kr: int
       Degree

    Returns
    -------
    hn : complex float
       Spherical Hankel function hn (second kind)
    """
    return spbessel(n, kr) - 1j * spneumann(n, kr)


def dsphankel(n, kr):
    """Derivative spherical Hankel function hn' (second kind)

    Parameters
    ----------
    n : int
       Order
    kr: int
       Degree

    Returns
    -------
    hn' : complex float
       Derivative of spherical Hankel function hn' (second kind)
    """
    return 0.5 * (sphankel(n - 1, kr) - sphankel(n + 1, kr) - sphankel(n, kr) / kr)


def bn_npf(n, krm, krs, ac):
    """ Microphone scaling

    Parameters
    ----------
    n : int
       Order

    krm : array of floats
       Microphone radius

    krs : array of floats
       Sphere radius

    ac : int {0, 1, 2, 3, 4}
       Array Configuration:
        - `0`:  Open Sphere with p Transducers (NO plc!) [Default]
        - `1`:  Open Sphere with pGrad Transducers
        - `2`:  Rigid Sphere with p Transducers
        - `3`:  Rigid Sphere with pGrad Transducers
        - `4`:  Dual Open Sphere with p Transducers

    Returns
    -------
    b : array of floats
    """
    if ac == 0:
        return bn_openP(n, krm)
    elif ac == 1:
        return bn_openPG(n, krm)
    elif ac == 2:
        return bn_rigidP(n, krm, krs)
    elif ac == 3:
        return bn_rigidPG(n, krm, krs)
    elif ac == 4:
        return bn_dualOpenP(n, krm, krs)


def bn_openP(n, krm):
    return spbessel(n, krm)


def bn_openPG(n, krm):
    return 0.5 * (spbessel(n, krm) - 1j * dspbessel(n, krm))


def bn_rigidP(n, krm, krs):
    return spbessel(n, krm) - (dspbessel(n, krs) / dsphankel(n, krs)) * sphankel(n, krm)


def bn_rigidPG(n, krm, krs):
    #  Rerence for Filter design for rigid sphere with cardioid microphones:
    #  P. Plessas, F. Zotter: Microphone arrays around rigid spheres for spatial recording and holography, DAGA 2010
    #  krm: for mic radius, krs: for sphere radius
    #  Implementation by Nils Peters, November 2011
    return spbessel(n, krm) - 1j * dspbessel(n, krm) + (1j * dsphankel(n, krm) - sphankel(n, krm)) * (dspbessel(n, krs) / dsphankel(n, krs))


def bn_dualOpenP(n, kr1, kr2):
    # Reference: Rafaely et al,
    #  High-resolution plane-wave decomposition in an auditorium using a dual-radius scanning spherical microphone array
    #  JASA, 122(5), 2007
    # kr1, kr2 are the kr values of the two different microphone spheres
    # Implementation by Nils Peters, November 2011*/
    bn1 = bn_openP(n, kr1)
    bn2 = bn_openP(n, kr2)

    if (abs(bn1) >= abs(bn2)):
        return bn1
    else:
        return bn2


def bn(n, krm, krs, ac):
    return bn_npf(n, krm, krs, ac) * 4 * _np.pi * pow(1j, n)


def sph_harm(m, n, az, el, type='complex'):
    '''Compute sphercial harmonics

    Parameters
    ----------
    m : (int)
        Order of the spherical harmonic. abs(m) <= n

    n : (int)
        Degree of the harmonic, sometimes called l. n >= 0

    az: (float)
        Azimuthal (longitudinal) coordinate [0, 2pi], also called Theta.

    el : (float)
        Elevation (colatitudinal) coordinate [0, pi], also called Phi.

    Returns
    -------
    y_mn : (complex float)
        Complex spherical harmonic of order m and degree n,
        sampled at theta = az, phi = el
    '''
    if type == 'legacy':
        return scy.sph_harm(m, n, az, el)
    elif type == 'real':
        Lnm = scy.lpmv(_np.abs(m), n, _np.cos(el))

        factor_1 = (2 * n + 1) / (4 * _np.pi)
        factor_2 = scy.factorial(n - _np.abs(m)) / scy.factorial(n + abs(m))

        if m != 0:
            factor_1 = 2 * factor_1

        if m < 0:
            return (-1) ** m * _np.sqrt(factor_1 * factor_2) * Lnm * _np.sin(m * az)
        else:
            return (-1) ** m * _np.sqrt(factor_1 * factor_2) * Lnm * _np.cos(m * az)
    else:
        # For the correct Condon–Shortley phase, all m>0 need to be increased by 1
        return (-1) ** (m - (m < 0) * (m % 2)) * scy.sph_harm(m, n, az, el)


def sph_harm_large(m, n, az, el):
    '''Compute sphercial harmonics for large orders > 84

    Parameters
    ----------
    m : (int)
        Order of the spherical harmonic. abs(m) <= n

    n : (int)
        Degree of the harmonic, sometimes called l. n >= 0

    az: (float)
        Azimuthal (longitudinal) coordinate [0, 2pi], also called Theta.

    el : (float)
        Elevation (colatitudinal) coordinate [0, pi], also called Phi.

    Returns
    -------
    y_mn : (complex float)
        Complex spherical harmonic of order m and degree n,
        sampled at theta = az, phi = el

    Y_n,m (theta, phi) = ((n - m)! * (2l + 1)) / (4pi * (l + m))^0.5 * exp(i m phi) * P_n^m(cos(theta))
    as per http://dlmf.nist.gov/14.30
    Pmn(z) is the associated Legendre function of the first kind, like scipy.special.lpmv
    scipy.special.lpmn calculates P(0...m 0...n) and its derivative but won't return +inf at high orders
    '''
    if _np.all(_np.abs(m) < 84):
        return scy.sph_harm(m, n, az, el)
    else:
        mAbs = _np.abs(m)
        if isinstance(el, _np.ndarray):
            P = _np.empty(el.size)
            for k in range(0, el.size):
                P[k] = scy.lpmn(mAbs, n, _np.cos(el[k]))[0][-1][-1]
        else:
            P = scy.lpmn(mAbs, n, _np.cos(el))[0][-1][-1]
        preFactor1 = _np.sqrt((2 * n + 1) / (4 * _np.pi))
        try:
            preFactor2 = _np.sqrt(fact(n - mAbs) / fact(n + mAbs))
        except OverflowError:  # integer division for very large orders
            preFactor2 = _np.sqrt(fact(n - mAbs) // fact(n + mAbs))

        Y = preFactor1 * preFactor2 * _np.exp(1j * m * az) * P
        if m < 0:
            return _np.conj(Y)
        else:
            return Y


def sph_harm_all(nMax, az, el, type='complex'):
    '''Compute all sphercial harmonic coefficients up to degree nMax.

    Parameters
    ----------
    nMax : (int)
        Maximum degree of coefficients to be returned. n >= 0

    az: (float), array_like
        Azimuthal (longitudinal) coordinate [0, 2pi], also called Theta.

    el : (float), array_like
        Elevation (colatitudinal) coordinate [0, pi], also called Phi.

    Returns
    -------
    y_mn : (complex float), array_like
        Complex spherical harmonics of degrees n [0 ... nMax] and all corresponding
        orders m [-n ... n], sampled at [az, el]. dim1 corresponds to az/el pairs,
        dim2 to oder/degree (m, n) pairs like 0/0, -1/1, 0/1, 1/1, -2/2, -1/2 ...
    '''
    m, n = mnArrays(nMax)
    mA, azA = _np.meshgrid(m, az)
    nA, elA = _np.meshgrid(n, el)
    return sph_harm(mA, nA, azA, elA, type=type)


def mnArrays(nMax):
    '''Returns degrees n and orders m up to nMax.

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
    '''

    # Degree n = 0, 1, 1, 1, 2, 2, 2, 2, 2 ...
    degs = _np.arange(nMax + 1)
    n = _np.repeat(degs, degs * 2 + 1)

    # Order m = 0, -1, 1, 1, -2, -1, 0, 1, 2 ...
    # http://oeis.org/A196199
    elementNumber = _np.arange((nMax + 1) ** 2) + 1
    t = _np.floor(_np.sqrt(elementNumber - 1)).astype(int)
    m = elementNumber - t * t - t - 1

    return m, n


def cart2sph(x, y, z):
    '''Converts cartesian coordinates x, y, z to spherical coordinates az, el, r.'''
    hxy = _np.hypot(x, y)
    r = _np.hypot(hxy, z)
    el = _np.arctan2(z, hxy)
    az = _np.arctan2(y, x)
    return az, el, r


def sph2cart(az, el, r):
    '''Converts spherical coordinates az, el, r to cartesian coordinates x, y, z.'''
    rcos_theta = r * _np.cos(el)
    x = rcos_theta * _np.cos(az)
    y = rcos_theta * _np.sin(az)
    z = r * _np.sin(el)
    return x, y, z


# DEBUG
def _printSPH(n, k):
    print(' '.join(('spbessel:', str(spbessel(n, k)))))
    print(' '.join(('dspbessel:', str(dspbessel(n, k)))))
    print(' '.join(('spneumann:', str(spneumann(n, k)))))
    print(' '.join(('dspneumann:', str(dspneumann(n, k)))))
    print(' '.join(('sphankel:', str(sphankel(n, k)))))
    print(' '.join(('dsphankel:', str(dsphankel(n, k)))))


def _printBNs(n, krm, krs):
    print(' '.join(('bn_openP:', str(bn(n, krm, krs, 0)))))
    print(' '.join(('bn_openPG:', str(bn(n, krm, krs, 1)))))
    print(' '.join(('bn_rigidP:', str(bn(n, krm, krs, 2)))))
    print(' '.join(('bn_rigidPG:', str(bn(n, krm, krs, 3)))))
    print(' '.join(('bn_dualopenP:', str(bn(n, krm, krs, 4)))))
