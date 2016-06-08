"""
Collection of spherical helper functions
"""

import numpy as _np
from scipy import special as scy


def spbessel(n, kr):
    """Spherical Bessel of order n"""
    # spb1 = scy.sph_jn(n, kr)  # returns j and j'
    # return spb1[0][-1]
    return _np.sqrt(_np.pi / (2 * kr)) * scy.jn(n + 0.5, kr)


def dspbessel(n, kr):
    """Derivative spherical Bessel of order n"""
    # spb1 = scy.sph_jn(n, kr)  # returns j and j'
    # return spb1[1][-1]
    return 1 / (2 * n + 1) * (n * spbessel(n - 1, kr) - (n + 1) * spbessel(n + 1, kr))


def spneumann(n, kr):
    """Spherical Neumann (Bessel second kind) of order n"""
    # spb2 = scy.sph_yn(n, kr)
    # return spb2[0][-1]
    return _np.sqrt(_np.pi / (2 * kr)) * scy.yv(n + 0.5, kr)


def dspneumann(n, kr):
    """Derivative spherical Neumann (Bessel second kind) of order n"""
    # spb2 = scy.sph_yn(n, kr)
    # return spb2[1][-1]
    return 1 / (2 * n + 1) * (n * spneumann(n - 1, kr) - (n + 1) * spneumann(n + 1, kr))


def sphankel(n, kr):
    """Spherical Hankel of order n"""
    return spbessel(n, kr) - 1j * spneumann(n, kr)


def dsphankel(n, kr):
    """Derivative spherical Hankel of order n"""
    return 0.5 * (sphankel(n - 1, kr) - sphankel(n + 1, kr) - sphankel(n, kr) / kr)


def bn_npf(n, krm, krs, ac):
    """ Spherical coefficients
    Parameters
    ----------
    n : (type of bar)
        A description of bar

    krm : (type of baz), optional
        A description of baz

    krs : (type of baz), optional
        A description of baz

    ac : (type of baz), optional
        A description of baz

    Returns
    -------
    foobar : (type of foobar)
        A description of foobar
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


def sph_harm(m, n, az, el):
    return scy.sph_harm(m, n, az, el)


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


def printSPH(n, k):
    print(' '.join(('spbessel:', str(spbessel(n, k)))))
    print(' '.join(('dspbessel:', str(dspbessel(n, k)))))
    print(' '.join(('spneumann:', str(spneumann(n, k)))))
    print(' '.join(('dspneumann:', str(dspneumann(n, k)))))
    print(' '.join(('sphankel:', str(sphankel(n, k)))))
    print(' '.join(('dsphankel:', str(dsphankel(n, k)))))


def printBNs(n, krm, krs):
    print(' '.join(('bn_openP:', str(bn(n, krm, krs, 0)))))
    print(' '.join(('bn_openPG:', str(bn(n, krm, krs, 1)))))
    print(' '.join(('bn_rigidP:', str(bn(n, krm, krs, 2)))))
    print(' '.join(('bn_rigidPG:', str(bn(n, krm, krs, 3)))))
    print(' '.join(('bn_dualopenP:', str(bn(n, krm, krs, 4)))))
