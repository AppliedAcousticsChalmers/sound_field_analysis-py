"""
Module contains various generator functions:

`whiteNoise`
   Generate additive White Gaussian noise
`gaussGrid`
   Gauss-Legendre quadrature grid and weights
`lebedev`
   Lebedev quadrature grid and weigths
`radial_filter`
   Modal radial filter
`radial_filter_fullspec`
   Modal radial filter over the full spectrum
`sampledWave`
   Sampled Wave generator, emulating discrete sampling
`ideal_wave`
   Ideal wave generator, returns spatial fourier coefficients
"""
import numpy as _np
from .sph import sph_harm, sph_harm_all, cart2sph, mnArrays, array_extrapolation, kr, sphankel2
from .io import ArrayConfiguration
from .process import iSpatFT
from .utils import progress_bar

pi = _np.pi


def whiteNoise(fftData, noiseLevel=80):
    '''Adds White Gaussian Noise of approx. 16dB crest to a FFT block.

    Parameters
    ----------
    fftData : array of complex floats
       Input fftData block (e.g. from F/D/T or S/W/G)
    noiseLevel : int, optional
       Average noise Level in dB [Default: -80dB]

    Returns
    -------
    noisyData : array of complex floats
       Output fftData block including white gaussian noise
    '''
    dimFactor = 10**(noiseLevel / 20)
    fftData = _np.atleast_2d(fftData)
    channels = fftData.shape[0]
    NFFT = fftData.shape[1] * 2 - 2
    nNoise = _np.random.rand(channels, NFFT)
    nNoise = dimFactor * nNoise / _np.mean(_np.abs(nNoise))
    nNoiseSpectrum = _np.fft.rfft(nNoise, axis=1)
    return fftData + nNoiseSpectrum


def gaussGrid(AZnodes=10, ELnodes=5, plot=False):
    '''Compute Gauss-Legendre quadrature nodes and weigths in the SOFiA/VariSphear data format.

    Parameters
    ----------
    AZnodes, ELnodes : int, optional
       Number of azimutal / elevation nodes  [Default: 10 / 5]
    plot : bool, optional
        Show a globe plot of the selected grid [Default: False]

    Returns
    -------
    gridData : matrix of floats
       Gauss-Legendre quadrature positions and weigths
       ::
          [AZ_0, EL_0, W_0
               ...
          AZ_n, EL_n, W_n]
    Npoints : int
       Total number of nodes
    Nmax : int
       Highest stable grid order
    '''

    # Azimuth: Gauss
    AZ = _np.linspace(0, AZnodes - 1, AZnodes) * 2 * pi / AZnodes
    AZw = _np.ones(AZnodes) * 2 * pi / AZnodes

    # Elevation: Legendre
    EL, ELw = _np.polynomial.legendre.leggauss(ELnodes)
    EL = _np.arccos(EL)

    # Weights
    W = _np.outer(AZw, ELw) / 3
    W /= W.sum()

    # VariSphere order: AZ increasing, EL alternating
    gridData = _np.empty((ELnodes * AZnodes, 3))
    for k in range(0, AZnodes):
        curIDX = k * ELnodes
        gridData[curIDX:curIDX + ELnodes, 0] = AZ[k].repeat(ELnodes)
        gridData[curIDX:curIDX + ELnodes, 1] = EL[::-1 + k % 2 * 2]  # flip EL every second iteration
        gridData[curIDX:curIDX + ELnodes, 2] = W[k][::-1 + k % 2 * 2]  # flip W every second iteration

    return gridData


def lebedev(max_order=None, degree=None):
    '''Compute Lebedev quadrature nodes and weigths given a maximum stable order. Alternatively, a degree may be supplied.

    Parameters
    ----------
    max_order : int
       Maximum stable order of the Lebedev grid, [0 ... 11]
    degree : int, optional
       Lebedev Degree, one of {6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194}

    Returns
    -------
    gridData : array_like
       Lebedev quadrature positions and weigths: [AZ, EL, W]
    '''
    if max_order is None and not degree:
        raise ValueError('Either a maximum order or a degree have to be given.')

    if max_order is 0:
        max_order = 1

    allowed_degrees = [6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194]

    if max_order and 0 <= max_order <= 11:
        degree = allowed_degrees[int(max_order) - 1]
    elif max_order:
        raise ValueError('Maximum order can only be between 0 and 11.')

    if degree not in allowed_degrees:
        raise ValueError(str(degree) + ' is an invalid quadrature degree. Choose one of the following: ' + str(allowed_degrees))

    from . import lebedev
    leb = lebedev.genGrid(degree)
    theta, phi, _ = cart2sph(leb.x, leb.y, leb.z)
    theta = theta % (2 * pi)
    gridData = _np.array([theta, phi + pi / 2, leb.w]).T
    gridData = gridData[gridData[:, 1].argsort()]  # ugly double sorting that keeps rows together
    gridData = gridData[gridData[:, 0].argsort()]

    return gridData


def radial_filter_fullspec(max_order, NFFT, fs, array_configuration, amp_maxdB=40):
    """Generate NFFT/2 + 1 modal radial filter of orders 0:max_order for frequencies 0:fs/2, wraps radial_filter()

    Parameters
    ----------
    max_order : int
       Maximum order
    NFFT : int
       Order of FFT (number of bins), should be a power of 2.
    fs : int
       Sampling frequency
    array_configuration : ArrayConfiguration
       List/Tuple/ArrayConfiguration, see io.ArrayConfiguration
    amp_maxdB : int, optional
       Maximum modal amplification limit in dB [Default: 40]

    Returns
    -------
    dn : array_like
       Vector of modal frequency domain filter of shape [max_order + 1 x NFFT / 2 + 1]
    """

    freqs = _np.linspace(0, fs / 2, NFFT / 2 + 1)
    orders = _np.r_[0:max_order + 1]
    return radial_filter(orders, freqs, array_configuration, amp_maxdB=amp_maxdB)


def radial_filter(order, freq, array_configuration, amp_maxdB=40):
    """Generate modal radial filter of specified order and frequency

    Parameters
    ----------
    order : array_like
       order of filter
    freq : array_like
       Frequency of modal filter
    array_configuration : ArrayConfiguration
       List/Tuple/ArrayConfiguration, see io.ArrayConfiguration
    amp_maxdB : int, optional
       Maximum modal amplification limit in dB [Default: 40]

    Returns
    -------
    dn : array_like
       Vector of modal frequency domain filter of shape [nOrders x nFreq]
    """
    array_configuration = ArrayConfiguration(*array_configuration)

    extrapolation_coeffs = array_extrapolation(order, freq, array_configuration)
    extrapolation_coeffs[extrapolation_coeffs == 0] = 1e-12

    a_max = 10 ** (amp_maxdB / 20)
    limiting_factor = 2 * a_max / _np.pi * _np.abs(extrapolation_coeffs) * _np.arctan(_np.pi / (2 * a_max * _np.abs(extrapolation_coeffs)))

    return limiting_factor / extrapolation_coeffs


def sampled_wave(fs, NFFT, array_configuration,
                 gridData, wave_azimuth, wave_colatitude, wavetype='plane', c=343, distance=1.0, limit_order=85):
    """Returns the frequency domain data of an ideal wave as recorded by a provided array.

    Parameters
    ----------
    fs : int
       Sampling frequency
    NFFT : int
       Order of FFT (number of bins), should be a power of 2.
    array_configuration : ArrayConfiguration
       List/Tuple/ArrayConfiguration, see io.ArrayConfiguration
    gridData : array_like
       Quadrature grid
       ::
          Columns : Position Number 1...M
          Rows    : [AZ EL Weight]
    wave_azimuth, wave_colatitude : float, optional
       Direction of incoming wave in radians [0-2pi].
    wavetype : {'plane', 'spherical'}, optional
       Type of the wave. [Default: plane]
    c : float, optional
       Speed of sound in [m/s] [Default: 343 m/s]
    distance : float, optional
       Distance of the source in [m] (For spherical waves only)
    limit_order : int, optional
       Sets the limit for wave generation

    Warning
    -------
    If NFFT is smaller than the time the wavefront
    needs to travel from the source to the array, the impulse
    response will by cyclically shifted (cyclic convolution).

    Returns
    -------
    fftData : array_like
        Complex sound pressures of size [(N+1)^2 x NFFT]

    Note
    ----
    This file is a wrapper generating the complex pressures at the positions given in 'gridData'
    for a full spectrum 0-FS/2 Hz (NFFT Bins) wave impinging on the array, emulating discrete sampling.
    """
    array_configuration = ArrayConfiguration(*array_configuration)

    freqs = _np.linspace(0, fs / 2, NFFT)
    kr_mic = kr(freqs, array_configuration.array_radius)

    max_order_fullspec = _np.ceil(_np.max(kr_mic) * 2)

    # TODO : Investigate if limit_order works as intended
    if max_order_fullspec > limit_order:
        print('Requested wave front needs a minimum order of ' + str(int(max_order_fullspec)) + ' but was limited to order ' + str(limit_order))

    Pnm = ideal_wave(min(max_order_fullspec, limit_order), fs, wave_azimuth, wave_colatitude, array_configuration, wavetype, distance, NFFT)
    azimuth_grid = gridData[:, 0]
    colatitude_grid = gridData[:, 1]
    fftData = iSpatFT(Pnm, azimuth_grid, colatitude_grid)

    return fftData


def ideal_wave(order, fs, azimuth, colatitude, array_configuration,
               wavetype='plane', distance=1.0, NFFT=128, delay=0.0, c=343.0):
    """Ideal wave generator, returns spatial Fourier coefficients `Pnm` of an ideal wave front hitting a specified array

    Parameters
    ----------
    order : int
        Maximum transform order.
    fs : int
       Sampling frequency
    NFFT : int
       Order of FFT (number of bins), should be a power of 2
    array_configuration : ArrayConfiguration
       List/Tuple/ArrayConfiguration, see io.ArrayConfiguration
    azimuth, colatitude : float
       Azimuth/Colatitude angle of the wave in [RAD]
    wavetype : {'plane', 'spherical'}, optional
       Select between plane or spherical wave [Default: Plane wave]
    distance : float, optional
       Distance of the source in [m] (for spherical waves only)
    delay : float, optional
       Time Delay in s [default: 0]
    c : float, optional
       Propagation veolcity in m/s [Default: 343m/s]

    Warning
    -------
    If NFFT is smaller than the time the wavefront needs to travel from the source to the array,
    the impulse response will by cyclically shifted.

    Returns
    -------
    Pnm : array of complex floats
       Spatial Fourier Coefficients with nm coeffs in cols and FFT coeffs in rows
    """
    array_configuration = ArrayConfiguration(*array_configuration)

    order = _np.int_(order)
    NFFT = int(NFFT / 2 + 1)
    NMLocatorSize = (order + 1) ** 2

    # SAFETY CHECKS
    if wavetype not in {'plane', 'spherical'}:
        raise ValueError('Invalid wavetype: Choose either plane or spherical.')
    if (delay * fs > NFFT - 1):
        raise ValueError('Delay t is large for provided NFFT. Choose t < NFFT/(2*FS).')

    w = _np.linspace(0, pi * fs, NFFT)
    freqs = _np.linspace(0, fs / 2, NFFT)

    radial_filters = _np.zeros([NMLocatorSize, NFFT], dtype=_np.complex_)
    time_shift = _np.exp(-1j * w * delay)

    for n in range(0, order + 1):
        if wavetype is 'plane':
            radial_filters[n] = time_shift * array_extrapolation(n, freqs, array_configuration)
        elif wavetype is 'spherical':
            k_dist = kr(freqs, distance)
            radial_filters[n] = 4 * pi * -1j * w / c * time_shift * sphankel2(n, k_dist) * array_extrapolation(n, freqs, array_configuration)

    # GENERATOR CORE
    Pnm = _np.empty([NMLocatorSize, NFFT], dtype=_np.complex_)
    m, n = mnArrays(order + 1)
    ctr = 0
    for n in range(0, order + 1):
        for m in range(-n, n + 1):
            Pnm[ctr] = _np.conj(sph_harm(m, n, azimuth, colatitude)) * radial_filters[n]
            ctr = ctr + 1

    return Pnm


def spherical_noise(azimuth_grid, colatitude_grid, order_max=8, spherical_harmonic_bases=None):
    ''' Returns band-limited random weights on a spherical surface

    Parameters
    ----------
    azimuth_grid, colatitude_grid : array_like, float
       Grids holding azimuthal and colatitudinal angles
    order_max : int, optional
        Spherical order limit [Default: 8]

    Returns
    -------
    noisy_weights : array_like, complex
       Noisy weigths
    '''
    if spherical_harmonic_bases is None:
        spherical_harmonic_bases = sph_harm_all(order_max, azimuth_grid, colatitude_grid)
    return _np.inner(spherical_harmonic_bases, _np.random.randn((order_max + 1) ** 2) + 1j * _np.random.randn((order_max + 1) ** 2))
