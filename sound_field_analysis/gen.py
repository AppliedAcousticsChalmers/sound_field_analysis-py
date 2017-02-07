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


def lebedev(degree):
    '''Compute Lebedev quadrature nodes and weigths.

    Parameters
    ----------
    Degree : int
       Lebedev Degree. Currently available: 6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194

    Returns
    -------
    gridData : array_like
       Lebedev quadrature positions and weigths: [AZ, EL, W]
    Nmax : int
       Highest stable grid order

    '''
    from . import lebedev

    deg_avail = _np.array([6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194])

    if degree not in deg_avail:
        raise ValueError('WARNING: Invalid quadrature degree', degree, '[deg] supplied. Choose one of the following:\n', deg_avail)

    leb = lebedev.genGrid(degree)
    theta, phi, _ = cart2sph(leb.x, leb.y, leb.z)
    theta = theta % (2 * pi)
    gridData = _np.array([theta, phi + pi / 2, leb.w]).T
    gridData = gridData[gridData[:, 1].argsort()]  # ugly double sorting that keeps rows together
    gridData = gridData[gridData[:, 0].argsort()]

    # TODO: turnover
    Nmax = _np.floor(_np.sqrt(degree / 1.3) - 1)

    return gridData, Nmax


def radial_filter_fullspec(max_order, NFFT, fs, array_radius, array_configuration='open', transducer_type='pressure', scatter_radius=None, amp_maxdB=40):
    """Generate NFFT/2 + 1 modal radial filter of orders 0:max_order for frequencies 0:fs/2, wraps radial_filter()

    Parameters
    ----------
    max_order : int
       Maximum Order
    array_configuration : string {open, rigid, dual}
       Array configuration [Default: open]
    transducer_type: string {pressure, velocity}
       Transducer type [Default: pressure]
    amp_maxdB : int, optional
       Maximum modal amplification limit in dB [Default: 40]

    Returns
    -------
    dn : array_like
       Vector of modal frequency domain filter of shape [max_order + 1 x NFFT / 2 + 1]
    """

    freqs = _np.linspace(0, fs / 2, NFFT / 2 + 1)
    orders = _np.r_[0:max_order + 1]
    return radial_filter(orders, freqs, array_radius, array_configuration, transducer_type, scatter_radius=scatter_radius, amp_maxdB=amp_maxdB)


def radial_filter(order, freq, array_radius, array_configuration, transducer_type, scatter_radius=None, amp_maxdB=40):
    """Generate modal radial filter of specified order and frequency

    Parameters
    ----------
    order : array_like
       order of filter
    freq : array_like
       Frequency of modal filter
    array_configuration : string {open, rigid, dual}
       Array configuration [Default: open]
    transducer_type: string {pressure, velocity}
       Transducer type [Default: pressure]
    amp_maxdB : int, optional
       Maximum modal amplification limit in dB [Default: 40]

    Returns
    -------
    dn : array_like
       Vector of modal frequency domain filter of shape [nOrders x nFreq]
    """

    extrapolation_coeffs = array_extrapolation(order, freq, array_radius, scatter_radius=scatter_radius, array_configuration=array_configuration, transducer_type=transducer_type)
    extrapolation_coeffs[extrapolation_coeffs == 0] = 1e-12

    a_max = 10 ** (amp_maxdB / 20)
    limiting_factor = 2 * a_max / _np.pi * _np.abs(extrapolation_coeffs) * _np.arctan(_np.pi / (2 * a_max * _np.abs(extrapolation_coeffs)))

    return limiting_factor / extrapolation_coeffs


def sampledWave(r=0.01, gridData=None, ac=0, FS=48000, NFFT=512, AZ=0, EL=_np.pi / 2,
                c=343, wavetype=0, ds=1, Nlim=120):
    """Sampled Wave Generator Wrapper

    Parameters
    ----------
    r : array_like, optional
       Microphone Radius [Default: 0.01]
       ::
          Can also be a vector for rigid sphere configurations:
          [1,1] => rm  Microphone Radius
          [2,1] => rs  Sphere Radius (Scatterer)
    gridData : array_like
       Quadrature grid [Default: 110 Lebebdev grid]
       ::
          Columns : Position Number 1...M
          Rows    : [AZ EL Weight]
    ac : int {0, 1, 2, 3, 4}
       Array Configuration:
        - `0`:  Open Sphere with p Transducers (NO plc!) [Default]
        - `1`:  Open Sphere with pGrad Transducers
        - `2`:  Rigid Sphere with p Transducers
        - `3`:  Rigid Sphere with pGrad Transducers
        - `4`:  Dual Open Sphere with p Transducers
    FS : int, optional
       Sampling frequency [Default: 48000 Hz]
    NFFT : int, optional
       Order of FFT (number of bins), should be a power of 2. [Default: 512]
    AZ : float, optional
       Azimuth angle in radians [0-2pi]. [Default: 0]
    EL : float, optional
       Elevation angle in in radians [0-pi]. [Default: pi / 2]
    c : float, optional
       Speed of sound in [m/s] [Default: 343 m/s]
    wavetype : int {0, 1}, optional
       Type of the wave:
        - 0: Plane wave [Default]
        - 1: Spherical wave
    ds : float, optional
       Distance of the source in [m] (For wavetype = 1 only)
    Nlim : int, optional
       Internal generator transform order limit [Default: 120]

    Warning
    -------
    If NFFT is smaller than the time the wavefront
    needs to travel from the source to the array, the impulse
    response will by cyclically shifted (cyclic convolution).

    Returns
    -------
    fftData : array_like
        Complex sound pressures of size [(N+1)^2 x NFFT]
    kr : array_like
       kr-vector
       ::
          Can also be a matrix [krm; krs] for rigid sphere configurations:
          [1,:] => krm referring to the microphone radius
          [2,:] => krs referring to the sphere radius (scatterer)

    Note
    ----
    This file is a wrapper generating the complex pressures at the
    positions given in 'gridData' for a full spectrum 0-FS/2 Hz (NFFT Bins)
    wave impinging to an array. The wrapper involves the ideal_wave generator
    and the spatFT spatial transform.

    sampledWave emulates discrete sampling. You can observe alias artifacts.
    """

    if gridData is None:
        gridData = lebedev(110)[0]

    if not isinstance(r, list):  # r [1,1] => rm  Microphone Radius
        kr = _np.linspace(0, r * pi * FS / c, (NFFT / 2 + 1))
        krRef = kr
    else:  # r [2,1] => rs  Sphere Radius (Scatterer)
        kr = _np.array([_np.linspace(0, r[0] * pi * FS / c, NFFT / 2 + 1),
                        _np.linspace(0, r[1] * pi * FS / c, NFFT / 2 + 1)])
        krRef = kr[0] if r[0] > r[1] else kr[1]

    minOrderLim = 70
    rqOrders = _np.ceil(krRef * 2).astype('int')
    maxReqOrder = _np.max(rqOrders)
    rqOrders[rqOrders <= minOrderLim] = minOrderLim
    rqOrders[rqOrders > Nlim] = Nlim
    Ng = _np.max(rqOrders)

    if maxReqOrder > Ng:
        print('WARNING: Requested wave needs a minimum order of ' + str(maxReqOrder) + ' but only order ' + str(Ng) + 'can be delivered.')
    else:
        print('Segmented generator orders: ' + str(minOrderLim) + ' to ' + str(Ng))

    # SEGMENTATION
    # index = 1
    # ctr = -1
    Pnm = _np.zeros([(Ng + 1) ** 2, int(NFFT / 2 + 1)], dtype=_np.complex_)
    unique_orders = _np.unique(rqOrders)

    for idx, order in enumerate(unique_orders):
        progress_bar(idx, _np.size(unique_orders), 'sampledWave - Sampled Wave Generator')
        fOrders = _np.flatnonzero(rqOrders == order)
        Pnm += ideal_wave(Ng, r, ac, FS, NFFT, AZ, EL, wavetype=wavetype, ds=ds, lowerSegLim=fOrders[0], upperSegLim=fOrders[-1], SegN=order)[0]
    fftData = iSpatFT(Pnm, gridData)

    return fftData, kr


def ideal_wave(order, azimuth, colatitude, array_radius, array_configuration='open', transducer_type='pressure', scatter_radius=None,
               wavetype='plane', distance=1.0, fs=44100, F_NFFT=512, delay=0.0, c=343.0, segment_order=None, lowerSegLim=0, upperSegLim=None):
    """Ideal wave generator, returns spatial Fourier coefficients `Pnm` of an ideal wave front hitting a specified array

    Parameters
    ----------
    order : int
        Maximum transform order.
    array_radius  : float
       Microphone array radius
    array_configuration : string {open, rigid, dual}
       Array configuration [Default: Open]
    transducer_type: string {pressure, velocity}
       Transducer type [Default: pressure]
    scatter_radius : float, optional
       Radius of scatter (for rigid configuration) or of second microphone array (for dual configuration)
    FS : int, optional
       Sampling frequency (Default: 44100)
    NFFT : int, optional
       Order of FFT (number of bins), should be a power of 2. (Default: 512)
    azimuth, colatitude : float
       Azimuth/Colatitude  angle in [RAD].
    delay : float, optional
       Time Delay in s.
    c : float, optional
       Propagation veolcity in m/s [Default: 343m/s]
    wavetype : string {'plane', 'spherical'}, optional
       Select between plane or spherical wave [Default: Plane wave]
    distance : float, optional
       Distance of the source in [m] (for spherical waves only)
    lSegLim : int, optional
       Lower Segment Limit (Used by sampledWave())
    uSegLim : int, optional
       Upper Segment Limit (Used by sampledWave())
    SegN : int, optional
        Segment Order (Used by sampledWave())

    Warning
    -------
    If NFFT is smaller than the time the wavefront needs to travel from the source to the array,
    the impulse response will by cyclically shifted.

    Returns
    -------
    Pnm : array of complex floats
       Spatial Fourier Coefficients with nm coeffs in cols and FFT coeffs in rows
    """

    NFFT = int(F_NFFT / 2 + 1)
    NMLocatorSize = (order + 1) ** 2

    if segment_order is None:
        segment_order = order
    if upperSegLim is None:
        upperSegLim = NFFT - 1

    # SAFETY CHECKS
    if upperSegLim < lowerSegLim:
        raise ValueError('Upper segment limit needs to be below lower limit.')
    if upperSegLim > NFFT - 1:
        raise ValueError('Upper segment limit needs to be below NFFT - 1.')
    if upperSegLim > NFFT - 1 or upperSegLim < 0:
        raise ValueError('Upper segment limit needs to be between 0 and NFFT - 1.')
    if lowerSegLim > NFFT - 1 or lowerSegLim < 0:
        raise ValueError('Lower segment limit needs to be between 0 and NFFT - 1.')
    if segment_order > order:
        raise ValueError("Segment order needs to be smaller than N.")
    if wavetype not in {'plane', 'spherical'}:
        raise ValueError('Invalid wavetype: Choose either plane or spherical.')
    if array_configuration not in {'open', 'rigid', 'dual'}:
        raise ValueError('Sphere configuration has to be either open (default), rigid, or dual.')
    if transducer_type not in {'pressure', 'velocity'}:
        raise ValueError('Transducer type has to be either pressure (default) or velocity.')
    if array_configuration is 'dual' and transducer_type is not 'pressure':
        raise ValueError('For dual sphere configuration, only pressure transducers are allowed.')
    if (delay * fs > F_NFFT / 2):
        raise ValueError('Delay t is large for provided NFFT. Choose t < NFFT/(2*FS).')
    if wavetype is 'plane' and distance <= array_radius:
        raise ValueError('Invalid source distance, source must be outside the microphone radius.')

    w = _np.linspace(0, pi * fs, NFFT)
    freqs = _np.linspace(0, fs / 2, NFFT)

    radial_filters = _np.zeros([NMLocatorSize, NFFT], dtype=_np.complex_)
    time_shift = _np.exp(-1j * w * delay)

    for n in range(0, segment_order + 1):
        if wavetype is 'plane':
            radial_filters[n] = time_shift * array_extrapolation(n, freqs, array_radius, scatter_radius=array_radius, array_configuration=array_configuration, transducer_type=transducer_type)
        elif wavetype is 'spherical':
            k_dist = kr(freqs, distance)
            radial_filters[n] = 4 * pi * -1j * w / c * time_shift * sphankel2(n, k_dist) * array_extrapolation(n, freqs, array_radius, scatter_radius=array_radius, array_configuration=array_configuration, transducer_type=transducer_type)

    # GENERATOR CORE
    Pnm = _np.empty([NMLocatorSize, NFFT], dtype=_np.complex_)
    m, n = mnArrays(segment_order + 1)
    ctr = 0
    for n in range(0, segment_order + 1):
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
