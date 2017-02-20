"""
Functions that act on the Spatial Fourier Coefficients

`FFT`
   (Fast) Fourier Transform
`iFFT`
   Inverse (Fast) Fourier Transform

`spatFT`
   Spatial Fourier Transform
`iSpatFT`
   Fast Inverse Spatial Fourier Transform

`PWDecomp`
   Plane Wave Decomposition
"""

import numpy as _np
from scipy.signal import fftconvolve
from scipy.linalg import lstsq
from .sph import besselj, hankel1, sph_harm_all
from .io import SphericalGrid


def BEMA(Pnm, ctSig, dn, transition, avgBandwidth, fade=True):
    '''BEMA Spatial Anti-Aliasing - NOT YET IMPLEMENTED

    Parameters
    ----------
    Pnm : array_like
       Spatial Fourier coefficients
    ctSig : array_like
       Signal of the center microphone
    dn : array_like
       Radial filters for the current array configuration
    transition : int
       Highest stable bin, approx: transition = (NFFT/FS+1) * (N*c)/(2*pi*r)
    avgBandwidth : int
       Averaging Bandwidth in oct
    fade : bool, optional
       Fade over if True, else hard cut {false} [Default: True]

    Returns
    -------
    Pnm : array_like
       Alias-free spatial Fourier coefficients

    Note
    ----
    This was presented at the 2012 AES convention, see [1]_.

    References
    ----------
    .. [1] B. Bernschütz, "Bandwidth Extension for Microphone Arrays",
       AES Convention 2012, Convention Paper 8751, 2012. http://www.aes.org/e-lib/browse.cfm?elib=16493
    '''

    print('!Warning, BEMA is not yet implemented. Continuing with initial coefficients!')
    return Pnm


def FFT(time_signals, fs=None, NFFT=None, oversampling=1, first_sample=0, last_sample=None):
    '''Real-valued Fast Fourier Transform.

    Parameters
    ----------
    time_signals : TimeSignal/tuple/object
       Time-domain signals to be transformed. If of length 2, fs is assumened as the second element, otherwise fs has to be specified.
    fs : int, optional
       Sampling frequency - only optional if a TimeSignal or tuple/array containing fs is passed
    NFFT : int, optional
       Number of frequency bins. Resulting array will have size NFFT//2+1 Default: Next power of 2
    oversampling : int, optional
       Oversamples the incoming signal to increase frequency resolution [Default: 1]
    firstSample : int, optional
       First time domain sample to be included. [Default: 0]
    lastSample : int, optional
       Last time domain sample to be included. [Default: -1]

    Returns
    -------
    fftData : ndarray
       Frequency-domain data
    f : ndarray
       Frequency scale

    Note
    ----
    An oversampling*NFFT point Fourier Transform is applied to the time domain data,
    where NFFT is the next power of two of the number of samples.
    Time-windowing can be used by providing a first_sample and last_sample index.
    '''
    try:
        signals = time_signals.signal
        fs = time_signals.fs
    except AttributeError:
        if len(time_signals) == 2:
            signals = time_signals[0]
            fs = time_signals[1]
        else:
            if fs is not None:
                signals = time_signals
            else:
                raise ValueError('No valid signal found. Either pass an io.TimeSignal, a tuple/array containg the signal and the sampling frequecy or use the fs argument.')

    signals = _np.atleast_2d(signals)
    nSig, nSamples = signals.shape

    if last_sample is None:  # assign lastSample to length of signals if not provided
        last_sample = nSamples

    if oversampling < 1:
        raise ValueError('oversampling must be >= 1.')

    if last_sample < first_sample or last_sample > nSamples:
        raise ValueError('lastSample must be between firstSample and nSamples.')

    if first_sample < 0 or first_sample > last_sample:
        raise ValueError('firstSample must be between 0 and lastSample.')

    total_samples = last_sample - first_sample
    signals = signals[:, first_sample:last_sample]

    if not NFFT:
        NFFT = int(2**_np.ceil(_np.log2(total_samples)))

    fftData = _np.fft.rfft(signals, NFFT * oversampling, 1)
    f = _np.fft.rfftfreq(NFFT * oversampling, d=1 / fs)

    return fftData, f


def spatFT(data, position_grid, order_max=10, spherical_harmonic_bases=None):
    ''' Spatial Fourier Transform

    Parameters
    ----------
    data : array_like
       Data to be transformed, with signals in rows and frequency bins in columns
    order_max : int, optional
       Maximum transform order (Default: 10)
    position_grid : array_like or io.SphericalGrid
       Azimuths/Colatitudes/Gridweights of spatial sampling points

    Returns
    -------
    Pnm : array_like
       Spatial Fourier Coefficients with nm coeffs in rows and FFT bins in columns
    '''
    data = _np.atleast_2d(data)
    number_of_signals, FFTLength = data.shape

    position_grid = SphericalGrid(*position_grid)

    # Re-generate spherical harmonic bases if they were not provided or their order is too small
    if (spherical_harmonic_bases is None or
            spherical_harmonic_bases.shape[0] < number_of_signals or
            spherical_harmonic_bases.shape[1] < (order_max + 1) ** 2):
        spherical_harmonic_bases = sph_harm_all(order_max, position_grid.azimuth, position_grid.colatitude)

    spherical_harmonic_bases = (_np.conj(spherical_harmonic_bases).T * (4 * _np.pi * position_grid.weight))
    return _np.inner(spherical_harmonic_bases, data.T)


def iSpatFT(spherical_coefficients, position_grid, order_max=None, spherical_harmonic_bases=None):
    """Inverse spatial Fourier Transform

    Parameters
    ----------
    spherical_coefficients : array_like
       Spatial Fourier coefficients with columns representing frequncy bins
    position_grid : array_like or io.SphericalGrid
       Azimuth/Colatitude angles of spherical coefficients
    order_max : int, optional
       Maximum transform order [Default: highest available order]

    Returns
    -------
    P : array_like
       Sound pressures with frequency bins in columnss and angles in rows
    """
    position_grid = SphericalGrid(*position_grid)
    spherical_coefficients = _np.atleast_2d(spherical_coefficients)
    number_of_coefficients, FFTLength = spherical_coefficients.shape

    # todo: check coeffs and bases length correspond with order_max
    if order_max is None:
        order_max = int(_np.sqrt(number_of_coefficients) - 1)

    # Re-generate spherical harmonic bases if they were not provided or their order is too small
    if spherical_harmonic_bases is None or spherical_harmonic_bases.shape[1] < number_of_coefficients or spherical_harmonic_bases.shape[1] == position_grid.azimuths.size:
        spherical_harmonic_bases = sph_harm_all(order_max, position_grid.azimuth, position_grid.colatitude)

    return _np.inner(spherical_harmonic_bases, spherical_coefficients.T)


def spatFT_LSF(data, position_grid, order_max, spherical_harmonic_bases=None):
    '''Returns spherical harmonics coefficients least square fitted to provided data

    Parameters
    ----------
    data : array_like, complex
       Data to be fitted to
    position_grid : array_like, or io.SphericalGrid
       Azimuth / colatitude data locations
    order_max: int
       Maximum order N of fit

    Returns
    -------
    coefficients: array_like, float
       Fitted spherical harmonic coefficients (indexing: n**2 + n + m + 1)
    '''
    position_grid = SphericalGrid(*position_grid)
    if spherical_harmonic_bases is None:
        spherical_harmonic_bases = sph_harm_all(order_max, position_grid.azimuth, position_grid.colatitude)
    return lstsq(spherical_harmonic_bases, data)[0]


def plane_wave_decomp(order, wave_direction, field_coeffs, radial_filter, weights=None):
    """Plane wave decomposition

    Parameters
    ----------
    order : int
       Decomposition order
    wave_direction : array_like
       Direction of plane wave as [azimuth, colatitude] pair. io.SphericalGrid is used internally
    field_coeffs : array_like
       Spatial fourier coefficients
    radial_filter : array_like
       Radial filters
    weights : array_like, optional
       Weighting function. Either scalar, one per directions or of dimension (nKR_bins x  nDirections). [Default: None]

    Returns
    -------
    Y : matrix of floats
       Matrix of the decomposed wavefield with kr bins in rows
    """

    wave_direction = SphericalGrid(*wave_direction)
    number_of_angles = wave_direction.azimuth.size
    field_coeffs = _np.atleast_2d(field_coeffs)
    radial_filter = _np.atleast_2d(radial_filter)

    if field_coeffs.shape[0] == 1:
        field_coeffs = field_coeffs.T
    if radial_filter.shape[0] == 1:
        radial_filter = radial_filter.T

    NMDeliveredSize, FFTBlocklengthPnm = field_coeffs.shape
    Ndn, FFTBlocklengthdn = radial_filter.shape

    if FFTBlocklengthdn != FFTBlocklengthPnm:
        raise ValueError('FFT Blocksizes of field coefficients (Pnm) and radial filter (dn) are not consistent.')

    max_order = int(_np.floor(_np.sqrt(NMDeliveredSize) - 1))
    if order > max_order:
        raise ValueError('The provided coefficients deliver a maximum order of ' + str(max_order) + ' but order ' + str(order) + ' was requested.')

    gaincorrection = 4 * _np.pi / ((order + 1) ** 2)

    if weights is not None:
        weights = _np.asarray(weights)
        if weights.ndim == 1:
            number_of_weights = weights.size
            if (number_of_weights != FFTBlocklengthPnm) and (number_of_weights != number_of_angles) and (number_of_weights != 1):
                raise ValueError('Weights is not a scalar nor consistent with shape of the field coefficients (Pnm).')
            if number_of_weights == number_of_angles:
                weights = _np.broadcast_to(weights, (FFTBlocklengthPnm, number_of_angles)).T
        else:
            if weights.shape != (number_of_angles, FFTBlocklengthPnm):
                raise ValueError('Weights is not a scalar nor consistent with shape of field coefficients (Pnm) and radial filter (dn).')
    else:
        weights = 1

    sph_harms = sph_harm_all(order, wave_direction.azimuth, wave_direction.colatitude)
    filtered_coeffs = field_coeffs * _np.repeat(radial_filter, _np.r_[:order + 1] * 2 + 1, axis=0)
    OutputArray = _np.dot(sph_harms, filtered_coeffs)
    OutputArray = _np.multiply(OutputArray, weights)
    return OutputArray * gaincorrection


def rfi(dn, kernelDownScale=2, highPass=0.0):
    '''R/F/I Radial Filter Improvement [NOT YET IMPLEMENTED!]

    Parameters
    ----------
    dn : array_like
       Analytical frequency domain radial filters (e.g. gen.radFilter())
    kernelDownScale : int, optional
       Downscale factor for the filter kernel [Default: 2]
    highPass : float, optional
       Highpass Filter from 0.0 (off) to 1.0 (maximum kr) [Default: 0.0]

    Returns
    -------
    dn : array_like
       Improved radial filters
    kernelSize : int
       Filter kernel size (total)
    latency : float
       Approximate signal latency due to the filters

    Note
    ----
    This function improves the FIR radial filters from gen.radFilter(). The filters
    are made causal and are windowed in time domain. The DC components are
    estimated. The R/F/I module should always be inserted to the filter
    path when treating measured data even if no use is made of the included
    kernel downscaling or highpass filters.

    Do NOT use R/F/I for single open sphere filters (e.g.simulations).

    IMPORTANT
       Remember to choose a fft-oversize factor (.FFT()) being large
       enough to cover all filter latencies and reponse slopes.
       Otherwise undesired cyclic convolution artifacts may appear
       in the output signal.

    HIGHPASS
       If HPF is on (highPass>0) the radial filter kernel is
       downscaled by a factor of two. Radial Filters and HPF
       share the available taps and the latency keeps constant.
       Be careful using very small signal blocks because there
       may remain too few taps. Observe the filters by plotting
       their spectra and impulse responses.
       > Be very carefull if NFFT/max(kr) < 25
       > Do not use R/F/I if NFFT/max(kr) < 15
    '''
    return dn


def sfe(Pnm_kra, kra, krb, problem='interior'):
    ''' S/F/E Sound Field Extrapolation. CURRENTLY WIP

    Parameters
    ----------
    Pnm_kra : array_like
       Spatial Fourier Coefficients (e.g. from spatFT())
    kra,krb : array_like
       k * ra/rb vector
    problem : string{'interior', 'exterior'}
       Select between interior and exterior problem [Default: interior]
    '''

    if kra.shape[1] != Pnm_kra.shape[1] or kra.shape[1] != krb.shape[1]:
        raise ValueError('FFTData: Complex Input Data expected.')

    FCoeff = Pnm_kra.shape[0]
    N = int(_np.floor(_np.sqrt(FCoeff) - 1))

    nvector = _np.zeros(FCoeff)
    IDX = 1

    for n in range(0, N + 1):
        for m in range(-n, n + 1):
            nvector[IDX] = n
            IDX += 1

    nvector = _np.tile(nvector, (1, Pnm_kra.shape[1]))
    kra = _np.tile(kra, (FCoeff, 1))
    krb = _np.tile(krb, (FCoeff, 1))

    if problem == 'interior':
        jn_kra = _np.sqrt(_np.pi / (2 * kra)) * besselj(n + 5, kra)
        jn_krb = _np.sqrt(_np.pi / (2 * krb)) * besselj(n + 5, krb)
        exp = jn_krb / jn_kra

        if _np.any(_np.abs(exp) > 1e2):  # 40dB
            print('WARNING: Extrapolation might be unstable for one or more frequencies/orders!')

    elif problem == 'exterior':
        hn_kra = _np.sqrt(_np.pi / (2 * kra)) * hankel1(nvector + 0.5, 1, kra)
        hn_krb = _np.sqrt(_np.pi / (2 * krb)) * hankel1(nvector + 0.5, 1, krb)
        exp = hn_krb / hn_kra
    else:
        raise ValueError('Problem selector ' + problem + ' not recognized. Please either choose "interior" [Default] or "exterior".')

    return Pnm_kra * exp.T


def iFFT(Y, output_length=None, window=False):
    """ Inverse real-valued Fourier Transform

    Parameters
    ----------
    Y : array_like
       Frequency domain data [Nsignals x Nbins]
    output_length : int, optional
       Lenght of returned time-domain signal (Default: 2 x len(Y) + 1)
    win : boolean, optional
       Weights the resulting time-domain signal with a Hann

    Returns
    -------
    y : array_like
       Reconstructed time-domain signal
    """
    Y = _np.atleast_2d(Y)
    y = _np.fft.irfft(Y, n=output_length)

    if window:
        if window not in {'hann', 'hamming', 'blackman', 'kaiser'}:
            raise ValueError('Selected window must be one of hann, hamming, blackman or kaiser')
        no_of_signals, no_of_samples = y.shape

        if window == 'hann':
            window_array = _np.hanning(no_of_samples)
        elif window == 'hamming':
            window_array = _np.hamming(no_of_samples)
        elif window == 'blackman':
            window_array = _np.blackman(no_of_samples)
        elif window == 'kaiser':
            window_array = _np.kaiser(no_of_samples, 3)
        y = window_array * y
    return y


def wdr(Pnm, xAngle, yAngle, zAngle):
    """W/D/R Wigner-D Rotation - NOT YET IMPLEMENTED

    Parameters
    ----------
    Pnm : array_like
       Spatial Fourier coefficients
    xAngle, yAngle, zAngle : float
       Rotation angle around the x/y/z-Axis

    Returns
    -------
    PnmRot: array_like
       Rotated spatial Fourier coefficients
    """
    print('!WARNING. Wigner-D Rotation is not yet implemented. Continuing with un-rotated coefficients!')

    return Pnm


def convolve(A, B, FFT=None):
    """ Convolve two arrrays A & B row-wise. One or both can be one-dimensional for SIMO/SISO convolution

    Parameters
    ----------
    A, B: array_like
       Data to perform the convolution on of shape [Nsignals x NSamples]
    FFT: bool, optional
       Selects wether time or frequency domain convolution is applied. Default: On if Nsamples > 500 for both

    Returns
    -------
    out: array
       Array containing row-wise, linear convolution of A and B
    """
    A = _np.atleast_2d(A)
    B = _np.atleast_2d(B)

    N_sigA, L_sigA = A.shape
    N_sigB, L_sigB = B.shape

    if FFT is None and (L_sigA > 500 and L_sigB > 500):
        FFT = True
    else:
        FFT = False

    if (N_sigA != N_sigB) and not (N_sigA == 1 or N_sigB == 1):
        raise ValueError('Number of rows must either match or at least one must be one-dimensional.')

    if N_sigA == 1 and N_sigB != 1:
        A = _np.broadcast_to(A, (N_sigB, L_sigA))
    elif N_sigA != 1 and N_sigB == 1:
        B = _np.broadcast_to(B, (N_sigA, L_sigB))

    out = []

    for IDX, cur_row in enumerate(A):
        if FFT:
            out.append(fftconvolve(cur_row, B[IDX]))
        else:
            out.append(_np.convolve(cur_row, B[IDX]))

    return _np.array(out)
