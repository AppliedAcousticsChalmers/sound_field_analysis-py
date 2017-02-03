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

Not yet implemented:
`BEMA`
   BEMA Spatial Anti-Aliasing
`rfi`
   Radial filter improvement
`sfe`
   Sound field extrapolation
`wdr`
   Wigner-D Rotation
"""

import numpy as _np
from scipy.signal import hann, fftconvolve
from scipy.linalg import lstsq
from .sph import sph_harm, besselj, hankel1, sph_harm_all
from .utils import progress_bar

pi = _np.pi


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


def FFT(time_signals, fs, NFFT=None, oversampling=1, first_sample=0, last_sample=None):
    '''Real-valued Fast Fourier Transform.

    Parameters
    ----------
    time_signals : array_like
       Time-domain signals to be transformed, of shapeb [nSig x nSamples]
    fs : int
       Sampling frequency of the time data
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

    time_signals = _np.atleast_2d(time_signals)
    nSig, nSamples = time_signals.shape

    if last_sample is None:  # assign lastSample to length of signals if not provided
        last_sample = nSamples

    if oversampling < 1:
        raise ValueError('oversampling must be >= 1.')

    if last_sample < first_sample or last_sample > nSamples:
        raise ValueError('lastSample must be between firstSample and nSamples.')

    if first_sample < 0 or first_sample > last_sample:
        raise ValueError('firstSample must be between 0 and lastSample.')

    total_samples = last_sample - first_sample
    time_signals = time_signals[:, first_sample:last_sample]

    if not NFFT:
        NFFT = int(2**_np.ceil(_np.log2(total_samples)))

    fftData = _np.fft.rfft(time_signals, NFFT * oversampling, 1)
    f = _np.fft.rfftfreq(NFFT * oversampling, d=1 / fs)

    return fftData, f


def spatFT(data, azimuths, colatitudes, gridweights, order_max=10, spherical_harmonic_bases=None):
    ''' Spatial Fourier Transform

    Parameters
    ----------
    data : array_like
       Data to be transformed, with signals in rows and frequency bins in columns
    order_max : int, optional
       Maximum transform order (Default: 10)
    azimuths, colatitudes, gridweights : array_like
       Azimuths/Colatitudes/Gridweights of spatial sampling points

    Returns
    -------
    Pnm : array_like
       Spatial Fourier Coefficients with nm coeffs in rows and FFT bins in columns
    '''
    data = _np.atleast_2d(data)
    number_of_signals, FFTLength = data.shape

    # Re-generate spherical harmonic bases if they were not provided or their order is too small
    if (spherical_harmonic_bases is None or
            spherical_harmonic_bases.shape[0] < number_of_signals or
            spherical_harmonic_bases.shape[1] < (order_max + 1) ** 2):
        print('Regenerating spherical harmonic bases')
        spherical_harmonic_bases = sph_harm_all(order_max, azimuths, colatitudes)

    spherical_harmonic_bases = (_np.conj(spherical_harmonic_bases).T * (4 * pi * gridweights))
    return _np.inner(spherical_harmonic_bases, data.T)


def iSpatFT(spherical_coefficients, azimuths, colatitudes, order_max=None, spherical_harmonic_bases=None):
    """Inverse spatial Fourier Transform

    Parameters
    ----------
    spherical_coefficients : array_like
       Spatial Fourier coefficients with columns representing frequncy bins
    azimuths, colatitudes : array_like
       Azimuth/Colatitude angles of spherical coefficients
    order_max : int, optional
       Maximum transform order [Default: highest available order]

    Returns
    -------
    P : array_like
       Sound pressures with frequency bins in columnss and angles in rows
    """

    spherical_coefficients = _np.atleast_2d(spherical_coefficients)
    number_of_coefficients, FFTLength = spherical_coefficients.shape

    # todo: check coeffs and bases length correspond with order_max
    if order_max is None:
        order_max = int(_np.sqrt(number_of_coefficients) - 1)

    # Re-generate spherical harmonic bases if they were not provided or their order is too small
    if spherical_harmonic_bases is None or spherical_harmonic_bases.shape[1] < number_of_coefficients or spherical_harmonic_bases.shape[1] == azimuths.size:
        print('Regenerating spherical harmonic bases')
        spherical_harmonic_bases = sph_harm_all(order_max, azimuths, colatitudes)

    return _np.inner(spherical_harmonic_bases, spherical_coefficients.T)


def spatFT_LSF(data, azimuths, colatitudes, order_max, spherical_harmonic_bases=None):
    '''Returns spherical harmonics coefficients least square fitted to provided data

    Parameters
    ----------
    data : array_like, complex
       Data to be fitted to
    azimuth_grid, colatitude_grid : array_like, float
       Azimuth / colatidunenal data locations
    order_max: int
       Maximum order N of fit

    Returns
    coefficients: array_like, float
       Fitted spherical harmonic coefficients (indexing: n**2 + n + m + 1)
    '''
    if spherical_harmonic_bases is None:
        spherical_harmonic_bases = sph_harm_all(order_max, azimuths, colatitudes)
    return lstsq(spherical_harmonic_bases, data)[0]


def PWDecomp(N, OmegaL, Pnm, dn, cn=None):
    """Plane Wave Decomposition

    Parameters
    ----------
    N : int
       Decomposition order
    OmegaL : array_like
       Look directions of shape
       ::
          [AZ1, EL1;
           AZ2, EL2;
             ...
           AZn, ELn]
    Pnm : matrix of complex floats
       Spatial Fourier Coefficients (e.g. from spatFT)
    dn : matrix of complex floats
       Radial filters (e.g. from radFilter)
    cn : array_like, optional
       Weighting Function. Either frequency invariant weights as 1xN array
       or with kr bins in rows over N cols. [Default: None]

    Returns
    -------
    Y : matrix of floats
       MxN Matrix of the decomposed wavefield with kr bins in rows
    """
    if N < 0:
        N = 0

    # Check shape of supplied look directions
    OmegaL = _np.asarray(OmegaL)
    if OmegaL.ndim == 1:  # only one dimension -> one AE/EL pair
        if OmegaL.size != 2:
            raise ValueError('Angle Matrix OmegaL is not valid. Must consist of AZ/EL pairs in one column [AZ1 EL1; AZ2 EL2; ... ; AZn ELn].\nRemember: All angles are in RAD.')
        numberOfAngles = 1
    else:                 # else: two or more AE/EL pairs
        if OmegaL.shape[1] != 2:
            raise ValueError('Angle Matrix OmegaL is not valid. Must consist of AZ/EL pairs in one column [AZ1 EL1; AZ2 EL2; ... ; AZn ELn].\nRemember: All angles are in RAD.')
        numberOfAngles = OmegaL.shape[0]

    Azimut = OmegaL[:, 0]
    Elevation = OmegaL[:, 1]

    # Expand Pnm and dn dims to 2D if necessary
    if Pnm.ndim == 1:
        Pnm = _np.expand_dims(Pnm, 1)

    NMDeliveredSize = Pnm.shape[0]
    FFTBlocklengthPnm = Pnm.shape[1]

    if dn.ndim == 1:
        dn = _np.expand_dims(dn, 1)
    Ndn = dn.shape[0]
    FFTBlocklengthdn = dn.shape[1]

    if cn is not None:
        pwdflag = 0
        Ncn = cn.shape[0]
        FFTBlocklengthcn = cn.shape[1]
        cnnofreqflag = 0 if _np.asarray(cn).ndim == 1 else 1
    else:
        pwdflag = 1

    # Check blocksizes
    if FFTBlocklengthdn != FFTBlocklengthPnm:
        raise ValueError('FFT Blocksizes of Pnm and dn are not consistent.')
    if cn is not None:
        if FFTBlocklengthcn != FFTBlocklengthPnm and FFTBlocklengthcn != 1:
            raise ValueError('FFT Blocksize of cn is not consistent to Pnm and dn.')

    NMLocatorSize = pow(N + 1, 2)
    # TODO: Implement all other warnings
    if NMLocatorSize > NMDeliveredSize:  # Maybe throw proper warning?
        print('WARNING: The requested order N=', N, 'cannot be achieved.\n'
              'The Pnm coefficients deliver a maximum of', int(_np.sqrt(NMDeliveredSize) - 1), '\n'
              'Will decompose on maximum available order.\n\n')

    gaincorrection = 4 * pi / pow(N + 1, 2)

    OutputArray = _np.squeeze(_np.zeros((numberOfAngles, FFTBlocklengthPnm), dtype=_np.complex_))

    ctr = 0

    # TODO: clean up for loops
    if pwdflag == 1:  # PWD CORE
        for n in range(0, N + 1):
            for m in range(-n, n + 1):
                Ynm = sph_harm(m, n, Azimut, Elevation)
                OutputArray += _np.squeeze(_np.outer(Ynm, Pnm[ctr] * dn[n]))
                ctr = ctr + 1
    else:  # BEAMFORMING CORE
        for n in range(0, N + 1):
            for m in range(-n, n + 1):
                Ynm = sph_harm(m, n, Azimut, Elevation)
                OutputArray += _np.squeeze(_np.outer(Ynm, Pnm[ctr] * dn[n] * cn[n]))
                ctr = ctr + 1
    # RETURN
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
        jn_kra = _np.sqrt(pi / (2 * kra)) * besselj(n + 5, kra)
        jn_krb = _np.sqrt(pi / (2 * krb)) * besselj(n + 5, krb)
        exp = jn_krb / jn_kra

        if _np.any(_np.abs(exp) > 1e2):  # 40dB
            print('WARNING: Extrapolation might be unstable for one or more frequencies/orders!')

    elif problem == 'exterior':
        hn_kra = _np.sqrt(pi / (2 * kra)) * hankel1(nvector + 0.5, 1, kra)
        hn_krb = _np.sqrt(pi / (2 * krb)) * hankel1(nvector + 0.5, 1, krb)
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
