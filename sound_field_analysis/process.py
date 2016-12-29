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
from scipy.signal import hann, resample
from .sph import sph_harm, besselj, besselh
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


def FFT(timeData, FFToversize=1, firstSample=0, lastSample=None):
    '''(Fast) Fourier Transform

    Parameters
    ----------
    timeData : named tuple
       timeData tuple with following fields
       ::
          .impulseResponses [Channels X Samples]
          .FS
          .radius           Array radius
          .averageAirTemp   Temperature in [C]
          (.centerIR        [1 x Samples] )
    FFToversize : int, optional
       FFToversize > 1 increase the FFT Blocksize. [Default: 1]
    firstSample : int, optional
       First time domain sample to be included. [Default: 0]
    lastSample : int, optional
       Last time domain sample to be included. [Default: -1]

    Returns
    -------
    fftData : array_like
       Frequency domain data ready for the Spatial Fourier Transform (stc)
    kr : array_like
       kr-Values of the delivered data
    f : array_like
       Absolute frequency scale
    ctSig : array_like
       Center signal, if available

    Note
    ----
    A FFT of the blocksize (FFToversize*NFFT) is applied
    to the time domain data,  where  NFFT is determinded
    as the next power of two of the signalSize  which is
    signalSize = (lastSample-firstSample).
    The function will pick a window of (lastSample-firstSample)
    for the FFT.

    Call this function with a running window (firstSample+td->lastSample+td)
    iteration increasing td to obtain time slices. This way you resolve the
    temporal information within the captured sound field.
    '''

    IR = timeData.impulseResponse
    FS = timeData.FS
    temperature = timeData.averageAirTemp
    radius = timeData.radius

    N, IRLength = timeData.impulseResponse.shape

    if lastSample is None:  # assign lastSample to length of IR if not provided
        lastSample = IRLength

    if FFToversize < 1:
        raise ValueError('FFToversize must be >= 1.')

    if lastSample < firstSample or lastSample > IRLength:
        raise ValueError('lastSample must be between firstSample and IRLength.')

    if firstSample < 0 or firstSample > lastSample:
        raise ValueError('firstSample must be between 0 and lastSample.')

    totalSamples = lastSample - firstSample
    IR = IR[:, firstSample:lastSample]
    NFFT = int(2**_np.ceil(_np.log2(totalSamples)))
    fftData = _np.fft.rfft(IR, NFFT * FFToversize, 1)

    if timeData.centerIR.any():
        centerIR = timeData.centerIR[:, firstSample:lastSample]
        ctSig = _np.fft.rfft(centerIR, NFFT * FFToversize)
    else:
        ctSig = []

    f = _np.fft.rfftfreq(NFFT, d=1 / FS)
    c = 331.5 + 0.6 * temperature
    kr = 2 * pi * f / c * radius

    return fftData, kr, f, ctSig


def iSpatFT(Pnm, angles, N=None, printInfo=True):
    """Inverse spatial Fourier Transform

    Parameters
    ----------
    Pnm : array_like
       Spatial Fourier coefficients with FFT bins as cols and nm coeffs as rows (e.g. from spatFT)
    angles : array_like
       Target angles of shape
       ::
          [AZ1, EL1;
           AZ2, EL2;
             ...
           AZn, ELn]
    [N] : int, optional
       Maximum transform order [Default: highest available order]

    Returns
    -------
    p : array of complex floats
       Sound pressures with FFT bins in cols and specified angles in rows

    Note
    ----
    This transform does not involve extrapolation. (=The pressures are referred to the original radius)
    """

    if angles.ndim == 1 and angles.shape[0] == 2:
        AzimuthAngles = _np.array([angles[0]])
        ElevationAngles = _np.array([angles[1]])
        numberOfAngles = 1
    elif angles.ndim == 2 and angles.shape[1] > 1:
        numberOfAngles = angles.shape[0]
        AzimuthAngles = _np.asarray(angles[:, 0])
        ElevationAngles = _np.asarray(angles[:, 1])
    else:
        raise ValueError('Error: Delivered angles are not valid. Must consist of [AZ1 EL1; AZ2 EL2; ...; AZn ELn] pairs.')

    try:
        PnmDataLength = Pnm.shape[0]
        FFTBlocklength = Pnm.shape[1]
    except:
        print('Supplied Pnm matrix needs to be of [m x n] dimensions, with [m] FFT bins of [n] coefficients.')

    Nmax = int(_np.sqrt(PnmDataLength - 1))
    if N is None:
        N = Nmax

    OutputArray = _np.zeros([numberOfAngles, FFTBlocklength], dtype=_np.complex_)

    ctr = 0
    for n in range(0, N + 1):
        if printInfo:
            progress_bar(ctr, N ** 2, 'iSpatFT - Inverse spatial Transform')
        for m in range(-n, n + 1):
            SHresults = sph_harm(m, n, AzimuthAngles, ElevationAngles)
            OutputArray += _np.outer(SHresults, Pnm[ctr])
            ctr += 1
    return OutputArray


def PWDecomp(N, OmegaL, Pnm, dn, cn=None, printInfo=True):
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

    if printInfo:
        print('PWDecomp - Plane Wave Decomposition')

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
        hn_kra = _np.sqrt(pi / (2 * kra)) . besselh(nvector + 0.5, 1, kra)
        hn_krb = _np.sqrt(pi / (2 * krb)) . besselh(nvector + 0.5, 1, krb)
        exp = hn_krb / hn_kra
    else:
        raise ValueError('Problem selector ' + problem + ' not recognized. Please either choose "interior" [Default] or "exterior".')

    return Pnm_kra * exp.T


def spatFT(N, fftData, grid):
    ''' Fast Spatial Fourier Transform

    Parameters
    ----------
    N : int
       Maximum transform order
    fftData : array_like
       Frequency domain soundfield data, (e.g. from FFT()) with spatial sampling positions in cols and FFT bins in rows
    grid : array_like
       Grid configuration of AZ [0 ... 2pi], EL [0...pi] and W of shape
       ::
          [AZ1, EL1, W1;
           AZ2, EL2, W2;
             ...
           AZn, ELn, Wn]

    Returns
    -------
    Pnm : array_like
       Spatial Fourier Coefficients with nm coeffs in cols and FFT bins in rows
    '''

    numberOfSpatialPositionsInFFTBlock, FFTBlocklength = fftData.shape
    numberOfGridPoints, numberOfGridInfos = grid.shape
    if numberOfGridInfos < 3:
        raise ValueError('GRID: Invalid grid data, must contain [az, el, r].')

    if numberOfSpatialPositionsInFFTBlock != numberOfGridPoints:
        raise ValueError('Inconsistent spatial sampling points between fftData (' + str(numberOfSpatialPositionsInFFTBlock) + ') and supplied grid  (' + str(numberOfGridPoints) + ').')

    AzimuthAngles = grid[:, 0]
    ElevationAngles = grid[:, 1]
    GridWeights = grid[:, 2]

    OutputArray = _np.zeros([(N + 1) ** 2, FFTBlocklength], dtype=_np.complex_)

    ctr = 0
    for n in range(0, N + 1):
        for m in range(-n, n + 1):
            SHarm = 4 * pi * GridWeights * _np.conj(sph_harm(m, n, AzimuthAngles, ElevationAngles))
            OutputArray[ctr] += _np.inner(SHarm, fftData.T)
            ctr += 1
    return OutputArray


def iFFT(Y, win=0, minPhase=False, resampleFactor=1, printInfo=True):
    """ Inverse (Fast) Fourier Transform

    Parameters
    ----------
    Y : array_like
       Frequency domain data over multiple channels (cols) with FFT data in rows
    win float, optional
       Window Signal tail [0...1] with a HANN window [Default: 0] - NOT YET IMPLEMENTED
    resampleFactor int, optional
       Resampling factor (FS_target/FS_source)
    minPhase bool, optional
       Ensure minimum phase reduction - NOT YET IMPLEMENTED [Default: False]

    Returns
    -------
    y : array_like
       Reconstructed time-domain signal of channels in cols and impulse responses in rows

    Note
    ----
    This function recombines time domain signals for multiple channels from
    frequency domain data. It is made to work with half-sided spectrum FFT
    data.  The impulse responses can be windowed.  The IFFT blocklength is
    determined by the Y data itself:

    Y should have a size [NumberOfChannels x ((2^n)/2)+1] with n=[1,2,3,...]
    and the function returns [NumberOfChannels x resampleFactor*2^n] samples.
    """
    if win > 1:
        raise ValueError('Argument window must be in range 0.0 ... 1.0!')

    if printInfo:
        print('iFFT - inverse Fourier Transform')

    # inverse real FFT
    y = _np.fft.irfft(Y)

    # TODO: minphase
    if minPhase != 0:
        # y    = [y, zeros(size(y))]';
        # Y    = fft(y);
        # Y(Y == 0) = 1e-21;
        # img  = imag(hilbert(log(abs(Y))));
        # y    = real(ifft(abs(Y) .* exp(-1i*img)));
        # y    = y(1:end/2,:)';
        pass

    # TODO: percentage(?) windowing
    if win != 0:
        winfkt = hann(y.shape[1])
        y = winfkt * y

    if resampleFactor != 1:
        y = resample(y, _np.round(y.shape[1] / resampleFactor), axis=1)

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
