"""
Processing functions:
- pdc: Plane Wave Decomposition
- tdt: Time Domain Reconstruction
- stc: Fast Spatial Fourier Transform
- itc: Fast Inverse Spatial Fourier Transform
"""

import numpy as _np
from scipy.signal import hann, resample
from .sph import sph_harm

pi = _np.pi


def pdc(N, OmegaL, Pnm, dn, cn=None, printInfo=True):
    """
        Y = pdc(N, OmegaL, Pnm, dn, [cn])
    ------------------------------------------------------------------------
    Y      MxN Matrix of the decomposed wavefield
           Col - Look Direction as specified in OmegaL
           Row - kr bins
    ------------------------------------------------------------------------
    N      Decomposition Order
    OmegaL Look Directions (Vector)
           Col - L1, L2, ..., Ln
           Row - AZn ELn
    Pnm    Spatial Fourier Coefficients from SOFiA S/T/C
    dn     Modal Array Filters from SOFiA M/F
    cn     (Optional) Weighting Function
           Can be used for N=0...N weigths:
           Col - n...N
           Row - 1
           Or n(f)...N(f) weigths:
           Col - n...N
           Row - kr bins
           If cn is not specified a PWD will be done
    """

    if printInfo:
        print('SOFiA P/D/C - Plane Wave Decomposition')

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


def tdt(Y, win=0, minPhase=False, resampleFactor=1, printInfo=True):
    """
    y = tdt(Y, [win], [resampleFactor], [minPhase])
    ------------------------------------------------------------------------
    y                  Reconstructed Time Domain Signal
                       Columns : Index / Channel: IR1, IR2, ... IRn
                       Rows    : Impulse responses (time domain)
    ------------------------------------------------------------------------
    Y                  Frequency domain FFT data for multiple channels
                       Columns : Index / Channel
                       Rows    : FFT data (frequency domain)

    [win]              Window Signal tail [0...1] with a HANN window
                       0    off (#default)
                       0-1  window coverage (1 full, 0 off)

    [resampleFactor]   Optional resampling: Resampling factor
                       e.g. FS_target/FS_source

    [minPhase]         Optional minimum phase reduction
                       0 off (#default)
                       1 on

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
        print('SOFiA T/D/T - Time Domain Transform')

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


def itc(Pnm, angles, N=None, printInfo=True):
    """I/T/C Fast Inverse spatial Fourier Transform Core
    p = sofia_itc(Pnm, angles, [N])
    ------------------------------------------------------------------------
    p      sound pressures (complex data)
           Columns : FFT bins
           Rows    : angles
    ------------------------------------------------------------------------
    Pnm    spatial Fourier coefficients (e.g. from SOFiA S/T/C)
           Columns : FFT bins
           Rows    : nm coeff

    angles target angles [AZ1 EL1; AZ2 EL2; ... AZn ELn]
           Columns : Angle Number 1...n
           Rows    : AZ EL

    [N]     *** Optional: Maximum transform order
               If not specified the highest order available included in
               the Pnm coefficients will be taken.

    This is a pure ISFT core that does not involve extrapolation.
    (=The pressures are referred to the original radius)
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

    if printInfo:
        print('SOFiA I/T/C - Inverse spatial Transform Core R13-0306')

    OutputArray = _np.zeros([numberOfAngles, FFTBlocklength], dtype=_np.complex_)

    ctr = 0
    for n in range(0, N + 1):
        for m in range(-n, n + 1):
            SHresults = sph_harm(m, n, AzimuthAngles, ElevationAngles)
            OutputArray += _np.outer(SHresults, Pnm[ctr])
            ctr += 1

    return OutputArray


def stc(N, fftData, grid):
    '''Pnm = process.stc(N, fftData, grid)
    ------------------------------------------------------------------------
    Pnm      Spatial Fourier Coefficients
             Columns : nm coeff
             Rows    : FFT bins
    ------------------------------------------------------------------------
    N        Maximum transform order

    fftData  Frequency domain sounfield data, e.g. from fdt
             Columns : number of spatial sampling position
             Rows    : FFT bins (complex sound pressure data)

    grid     Sample grid configuration
             Columns : s=1...S spatial positions
             Rows    : [AZ_s EL_s GridWeight_s]
             AZ in [0...2pi] and EL [0...pi] in RAD
    '''

    if not _np.max(_np.iscomplex(fftData)):
        raise ValueError('FFTData: Complex Input Data expected.')

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


def fdt(timeData, FFToversize=1, firstSample=0, lastSample=None):
    '''F/D/T frequency domain transform
    [fftData, kr, f, ctSig] = sofia_fdt(timeData, FFToversize, firstSample, lastSample)
     ------------------------------------------------------------------------
     fftData           Frequency domain data ready for the Spatial Fourier Transform (stc)
     kr                kr-Values of the delivered data
     f                 Absolute frequency scale
     ctSig             Center signal, if available
     ------------------------------------------------------------------------
     timeData          Named tuple with minimum fields:
                       * .impulseResponses     [Channels X Samples]
                       * .FS
                       * .radius               Array radius
                       * .averageAirTemp       Temperature in [C]
                       (* .centerIR            [1 x Samples] )

     FFToversize       FFToversize rises the FFT Blocksize.   [default = 2]
                       A FFT of the blocksize (FFToversize*NFFT) is applied
                       to the time domain data,  where  NFFT is determinded
                       as the next power of two of the signalSize  which is
                       signalSize = (lastSample-firstSample).
                       The function will pick a window of
                       (lastSample-firstSample) for the FFT:
     firstSample       First time domain sample to be included. Default=0
     lastSample        Last time domain sample to be included. Default=None

    Call this function with a running window (firstSample+td->lastSample+td)
    iteration increasing td to obtain time slices. This way you resolve the
    temporal information within the captured sound field.
    '''

    IR = timeData.impulseResponse
    FS = timeData.FS
    temperature = timeData.averageAirTemp
    radius = timeData.radius

    N = IR.shape[1]

    if lastSample is None:  # assign lastSample to length of IR if not provided
        lastSample = N

    if FFToversize < 1:
        raise ValueError('FFToversize must be >= 1.')

    if lastSample < firstSample or lastSample > N:
        raise ValueError('lastSample must be between firstSample and N (length of impulse response).')

    if firstSample < 0 or firstSample > lastSample:
        raise ValueError('firstSample must be between 0 and lastSample.')

    totalSamples = lastSample - firstSample + 1
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


def rfi(dn, kernelDownScale=2, highPass=False):
    '''R/F/I Radial Filter Improvement
    [dn, kernelSize, latency] = rfi(dn, kernelDownScale, highPass)
    ------------------------------------------------------------------------
    dn                 Improved radial filters
    kernelSize         Filter kernel size (total)
    latency            Approximate signal latency due to the filters

    ------------------------------------------------------------------------
    dn                 Analytical frequency domain radial filters from SOFiA M/F
    kernelDownScale    Downscale factor for the filter kernel #default: 2
    highPass           Highpass Filter 0:highPass:1
                    highPass = 1 corresponds to the maximum kr available.
                    highPass = 0 filter off (#default)
    INFORMATION: If HPF is on (highPass>0) the radial filter kernel is
              downscaled by a factor of two. Radial Filters and HPF
              share the available taps and the latency keeps constant.
              Be careful using very small signal blocks because there
              may remain too few taps. Observe the filters by plotting
              their spectra and impulse responses.
              > Be very carefull if NFFT/max(kr) < 25
              > Do not use R/F/I if NFFT/max(kr) < 15

    This function improves the FIR radial filters from SOFiA M/F. The filters
    are made causal and are windowed in time domain. The DC components are
    estimated. The R/F/I module should always be inserted to the filter
    path when treating measured data even if no use is made of the included
    kernel downscaling or highpass filters.

    Do NOT use R/F/I for single open sphere filters (e.g.simulations).

    IMPORTANT: Remember to choose a fft-oversize factor (F/D/T) being large
            enough to cover all filter latencies and reponse slopes.
            Otherwise undesired cyclic convolution artifacts may appear
            in the output signal.
    '''
    return dn
