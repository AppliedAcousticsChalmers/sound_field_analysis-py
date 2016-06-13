"""
Processing functions:
- Plane Wave Decomposition (pdc)
- Time domain signals reconstruction (tdt)
"""

import numpy as _np
from scipy.signal import hann, resample
from .sph import sph_harm

pi = _np.pi


def pdc(N, OmegaL, Pnm, dn, **kargs):
    """
        Y = ASAR_PDC(N, OmegaL, Pnm, dn, [cn])
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

    printInfo = kargs['printInfo'] if 'printInfo' in kargs else True

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
        numberOfAngles = OmegaL.shape[1]

    Azimut = OmegaL[:, 0]
    Elevation = OmegaL[:, 1]

    # Extract blocksizes. TODO: Error handle
    NMDeliveredSize = Pnm.shape[0]
    FFTBlocklengthPnm = Pnm.shape[1]

    Ndn = dn.shape[0]
    FFTBlocklengthdn = dn.shape[1]

    if 'cn' in kargs:
        cn = kargs['cn']
        pwdflag = 0
        Ncn = cn.shape[0]
        FFTBlocklengthcn = cn.shape[1]
        cnnofreqflag = 0 if _np.asarray(cn).ndim == 1 else 1
    else:
        pwdflag = 1

    # Check blocksizes
    if FFTBlocklengthdn != FFTBlocklengthPnm:
        raise ValueError('FFT Blocksizes of Pnm and dn are not consistent.')
    if 'cn' in kargs:
        if FFTBlocklengthcn != FFTBlocklengthPnm and FFTBlocklengthcn != 1:
            raise ValueError('FFT Blocksize of cn is not consistent to Pnm and dn.')

    NMLocatorSize = pow(N + 1, 2)
    # TODO: Implement all other warnings
    if NMLocatorSize > NMDeliveredSize:  # Maybe throw proper warning?
        print('WARNING: The requested order N=', N, 'cannot be achieved.\n'
              'The Pnm coefficients deliver a maximum of', int(_np.sqrt(NMDeliveredSize) - 1), '\n'
              'Will decompose on maximum available order.\n\n')

    gaincorrection = 4 * pi / pow(N + 1, 2)

    OutputArray = _np.zeros((numberOfAngles, FFTBlocklengthPnm), dtype=_np.complex_)

    ctr = 0

    # TODO: clean up for loops
    if pwdflag == 1:  # PWD CORE
        for n in range(0, N + 1):
            for m in range(-n, n + 1):
                for omegactr in range(0, numberOfAngles):
                    Ynm = sph_harm(m, n, Azimut[omegactr], Elevation[omegactr])
                    for f in range(0, FFTBlocklengthPnm):
                        OutputArray[omegactr][f] = OutputArray[omegactr][f] + Ynm * Pnm[ctr][f] * dn[n][f]
                ctr = ctr + 1
    else:  # BEAMFORMING CORE
        for n in range(0, N + 1):
            for m in range(-n, n + 1):
                for omegactr in range(0, numberOfAngles):
                    Ynm = sph_harm(m, n, Azimut[omegactr], Elevation[omegactr])
                    for f in range(0, FFTBlocklengthPnm):
                        OutputArray[omegactr][f] = OutputArray[omegactr][f] + Ynm * Pnm[ctr][f] * dn[n][f] * cn[n][f]
                ctr = ctr + 1
    # RETURN
    return OutputArray * gaincorrection


def tdt(Y, **kargs):
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

    # Get optional arguments
    win = kargs['win'] if 'win' in kargs else 0
    if win > 1:
        raise ValueError('Argument window must be in range 0.0 ... 1.0!')
    minPhase = kargs['minPhase'] if 'minPhase' in kargs else 0
    resampleFactor = kargs['resampleFactor'] if 'resampleFactor' in kargs else 1
    printInfo = kargs['printInfo'] if 'printInfo' in kargs else True

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


def itc(Pnm, angles, **kargs):
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

    N = kargs['N'] if 'N' in kargs else Nmax
    printInfo = kargs['printInfo'] if 'printInfo' in kargs else True

    if printInfo:
        print('SOFiA I/T/C - Inverse spatial Transform Core R13-0306')

    OutputArray = _np.zeros([numberOfAngles, FFTBlocklength], dtype=_np.complex_)

    ctr = 0
    for n in range(0, N + 1):
        for m in range(-n, n + 1):
            for j in range(0, numberOfAngles):
                SHresults = sph_harm(m, n, AzimuthAngles[j], ElevationAngles[j])

                for f in range(0, FFTBlocklength):
                    OutputArray[j][f] += SHresults * Pnm[ctr][f]
            ctr += 1

    return OutputArray
