"""
Module contains various generator functions:

`awgn`
   Generate additive White Gaussian noise
`gaussGrid`
   Gauss-Legendre quadrature grid and weights
`lebedev`
   Lebedev quadrature grid and weigths
`mf`
   Modal Radial Filter
`swg`
   Sampled Wave Generator, emulating discrete sampling
`wgc`
   Wave Generator, returns spatial Fourier coefficients
"""
import numpy as _np
from .sph import bn, bn_npf, sphankel, sph_harm, cart2sph, sph2cart
from .process import itc
from .utils import progress_bar

pi = _np.pi


def awgn(fftData, noiseLevel=80, printInfo=True):
    '''Adds White Gaussian Noise of approx. 16dB crest to a FFT block.

    Parameters
    ----------
    fftData : array of complex floats
       Input fftData block (e.g. from F/D/T or S/W/G)
    noiseLevel : int, optional
       Average noise Level in dB [Default: -80dB]
    printInfo : bool, optional
       Toggle print statements [Default: True]

    Returns
    -------
    noisyData : array of complex floats
       Output fftData block including white gaussian noise
    '''
    if printInfo:
        print('SOFiA A/W/G/N - Additive White Gaussian Noise Generator')

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


def lebedev(degree, plot=False, printInfo=True):
    '''Compute Lebedev quadrature nodes and weigths.

    Parameters
    ----------
    Degree : int
       Lebedev Degree. Currently available: 6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194
    plot : bool, optional
       Plot selected Lebedev grid [Default: False]

    Returns
    -------
    gridData : array_like
       Lebedev quadrature positions and weigths: [AZ, EL, W]
    Nmax : int
       Highest stable grid order
    '''
    from sofia import lebedev

    if printInfo:
        print('SOFiA Lebedev Grid')

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

    if plot:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_zlim(-1.01, 1.01)

        # Create sphere
        u = _np.linspace(0, 2 * pi, 20)
        v = _np.linspace(0, pi, 20)

        X = _np.outer(_np.cos(u), _np.sin(v))
        Y = _np.outer(_np.sin(u), _np.sin(v))
        Z = _np.outer(_np.ones(_np.size(u)), _np.cos(v))

        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=1,
                        color='gray', alpha=0.6, antialiased=True)

        # Scatter points
        ax.scatter(leb.x, leb.y, leb.z, color='black', s=50)

        # Finish up
        plt.title('Lebedev configuration with ' + str(degree) + ' degrees')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    return gridData, Nmax


def mf(N, kr, ac, amp_maxdB=0, plc=0, fadeover=0, printInfo=True):
    """Generate modal radial filters

    Parameters
    ----------
    N : int
       Maximum Order
    kr : array_like
       Vector or Matrix of kr values
       ::
          First Row  (M=1) N: kr values microphone radius
          Second Row (M=2) N: kr values sphere/microphone radius
          [kr_mic;kr_sphere] for rigid/dual sphere configurations
          ! If only one kr-vector is given using a rigid/dual sphere
          Configuration: kr_sphere = kr_mic
    ac : int {0, 1, 2, 3, 4}
       Array configuration
         - `0`:  Open Sphere with p Transducers (NO plc!)
         - `1`:  Open Sphere with pGrad Transducers
         - `2`:  Rigid Sphere with p Transducers
         - `3`:  Rigid Sphere with pGrad Transducers
         - `4`:  Dual Open Sphere with p Transducers
    amp_maxdB : int, optional
       Maximum modal amplification limit in dB [Default: 0]
    plc : int {0, 1, 2}, optional
        OnAxis powerloss-compensation:
        - `0`:  Off [Default]
        - `1`:  Full kr-spectrum plc
        - `2`:  Low kr only -> set fadeover
    fadeover : int, optional
       Number of kr values to fade over +/- around min-distance
       gap of powerloss compensated filter and normal N0 filters.
       0 is auto fadeover [Default]

    Returns
    -------
    dn : array_like
       Vector of modal 0-N frequency domain filters
    beam : array_like
       Expected free field on-axis kr-response
    """
    a_max = pow(10, (amp_maxdB / 20))
    if amp_maxdB != 0:
        limiteronflag = True
    else:
        limiteronflag = False

    if printInfo:
        print('SOFiA M/F - Modal radial filter generator')

    if kr.ndim == 1:
        krN = kr.size
        krM = 1
    else:
        krM, krN = kr.shape

    # TODO: check input

    # TODO: correct krm, krs?
    # TODO: check / replace zeros in krm/krs
    if kr[0] == 0:
        kr[0] = kr[1]
    krm = kr
    krs = kr

    OutputArray = _np.empty((N + 1, krN), dtype=_np.complex_)

    # BN filter calculation
    amplicalc = 1
    ctrb = _np.array(range(0, krN))
    for ctr in range(0, N + 1):
        bnval = bn(ctr, krm[ctrb], krs[ctrb], ac)
        if limiteronflag:
            amplicalc = 2 * a_max / pi * abs(bnval) * _np.arctan(pi / (2 * a_max * abs(bnval)))
        OutputArray[ctr] = amplicalc / bnval

    if(krN < 32 & plc != 0):
        plc = 0
        print("Not enough kr values for PLC fading. PLC disabled.")

    # Powerloss Compensation Filter (PLC)
    noplcflag = 0
    if not plc:
        xi = _np.zeros(krN)
        for ctr in range(0, krN):
            for ctrb in range(0, N + 1):
                xi = xi + (2 * ctrb + 1) * (1 - OutputArray[ctrb][ctr] * bn(ctrb, krm[ctr], krs[ctr], ac))
            xi[ctr] = xi[ctr] * 1 / bn(0, krm[ctr], krs[ctr], ac)
            xi[ctr] = xi[ctr] + OutputArray[0][ctr]

    if plc == 1:  # low kr only
        minDisIDX = _np.argmin(_np.abs(OutputArray[0] - xi))
        minDis = _np.abs(OutputArray[0][minDisIDX] - xi[minDisIDX])

        filtergap = 20 * _np.log10(1 / _np.abs(OutputArray[0][minDisIDX] / xi[minDisIDX]))
        if printInfo:
            print("Filter fade gap: ", filtergap)
        if _np.abs(filtergap) > 20:
            print("Filter fade gap too large, no powerloss compensation applied.")
            noplcflag = 1

        if not noplcflag:
            if abs(filtergap) > 5:
                print("Filtergap is large (> 5 dB).")

            if fadeover == 0:
                fadeover = krN / 100
                if amp_maxdB > 0:
                    fadeover = fadeover / _np.ceil(a_max / 4)

            if fadeover > minDis | (minDis + fadeover) > krN:
                if minDisIDX - fadeover < krN - minDisIDX + fadeover:
                    fadeover = minDisIDX
                else:
                    fadeover = krN - minDisIDX
            if printInfo:
                print("Auto filter size of length: ", fadeover)
        # TODO: Auto reduce filter length
    elif plc == 2:  # Full spectrum
        OutputArray[0] = xi

    normalizeBeam = pow(N + 1, 2)

    BeamResponse = _np.zeros((krN), dtype=_np.complex_)
    for ctrb in range(0, N + 1):             # ctrb = n
        # m = 2 * ctrb + 1
        BeamResponse += (2 * ctrb + 1) * bn(ctrb, krm, krs, ac) * OutputArray[ctrb]
    BeamResponse /= normalizeBeam

    return OutputArray, BeamResponse


def swg(r=0.01, gridData=None, ac=0, FS=48000, NFFT=512, AZ=0, EL=_np.pi / 2,
        c=343, wavetype=0, ds=1, Nlim=120, printInfo=True):
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
    wave impinging to an array. The wrapper involves the W/G/C wave
    generator core and the I/T/C spatial transform core.

    S/W/G emulates discrete sampling. You can observe alias artifacts.
    """

    if gridData is None:
        gridData = lebedev(110)

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
    elif minOrderLim == Ng:
        if printInfo:
            print('Full spectrum generator order: ' + str(Ng))
    else:
        if printInfo:
            print('Segmented generator orders: ' + str(minOrderLim) + ' to ' + str(Ng))

    # SEGMENTATION
    # index = 1
    # ctr = -1
    Pnm = _np.zeros([(Ng + 1) ** 2, int(NFFT / 2 + 1)], dtype=_np.complex_)

    for idx, order in progress_bar(enumerate(_np.unique(rqOrders))):
        fOrders = _np.flatnonzero(rqOrders == order)
        Pnm += wgc(Ng, r, ac, FS, NFFT, AZ, EL, wavetype=wavetype, ds=ds, lowerSegLim=fOrders[0], upperSegLim=fOrders[-1], SegN=order, printInfo=False)[0]
    fftData = itc(Pnm, gridData)

    return fftData, kr


def wgc(N, r, ac, fs, F_NFFT, az, el, t=0.0, c=343.0, wavetype=0, ds=1.0, lowerSegLim=0,
        SegN=None, upperSegLim=None, printInfo=True):
    """
    Wave Generator Core:
    Returns Spatial Fourier Coefficients `Pnm` and `kr` vector

    Parameters
    ----------
    N : int
        Maximum transform order.
    r  : list of ints
       Microphone radius
       ::
          Can also be a vector for rigid/dual sphere configurations:
          [1,1] => rm  Microphone radius
          [2,1] => rs  Sphere or microphone radius
          ! If only one radius (rm) is given using a rigid/dual sphere
          Configuration: rs = rm and only one kr-vector is returned!
    ac : int {0, 1, 2, 3, 4}
       Array Configuration:
        - `0`:  Open Sphere with p Transducers (NO plc!)
        - `1`:  Open Sphere with pGrad Transducers
        - `2`:  Rigid Sphere with p Transducers
        - `3`:  Rigid Sphere with pGrad Transducers
        - `4`:  Dual Open Sphere with p Transducers
    FS : int
       Sampling frequency
    NFFT : int
       Order of FFT (number of bins), should be a power of 2.
    AZ : float
       Azimuth angle in radians [0-2pi].
    EL : float
       Elevation angle in in radians [0-pi].
    t : float, optional
       Time Delay in s.
    c : float, optional
       Speed of sound in [m/s] [Default: 343m/s]
    wavetype : int {0, 1}, optional
       Type of the Wave:
        - 0: Plane Wave [Default]
        - 1: Spherical Wave
    ds : float, optional
       Distance of the source in [m] (For wavetype = 1 only)
    lSegLim : int, optional
       (Lower Segment Limit) Used by the S/W/G wrapper
    uSegLim : int, optional
       (Upper Segment Limit) Used by the S/W/G wrapper
    SegN : int, optional
        (Sement Order) Used by the S/W/G wrapper
    printInfo: bool, optional
       Toggle print statements

    Warning
    -------
    If NFFT is smaller than the time the wavefront
    needs to travel from the source to the array, the impulse
    response will by cyclically shifted (cyclic convolution).

    Returns
    -------
    Pnm : array of complex floats
       Spatial Fourier Coefficients with nm coeffs in cols and FFT coeffs in rows
    kr : array_like
       kr-vector
       ::
          Can also be a matrix [krm; krs] for rigid sphere configurations:
          [1,:] => krm referring to the microphone radius
          [2,:] => krs referring to the sphere radius (scatterer)
    """

    NFFT = int(F_NFFT / 2 + 1)
    NMLocatorSize = (N + 1) ** 2

    if SegN is None:
        SegN = N
    if upperSegLim is None:
        upperSegLim = NFFT - 1

    if printInfo:
        print('SOFiA W/G/C - Wave Generator')

    # SAFETY CHECKS
    if upperSegLim < lowerSegLim:
        raise ValueError('Upper segment limit needs to be below lower limit.')
    if upperSegLim > NFFT - 1:
        raise ValueError('Upper segment limit needs to be below NFFT - 1.')
    if upperSegLim > NFFT - 1 or upperSegLim < 0:
        raise ValueError('Upper segment limit needs to be between 0 and NFFT - 1.')
    if lowerSegLim > NFFT - 1 or lowerSegLim < 0:
        raise ValueError('Lower segment limit needs to be between 0 and NFFT - 1.')
    if SegN > N:
        raise ValueError("Segment order needs to be smaller than N.")
    if wavetype != 0 and wavetype != 1:
        raise ValueError('Invalid wavetype: Either 0 (plane wave) or 1 (spherical wave).')
    if ac != 0 and ac != 1 and ac != 2 and ac != 3 and ac != 4:
        raise ValueError('Invalid sphere array configuration: Either 0 (open), 1 (open, pGrad), 2 (rigid), 3 (rigid, pGrad), 4 (dual open)')
    if (t * fs > F_NFFT / 2):
        raise ValueError('Delay t is large for provided NFFT. Choose t < NFFT/(2*FS).')

    # Check source distance

    if not isinstance(r, list):
        rm = r
        rs = r
        nor = 1
    else:  # r is matrix
        rm = r[0]
        m, n = r.shape
        if m == 2 | n == 2:
            rs = r[1]
            nor = 2
        else:
            rs = r[0]
            nor = 1

    if nor == 2 and (ac == 0 or ac == 1):
        nor = 1

    if wavetype == 1 and ds <= rm:
        raise ValueError('Invalid source distance, source must be outside the microphone radius.')

    w = _np.linspace(0, pi * fs, NFFT)
    # w = 0 breaks stuff ?
    w[0] = w[1]
    k = w / c
    krm = k * rm
    krs = k * rs
    kds = k * ds

    f = list(range(lowerSegLim, upperSegLim + 1))

    # RADIAL FILTERS
    rfArray = _np.zeros([NMLocatorSize, NFFT], dtype=_np.complex_)
    timeShift = _np.exp(- 1j * w[f] * t)

    for n in range(0, SegN + 1):
        if wavetype == 0:    # Plane wave
            rfArray[n][f] = bn(n, krm[f], krs[f], ac) * timeShift
        elif wavetype == 1:  # Spherical wave
            rfArray[n][f] = 4 * pi * -1j * k[f] * timeShift * sphankel(n, kds[f]) * bn_npf(n, krm[f], krs[f], ac)

    # GENERATOR CORE
    Pnm = _np.empty([NMLocatorSize, NFFT], dtype=_np.complex_)
    ctr = 0
    for n in range(0, SegN + 1):
        for m in range(-n, n + 1):
            SHarms = _np.conj(sph_harm(m, n, az, el))
            Pnm[ctr] = SHarms * rfArray[n]
            ctr = ctr + 1

    if nor == 2:
        kr = krs
    else:
        kr = krm
    kr[0] = 0  # resubstitute kr = 0

    return Pnm, kr
