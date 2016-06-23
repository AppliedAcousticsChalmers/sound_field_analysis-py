"""
Generator functions:
- wgc: Wave Generator
- mf: Modal Radial Filter Generator
- lebedev: Lebedev quadrature nodes and weigths
- swg: Sampled Wave Generator
"""
import numpy as _np
from .sph import bn, bn_npf, sphankel, sph_harm, cart2sph
from .process import itc

pi = _np.pi


def wgc(N, r, ac, fs, F_NFFT, az, el, **kargs):
    """
    Wave Generator Core:
    Returns Spatial Fourier Coefficients Pnm and kr vector
    Pnm, kr = sofia.wgc(N, r, ac, FS, NFFT, AZ, EL)

    Optional keyword parameters: t, c, wavetype, ds, lSegLim, uSegLim, SeqN
    ------------------------------------------------------------------------
    Pnm      Spatial Fourier Coefficients
             Columns : nm coeff
             Rows    : FFT bins
    kr       kr-Vector
             Can also be a matrix [krm; krs] for rigid sphere configurations:
             [1,:] => krm referring to the Microphone Radius
             [2,:] => krs referring to the Sphere Radius (Scatterer)
    ------------------------------------------------------------------------
    N        Maximum transform order
    r        Microphone Radius
             Can also be a vector for rigid/dual sphere configurations:
             [1,1] => rm  Microphone Radius
             [2,1] => rs  Sphere Radius or Microphone2 Radius
             ! If only one radius (rm) is given using a Rigid/Dual Sphere
               Configuration: rs = rm and only one kr-vector is returned!
    ac       Array Configuration
             0  Open Sphere with p Transducers (NO plc!)
             1  Open Sphere with pGrad Transducers
             2  Rigid Sphere with p Transducers
             3  Rigid Sphere with pGrad Transducers (Thx to Nils Peters!)
             4  Dual Open Sphere with p Transducers (Thx to Nils Peters!)
    FS       Sampling Frequency
    NFFT     FFT Order (Number of bins) should be 2^x, x=1,2,3,...
    AZ       Azimuth   angle in [DEG] 0-2pi
    EL       Elevation angle in [DEG] 0-pi
    t        Time Delay in s. The delay has: (t*FS) Samples
    c        Speed of sound in [m/s] (Default: 343m/s)
    wavetype Type of the Wave. 0: Plane Wave (default) 1: Spherical Wave
    ds       Distance of the source in [m] (For wavetype = 1 only)
             Warning: If NFFT is smaller than the time the wavefront
             needs to travel from the source to the array, the impulse
             response will by cyclically shifted (cyclic convolution).
    ---
    lSegLim  (Lower Segment Limit) Used by the S/W/G wrapper
    uSegLim  (Upper Segment Limit) Used by the S/W/G wrapper
    SegN     (Sement Order)        Used by the S/W/G wrapper
    """

    NFFT = F_NFFT / 2 + 1

    SegN = kargs['SegN'] if 'SegN' in kargs else N
    upperSegLim = int(kargs['upperSegLim'] if 'upperSegLim' in kargs else NFFT - 1)
    lowerSegLim = int(kargs['lowerSegLim'] if 'lowerSegLim' in kargs else 0)
    ds = kargs['ds'] if 'ds' in kargs else 1.0
    wavetype = kargs['wavetype'] if 'wavetype' in kargs else 0
    c = kargs['c'] if 'c' in kargs else 343.0
    t = kargs['t'] if 't' in kargs else 0.0
    printInfo = kargs['printInfo'] if 'printInfo' in kargs else True

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
    rfArray = _np.empty([SegN + 1, upperSegLim + 1 - lowerSegLim], dtype=_np.complex_)
    timeShift = _np.exp(- 1j * w[f] * t)

    for n in range(0, SegN + 1):
        if wavetype == 0:    # Plane wave
            rfArray[n][f] = bn(n, krm[f], krs[f], ac) * timeShift
        elif wavetype == 1:  # Spherical wave
            rfArray[n][f] = 4 * pi * -1j * k[f] * timeShift * sphankel(n, kds[f]) * bn_npf(n, krm[f], krs[f], ac)

    # GENERATOR CORE
    Pnm = _np.empty([pow(N + 1, 2), upperSegLim + 1 - lowerSegLim], dtype=_np.complex_)
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


def mf(N, kr, ac, **kargs):
    """M/F Modal radial filters
    dn, beam = mf(N, kr, ac)
    Optional keyword parameters: a_max, plc, fadeover
    ------------------------------------------------------------------------
    dn          Vector of modal 0-N frequency domain filters
    beam        Expected free field On-Axis kr-response
    ------------------------------------------------------------------------
    N           Maximum Order
    kr          Vector or Matrix of kr values
                First Row   (M=1) N: kr values Microphone Radius
                Second Row  (M=2) N: kr values Sphere/Microphone2 Radius
                [kr_mic;kr_sphere] for Rigid/Dual Sphere Configurations
                ! If only one kr-vector is given using a Rigid/Dual Sphere
                Configuration: kr_sphere = kr_mic
    ac          Array Configuration:
                0  Open Sphere with pressure Transducers (NO plc!)
                1  Open Sphere with cardioid Transducers
                2  Rigid Sphere with pressure Transducers
                3  Rigid Sphere with cardioid Transducers
                4  Dual Open Sphere with pressure Transducers
    a_max       Maximum modal amplification limit in [dB]
    plc         OnAxis powerloss-compensation:
                0  Off
                1  Full kr-spectrum plc
                2  Low kr only -> set fadeover
    fadeover    Number of kr values to fade over +/- around min-distance
                gap of powerloss compensated filter and normal N0 filters.
                0 = auto fadeover
    """

    # Get optional arguments
    a_maxdB = kargs['a_maxdB'] if 'a_maxdB' in kargs else 0
    a_max = pow(10, (a_maxdB / 20)) if 'a_maxdB' in kargs else 1
    limiteronflag = True if 'a_maxdB' in kargs else False
    plc = kargs['plc'] if 'plc' in kargs else 0
    fadeover = kargs['fadeover'] if 'fadeover' in kargs else 0
    printInfo = kargs['printInfo'] if 'printInfo' in kargs else True

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
                if a_maxdB > 0:
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


def lebedev(degree, **kargs):
    '''
    [gridData, Nmax] = sofia_lebedev(degree, plot)
    This function computes Lebedev quadrature nodes and weigths.
    ------------------------------------------------------------------------
    gridData            Lebedev quadrature including weigths(W):
                        [AZ_1 EL_1 W_1;
                        ...
                        AZ_n EL_n W_n]

    Nmax                Highest stable grid order
    ------------------------------------------------------------------------
    Degree              Lebedev Degree (Number of nodes). Currently available:
                        6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194

    plot                Show a globe plot of the selected grid
                        0: Off [default], 1: On
    '''
    from sofia import lebedev

    plot = kargs['plot'] if 'plot' in kargs else 0
    printInfo = kargs['printInfo'] if 'printInfo' in kargs else True

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


def swg(**kargs):
    """S/W/G Sampled Wave Generator Wrapper
    [fftdata, kr] = sofia_swg(r, gridData, ac, FS, NFFT, ...
                                          AZ, EL, Nlim, t, c, wavetype, ds)
    ------------------------------------------------------------------------
    fftData  Complex sound pressures                   [(N+1)^2 x NFFT]
    kr       kr-Vector
             Can also be a matrix [krm; krs] for rigid sphere configurations:
              [1,:] => krm referring to the Microphone Radius
              [2,:] => krs referring to the Sphere/Microphone2 Radius
    ------------------------------------------------------------------------
    r        Microphone Radius
             Can also be a vector for rigid sphere configurations:
              [1,1] => rm  Microphone Radius
              [2,1] => rs  Sphere Radius (Scatterer)
    gridData Quadrature grid                           [default LEB110]
              Columns : Position Number 1...M
              Rows    : [AZ EL Weight]
              Angles AZ, EL in [RAD]
    ac       Array Configuration
              0  Open Sphere with p Transducers
              1  Open Sphere with pGrad Transducers
              2  Rigid Sphere with p Transducers
              3  Rigid Sphere with pGrad Transducers (Thx to Nils Peters!)
              4  Dual Open Sphere with p Transducers (Thx to Nils Peters!)
    FS       Sampling Frequency
    NFFT     FFT Order (Number of bins) should be 2^x, x=1,2,3,...
    AZ       Azimuth   angle in [DEG] 0-2pi
    EL       Elevation angle in [DEG] 0-pi
    Nlim     Internal generator transform order limit
    c        Speed of sound in [m/s] (Default: 343m/s)
    t        Time Delay in s
    wavetype Type of the Wave. 0: Plane Wave (default) 1: Spherical Wave
    ds       Distance of the source in [m] (For wavetype = 1 only)
             Warning: If NFFT is smaller than the time the wavefront
             needs to travel from the source to the array, the impulse
             response will by cyclically shifted (cyclic convolution).

    This file is a wrapper generating the complex pressures at the
    positions given in 'gridData' for a full spectrum 0-FS/2 Hz (NFFT Bins)
    wave impinging to an array. The wrapper involves the W/G/C wave
    generator core and the I/T/C spatial transform core.

    S/W/G emulates discrete sampling. You can observe alias artifacts.
    """

    # Get optional arguments - most probably could be proper optional arguments
    r = kargs['r'] if 'r' in kargs else 0.1
    gridData = kargs['gridData'] if 'gridData' in kargs else lebedev(110)
    ac = kargs['ac'] if 'ac' in kargs else 0
    FS = kargs['FS'] if 'FS' in kargs else 48000
    NFFT = kargs['NFFT'] if 'NFFT' in kargs else 512
    AZ = kargs['AZ'] if 'AZ' in kargs else 0
    EL = kargs['EL'] if 'EL' in kargs else _np.pi / 2
    Nlim = kargs['Nlim'] if 'Nlim' in kargs else 120
    # t = kargs['t'] if 't' in kargs else 0
    c = kargs['c'] if 'c' in kargs else 343
    wavetype = kargs['wavetype'] if 'wavetype' in kargs else 0
    ds = kargs['ds'] if 'ds' in kargs else 1
    printInfo = kargs['printInfo'] if 'printInfo' in kargs else True

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

    for idx, order in enumerate(_np.unique(rqOrders)):
        if printInfo:
            amtDone = idx / (_np.unique(rqOrders).size - 1)
            print('\rProgress: [{0:50s}] {1:.1f}%'.format('#' * int(amtDone * 50), amtDone * 100), end="", flush=True)
        fOrders = _np.flatnonzero(rqOrders == order)
        temp, _ = wgc(Ng, r, ac, FS, NFFT, AZ, EL, wavetype=wavetype, ds=ds, lSegLim=fOrders[0], uSegLim=fOrders[-1], SeqN=order, printInfo=False)
        Pnm += temp
    print('\n')
    fftData = itc(Pnm, gridData)

    return fftData, kr
