"""This is a reference implementation to compare the binaural rendering of a
spherical microphone array measurement based on the different Spherical
Harmonics conventions available in `sph.sph_harm()`.

The implementation is a simplified version of "Exp4_BinauralRendering.ipynb".
"""

from itertools import combinations

import numpy as np

from sound_field_analysis import gen, io, plot, process, sph, utils
from scipy.io import wavfile

################################################################################
sh_max_order = 8  # maximal utilized spherical harmonic rendering order
rf_nfft = 4096  # target length of radial filter in samples
rf_amp_max_db = 20  # soft limiting level of radial filter in dB
azim_offset_deg = -37  # azimuth head rotation offset in degrees
pre_azims_deg = [0, 45]  # azimuth head orientations in degrees
pre_len_s = 5.2  # length of rendered preview BRIR auralizations in seconds
pre_src_file = "../examples/data/audio.wav"  # audio source file for rendered preview
sh_kinds = ["real", "real_Zotter", "complex", "complex_GumDur"]

################################################################################
# load impulse responses from SOFA file
DRIR = io.read_SOFA_file("../examples/data/DRIR_CR1_VSA_110RS_L.sofa")
HRIR = io.read_SOFA_file("../examples/data/HRIR_L2702.sofa")
FS = int(HRIR.l.fs)
NFFT = HRIR.l.signal.shape[-1]

# check match of sampling frequencies
if DRIR.signal.fs != FS:
    raise ValueError("Mismatch between sampling frequencies of DRIR and HRIR.")

# automatically calculate target processing length
# by summing impulse response lengths of DRIR, HRIR and radial filters
NFFT += DRIR.signal.signal.shape[-1] + rf_nfft

################################################################################
# HRIR SH coefficients
# transform SOFA data
Hnm = np.stack(
    [
        [
            process.spatFT(
                process.FFT(HRIR.l.signal, fs=FS, NFFT=NFFT, calculate_freqs=False),
                position_grid=HRIR.grid,
                order_max=sh_max_order,
                kind=sh_kind,
            ),
            process.spatFT(
                process.FFT(HRIR.r.signal, fs=FS, NFFT=NFFT, calculate_freqs=False),
                position_grid=HRIR.grid,
                order_max=sh_max_order,
                kind=sh_kind,
            ),
        ]
        for sh_kind in sh_kinds
    ]
)

# DRIR SH coefficients
Pnm = np.stack(
    [
        process.spatFT(
            process.FFT(DRIR.signal.signal, fs=FS, NFFT=NFFT, calculate_freqs=False),
            position_grid=DRIR.grid,
            order_max=sh_max_order,
            kind=sh_kind,
        )
        for sh_kind in sh_kinds
    ]
)

# compute radial filters
dn = gen.radial_filter_fullspec(
    max_order=sh_max_order,
    NFFT=rf_nfft,
    fs=FS,
    array_configuration=DRIR.configuration,
    amp_maxdB=rf_amp_max_db,
)

# make radial filters causal
dn_delay_samples = rf_nfft / 2
dn *= gen.delay_fd(target_length_fd=dn.shape[-1], delay_samples=dn_delay_samples)

# adjust length of radial filters
dn = utils.zero_pad_fd(dn, target_length_td=NFFT)

# adjust radial filter data shape according to SH order as grades
dn = np.repeat(dn, np.arange(sh_max_order * 2 + 1, step=2) + 1, axis=0)

################################################################################
# SH grades stacked by order
m = sph.mnArrays(sh_max_order)[0]

# reverse indices for stacked SH grades
m_rev_id = sph.reverseMnIds(sh_max_order)

# select azimuth head orientations to compute (according to SSR BRIR requirements)
azims_SSR_rad = np.deg2rad(np.arange(0, 360) - azim_offset_deg)

S = np.zeros(
    [len(sh_kinds), azims_SSR_rad.size, Hnm.shape[-3], Hnm.shape[-1]], dtype=Hnm.dtype
)
for sh_id, sh_kind in enumerate(sh_kinds):

    if sh_kind == "real":
        # compute possible components before the loop
        # apply term according to spherical harmonic kind
        Pnm_dn_Hnm = Pnm[sh_id] * dn * Hnm[sh_id]

        # loop over all head orientations that are to be computed
        # this could be done with one inner product but the loop helps clarity
        for azim_id, alpha in enumerate(azims_SSR_rad):
            # From https://github.com/polarch/Spherical-Harmonic-Transform/blob/master/TEST_SCRIPTS_SHT.m
            # R_{nm}(\theta,\phi) = \sqrt{\frac{2n+1}{4\pi}\frac{(n-|m|)!}{(n+|m|)!}} P_l^{|m|}(\cos\theta) N_m(\phi)
            # N_m(\phi) = \sqrt{2}\cos(m\phi) $$ m > 0 $$
            # N_m(\phi) = 1 $$ m = 0 $$
            # N_m(\phi) = \sqrt{2}\sin(|m|\phi) $$ m < 0 $$
            Nm = np.ones(shape=m.shape)
            Nm[m > 0] = np.sqrt(2) * np.cos(m[m > 0] * alpha)
            Nm[m < 0] = np.sqrt(2) * np.sin(abs(m[m < 0]) * alpha)

            # these are the spectra of the ear signals
            S[sh_id, azim_id] = np.sum(Pnm_dn_Hnm * Nm[:, np.newaxis], axis=1)
            # TODO: This does not match the result of the complex
            #  conventions so far (not

    elif sh_kind == "real_Zotter":  # CORRECT
        # compute possible components before the loop
        Pnm_dn = Pnm[sh_id] * dn

        # loop over all head orientations that are to be computed
        # this could be done with one inner product but the loop helps clarity
        for azim_id, alpha in enumerate(azims_SSR_rad + np.pi):
            # the rotation by 180 degrees is equivalent to the (-1)^m term
            alpha_cos = np.cos(m * alpha)[:, np.newaxis]
            alpha_sin = np.sin(m * alpha)[:, np.newaxis]

            # these are the spectra of the ear signals
            S[sh_id, azim_id] = np.sum(
                (alpha_cos * Pnm_dn - alpha_sin * Pnm_dn[m_rev_id]) * Hnm[sh_id], axis=1
            )
            # TODO: This matches the result of the complex conventions,
            #  but may be written in a different and more efficient way?

    elif sh_kind in ["complex", "complex_GumDur"]:  # CORRECT
        # compute possible components before the loop
        Pnm_dn_Hnm = Pnm[sh_id, m_rev_id] * dn * Hnm[sh_id]

        # loop over all head orientations that are to be computed
        # this could be done with one inner product but the loop helps clarity
        for azim_id, alpha in enumerate(
            azims_SSR_rad + (np.pi if sh_kind == "complex_GumDur" else 0)
        ):
            # the rotation by 180 degrees is equivalent to the (-1)^m term
            alpha_exp = np.exp(-1j * m * alpha)[:, np.newaxis]

            # these are the spectra of the ear signals
            S[sh_id, azim_id] = np.sum(Pnm_dn_Hnm * alpha_exp, axis=1)

    else:
        raise NotImplementedError(f"SH convention {sh_kind}")

# IFFT to yield ear impulse responses
BRIR = process.iFFT(S)

# normalize BRIRs
BRIR *= 0.9 / np.max(np.abs(BRIR))

################################################################################
# read source file
source, source_fs = io.read_wavefile(pre_src_file)
if len(source.shape) > 1:
    source = source[0]  # consider only first channel
if source.dtype in [np.int16, np.int32]:
    source = source.astype(np.float_) / np.iinfo(source.dtype).max

# select shorter extract from source
source = np.atleast_2d(source[: int(pre_len_s * source_fs)])

# resample source to match BRIRs
source = utils.simple_resample(source, original_fs=source_fs, target_fs=FS)

# export preview per specified orientation
print("\nGenerate audio preview")
for sh_id, sh_kind in enumerate(sh_kinds):
    for azim in pre_azims_deg:
        filename = f"plots/preview_conv_sh{sh_max_order}_{sh_kind}_{azim}deg.wav"
        data = process.convolve(source, BRIR[sh_id, azim])

        print(f' --> exporting "{filename}"')
        wavfile.write(filename=filename, rate=FS, data=data.T.astype(np.float32))

################################################################################
print("\nGenerate plot preview")
for azim in pre_azims_deg:
    filename = f"plots/preview_time_sh{sh_max_order}_{azim}deg_left"
    print(f' --> exporting "{filename}"')
    plot.plot2D(
        BRIR[:, azim, 0],
        fs=FS,
        viz_type="Time",
        line_names=[f"Left ear, {sh_kind}" for sh_kind in sh_kinds],
        title=filename,
    )

    filename = f"plots/preview_spec_sh{sh_max_order}_{azim}deg_left"
    print(f' --> exporting "{filename}"')
    plot.plot2D(
        BRIR[:, azim, 0],
        fs=FS,
        viz_type="LogFFT",
        log_fft_frac=3,
        line_names=[f'Left ear, "{sh_kind}"' for sh_kind in sh_kinds],
        title=filename,
    )

################################################################################
print("\nEvaluate result match with `np.allclose()`")
for sh_id1, sh_id2 in combinations(range(len(sh_kinds)), 2):
    print(
        f" -->   {np.allclose(BRIR[sh_id1], BRIR[sh_id2])}"
        f'   ("{sh_kinds[sh_id1]}" vs. "{sh_kinds[sh_id2]}")'
    )
