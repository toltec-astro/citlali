#!/usr/bin/env python3
"""Offline explicit development plans; no runtime design or automatic selection.

The two new entries retain the 235 arcsec/s design ceiling. Their full optical
bands require native output cadence (M=1). The accepted a2000 coefficients are
copied verbatim from the exact existing request, not regenerated.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import signal, special

REFERENCE_INPUT = "bbb85e2b943c0f6d21ebebf990aef1286906fab8f6971c3026020603e9dcd8e3"
FREQUENCIES = {"a1100": 272e9, "a1400": 214e9, "a2000": 150e9}
INTERVAL = .008192
SPEED = 235.0
CADENCE_MARGIN = .0001


def optical(array):
    radians = math.pi / (180 * 3600)
    wavelength = 299792458.0 / FREQUENCIES[array]
    return (SPEED * radians * 50 / wavelength,
            1.028993969962188 * wavelength / 50 / radians)


def prepare(array, reference=None):
    band, fwhm = optical(array)
    # These are explicit experiment choices, not a largest-factor algorithm.
    factor = {"a1100": 1, "a1400": 1, "a2000": 2}[array]
    fs = 1 / INTERVAL
    stop = fs / (2 * factor) * (1 - CADENCE_MARGIN)
    if array == "a2000":
        if reference is None:
            raise ValueError("a2000 requires the preserved exact input")
        h = np.asarray(reference["fir"], dtype=np.float64)
        identity = reference["lowpass_identity"]
        design = "unchanged accepted a2000 coefficient bytes"
    else:
        # Protect the full band even at the longest allowed sample interval.
        pass_edge = band * (1 + CADENCE_MARGIN)
        taps, beta = signal.kaiserord(80, (stop - pass_edge) / (fs / 2))
        taps |= 1
        h = signal.firwin(taps, (pass_edge + stop) / 2,
                         window=("kaiser", beta), fs=fs, scale=True)
        h = (h + h[::-1]) / 2
        h /= sum(h)
        identity = f"offline-explicit-{array}-235arcsec-per-sec-Kaiser-F1-20260919"
        design = "offline Kaiser order estimate target80dB; actual response measured separately"
    if (h.ndim != 1 or len(h) % 2 != 1 or not np.isfinite(h).all()
            or not np.array_equal(h, h[::-1]) or abs(sum(h) - 1) > 1e-12):
        raise ValueError("invalid centered unit-DC coefficient realization")
    plan = dict(schema="rtc-explicit-array-lowpass-v1", identity=identity,
                array=array, factor=factor,
                nominal_interval_seconds=INTERVAL,
                cadence_relative_bound=CADENCE_MARGIN,
                speed_ceiling_arcsec_per_sec=SPEED, coefficients=h.tolist(),
                coefficients_sha256=hashlib.sha256(h.astype('<f8').tobytes()).hexdigest(),
                scientific_authority="ADR0020+ADR0022+fixed-speed-support-owner-20260916",
                beam="50m-unobscured-circular-Airy", reference_frequency_hz=FREQUENCIES[array],
                design=design, use="explicit-development-only",
                automatic_selection=False, production_certified=False)
    f, response = signal.freqz(h, worN=262145, include_nyquist=True, fs=fs)
    # Evaluate physical passband for the worst accepted cadence endpoint.
    pass_error = float(max(abs(abs(response[f <= band * (1 + CADENCE_MARGIN)]) - 1)))
    stop_amplitude = float(max(abs(response[f >= stop])))
    if pass_error >= .01 or stop_amplitude >= .001:
        raise ValueError("explicit development response check failed")
    numerical = dict(taps=len(h), half_support_seconds=len(h)//2 * INTERVAL,
                     optical_support_hz=band, output_nyquist_hz=fs/(2*factor),
                     maximum_passband_magnitude_error=pass_error,
                     maximum_stopband_amplitude=stop_amplitude,
                     dc_error=abs(float(sum(h))-1),
                     new_decimation_alias_images=0 if factor == 1 else "existing factor2 plan unchanged",
                     four_sample_center_speed_limit_arcsec_per_sec=fwhm*(1-CADENCE_MARGIN)/(4*factor*INTERVAL*1.05),
                     qualification="engineering transfer/support only; no MAP/OOF/FRUIT or cleaned-PSD certification")
    return plan, numerical


def source_probes(plan):
    """Paired runs through one frozen FIR; original uniform-average Airy inputs."""
    array = plan["array"]; h = np.array(plan["coefficients"])
    _, fwhm = optical(array)
    scale = math.pi * 50 * FREQUENCIES[array] / 299792458 / (180*3600)
    max_admitted = fwhm*(1-CADENCE_MARGIN)/(4*plan["factor"]*INTERVAL*1.05)
    records = []
    for speed in (1., 25., .95*max_admitted, SPEED):
        half = len(h)//2
        n = int(math.ceil(12*fwhm/speed/INTERVAL)) + half + 2
        t = np.arange(-n, n+1)*INTERVAL
        background = .2*np.sin(2*math.pi*.37*t) + .04*np.cos(2*math.pi*55*t)
        for phase in (0., .25, .5, .75):
            fine = (t[:,None] + ((np.arange(128)+.5)/128-.5)*INTERVAL - phase*INTERVAL)
            z = scale*speed*fine
            with np.errstate(divide='ignore', invalid='ignore'):
                sky = np.where(z == 0, 1., (2*special.j1(z)/z)**2).mean(axis=1)
            reference = sky[half:-half or None:plan["factor"]]
            delta = (signal.convolve(background+sky,h,mode='valid',method='direct')-
                     signal.convolve(background,h,mode='valid',method='direct'))[::plan["factor"]]
            records.append(dict(speed_arcsec_per_sec=speed, phase_native_samples=phase,
                                center_sampling_admitted=speed <= max_admitted,
                                peak_relative_change=float(delta.max()/reference.max()-1),
                                integrated_relative_change=float(delta.sum()/reference.sum()-1),
                                maximum_residual_over_native_peak=float(max(abs(delta-reference))/reference.max()),
                                comparison="matched complete support and exact output schedule; provisional uniform averaging"))
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if hashlib.sha256(a.reference_input.read_bytes()).hexdigest()!=REFERENCE_INPUT:
        raise ValueError("preserved 2-mm request changed")
    reference=json.loads(a.reference_input.read_text());a.output.mkdir(parents=True,exist_ok=False)
    results={}
    for array in FREQUENCIES:
        plan,metrics=prepare(array,reference)
        metrics['source_probes']=source_probes(plan)
        if any(r['maximum_residual_over_native_peak'] >= .01 for r in metrics['source_probes']):
            raise ValueError("controlled source transfer failed")
        path=a.output/(array+'.json')
        path.write_text(json.dumps(plan,indent=2,allow_nan=False)+'\n')
        metrics['artifact_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();results[array]=metrics
    (a.output/'verification.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
