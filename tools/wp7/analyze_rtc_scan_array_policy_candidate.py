#!/usr/bin/env python3
"""Generate evidence for the proposed WP-7 scan/array RTC numerical policy.

This is an evidence-only calculator.  Its constants are candidates, not
scientific authority, and its Kaiser-window order calculation is a feasibility
estimate rather than a coefficient or response certification.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scipy.signal import kaiserord


SCHEMA = "citlali-wp7-rtc-scan-array-policy-candidate-v1"
SPEEDS_ARCSEC_PER_SEC = (1.0, 10.0, 25.0, 50.0, 100.0)
SPEED_OF_LIGHT_M_PER_SEC = 299_792_458.0
ARCSEC_PER_RADIAN = 180.0 * 3600.0 / math.pi
AIRY_INTENSITY_FWHM_COEFFICIENT = 1.028993969962188


@dataclass(frozen=True)
class CandidatePolicy:
    input_sample_rate_hz: float = 122.0703125
    aperture_m: float = 50.0
    passband_amplitude_ripple: float = 1.0e-4
    folded_alias_power_gain: float = 1.0e-6
    minimum_samples_per_airy_fwhm: float = 4.0
    maximum_filter_half_support_sec: float = 5.0
    velocity_fractional_margin: float = 0.05
    cadence_fractional_margin: float = 1.0e-4
    minimum_factor: int = 1
    maximum_factor: int = 256


ARRAY_CENTER_FREQUENCY_GHZ = {
    "a1100": 272.0,
    "a1400": 214.0,
    "a2000": 150.0,
}


def array_model(
    array: str,
    center_frequency_ghz: float,
    aperture_m: float,
) -> dict[str, float | str]:
    wavelength_m = SPEED_OF_LIGHT_M_PER_SEC / (center_frequency_ghz * 1.0e9)
    lambda_over_d_arcsec = wavelength_m / aperture_m * ARCSEC_PER_RADIAN
    fwhm_arcsec = AIRY_INTENSITY_FWHM_COEFFICIENT * lambda_over_d_arcsec
    return {
        "array": array,
        "center_frequency_ghz": center_frequency_ghz,
        "wavelength_mm": wavelength_m * 1.0e3,
        "lambda_over_d_arcsec": lambda_over_d_arcsec,
        "airy_intensity_fwhm_arcsec": fwhm_arcsec,
        "optical_cutoff_hz_per_arcsec_per_sec": 1.0 / lambda_over_d_arcsec,
    }


def estimated_kaiser_realization(
    *,
    factor: int,
    science_passband_hz: float,
    safe_sample_rate_hz: float,
    policy: CandidatePolicy,
) -> dict[str, float | int] | None:
    if factor <= 1:
        return None

    output_sample_rate_hz = safe_sample_rate_hz / factor
    alias_stopband_start_hz = output_sample_rate_hz - science_passband_hz
    if alias_stopband_start_hz <= science_passband_hz:
        return None

    per_image_stopband_amplitude = math.sqrt(
        policy.folded_alias_power_gain / (factor - 1)
    )
    attenuation_db = max(
        -20.0 * math.log10(policy.passband_amplitude_ripple),
        -20.0 * math.log10(per_image_stopband_amplitude),
    )
    normalized_transition_width = (
        (alias_stopband_start_hz - science_passband_hz)
        / (policy.input_sample_rate_hz / 2.0)
    )
    tap_count, beta = kaiserord(attenuation_db, normalized_transition_width)
    tap_count = max(tap_count, 3)
    if tap_count % 2 == 0:
        tap_count += 1
    half_support_sec = (tap_count - 1) / (2.0 * policy.input_sample_rate_hz)
    return {
        "tap_count_estimate": tap_count,
        "kaiser_beta_estimate": beta,
        "half_support_sec_estimate": half_support_sec,
        "alias_stopband_start_hz": alias_stopband_start_hz,
        "per_image_stopband_amplitude_bound": per_image_stopband_amplitude,
        "attenuation_db_estimate": attenuation_db,
        "normalized_transition_width": normalized_transition_width,
    }


def choose_factor(
    *,
    model: dict[str, float | str],
    speed_arcsec_per_sec: float,
    policy: CandidatePolicy,
) -> dict[str, Any]:
    safe_sample_rate_hz = policy.input_sample_rate_hz * (
        1.0 - policy.cadence_fractional_margin
    )
    planning_speed = speed_arcsec_per_sec * (
        1.0 + policy.velocity_fractional_margin
    )
    lambda_over_d_arcsec = float(model["lambda_over_d_arcsec"])
    fwhm_arcsec = float(model["airy_intensity_fwhm_arcsec"])
    science_passband_hz = planning_speed / lambda_over_d_arcsec

    input_samples_per_fwhm = (
        safe_sample_rate_hz * fwhm_arcsec / planning_speed
    )
    base = {
        "speed_arcsec_per_sec": speed_arcsec_per_sec,
        "planning_speed_arcsec_per_sec": planning_speed,
        "science_passband_hz": science_passband_hz,
        "safe_sample_rate_hz": safe_sample_rate_hz,
        "input_samples_per_airy_fwhm": input_samples_per_fwhm,
    }

    if input_samples_per_fwhm < policy.minimum_samples_per_airy_fwhm:
        return {
            **base,
            "disposition": "input_cadence_inadequate",
            "selected_factor": None,
        }

    selected: dict[str, Any] | None = None
    for factor in range(policy.minimum_factor + 1, policy.maximum_factor + 1):
        output_sample_rate_hz = safe_sample_rate_hz / factor
        output_samples_per_fwhm = (
            output_sample_rate_hz * fwhm_arcsec / planning_speed
        )
        if output_samples_per_fwhm < policy.minimum_samples_per_airy_fwhm:
            continue
        estimate = estimated_kaiser_realization(
            factor=factor,
            science_passband_hz=science_passband_hz,
            safe_sample_rate_hz=safe_sample_rate_hz,
            policy=policy,
        )
        if estimate is None:
            continue
        if (
            float(estimate["half_support_sec_estimate"])
            > policy.maximum_filter_half_support_sec
        ):
            continue
        selected = {
            **base,
            "disposition": "prototype_decimation_candidate",
            "selected_factor": factor,
            "output_sample_rate_hz": output_sample_rate_hz,
            "output_samples_per_airy_fwhm": output_samples_per_fwhm,
            **estimate,
        }

    if selected is not None:
        return selected

    return {
        **base,
        "disposition": "identity_fallback_candidate",
        "selected_factor": 1,
        "output_sample_rate_hz": safe_sample_rate_hz,
        "output_samples_per_airy_fwhm": input_samples_per_fwhm,
        "tap_count_estimate": 1,
        "half_support_sec_estimate": 0.0,
    }


def make_report(policy: CandidatePolicy | None = None) -> dict[str, Any]:
    candidate = policy or CandidatePolicy()
    models = {
        name: array_model(name, frequency, candidate.aperture_m)
        for name, frequency in ARRAY_CENTER_FREQUENCY_GHZ.items()
    }
    sweep = []
    for speed in SPEEDS_ARCSEC_PER_SEC:
        sweep.append(
            {
                "speed_arcsec_per_sec": speed,
                "arrays": {
                    name: choose_factor(
                        model=model,
                        speed_arcsec_per_sec=speed,
                        policy=candidate,
                    )
                    for name, model in models.items()
                },
            }
        )
    return {
        "schema": SCHEMA,
        "status": "evidence_only_not_scientific_authority",
        "method": {
            "beam": (
                "uniformly_illuminated unobscured circular-aperture Airy "
                "intensity profile"
            ),
            "science_passband": (
                "full one-dimensional temporal support of the scanned Airy "
                "profile: f_sci = v_plan D / lambda"
            ),
            "factor_selection": (
                "largest factor whose conservative cadence, beam sampling, "
                "transition, alias-amplitude allocation, and estimated "
                "Kaiser half-support satisfy the candidate limits"
            ),
            "filter_warning": (
                "Kaiser order and beta are feasibility estimates only; no "
                "coefficient artifact or certified response is produced"
            ),
        },
        "constants": {
            "speed_of_light_m_per_sec": SPEED_OF_LIGHT_M_PER_SEC,
            "arcsec_per_radian": ARCSEC_PER_RADIAN,
            "airy_intensity_fwhm_coefficient": (
                AIRY_INTENSITY_FWHM_COEFFICIENT
            ),
        },
        "candidate_policy": asdict(candidate),
        "array_models": models,
        "synthetic_speed_sweep": sweep,
        "representative_observation": {
            "observation": 152390,
            "header_scan_program": "Lissajous",
            "header_scan_rate_stored_value": 0.00024240684055476798,
            "header_scan_rate_attribute_unit": "arcsec/sec",
            "header_scan_rate_if_interpreted_as_radian_per_sec_arcsec_per_sec": (
                50.0
            ),
            "disposition": (
                "representative workload evidence only; the stored value and "
                "unit attribute disagree, and no accepted AST-valid v_max "
                "product is available"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="write the deterministic JSON report to this path",
    )
    args = parser.parse_args()
    rendered = json.dumps(make_report(), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
