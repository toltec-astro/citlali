#!/usr/bin/env python3
"""Generate the deterministic SCI-CAL-001 phase-0 continuity evidence.

This script is intentionally independent of the application build.  It reads
the exact repair-base source literals, verifies their frozen file digests, and
evaluates the one-sided atmospheric q-model limits in IEEE-754 binary64.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import re
import sys
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path


SOURCE_SHA = "9aae0e669384c5c0c0dda93debc194d6b8dac787"
AUDIT_DISPATCH_SHA = "e6174dd9f49afe9ec57c150a7a97db3f0f4910e0"
HANDOFF_SHA256 = (
    "9d2c0ae89244d355070d6b300f431ac1799787b835c7e4cb76c8d7fc51cde106"
)
AMENDMENT_SHA256 = (
    "64fd3ae9788c6a8e3db18ac5ea4f04799586b548f9e7ec12cc8c18f9cbf96e09"
)

CALIBRATE_REL = Path("include/citlali/core/timestream/rtc/calibrate.h")
SELECTOR_REL = Path(
    "include/citlali/core/timestream/extinction_model_selection.h"
)
EXPECTED_SOURCE_DIGESTS = {
    CALIBRATE_REL: (
        "d70a55278227b43cdd7de19bc67e4ddb332524d40e1455c5fa7a80ae5e2d11ee"
    ),
    SELECTOR_REL: (
        "45cf86bbb2318c22514411f6d2a0e0371e22e9e355e61b293d93c628d9f3469d"
    ),
}

MODELS = ("am_q0", "am_q25", "am_q50", "am_q75", "am_q95")
BANDS = ("a1100", "a1400", "a2000")
EVALUATION_GRID = (
    (Decimal("30"), "diagnostic_lower_point"),
    (Decimal("50"), "audit_representative_interior"),
    (Decimal("70"), "audit_representative_interior"),
    (Decimal("80"), "selector_reference_elevation"),
)
TABLE_NAME = "q_model_continuity_table.csv"
REPORT_NAME = "Q_MODEL_CONTINUITY_REPORT.md"
SUMS_NAME = "SHA256SUMS"
SCRIPT_NAME = "generate_q_model_continuity.py"


@dataclass(frozen=True)
class SourceModel:
    transmissions: dict[str, str]
    coefficients: dict[str, dict[str, tuple[str, ...]]]
    pi_literal: str
    reference_elevation_deg_literal: str
    degree_divisor_literal: str
    airmass_correction_literal: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def require_source_digests(repo_root: Path) -> dict[Path, str]:
    actual: dict[Path, str] = {}
    for relative, expected in EXPECTED_SOURCE_DIGESTS.items():
        digest = sha256_path(repo_root / relative)
        if digest != expected:
            raise RuntimeError(
                f"source digest mismatch for {relative}: {digest} != {expected}"
            )
        actual[relative] = digest
    return actual


def parse_number_literals(expression: str) -> tuple[str, ...]:
    return tuple(
        re.findall(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
            expression,
        )
    )


def parse_source(repo_root: Path) -> SourceModel:
    calibrate_text = (repo_root / CALIBRATE_REL).read_text(encoding="utf-8")
    selector_text = (repo_root / SELECTOR_REL).read_text(encoding="utf-8")

    tx_match = re.search(
        r"tx_225_zenith\s*=\s*\{(?P<body>.*?)\n\s*\};",
        calibrate_text,
        flags=re.DOTALL,
    )
    if tx_match is None:
        raise RuntimeError("could not parse tx_225_zenith source map")
    transmissions = dict(
        re.findall(
            r'\{"(?P<name>am_q\d+)",\s*'
            r"(?P<value>[-+0-9.eE]+)\s*\}",
            tx_match.group("body"),
        )
    )

    coefficients: dict[str, dict[str, tuple[str, ...]]] = {}
    block_pattern = re.compile(
        r'(?:else\s+)?if\s*\(extinction_model=="(?P<model>am_q\d+)"\)\s*'
        r"\{(?P<body>.*?)^\s{8}\}",
        flags=re.DOTALL | re.MULTILINE,
    )
    assignment_pattern = re.compile(
        r'tx_ratio_coeff\["(?P<band>a\d+)"\]\s*<<(?P<values>.*?);',
        flags=re.DOTALL,
    )
    for block in block_pattern.finditer(calibrate_text):
        parsed_assignments = {
            match.group("band"): parse_number_literals(match.group("values"))
            for match in assignment_pattern.finditer(block.group("body"))
        }
        if parsed_assignments:
            coefficients[block.group("model")] = parsed_assignments

    pi_match = re.search(
        r"constexpr double pi\s*=\s*(?P<value>[-+0-9.eE]+);",
        selector_text,
    )
    reference_match = re.search(
        r"reference_elevation_rad\s*=\s*(?P<degrees>[-+0-9.eE]+)\s*"
        r"\*\s*pi\s*/\s*(?P<divisor>[-+0-9.eE]+);",
        selector_text,
    )
    correction_match = re.search(
        r"1\.0\s*-\s*(?P<value>[-+0-9.eE]+)\s*\*\s*"
        r"\(std::pow\(secant_zenith,\s*2\)\s*-\s*1\.0\)",
        selector_text,
    )
    if pi_match is None or reference_match is None or correction_match is None:
        raise RuntimeError("could not parse selector constants")

    if tuple(transmissions) != MODELS:
        raise RuntimeError(
            f"unexpected transmission model order: {tuple(transmissions)!r}"
        )
    if tuple(coefficients) != MODELS:
        raise RuntimeError(
            f"unexpected coefficient model order: {tuple(coefficients)!r}"
        )
    for model in MODELS:
        if tuple(coefficients[model]) != BANDS:
            raise RuntimeError(
                f"unexpected band order for {model}: "
                f"{tuple(coefficients[model])!r}"
            )
        for band in BANDS:
            if len(coefficients[model][band]) != 7:
                raise RuntimeError(
                    f"expected seven coefficients for {model}/{band}"
                )

    return SourceModel(
        transmissions=transmissions,
        coefficients=coefficients,
        pi_literal=pi_match.group("value"),
        reference_elevation_deg_literal=reference_match.group("degrees"),
        degree_divisor_literal=reference_match.group("divisor"),
        airmass_correction_literal=correction_match.group("value"),
    )


def source_order_polynomial(coefficients: tuple[float, ...], elevation: float) -> float:
    terms = [
        coefficients[index] * math.pow(elevation, 6 - index)
        for index in range(7)
    ]
    result = terms[0]
    for term in terms[1:]:
        result += term
    return result


def exact_decimal_polynomial(
    coefficient_literals: tuple[str, ...], elevation: Decimal
) -> Decimal:
    result = Decimal(coefficient_literals[0]) * elevation**6
    for index, literal in enumerate(coefficient_literals[1:], start=1):
        result += Decimal(literal) * elevation ** (6 - index)
    return result


def airmass(elevation: float, pi_value: float, correction: float) -> float:
    cosine_zenith = math.cos(pi_value / 2.0 - elevation)
    secant_zenith = 1.0 / cosine_zenith
    return secant_zenith * (
        1.0 - correction * (math.pow(secant_zenith, 2) - 1.0)
    )


def select_model(tau_225: float, thresholds: dict[str, float]) -> str:
    selected = "am_q0"
    for name in MODELS:
        if thresholds[name] <= tau_225:
            selected = name
    return selected


def f64(value: float) -> str:
    return format(value, ".17e")


def d40(value: Decimal) -> str:
    return format(value, ".40E")


def build_rows(model: SourceModel) -> tuple[list[dict[str, str]], dict[str, float]]:
    pi_value = float(model.pi_literal)
    degrees_divisor = float(model.degree_divisor_literal)
    correction = float(model.airmass_correction_literal)
    reference_elevation = (
        float(model.reference_elevation_deg_literal)
        * pi_value
        / degrees_divisor
    )
    reference_airmass = airmass(reference_elevation, pi_value, correction)
    thresholds = {
        name: -math.log(float(transmission)) / reference_airmass
        for name, transmission in model.transmissions.items()
    }

    rows: list[dict[str, str]] = []
    epsilon = sys.float_info.epsilon
    with localcontext() as context:
        context.prec = 80
        decimal_pi = Decimal(model.pi_literal)
        decimal_divisor = Decimal(model.degree_divisor_literal)

        for boundary_index in range(1, len(MODELS)):
            left_model = MODELS[boundary_index - 1]
            right_model = MODELS[boundary_index]
            threshold = thresholds[right_model]
            threshold_below = math.nextafter(threshold, -math.inf)
            threshold_above = math.nextafter(threshold, math.inf)
            selected_below = select_model(threshold_below, thresholds)
            selected_at = select_model(threshold, thresholds)
            selected_above = select_model(threshold_above, thresholds)
            if (
                selected_below != left_model
                or selected_at != right_model
                or selected_above != right_model
            ):
                raise RuntimeError(
                    f"unexpected selector identities at {right_model}: "
                    f"{selected_below}/{selected_at}/{selected_above}"
                )
            strictly_above_q25 = boundary_index > 1

            for elevation_deg, point_role in EVALUATION_GRID:
                elevation_decimal = elevation_deg * decimal_pi / decimal_divisor
                elevation = float(elevation_decimal)
                sample_airmass = airmass(elevation, pi_value, correction)
                common_225_transmission = math.exp(-sample_airmass * threshold)

                for band in BANDS:
                    left_literals = model.coefficients[left_model][band]
                    right_literals = model.coefficients[right_model][band]
                    left_coefficients = tuple(map(float, left_literals))
                    right_coefficients = tuple(map(float, right_literals))

                    left_poly_decimal = exact_decimal_polynomial(
                        left_literals, elevation_decimal
                    )
                    right_poly_decimal = exact_decimal_polynomial(
                        right_literals, elevation_decimal
                    )
                    analytically_equal = (
                        left_model != "am_q0"
                        and left_poly_decimal == right_poly_decimal
                    )

                    left_polynomial = source_order_polynomial(
                        left_coefficients, elevation
                    )
                    right_polynomial = source_order_polynomial(
                        right_coefficients, elevation
                    )
                    left_transmission = (
                        1.0
                        if left_model == "am_q0"
                        else left_polynomial * common_225_transmission
                    )
                    right_transmission = (
                        right_polynomial * common_225_transmission
                    )
                    if not (
                        math.isfinite(left_transmission)
                        and math.isfinite(right_transmission)
                        and left_transmission > 0.0
                        and right_transmission > 0.0
                    ):
                        raise RuntimeError(
                            "nonpositive/nonfinite transmission at "
                            f"{right_model}/{band}/{elevation_deg} deg"
                        )

                    left_los_tau = -math.log(left_transmission)
                    right_los_tau = -math.log(right_transmission)
                    absolute_jump = abs(right_transmission - left_transmission)
                    relative_jump = absolute_jump / abs(left_transmission)
                    signed_los_tau_jump = right_los_tau - left_los_tau
                    absolute_los_tau_jump = abs(signed_los_tau_jump)
                    left_term_sum = sum(
                        abs(coefficient * math.pow(elevation, 6 - index))
                        for index, coefficient in enumerate(left_coefficients)
                    )
                    right_term_sum = sum(
                        abs(coefficient * math.pow(elevation, 6 - index))
                        for index, coefficient in enumerate(right_coefficients)
                    )
                    left_condition = (
                        0.0
                        if left_model == "am_q0"
                        else left_term_sum / abs(left_polynomial)
                    )
                    right_condition = right_term_sum / abs(right_polynomial)
                    # Conservative conditioned binary64 screening bound.  The
                    # 4096-u envelope covers both seven-term polynomial
                    # evaluations, coefficient conversion, powers, sums,
                    # modified-secant arithmetic, exp/log, division, and the
                    # independent rounding of the two one-sided values.
                    unit_roundoff = epsilon / 2.0
                    airmass_condition_scale = max(
                        1.0,
                        abs(sample_airmass),
                        1.0 / abs(sample_airmass),
                    )
                    los_tau_roundoff_bound = 4096.0 * unit_roundoff * (
                        (1.0 + left_condition + right_condition)
                        * airmass_condition_scale
                        + 1.0
                        + abs(left_los_tau)
                        + abs(right_los_tau)
                    )
                    transmission_roundoff_bound = (
                        max(abs(left_transmission), abs(right_transmission))
                        * math.expm1(los_tau_roundoff_bound)
                    )
                    exceeds_roundoff = (
                        absolute_los_tau_jump > los_tau_roundoff_bound
                    )
                    stop_condition = strictly_above_q25 and (
                        (not analytically_equal) or exceeds_roundoff
                    )

                    rows.append(
                        {
                            "source_sha": SOURCE_SHA,
                            "boundary": right_model,
                            "left_model": left_model,
                            "right_model": right_model,
                            "strictly_above_q25": str(strictly_above_q25).lower(),
                            "tau225_threshold_binary64": f64(threshold),
                            "tau225_threshold_hex": threshold.hex(),
                            "tau225_immediate_below_binary64": f64(
                                threshold_below
                            ),
                            "tau225_immediate_above_binary64": f64(
                                threshold_above
                            ),
                            "selected_immediate_below": selected_below,
                            "selected_at_threshold": selected_at,
                            "selected_immediate_above": selected_above,
                            "threshold_reference_transmission_literal": model.transmissions[
                                right_model
                            ],
                            "threshold_reference_elevation_deg_literal": model.reference_elevation_deg_literal,
                            "threshold_reference_airmass_binary64": f64(
                                reference_airmass
                            ),
                            "evaluation_elevation_deg": format(elevation_deg, "f"),
                            "evaluation_point_role": point_role,
                            "evaluation_elevation_rad_binary64": f64(elevation),
                            "evaluation_elevation_rad_hex": elevation.hex(),
                            "evaluation_airmass_binary64": f64(sample_airmass),
                            "evaluation_airmass_hex": sample_airmass.hex(),
                            "band": band,
                            "left_coefficients_source_literals": ";".join(
                                left_literals
                            ),
                            "right_coefficients_source_literals": ";".join(
                                right_literals
                            ),
                            "left_polynomial_decimal80": d40(left_poly_decimal),
                            "right_polynomial_decimal80": d40(right_poly_decimal),
                            "left_polynomial_condition_number_binary64": (
                                "not_applicable_q0_special_case"
                                if left_model == "am_q0"
                                else f64(left_condition)
                            ),
                            "right_polynomial_condition_number_binary64": f64(
                                right_condition
                            ),
                            "analytic_polynomial_equality": str(
                                analytically_equal
                            ).lower(),
                            "left_transmission_binary64": f64(left_transmission),
                            "right_transmission_binary64": f64(right_transmission),
                            "left_line_of_sight_tau_binary64": f64(left_los_tau),
                            "right_line_of_sight_tau_binary64": f64(right_los_tau),
                            "signed_line_of_sight_tau_jump_binary64": f64(
                                signed_los_tau_jump
                            ),
                            "absolute_line_of_sight_tau_jump_binary64": f64(
                                absolute_los_tau_jump
                            ),
                            "absolute_transmission_jump_binary64": f64(
                                absolute_jump
                            ),
                            "relative_transmission_jump_to_left_binary64": f64(
                                relative_jump
                            ),
                            "line_of_sight_tau_roundoff_bound_4096u": f64(
                                los_tau_roundoff_bound
                            ),
                            "transmission_roundoff_bound_from_tau_bound": f64(
                                transmission_roundoff_bound
                            ),
                            "jump_exceeds_roundoff_bound": str(
                                exceeds_roundoff
                            ).lower(),
                            "phase0_stop_condition": str(stop_condition).lower(),
                        }
                    )

    return rows, thresholds


def render_csv(rows: list[dict[str, str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=list(rows[0]),
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def render_report(
    rows: list[dict[str, str]],
    thresholds: dict[str, float],
    source_digests: dict[Path, str],
    table_digest: str,
    script_digest: str,
) -> bytes:
    above_q25 = [row for row in rows if row["strictly_above_q25"] == "true"]
    stopping = [row for row in above_q25 if row["phase0_stop_condition"] == "true"]
    jump_to_bound_ratios = [
        float(row["absolute_line_of_sight_tau_jump_binary64"])
        / float(row["line_of_sight_tau_roundoff_bound_4096u"])
        for row in above_q25
    ]
    maximum_roundoff_bound = max(
        float(row["line_of_sight_tau_roundoff_bound_4096u"])
        for row in above_q25
    )
    reference_rows = [
        row
        for row in above_q25
        if row["evaluation_point_role"] == "selector_reference_elevation"
    ]

    lines = [
        "# SCI-CAL-001 q-model continuity preflight",
        "",
        "## Identity and authority",
        "",
        f"- Governing application source: `{SOURCE_SHA}`.",
        f"- Audit-framework dispatch: `{AUDIT_DISPATCH_SHA}` (read-only and clean at dispatch verification).",
        f"- Bounded repair handoff SHA-256: `{HANDOFF_SHA256}`.",
        f"- Opacity amendment SHA-256: `{AMENDMENT_SHA256}`.",
        f"- `{CALIBRATE_REL}` SHA-256: `{source_digests[CALIBRATE_REL]}`.",
        f"- `{SELECTOR_REL}` SHA-256: `{source_digests[SELECTOR_REL]}`.",
        "",
        "The generator rejects source bytes that do not match those frozen digests.",
        "",
        "## Domain statement",
        "",
        "Neither the exact-base application source nor the supplied owner decision, opacity amendment, or bounded handoff gives a numeric approved elevation/airmass interval. This preflight therefore does not invent one. It evaluates the audit's representative elevations 30, 50, and 70 degrees and the selector's exact 80-degree reference. The 30--80 degree grid is diagnostic only, not a production-validity declaration. The stop result is already present at 80 degrees, so it does not depend on choosing a wider eventual validity domain.",
        "",
        "## Method",
        "",
        "The selector threshold for model `m` is derived from the exact source literal as `-log(T225_m) / A(80 deg)`, using the source pi and modified-secant coefficient. The left limit uses the preceding model and the exact-boundary/right limit uses model `m`, matching the source `<=` selection. Each band transmission is the source-order degree-six elevation polynomial multiplied by `exp(-A(e) * tau225_boundary)`. Line-of-sight optical depth is `-log(T)`.",
        "",
        "Above q25, analytic equality is tested at decimal precision 80 using the exact coefficient literals: the common 225-GHz attenuation cancels, so continuity requires identical adjacent polynomial values. Runtime values use IEEE-754 binary64. For each row, `kappa = sum(|c_i e^(6-i)|) / |P(e)|` and the conservative line-of-sight optical-depth comparison bound is `4096 * u * ((1 + kappa_left + kappa_right) * max(1, |A|, 1/|A|) + 1 + |tau_left| + |tau_right|)`, where `u = 2^-53`. The 4096-u envelope covers coefficient conversion, powers/products/sums, modified-secant arithmetic, exp/log, division, and independent left/right rounding. All reported above-q25 jumps exceed this conditioned bound by many orders of magnitude.",
        "",
        "## Exact source-derived thresholds",
        "",
        "| Boundary | Left model | Right model | tau225 binary64 | hex |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for index in range(1, len(MODELS)):
        right = MODELS[index]
        left = MODELS[index - 1]
        threshold = thresholds[right]
        lines.append(
            f"| `{right}` | `{left}` | `{right}` | `{f64(threshold)}` | `{threshold.hex()}` |"
        )

    lines.extend(
        [
            "",
            "## Above-q25 result at the selector reference",
            "",
            "| Boundary | Band | T left | T right | LOS tau left | LOS tau right | signed LOS tau jump | abs T jump | relative T jump | LOS tau roundoff bound | Analytically equal |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in reference_rows:
        lines.append(
            "| `{boundary}` | `{band}` | `{left_transmission_binary64}` | "
            "`{right_transmission_binary64}` | `{left_line_of_sight_tau_binary64}` | "
            "`{right_line_of_sight_tau_binary64}` | `{signed_line_of_sight_tau_jump_binary64}` | "
            "`{absolute_transmission_jump_binary64}` | "
            "`{relative_transmission_jump_to_left_binary64}` | "
            "`{line_of_sight_tau_roundoff_bound_4096u}` | `{analytic_polynomial_equality}` |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Disposition",
            "",
            f"**Phase 0 fails.** All {len(stopping)} of {len(above_q25)} above-q25 band/elevation rows are analytically unequal and exceed the documented binary64 roundoff bound. The largest row bound is `{f64(maximum_roundoff_bound)}` and the smallest observed absolute-jump-to-bound ratio is `{f64(min(jump_to_bound_ratios))}`. The q25 mismatch in the assessed source is recorded separately in the table but is the already authorized low-opacity repair and is not itself used as the stop condition.",
            "",
            "Per the bounded handoff, application-code work must stop. No q25/q50/q75/q95 model is modified. Only these phase-0 evidence artifacts are committed for the project owner's successor scope decision.",
            "",
            "## Artifact digests",
            "",
            f"- `{SCRIPT_NAME}`: `{script_digest}`.",
            f"- `{TABLE_NAME}`: `{table_digest}`.",
            f"- `{SUMS_NAME}` additionally records the report digest without creating a self-referential report.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def expected_artifacts(script_path: Path) -> dict[str, bytes]:
    output_dir = script_path.parent
    repo_root = output_dir.parents[1]
    source_digests = require_source_digests(repo_root)
    model = parse_source(repo_root)
    rows, thresholds = build_rows(model)
    table_bytes = render_csv(rows)
    script_digest = sha256_path(script_path)
    table_digest = sha256_bytes(table_bytes)
    report_bytes = render_report(
        rows,
        thresholds,
        source_digests,
        table_digest,
        script_digest,
    )
    sums = {
        SCRIPT_NAME: script_digest,
        TABLE_NAME: table_digest,
        REPORT_NAME: sha256_bytes(report_bytes),
    }
    sums_bytes = "".join(
        f"{digest}  {name}\n" for name, digest in sorted(sums.items())
    ).encode("utf-8")
    return {
        TABLE_NAME: table_bytes,
        REPORT_NAME: report_bytes,
        SUMS_NAME: sums_bytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify checked-in artifacts instead of rewriting them",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    artifacts = expected_artifacts(script_path)
    failed = False
    for name, expected in artifacts.items():
        path = script_path.parent / name
        if args.check:
            if not path.exists() or path.read_bytes() != expected:
                print(f"stale or missing phase-0 artifact: {path}", file=sys.stderr)
                failed = True
        else:
            path.write_bytes(expected)
            print(f"wrote {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
