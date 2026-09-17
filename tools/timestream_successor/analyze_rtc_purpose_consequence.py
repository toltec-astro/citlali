#!/usr/bin/env python3
"""Summarize runtime RTC consequences without selecting acceptance limits."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy import signal

from analyze_rtc_notch_recovery import measured_psd
from run_rtc_notch_recovery import binding, write_json


def projection_noise_sigma(source, rows, kernel, sigma):
    """Conditional original white-noise error of the existing linear projection.

    Uses the full finite-filter adjoint, including correlated final samples.
    This is a fixture calculation, not a measured sky/noise model or weights.
    No source-response correction or calibration renormalization is performed.
    """
    energy = np.dot(source[rows], source[rows])
    if energy <= 0 or not len(rows):
        return None
    weights = np.zeros(len(source))
    weights[rows] = source[rows] / energy
    original_weights = signal.convolve(
        weights, np.asarray(kernel)[::-1], mode="same", method="auto"
    )
    return float(sigma * np.linalg.norm(original_weights))


def extent(values):
    a = np.array([x for x in values if x is not None and np.isfinite(x)])
    return [float(a.min()), float(a.max())] if len(a) else None


def analyze(args):
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((args.prepared / "input.json").read_text())
    study = json.loads((args.prepared / "study.json").read_text())
    receipt = yaml.load(
        (args.replay / "receipt.yaml").read_text(), Loader=yaml.CSafeLoader
    )
    assert receipt["original_pair_unchanged"] and receipt["admitted_parent_unchanged"]
    old = json.loads((args.prior / "verification.json").read_text())
    unchanged = {}
    for name, digest in dict(old["file_digests"], **old["outcome_digests"]).items():
        assert binding(args.replay / name)["sha256"] == digest, name
        unchanged[name] = digest
    channels = [d["channel"] for d in cfg["detectors"]]
    t = np.fromfile(args.replay / "native-time.f64", "<f8")
    t -= t[0]
    n = len(t)
    q = np.arange(0, n, 2)
    per_arm = {}
    for arm in ("exclusion-control", "donor-continuity"):
        root = args.replay / arm
        report = yaml.load(
            (root / "purpose-study/consequences.yaml").read_text(),
            Loader=yaml.CSafeLoader,
        )
        assert report["study_sha256"] == binding(args.prepared / "study.json")["sha256"]
        arm_receipt = receipt[arm.replace("-", "_")]
        assert arm_receipt["reassessment"]["execution_disposition"] == "retain"
        assert not arm_receipt["reassessment"]["scientifically_qualified"]
        assert len(arm_receipt["reassessment"]["purpose_consequences"]) == len(
            report["cases"]
        )
        assert sum(d["replaced_rows"] for d in arm_receipt["realized_detectors"]) == 0
        for i, d in enumerate(arm_receipt["realized_detectors"]):
            assert d["lowpass_FIR"] == cfg["fir"]
            assert (d["finite_notch_FIR"] or []) == (
                cfg["finite_notch"] if cfg["detectors"][i]["filter"] == "w1-t3" else []
            )
        per_arm[arm] = report

    # Same no-natural-donor fixture must preserve both arms' diagnostics too.
    def stable(record):
        a = json.loads(json.dumps(record))
        a.pop("total_seconds")
        a.pop("lowpass_plan_attempt")
        a.pop("baseline_plan_attempt")
        for c in a["cases"]:
            for key in (
                "Apply_seconds",
                "Learn_seconds",
                "plan_attempt",
                "comparison_plan_attempt",
            ):
                c.pop(key)
        return a

    assert stable(per_arm["exclusion-control"]) == stable(per_arm["donor-continuity"])
    report = per_arm["exclusion-control"]
    root = args.replay / "exclusion-control"
    summaries = []
    for c in report["cases"]:
        template = next(v for v in study["cases"] if v["identity"] == c["domain"])
        assert c["stage"] == "scheduled-final-output-F2-phase0"
        good = [r for r in c["records"] if r["available"]]
        for r in c["records"]:
            rows = np.array(r["rows"], dtype=int)
            assert np.all(rows % 2 == 0)
            if r["available"]:
                assert len(rows) == r["expected"]
        summary = dict(
            case=c["domain"],
            treatment=c["treatment"],
            controlled=template["controlled"],
            measured_detectors=len(good),
            unavailable=[
                dict(channel=r["channel"], reason=r["unavailable"])
                for r in c["records"]
                if not r["available"]
            ],
            transfer_projection_error_percent=extent(
                [100 * (r["measured"]["projection"] - 1) for r in good]
            ),
            centroid_magnitude_arcsec=extent(
                [
                    np.hypot(
                        r["measured"]["centroid_x_arcsec"],
                        r["measured"]["centroid_y_arcsec"],
                    )
                    for r in good
                    if r["measured"]["centroid_x_arcsec"] is not None
                ]
            ),
            negative_fraction=extent(
                [r["measured"]["negative_ringing_fraction"] for r in good]
            ),
            waveform_error=extent([r["measured"]["waveform_error"] for r in good]),
            own_outputs=sum(r["own_total"] for r in c["records"]),
            common_outputs=sum(r["common_total"] for r in c["records"]),
            Apply_seconds=c["Apply_seconds"],
            Learn_seconds=c["Learn_seconds"],
        )
        if template["controlled"]:
            energy = sum(r["source_energy"] for r in good)
            summary.update(
                phase_rad=template["phase_rad"],
                line_population=template["line_population"],
                added_parameter_error_percent=extent(
                    [100 * r["additional_projection_error"] for r in good]
                ),
                cohort_projection_error_percent=(
                    100
                    * sum(
                        r["source_energy"] * r["additional_projection_error"]
                        for r in good
                    )
                    / energy
                    if energy
                    else None
                ),
                absolute_residual_line_rms_native_x=extent(
                    [r.get("added_line_rms_native_x") for r in c["records"]]
                ),
                centroid_increment_arcsec=extent(
                    [
                        np.hypot(
                            r["measured"]["centroid_x_arcsec"]
                            - r["line_free"]["centroid_x_arcsec"],
                            r["measured"]["centroid_y_arcsec"]
                            - r["line_free"]["centroid_y_arcsec"],
                        )
                        for r in good
                        if r["measured"]["centroid_x_arcsec"] is not None
                        and r["line_free"]["centroid_x_arcsec"] is not None
                    ]
                ),
            )
        else:
            errors = []
            variance_numerator = 0
            total_energy = 0
            for r in good:
                d = r["channel"]
                di = channels.index(d)
                values = np.fromfile(
                    template["detectors"][di]["source"]["path"], "<f8"
                ).reshape(n, 2)[:, 0]
                h = cfg["fir"]
                if (
                    c["treatment"] == "selected-notch-lowpass"
                    and cfg["detectors"][di]["filter"] == "w1-t3"
                ):
                    h = np.convolve(cfg["finite_notch"], h)
                noise = projection_noise_sigma(
                    values, np.array(r["rows"], int), h, 1e-6
                )
                errors.append(noise)
                variance_numerator += (r["source_energy"] * noise) ** 2
                total_energy += r["source_energy"]
            summary["conditional_white_noise_projection_sigma"] = extent(errors)
            summary["cohort_conditional_white_noise_sigma"] = (
                float(np.sqrt(variance_numerator) / total_energy)
                if total_energy
                else None
            )
        summaries.append(summary)
    real = []
    for di, d in enumerate(channels):
        low = np.fromfile(root / f"purpose-study/{d}-lowpass.f64", "<f8").reshape(n, 2)
        current = np.fromfile(root / f"{d}-filtered.f64", "<f8").reshape(n, 2)
        own = np.fromfile(root / f"{d}-map-center.u8", "u1").astype(bool)
        other = np.fromfile(
            root / f"purpose-study/{d}-lowpass-centers.u8", "u1"
        ).astype(bool)
        common = own & other
        spectral = (
            common
            & np.fromfile(root / f"purpose-study/{d}-selected-review.u8", "u1").astype(
                bool
            )
            & np.fromfile(root / f"purpose-study/{d}-lowpass-review.u8", "u1").astype(
                bool
            )
        )
        for coord in (0, 1):
            a, b = low[q, coord], current[q, coord]
            if (
                not np.isfinite(a[spectral[q]]).all()
                or not np.isfinite(b[spectral[q]]).all()
            ):
                raise ValueError("nonfinite admitted final-output spectrum")
            f, pa, wa = measured_psd(a, spectral[q], 0.016384)
            ff, pb, wb = measured_psd(b, spectral[q], 0.016384)
            assert np.array_equal(f, ff) and wa == wb
            band = (f >= 10.380900) & (f <= 11.631611)
            df = f[1] - f[0]
            a_power = float(pa[band].sum() * df)
            b_power = float(pb[band].sum() * df)
            record = dict(
                channel=d,
                coordinate="xr"[coord],
                stage="actual-scheduled-final-output",
                lowpass_feature_stored_power=a_power,
                selected_feature_stored_power=b_power,
                ratio=b_power / a_power if a_power else None,
                feature_hz=[10.380900, 11.631611],
                units="native-coordinate squared",
                windows_scheduled_indices=wa,
                windows_native_rows=[[2 * x, 2 * y] for x, y in wa],
                own_outputs=int(own[q].sum()),
                lowpass_outputs=int(other[q].sum()),
                common_outputs=int(common[q].sum()),
                spectral_eligible_outputs=int(spectral[q].sum()),
                estimated_line_excess=None,
            )
            real.append(record)
            np.column_stack([f, pa, pb]).astype("<f8").tofile(
                args.output / f"final-psd-{d}-{'xr'[coord]}.f64"
            )
    result = dict(
        status="PASS",
        baseline_unchanged_files=len(unchanged),
        unchanged_sha256=unchanged,
        diagnostic_arms_equal=True,
        science_qualification="unresolved",
        source=binding(args.replay / "receipt.yaml"),
        study=binding(args.prepared / "study.json"),
        summary=summaries,
        real=real,
        estimator_limits="projection and centroid are diagnostic surrogates; no new downstream fit; fractional unrenormalized response; independent white noise sigma1e-6 is a fixture assumption; four phases are not a population sample",
        unsupported=[
            "OOF defocused/focus/wavefront response",
            "POINT full2D fitted offset and repeatability",
            "BEAM fitted width, far wings, calibrated normalization",
            "SCIENCE general morphology/orientation and calibrated photometry",
        ],
    )
    write_json(args.output / "results.json", result)
    plot(args, study, root, n, t)
    print(
        json.dumps(
            dict(
                status="PASS",
                unchanged=len(unchanged),
                cases=len(summaries),
                real_coordinate_spectra=len(real),
                both_arms_equal=True,
            )
        )
    )


def plot(args, study, root, n, t):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = 269
    di = next(i for i, v in enumerate(study["positions"]) if v["channel"] == d)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    for ax, name in zip(axes, ["quiet-faint", "maximum-speed-faint", "extended-25"]):
        c = next(c for c in study["cases"] if c["identity"] == name)
        entry = c["detectors"][di]
        lo, hi = entry["ringing_window"]
        rr = np.arange(lo + (lo % 2), hi, 2)
        ref = np.fromfile(entry["source"]["path"], "<f8").reshape(n, 2)[:, 0]
        peak = ref.max()
        origin = t[round(np.mean(entry["window"]))]
        mask = np.fromfile(root / f"{d}-map-center.u8", "u1").astype(
            bool
        ) & np.fromfile(root / f"purpose-study/{d}-lowpass-centers.u8", "u1").astype(
            bool
        )
        ax.plot(t[rr] - origin, ref[rr] / peak, "k--", label="sampled source")
        for arm, label in [("lowpass", "low-pass"), ("selected", "notch + low-pass")]:
            y = np.fromfile(
                root / f"purpose-study/{name}-{arm}-{d}.f64", "<f8"
            ).reshape(n, 2)[:, 0]
            z = np.where(mask[rr], y[rr] / peak, np.nan)
            ax.plot(t[rr] - origin, z, label=label)
        ax.set(
            title=f"{name}; channel269; actual F2 output times",
            xlabel="Seconds relative to crossing",
            ylabel="Fraction of input peak",
        )
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=3)
    fig.savefig(args.output / "transfer.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "replay", "prior", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    analyze(p.parse_args())


if __name__ == "__main__":
    main()
