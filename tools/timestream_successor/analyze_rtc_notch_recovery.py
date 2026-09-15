#!/usr/bin/env python3
"""Measure finite RTC outputs and paired injected sources; no acceptance policy."""

import argparse
import json
from pathlib import Path
import numpy as np
from run_rtc_notch_recovery import CASES, binding, write_json


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def measured_psd(values, mask, dt):
    """Explicit conditioned diagnostic: accepted 4s Hann/50% median convention.

    Full qualifying windows only, with end anchoring on each contiguous run;
    no filling, no joining gaps. This is not a new original Learn product.
    """
    n = round(4 / dt)
    if n % 2:
        raise ValueError(
            "bounded diagnostic requires the even FFT lengths used by this experiment"
        )
    hop = n // 2
    w = np.hanning(n)
    norm = (1 / dt) * np.sum(w * w)
    values = np.asarray(values)
    population = np.median(values[mask])
    psds = []
    support = []
    for lo, hi in runs(mask):
        if hi - lo < n:
            continue
        starts = list(range(lo, hi - n + 1, hop))
        if starts[-1] != hi - n:
            starts.append(hi - n)
        for start in starts:
            v = values[start : start + n] - population
            v = v - np.median(v)
            p = abs(np.fft.rfft(v * w)) ** 2 / norm
            p[1:-1] *= 2
            psds.append(p)
            support.append([int(start), int(start + n)])
    if len(psds) < 2:
        raise ValueError("fewer than two complete diagnostic windows")
    return np.fft.rfftfreq(n, dt), np.mean(psds, axis=0), support


def band_measure(f, p, center):
    # Raw band integral is primary. A separately labeled fixed-sideband
    # subtraction is only an exploratory residual descriptor, not truth.
    band = abs(f - center) <= 0.75
    side = (abs(f - center) >= 1.25) & (abs(f - center) <= 2.5)
    df = f[1] - f[0]
    background = float(np.median(p[side]))
    power = float(p[band].sum() * df)
    return dict(
        band_hz=[max(0, center - 0.75), center + 0.75],
        stored_power=power,
        sideband_median_psd=background,
        signed_excess=power - background * int(band.sum()) * df,
        peak_to_sideband=float(max(p[band]) / background),
    )


def overlap_seconds(time, dt, mask, intervals):
    # Prior audit microseconds are relative to the same native origin. Use
    # actual half-open integration-cell intersections, not rounded row tags.
    lo = (time - time[0]) - dt / 2
    hi = lo + dt
    total = 0.0
    for first, last in intervals:
        total += np.maximum(
            0, np.minimum(hi, last / 1e6) - np.maximum(lo, first / 1e6)
        )[mask].sum()
    return float(total)


def difference_covariance(values, mask):
    """Actual output first-difference covariance; includes residual sky/lines.

    A diagnostic of changed correlations, not a white-noise sensitivity claim.
    No pair or lag is permitted to cross an excluded stretch.
    """
    parts = [np.diff(values[lo:hi]) for lo, hi in runs(mask) if hi - lo > 5]
    count = sum(len(p) for p in parts)
    if not count:
        return None
    mean = sum(p.sum() for p in parts) / count
    return [
        float(
            sum(np.dot(p[: len(p) - lag] - mean, p[lag:] - mean) for p in parts)
            / sum(len(p) - lag for p in parts)
        )
        for lag in range(5)
    ]


def source_metrics(delta, source, time, rows, center):
    local = abs(time[rows] - center) <= 3
    idx = rows[local]
    ref = source[idx]
    expected = (abs(time - center) <= 3) & (np.arange(len(time)) % 2 == 0)
    total = float(np.dot(source[expected], source[expected]))
    energy = float(np.dot(ref, ref))
    fraction = energy / total if total > 0 else 0
    result = dict(retained_local_source_energy_fraction=fraction)
    if not len(idx) or energy <= 1e-10 * total:
        return dict(
            **result,
            transfer_available=False,
            reason="selected crossing excluded by support",
        )
    got = delta[idx]
    tau = time[idx] - center
    refarea = float(ref.sum())
    area = float(got.sum())
    result.update(
        transfer_available=True,
        template_amplitude_ratio=float(np.dot(ref, got) / energy),
        peak_ratio=float(max(got) / max(ref)),
        signed_area_ratio=area / refarea,
        waveform_error_fraction=float(np.linalg.norm(got - ref) / np.sqrt(energy)),
        centroid_shift_ms=float(
            (np.dot(tau, got) / area - np.dot(tau, ref) / refarea) * 1000
        ),
        maximum_negative_fraction=float(max(0, -min(got)) / max(ref)),
    )
    return result


def analyze(a):
    a.output.mkdir(parents=True, exist_ok=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = []
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    sourcefig, srcax = plt.subplots(3, 2, figsize=(13, 10))
    for ci, (name, nw, d, _) in enumerate(CASES):
        cfg = json.loads((a.plans / f"{name}.json").read_text())
        root = a.campaign / name
        receipt = json.loads((root / "receipt.json").read_text())
        n = receipt["rows"]
        dt = cfg["nominal_interval"]
        mdt = cfg["measured_interval"]
        geometry = np.fromfile(root / "geometry.f64", "<f8").reshape(n, 4)
        time = geometry[:, 0]
        original = np.fromfile(root / "original.f64", "<f8").reshape(n, 2)
        export = np.fromfile(cfg["samples"]["path"], "<f8").reshape(n, 4)
        assert np.array_equal(original, export[:, :2], equal_nan=True)
        # Recovered accepted spectra must reproduce the prior sealed audit.
        folder = Path(cfg["audit_receipt"]["path"]).parent
        pr = json.loads((folder / "receipt.json").read_text())
        prior = np.memmap(
            folder / "spectra.f64",
            "<f8",
            mode="r",
            shape=(pr["channels"], 2, 4, pr["bins"]),
        )
        measured = np.fromfile(root / "initial-psd.f64", "<f8").reshape(2, pr["bins"])
        assert np.allclose(measured, prior[d, :, 0, :], rtol=1e-12, atol=0)
        trial_ids = [t["id"] for t in cfg["trials"]]
        active_ids = [t["id"] for t in cfg["trials"] if not t["reject"]]
        cause = {k: np.fromfile(root / f"{k}-causes.u8", "u1") for k in trial_ids}
        output = {
            k: np.fromfile(root / f"{k}-native.f64", "<f8").reshape(n, 2) for k in cause
        }
        selected = {k: np.fromfile(root / f"{k}-rows.i64", "<i8") for k in cause}
        assert all(np.all(v % 2 == 0) for v in selected.values())
        common = np.logical_and.reduce([cause[k] == 0 for k in active_ids])
        conditional = cfg["conditional_transient_accounting"]["intervals_us"]["direct"]
        acct = cfg["conditional_transient_accounting"]
        eligible = n * dt
        metrics = dict(
            case=name,
            network=nw,
            channel=d,
            array="a2000",
            source_revision=receipt["source_revision"],
            input=cfg["samples"],
            initial_spectra_reproduce_accepted=True,
            eligible_seconds=eligible,
            existing_screening_seconds=receipt["transient_excluded_cells"] * dt,
            prior_conditional_direct_seconds=acct["durations_us"]["direct_us"] / 1e6,
            prior_scan_binding=acct["scan_losses"],
            cost={},
            coordinates=[],
            injections=[],
            bindings=[
                binding(root / "receipt.json"),
                binding(a.plans / f"{name}.json"),
            ],
            treatment_trials=active_ids,
        )
        for k in cause:
            retained = cause[k] == 0
            durations = {
                str(int(code)): int(np.count_nonzero(cause[k] == code)) * dt
                for code in np.unique(cause[k])
            }
            duration = float(retained.sum() * dt)
            direct_overlap = overlap_seconds(time, dt, retained, conditional)
            metrics["cost"][k] = dict(
                causes_seconds=durations,
                retained_native_seconds=duration,
                candidate_retained_beyond_screening_seconds=duration,
                scientifically_qualified_recovery=False,
                conditional_recovery_after_direct_seconds=max(
                    0, duration - direct_overlap
                ),
                additional_loss_relative_lowpass_seconds=float(
                    (cause["lowpass"] == 0).sum() * dt - duration
                ),
                retained_fraction_of_screening_baseline=duration
                / (eligible - metrics["existing_screening_seconds"]),
                decimated_output_rows=len(selected[k]),
            )
        # Report both coordinates separately on the identical admitted support.
        for c in range(2):
            freq, raw, support = measured_psd(original[:, c], common, mdt)
            spectra = {
                k: measured_psd(output[k][:, c], common, mdt)[1] for k in active_ids
            }
            center = cfg["notch_hz"]
            fold = abs((center + 0.25 / mdt) % (0.5 / mdt) - 0.25 / mdt)
            target = {
                k: band_measure(freq, p, center)
                for k, p in {"original": raw, **spectra}.items()
            }
            native_common = common[::2]
            outfreq, lpout, outsupport = measured_psd(
                output["lowpass"][::2, c], native_common, 2 * mdt
            )
            output_spectra = {
                k: measured_psd(output[k][::2, c], native_common, 2 * mdt)[1]
                for k in active_ids
            }
            record = dict(
                coordinate="x" if c == 0 else "r",
                native_target_hz=center,
                folded_target_hz=fold,
                native_target=target,
                output_folded_band={
                    k: band_measure(outfreq, p, fold) for k, p in output_spectra.items()
                },
                actual_common_native_seconds=float(common.sum() * dt),
                predicted_whole_original_support={
                    k: float(
                        np.fromfile(root / f"{k}-predicted.f64", "<f8")
                        .reshape(2, pr["bins"], 5)[c][
                            abs(np.fft.rfftfreq(pr["fft_samples"], mdt) - center)
                            <= 0.75,
                            4,
                        ]
                        .sum()
                    )
                    for k in active_ids
                },
                output_first_difference_covariance={
                    k: difference_covariance(output[k][::2, c], native_common)
                    for k in active_ids
                },
                covariance_lag_seconds=2 * mdt,
                prediction_support="whole original accepted spectrum; bin-center steady-state proxy; finite measured comparison uses common retained support",
                native_windows=support,
                output_windows=outsupport,
            )
            for k in active_ids:
                record[k + "_native_band_power_ratio"] = (
                    target[k]["stored_power"] / target["original"]["stored_power"]
                )
            metrics["coordinates"].append(record)
            axes[ci, c].semilogy(
                freq, raw, label="original on common support", alpha=0.6
            )
            for k, ps in spectra.items():
                axes[ci, c].semilogy(freq, ps, label=k)
            axes[ci, c].axvline(center, color="k", lw=0.5)
            axes[ci, c].set_title(
                f"{name}: 152390 / n{nw} ch{d} / " + record["coordinate"]
            )
            axes[ci, c].set_xlabel("Native frequency (Hz)")
            axes[ci, c].set_ylabel("PSD (native coordinate²/Hz)")
            axes[ci, c].legend(fontsize=7)
        for inj in cfg["injections"]:
            source = np.fromfile(inj["path"], "<f8").reshape(n, 2)
            record = dict(
                identity=inj["identity"],
                speed_arcsec_per_sec=inj["measured_speed_arcsec_per_sec"],
                boundary_challenge=inj["boundary_challenge"],
                trials={},
            )
            for k in cause:
                paired = np.fromfile(
                    root / f"{k}-{inj['identity']}.f64", "<f8"
                ).reshape(n, 2)
                delta = paired - output[k]
                record["trials"][k] = {
                    ("x" if c == 0 else "r"): source_metrics(
                        delta[:, c],
                        source[:, c],
                        time,
                        selected[k],
                        inj["center_unix_sec"],
                    )
                    for c in range(2)
                }
                if inj["identity"] == "maxp0" and k != "reject":
                    for c in range(2):
                        rows = selected[k]
                        idx = rows[abs(time[rows] - inj["center_unix_sec"]) < 0.5]
                        srcax[ci, c].plot(
                            time[idx] - inj["center_unix_sec"],
                            delta[idx, c] / (1e-5 if c == 0 else 2e-6),
                            label=k,
                        )
            if inj["identity"] == "maxp0":
                for c in range(2):
                    idx = np.flatnonzero(abs(time - inj["center_unix_sec"]) < 0.5)
                    srcax[ci, c].plot(
                        time[idx] - inj["center_unix_sec"],
                        source[idx, c] / (1e-5 if c == 0 else 2e-6),
                        "k--",
                        lw=0.8,
                        label="injected original",
                    )
                    srcax[ci, c].set_title(
                        f"{name} / {'x' if c == 0 else 'r'}: fixed crossing {inj['measured_speed_arcsec_per_sec']:.1f} arcsec/s"
                    )
                    srcax[ci, c].set_xlabel("Seconds from injected crossing")
                    srcax[ci, c].legend(fontsize=8)
            metrics["injections"].append(record)
        results.append(metrics)
    fig.tight_layout()
    fig.savefig(a.output / "spectra.png", dpi=150)
    plt.close(fig)
    sourcefig.tight_layout()
    sourcefig.savefig(a.output / "source-transfer.png", dpi=150)
    plt.close(sourcefig)
    write_json(a.output / "results.json", results)
    print(
        json.dumps(
            [
                dict(
                    case=r["case"],
                    cost=r["cost"],
                    bands=[
                        {k: v for k, v in c.items() if k.endswith("ratio")}
                        for c in r["coordinates"]
                    ],
                    fastest=next(
                        i for i in r["injections"] if i["identity"] == "maxp0"
                    ),
                )
                for r in results
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plans", type=Path, required=True)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    analyze(p.parse_args())
