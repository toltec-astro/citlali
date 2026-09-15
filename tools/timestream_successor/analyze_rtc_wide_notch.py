#!/usr/bin/env python3
"""Measure fixed-band power, exact retention and fixed-plan source response."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import signal

from analyze_rtc_notch_recovery import measured_psd, runs
from rtc_wide_notch_experiment import COHERENT_SEAL, IDS, sealed_reader
from rtc_wide_notch_replay import ARMS, ASEC
from run_rtc_notch_recovery import binding, digest, write_json


def power_record(values, mask, dt, feature):
    if not np.isfinite(values[mask]).all():
        raise ValueError("unexpected nonfinite admitted diagnostic input")
    try:
        f, p, windows = measured_psd(values, mask, dt)
    except ValueError as e:
        if "fewer than two" not in str(e):
            raise
        return dict(available=False, reason=str(e))
    df = f[1] - f[0]

    def integral(edges):
        return float(np.sum(p[(f >= edges[0]) & (f <= edges[1])]) * df)

    return dict(
        available=True,
        feature_total_power=integral(feature),
        broad_total_power=integral([9.75, 12.25]),
        total_power=float(np.sum(p) * df),
        windows=windows,
        native_interval_seconds=dt,
        line_excess=None,
        source_and_continuum_retained=True,
    )


def source_record(got, source, mask, t, xy, component):
    """Only complete principal crossings get response metrics; tails are not recovery."""
    center = np.asarray(component["center_arcsec"])
    radius = np.linalg.norm(xy - center, axis=1)
    search = abs(t - component["time"]) <= 10
    if not search.any():
        return dict(available=False, reason="no local source inspection domain")
    idx = np.flatnonzero(search)[np.argmin(radius[search])]
    extended = component["kind"] == "extended"
    sigma = component.get("apparent_FWHM_arcsec", 8.483936) / np.sqrt(8 * np.log(2))
    core_radius = (
        5 * sigma
        if extended
        else 3.8317059702075125 / np.pi * (299792458 / 150e9) / 50 * ASEC
    )
    center_radius = (
        component.get("apparent_FWHM_arcsec", 8.483936) / 2 if extended else core_radius
    )
    if radius[idx] > center_radius:
        return dict(
            available=False,
            reason="no principal crossing at this detector's own pointing in fixed local domain",
            nearest_radius_arcsec=float(radius[idx]),
        )
    # Search chooses a crossing, but must not truncate its actual domain.
    core = radius <= core_radius
    lo, hi = next((a, b) for a, b in runs(core) if a <= idx < b)
    rr = np.arange(lo + (lo % 2), hi, 2)
    count = int(mask[rr].sum())
    record = dict(
        available=False,
        central_domain_native_rows=[int(lo), int(hi)],
        expected_output_samples=len(rr),
        retained_output_samples=count,
        central_domain="5sigma apparent extended Gaussian"
        if extended
        else "Airy first-null central lobe",
        closest_time_seconds=float(t[idx]),
        minimum_radius_arcsec=float(radius[idx]),
        complete_crossing=bool(
            lo > 0 and hi < len(t) and len(rr) >= 4 and count == len(rr)
        ),
    )
    if not record["complete_crossing"]:
        return dict(
            record,
            reason="principal crossing incomplete; remaining tails not qualified",
        )
    ref = source[rr]
    y = got[rr]
    energy = np.sum(ref * ref)
    peak = np.max(ref)
    if not energy > 0 or not np.isfinite(y).all():
        raise ValueError("invalid complete source-response input")
    area = np.sum(y)
    refarea = np.sum(ref)
    centroid = (
        np.sum(xy[rr] * y[:, None], axis=0) / area
        - np.sum(xy[rr] * ref[:, None], axis=0) / refarea
    )
    local = (abs(t - t[idx]) <= 3) & mask & (np.arange(len(t)) % 2 == 0)
    record.update(
        available=True,
        peak_ratio=float(np.max(y) / peak),
        template_amplitude_ratio=float(np.sum(y * ref) / energy),
        waveform_rms_error_fraction=float(np.sqrt(np.sum((y - ref) ** 2) / energy)),
        maximum_profile_error_fraction=float(np.max(abs(y - ref)) / peak),
        centroid_shift_arcsec=centroid.tolist(),
        centroid_shift_ms=float(
            1000 * (np.sum(t[rr] * y) / area - np.sum(t[rr] * ref) / refarea)
        ),
        observed_local_negative_fraction=float(max(0, -np.min(got[local])) / peak),
        ringing_domain="available phase-zero outputs within3s; mask retains missing support",
        # The compact central lobe is not a complete ringing/area
        # domain. Never promote a partial-core sum to total response.
        integrated_response_ratio=float(area / refarea) if extended else None,
        integrated_response_scope="complete5sigma apparent Gaussian only"
        if extended
        else "unavailable: finite compact-core coverage is not total impulse-response support",
        mapped_response_qualification=False,
    )
    return record


def analyze(a):
    a.output.mkdir(parents=True, exist_ok=False)
    design = json.loads((a.prepared / "design.json").read_text())
    replay = json.loads((a.campaign / "receipt.json").read_text())
    if digest(a.prepared / "design.json") != replay["design"]["sha256"]:
        raise ValueError("replay design changed")
    old, used = sealed_reader(Path(design["prior_root"]), COHERENT_SEAL)
    n = json.loads(old("native-full/receipt.json").read_text())["rows"]
    t = np.fromfile(old("native-full/native-time.f64"), "<f8")
    apt = np.fromfile(old("geometry-02/apt.f64"), "<f8").reshape(491, 6)
    baseline = np.memmap(
        old("prepare-03/lowpass-good.u8"), "?", mode="r", shape=(n, 491)
    )
    weights = np.where(apt[:, 4] == 0, 1 / apt[:, 3] ** 2, 0)
    dt = design["interval_seconds"]
    ndt = design["nominal_interval"]
    output_base_seconds = np.sum(baseline[::2], axis=0) * 2 * ndt
    total_weight = float(np.sum(output_base_seconds * weights))
    retained = {
        k: dict(
            target_output_detector_seconds=0.0,
            target_native_cell_seconds=0.0,
            network_static_weight_seconds=total_weight,
            detectors={},
        )
        for k in ARMS
    }
    measurements = []
    response_arrays = []
    max_error = 0.0
    config_bindings = []
    for d in IDS:
        cfg = json.loads((a.prepared / f"{d}.json").read_text())
        root = a.campaign / str(d)
        record = next(r for r in replay["records"] if r["detector"] == d)
        if digest(a.prepared / f"{d}.json") != record["specification"]["sha256"]:
            raise ValueError("changed replay specification")
        config_bindings.append(binding(a.prepared / f"{d}.json"))
        arms = ARMS if d in IDS[:8] else ARMS[:1]
        original = np.fromfile(root / "original.f64", "<f8").reshape(n, 2)
        parent = np.fromfile(cfg["samples"]["path"], "<f8").reshape(n, 4)
        producer_valid = parent[:, 2:].all(1)
        if not np.isfinite(original[producer_valid]).all():
            raise ValueError("unexpected nonfinite in admitted original input")
        expected_input = np.where(producer_valid[:, None], original, 0.0)
        xy = (
            np.fromfile(old(f"geometry-02/pointing-{d}.f64"), "<f8").reshape(n, 2)
            * ASEC
        )
        masks = {k: np.fromfile(root / f"{k}-causes.u8", "u1") == 0 for k in arms}
        arrays = {
            k: np.fromfile(root / f"{k}-native.f64", "<f8").reshape(n, 2) for k in arms
        }
        if not np.array_equal(masks["lowpass"], baseline[:, d]):
            raise ValueError("existing baseline support changed")
        common = np.logical_and.reduce(list(masks.values()))
        for arm in arms:
            rr = np.fromfile(root / f"{arm}-rows.i64", "<i8")
            if not np.array_equal(
                rr, np.flatnonzero(masks[arm] & (np.arange(n) % 2 == 0))
            ):
                raise ValueError("phase/support mismatch")
            if d in IDS[:8]:
                outsec = len(rr) * 2 * ndt
                native_seconds = float(masks[arm].sum() * ndt)
                retained[arm]["target_output_detector_seconds"] += outsec
                retained[arm]["target_native_cell_seconds"] += native_seconds
                retained[arm]["network_static_weight_seconds"] += (
                    outsec - output_base_seconds[d]
                ) * weights[d]
                retained[arm]["detectors"][str(d)] = dict(
                    output_seconds=outsec,
                    native_cell_seconds=native_seconds,
                    weight=float(weights[d]),
                    weight_seconds=float(outsec * weights[d]),
                )
            trial = next(z for z in cfg["trials"] if z["id"] == arm)
            h = np.array(trial.get("finite_notch", [1.0]))
            lp = np.array(cfg["fir"])
            # Explicit conditional numerical comparison to actual sequential
            # binary64-FMA Apply; all claims use the actual C++ outputs.
            expected = signal.fftconvolve(
                signal.fftconvolve(expected_input, h[:, None], axes=0, mode="same"),
                lp[:, None],
                axes=0,
                mode="same",
            )
            error = float(np.max(abs(expected[masks[arm]] - arrays[arm][masks[arm]])))
            max_error = max(max_error, error)
            if not np.isfinite(error) or error > 1e-11 * np.max(
                abs(original[masks[arm]])
            ):
                raise ValueError("finite replay disagrees with coefficient prediction")
            for support, mask in (("common", common), ("own", masks[arm])):
                for coordinate in (0, 1):
                    for stride in (1, 2):
                        row = dict(
                            detector=d,
                            arm=arm,
                            support=support,
                            coordinate="xr"[coordinate],
                            stage="native" if stride == 1 else "decimated",
                            unit="native-coordinate-squared",
                            calibration_reference_scale=float(apt[d, 2]),
                            spectrum=power_record(
                                arrays[arm][::stride, coordinate],
                                mask[::stride],
                                stride * dt,
                                design["fixed_feature_edges_hz"],
                            ),
                        )
                        measurements.append(row)
                    # Original spectrum on exactly the same complete windows.
                    measurements.append(
                        dict(
                            detector=d,
                            arm=arm,
                            support=support,
                            coordinate="xr"[coordinate],
                            stage="original",
                            unit="native-coordinate-squared",
                            spectrum=power_record(
                                original[:, coordinate],
                                mask,
                                dt,
                                design["fixed_feature_edges_hz"],
                            ),
                        )
                    )
            for inj in cfg["injections"]:
                if digest(inj["path"]) != inj["sha256"]:
                    raise ValueError("changed fixed source/line probe")
                source = np.fromfile(inj["path"], "<f8").reshape(n, 2)
                actual = (
                    np.fromfile(root / f"{arm}-{inj['identity']}.f64", "<f8").reshape(
                        n, 2
                    )
                    - arrays[arm]
                )
                for support, mask in (("common", common), ("own", masks[arm])):
                    for coordinate in (0, 1):
                        if inj["identity"] == "known-line":
                            for stride in (1, 2):
                                measurements.append(
                                    dict(
                                        detector=d,
                                        arm=arm,
                                        support=support,
                                        coordinate="xr"[coordinate],
                                        stage="known-line-native"
                                        if stride == 1
                                        else "known-line-decimated",
                                        unit="known-native-coordinate-squared",
                                        spectrum=power_record(
                                            actual[::stride, coordinate],
                                            mask[::stride],
                                            stride * dt,
                                            design["fixed_feature_edges_hz"],
                                        ),
                                    )
                                )
                        else:
                            metrics = source_record(
                                actual[:, coordinate],
                                source[:, coordinate],
                                mask,
                                t,
                                xy,
                                inj["source"],
                            )
                            measurements.append(
                                dict(
                                    detector=d,
                                    arm=arm,
                                    support=support,
                                    coordinate="xr"[coordinate],
                                    stage="source",
                                    source=inj["identity"],
                                    metrics=metrics,
                                )
                            )
                if d == 269 and inj["identity"] != "known-line":
                    c = inj["source"]
                    local = abs(t - c["time"]) <= 5
                    data = np.column_stack(
                        [
                            t[local] - c["time"],
                            source[local, 0],
                            actual[local, 0],
                            masks[arm][local],
                            common[local],
                            np.arange(n)[local] % 2,
                        ]
                    )
                    p = a.output / f"{arm}-{inj['identity']}-source.f64"
                    data.astype("<f8").tofile(p)
                    response_arrays.append(
                        dict(
                            arm=arm,
                            source=inj["identity"],
                            data=binding(p),
                            columns=[
                                "relative_seconds",
                                "input",
                                "response",
                                "own_valid",
                                "common_valid",
                                "row_phase",
                            ],
                        )
                    )
    result = dict(
        retention=retained,
        measurements=measurements,
        source_plot_arrays=response_arrays,
        fixed_feature_edges_hz=design["fixed_feature_edges_hz"],
        broad_diagnostic_hz=[9.75, 12.25],
        total_power_is_not_contaminant_power=True,
        line_excess=None,
        static_weights_are_not_achieved_sensitivity=True,
        unknown_source_retained=True,
        maximum_conditional_FFT_vs_actual_Apply_absolute_error=max_error,
        bindings=used + config_bindings + [binding(a.campaign / "receipt.json")],
        RTC_fixed_plan_only=True,
        FRUIT_qualification=False,
        production=False,
    )
    write_json(a.output / "comparison.json", result)
    plot(a.output, result)
    print(json.dumps(retained, indent=2))


def plot(out, result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = ["compact-1", "compact-4", "fast-0", "extended"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, name in zip(axes.flat, sources):
        for j, arm in enumerate(ARMS):
            entry = next(
                r
                for r in result["source_plot_arrays"]
                if r["source"] == name and r["arm"] == arm
            )
            z = np.fromfile(entry["data"]["path"], "<f8").reshape(-1, 6)
            scale = np.max(z[:, 1])
            valid = (z[:, 3] > 0) & (z[:, 5] == 0)
            if j == 0:
                ax.plot(z[:, 0], z[:, 1] / scale, "k--", label="native source")
            phase = z[:, 5] == 0
            y = z[:, 2] / scale
            y[~valid] = np.nan
            ax.plot(z[phase, 0], y[phase], label=arm, lw=1)
        ax.set(
            title=f"269 / {name}",
            xlabel="Seconds from declared crossing",
            ylabel="Response / native local peak",
        )
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Actual frozen-plan paired source response — missing output support remains blank"
    )
    fig.tight_layout()
    fig.savefig(out / "source-response.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for j, d in enumerate(IDS[:8]):
        axes[0].plot(
            ARMS,
            [
                result["retention"][a]["detectors"][str(d)]["output_seconds"]
                for a in ARMS
            ],
            "o-",
            label=str(d),
            alpha=0.6,
        )
    axes[0].set(ylabel="Retained output-cell seconds per affected detector")
    axes[1].bar(
        ARMS, [result["retention"][a]["network_static_weight_seconds"] for a in ARMS]
    )
    axes[1].set(ylabel="Whole-network static reference-weight seconds")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle(
        "Retention beyond existing exclusions; reference weights are not achieved sensitivity"
    )
    fig.tight_layout()
    fig.savefig(out / "retention.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "campaign", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    analyze(p.parse_args())


if __name__ == "__main__":
    main()
