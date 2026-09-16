#!/usr/bin/env python3
"""Audit fixed-filter support, direct centers and frozen-plan probe responses."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy import signal

from analyze_rtc_wide_notch import power_record, source_record
from rtc_speed_support_replay import ARMS, ASEC
from run_rtc_notch_recovery import binding, write_json


def erode(mask, half):
    counts = np.r_[0, np.cumsum(~mask, dtype=np.int64)]
    out = np.zeros(len(mask), bool)
    rows = np.arange(half, len(mask) - half)
    out[rows] = counts[rows + half + 1] == counts[rows - half]
    return out


def read(root, d, what, dtype="u1"):
    return np.fromfile(root / f"{d}-{what}", dtype)


def metrics(y, reference):
    energy = np.dot(reference, reference)
    if not len(y) or energy == 0:
        return dict(available=False, samples=len(y))
    return dict(
        available=True,
        samples=len(y),
        rms=float(np.sqrt(np.mean(y * y))),
        reference_rms=float(np.sqrt(np.mean(reference * reference))),
        power_ratio=float(np.dot(y, y) / energy),
        template_amplitude_ratio=float(np.dot(y, reference) / energy),
        waveform_rms_error_fraction=float(
            np.sqrt(np.dot(y - reference, y - reference) / energy)
        ),
    )


def analyze(a):
    a.output.mkdir(parents=True, exist_ok=False)
    design = json.loads((a.prepared / "design.json").read_text())
    cfg = json.loads((a.prepared / "both.json").read_text())
    channels = design["channels"]
    old = Path(design["original_control"])
    prior = Path(design["original_prior"])
    unix = np.fromfile(prior / "native-unix.f64", "<f8")
    t = unix - unix[0]
    n = len(t)
    q = np.arange(0, n, 2)
    records = []
    science = []
    totals = {k: 0 for k in ARMS}
    direct_lost = 0
    checks = dict(
        original_control_byte_parity=True,
        centers_and_evidence_unchanged=True,
        fixed_coefficients=True,
        exact_dependency_masks=True,
        common_dependency_bitwise_equal=True,
        conditioned_spectral_population_and_numerics_unchanged=True,
        zero_speed_center_hits=True,
        non_speed_support_unchanged=True,
        identical_donor_exclusion_arms=True,
    )
    map_jobs = []
    apt = np.fromfile(prior / "geometry-02/apt.f64", "<f8").reshape(491, 6)
    for arm in ARMS:
        root = a.campaign / arm
        for name in (
            "events.yaml",
            "event-decisions.yaml",
            "support-decisions.yaml",
            "processing-scans.yaml",
            "native-time.f64",
            "original-psd.f64",
        ):
            assert (root / name).read_bytes() == (old.parent / name).read_bytes(), (
                arm,
                name,
            )
        receipt = yaml.load(
            (root / "receipt.yaml").read_text(), Loader=yaml.CSafeLoader
        )
        assert (
            receipt["original_pair_unchanged"] and receipt["admitted_parent_unchanged"]
        )
        for det in receipt["exclusion_control"]["realized_detectors"]:
            notch = (
                cfg["detectors"][channels.index(det["channel"])]["filter"] == "w1-t3"
            )
            assert np.array_equal(det["lowpass_FIR"], cfg["fir"])
            assert np.array_equal(
                det["finite_notch_FIR"] or [], cfg["finite_notch"] if notch else []
            )
            assert det["replaced_rows"] == 0 and not det["map_route_authorized"]
    all_masks = {}
    all_outputs = {}
    all_probe = {}
    for d in channels:
        roots = {k: a.campaign / k / "exclusion-control" for k in ARMS}
        original = read(old, d, "input-causes.u8")
        speed = read(roots["control"], d, "speed-evidence.u8")
        centers = original == 0
        direct_lost += int(np.count_nonzero(~centers[q]))
        notch = cfg["detectors"][channels.index(d)]["filter"] == "w1-t3"
        kernel = (
            signal.convolve(cfg["finite_notch"], cfg["fir"])
            if notch
            else np.array(cfg["fir"])
        )
        half = len(kernel) // 2
        masks = {k: read(r, d, "map-center.u8").astype(bool) for k, r in roots.items()}
        values = {
            k: read(r, d, "filtered.f64", "<f8").reshape(n, 2) for k, r in roots.items()
        }
        all_masks[d] = masks
        all_outputs[d] = values
        for file in old.glob(f"{d}-*"):
            if file.name.endswith((".f64", ".u8", ".i64")):
                assert (
                    file.read_bytes() == (roots["control"] / file.name).read_bytes()
                ), file
        for arm, r in roots.items():
            assert np.array_equal(read(r, d, "input-causes.u8"), original)
            assert np.array_equal(read(r, d, "speed-evidence.u8"), speed)
            support = read(r, d, "support-causes.u8")
            want = original.copy()
            if arm in ("low-only", "both"):
                want[want == 7] = 0
            if arm in ("high-only", "both"):
                want[want == 8] = 0
            assert np.array_equal(support, want), (
                d,
                arm,
                "unexpected non-speed change",
            )
            numeric = erode(support == 0, half)
            state = read(r, d, "state.u8")
            assert np.array_equal(numeric, (state & 3) == 3), (
                d,
                arm,
                "numeric footprint",
            )
            expected = numeric & centers & (np.arange(n) % 2 == 0)
            assert np.array_equal(masks[arm], expected), (d, arm, "registered center")
            assert not masks[arm][speed != 0].any()
            assert np.array_equal(
                values[arm][masks["control"]].view("u8"),
                values["control"][masks["control"]].view("u8"),
            )
            for name in ["post-notch-psd.f64", "post-lowpass-psd.f64"]:
                assert (old / name).read_bytes() == (r / name).read_bytes()
            for file in r.glob(f"{d}-*"):
                assert (
                    file.read_bytes()
                    == (r.parent / "donor-continuity" / file.name).read_bytes()
                ), file
            totals[arm] += int(masks[arm].sum())
        populations = dict(
            control=masks["control"],
            low_recovered=masks["low-only"] & ~masks["control"],
            high_recovered=masks["high-only"] & ~masks["control"],
            both_required=masks["both"] & ~masks["low-only"] & ~masks["high-only"],
        )
        records.append(
            dict(
                channel=d,
                half_samples=half,
                direct_excluded=int((~centers[q]).sum()),
                retained={k: int(m.sum()) for k, m in masks.items()},
            )
        )
        # All source/noise probes execute the exact same complete frozen plan.
        # The convolution here is an independent numerical oracle only.
        for probe in design["probes"]:
            ident = probe["identity"]
            bound = probe["detectors"][channels.index(d)]
            path = Path(bound["path"])
            assert binding(path)["sha256"] == bound["sha256"]
            raw = np.fromfile(path, "<f8").reshape(n, 2)
            oracle = np.column_stack(
                [signal.fftconvolve(raw[:, c], kernel, mode="same") for c in range(2)]
            )
            deltas = {
                arm: read(r / f"injection-{ident}", d, "delta.f64", "<f8").reshape(n, 2)
                for arm, r in roots.items()
            }
            all_probe[(d, ident)] = deltas
            for arm, delta in deltas.items():
                mask = masks[arm]
                err = float(np.max(abs(delta[mask] - oracle[mask])))
                # Tolerance is roundoff of small injections added to native data,
                # not a new scientific acceptance percentage.
                assert err < 1e-14, (d, ident, arm, err)
                assert np.allclose(
                    delta[masks["control"]],
                    deltas["control"][masks["control"]],
                    rtol=0,
                    atol=1e-14,
                )
            for pop, mask in populations.items():
                delta = deltas["control" if pop == "control" else "both"]
                for c in range(2):
                    rec = dict(
                        channel=d,
                        probe=ident,
                        population=pop,
                        coordinate="xr"[c],
                        measurement=metrics(delta[mask, c], raw[mask, c]),
                        maximum_frozen_operator_error=float(
                            np.max(abs(delta[mask, c] - oracle[mask, c]))
                        )
                        if mask.any()
                        else None,
                    )
                    if ident == "white-noise":
                        rec["predicted_output_rms"] = float(
                            1e-6 * np.sqrt(np.dot(kernel, kernel))
                        )
                        rec["raw_noise_seed"] = 152390 + d
                    science.append(rec)
        # Actual data PSD: the accepted 4s/Hann/50% profile and exact disjoint
        # support. Empty/short recovered islands remain unavailable, no joining.
        for pop, mask in populations.items():
            v = values["control" if pop == "control" else "both"]
            for c in range(2):
                science.append(
                    dict(
                        channel=d,
                        probe="observed-spectrum",
                        population=pop,
                        coordinate="xr"[c],
                        measurement=power_record(
                            v[::2, c],
                            mask[::2],
                            0.016384,
                            [10.380900398862574, 11.631611290291799],
                        ),
                    )
                )
    den = len(q) * len(channels)
    expected = dict(
        control=440542, **{"low-only": 556356, "high-only": 640512}, both=851646
    )
    discrepancy = {k: totals[k] - expected[k] for k in ARMS}
    summary = dict(
        denominator=den,
        direct_excluded=direct_lost,
        retained=totals,
        percentages={k: 100 * v / den for k, v in totals.items()},
        recovered=totals["both"] - totals["control"],
        remaining_support_loss=den - direct_lost - totals["both"],
        reconciliation_discrepancy=discrepancy,
        checks=checks,
        detectors=records,
        design=binding(a.prepared / "design.json"),
    )
    write_json(a.output / "accounting.json", summary)
    write_json(a.output / "science-metrics.json", science)
    source = []
    # Report full crossing diagnostics and explicitly partial retained-center
    # transfer separately; missing centers are never called recovered sources.
    for comp in design["components"]:
        ident = comp["identity"]
        for d in channels:
            xy = (
                np.fromfile(prior / f"geometry-02/pointing-{d}.f64", "<f8").reshape(
                    n, 2
                )
                * ASEC
            )
            raw = np.fromfile(a.prepared / f"{d}-{ident}.f64", "<f8").reshape(n, 2)[
                :, 0
            ]
            lowpass = signal.fftconvolve(raw, cfg["fir"], mode="same")
            for arm in ARMS:
                mask = all_masks[d][arm]
                delta = all_probe[d, ident][arm][:, 0]
                local = mask & (abs(t - comp["time"]) < 3)
                source.append(
                    dict(
                        channel=d,
                        identity=ident,
                        arm=arm,
                        complete_crossing=source_record(delta, raw, mask, t, xy, comp),
                        retained_local_center_response=metrics(
                            delta[local], raw[local]
                        ),
                        max_local_error_fraction_of_injected_peak=(
                            float(np.max(abs(delta[local] - raw[local])))
                            * apt[d, 2]
                            / 5
                            if local.any()
                            else None
                        ),
                        max_local_error_mJy=(
                            float(np.max(abs(delta[local] - raw[local]))) * apt[d, 2]
                            if local.any()
                            else None
                        ),
                        most_negative_local_fraction_of_injected_peak=(
                            float(np.min(delta[local])) * apt[d, 2] / 5
                            if local.any()
                            else None
                        ),
                        diagnostic_lowpass_only_max_error_fraction=(
                            float(np.max(abs(lowpass[local] - raw[local])))
                            * apt[d, 2]
                            / 5
                            if local.any()
                            else None
                        ),
                        diagnostic_lowpass_scope="source-only analytic decomposition on identical eligible centers; not an additional runtime treatment or use of excluded real data",
                        partial_response_is_not_source_recovery=True,
                    )
                )
    write_json(a.output / "source-response.json", source)
    # Existing naive-map adapter, limited to +/-5s around four prescribed
    # injections; twelve-detector source-only diagnostics, no PCA, no map
    # qualification or new integration. Exact speed masks supply all hits.
    tel = np.fromfile(prior / "geometry-02/telescope.f64", "<f8").reshape(n, 6)
    maps = a.output / "map-input"
    maps.mkdir()
    for comp in design["components"]:
        ident = comp["identity"]
        rows = q[abs(t[q] - comp["time"]) <= 5]
        tp = maps / f"{ident}-telescope.f64"
        tel[rows].astype("<f8").tofile(tp)
        for arm in ARMS:
            mask = np.zeros((len(rows), 491))
            raw = np.zeros_like(mask)
            filtered = np.zeros_like(mask)
            for d in channels:
                mask[:, d] = all_masks[d][arm][rows]
                raw[:, d] = (
                    np.fromfile(a.prepared / f"{d}-{ident}.f64", "<f8").reshape(n, 2)[
                        rows, 0
                    ]
                    * apt[d, 2]
                )
                filtered[:, d] = np.where(
                    mask[:, d], all_probe[d, ident][arm][rows, 0] * apt[d, 2], 0
                )
            good = maps / f"{ident}-{arm}-good.f64"
            mask.astype("<f8").tofile(good)
            for mode, data in [("reference", raw), ("filtered", filtered)]:
                name = f"{ident}-{arm}-{mode}"
                path = maps / f"{name}.f64"
                data.astype("<f8").tofile(path)
                map_jobs.append(
                    dict(
                        name=name,
                        rows=len(rows),
                        data=binding(path),
                        good=binding(good),
                        telescope=binding(tp),
                        pca=False,
                        map_supports=[dict(name="actual", good=binding(good))],
                    )
                )
    write_json(
        a.output / "map-input.json",
        dict(
            mode="downstream",
            apt=binding(prior / "geometry-02/apt.f64"),
            jobs=map_jobs,
            map_pixels=401,
            pixel_arcsec=2,
            output_hz=1 / 0.016384,
        ),
    )
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("prepared", "campaign", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    analyze(p.parse_args())


if __name__ == "__main__":
    main()
