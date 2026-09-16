#!/usr/bin/env python3
"""Read-only RTC support accounting; diagnostic masks never authorize samples.

Compare exact native phase-zero positions, LPF-only replay and the selected
mixed LPF/notch replay. Preserve overlapping original causes as bit sets;
attribute expanded losses only after removing directly ineligible centers.
No signal values, thresholds, coefficients or runtime plans are modified.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


NAMES = (
    "producer_invalid",
    "failed_screen",
    "accepted_jump_scan",
    "pending_event",
    "processing_selection",
    "AST_unavailable",
    "below_minimum_speed",
    "above_F2_sampling_speed",
    "outside_optical_domain",
    "physical_run_edge",
)


def load(path):
    return yaml.load(Path(path).read_text(), Loader=yaml.CSafeLoader)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def intervals(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(
        zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist())
    )


def range_mask(n, ranges):
    out = np.zeros(n, dtype=bool)
    for begin, end in ranges:
        assert 0 <= begin < end <= n
        out[begin:end] = True
    return out


def dilate(mask, half):
    """Any unavailable occurrence in the inclusive [q-half,q+half] footprint."""
    assert half >= 0
    prefix = np.r_[0, np.cumsum(mask, dtype=np.int64)]
    rows = np.arange(len(mask))
    return (
        prefix[np.minimum(rows + half + 1, len(mask))]
        != prefix[np.maximum(rows - half, 0)]
    )


def support_bits(masks, half, physical_runs):
    n = masks.shape[1]
    bits = np.zeros(n, dtype=np.uint16)
    for i, mask in enumerate(masks):
        for begin, end in physical_runs:
            bits[begin:end] |= dilate(mask[begin:end], half).astype(np.uint16) << i
    interior = np.zeros(n, dtype=bool)
    for begin, end in physical_runs:
        if end - begin > 2 * half:
            interior[begin + half : end - half] = True
    bits[~interior] |= 1 << (len(NAMES) - 1)
    return bits


def combinations(bits):
    return {
        " + ".join(name for i, name in enumerate(NAMES) if value & (1 << i)): int(count)
        for value, count in sorted(Counter(map(int, bits)).items())
        if value
    }


def counts(bits):
    return {
        "union": int(np.count_nonzero(bits)),
        "shared_multiple_causes": int(
            sum(
                count
                for v, count in Counter(map(int, bits)).items()
                if v.bit_count() > 1
            )
        ),
        "by_cause": {
            name: {
                "inclusive": int(np.count_nonzero(bits & (1 << i))),
                "only_this_cause": int(np.count_nonzero(bits == (1 << i))),
            }
            for i, name in enumerate(NAMES)
        },
        "disjoint_combinations": combinations(bits),
    }


def verify_seal(root):
    count = 0
    for line in (root / "EVIDENCE_SHA256SUMS").read_text().splitlines():
        expected, rel = line.split(maxsplit=1)
        assert digest(root / rel) == expected, rel
        count += 1
    return {
        "root": str(root),
        "manifest_sha256": digest(root / "EVIDENCE_SHA256SUMS"),
        "files_verified": count,
    }


def donor_accounting(previous, output):
    """Separate numerical support, representative exclusion and consumer policy."""
    channel = 269

    def read(where):
        root = previous / where
        state = np.fromfile(root / f"{channel}-state.u8", "u1")
        rows = np.fromfile(root / f"{channel}-rows.i64", "<i8")
        values = np.fromfile(root / f"{channel}-filtered.f64", "<f8").reshape(-1, 2)
        scheduled = np.zeros(len(state), bool)
        scheduled[rows] = True
        return state, scheduled, values

    ds, dq, dv = read("injected-d92cd3a51/donor-continuity")
    cs, cq, cv = read("injected-d92cd3a51/exclusion-control")
    rs, rq, rv = read("natural-d92cd3a51/exclusion-control")
    x_available = dq & ((ds & 1) != 0)
    ref_available = rq & ((rs & 1) != 0)
    extra = x_available & ~(cq & ((cs & 1) != 0))
    replaced = extra & ((ds & 4) != 0)
    neighbors = extra & ((ds & 16) != 0) & ~replaced
    no_influence = extra & ((ds & 16) == 0)
    assert np.all(ref_available[extra])
    assert not np.any(ds[neighbors] & 8), (
        "nonrepresentative influence is not universal exclusion"
    )
    assert not np.any(ds[neighbors] & 2), "record the actual r unavailability"
    assert not np.any(ds[extra] & 32), "no failed reconstruction influence"
    receipt = load(previous / "injected-d92cd3a51/donor-continuity/apply-receipt.yaml")
    plan = next(d for d in receipt["realized_detectors"] if d["channel"] == channel)
    assert len(plan["donors"]) == 1 and plan["donors"][0]["cause"] == 0
    assert plan["donors"][0]["included_in_arm"]
    summary = {
        "channel": channel,
        "native_replacement": [19340, 19341],
        "extra_finite_x": int(extra.sum()),
        "direct_representative_exclusion": int(replaced.sum()),
        "neighbor_original_representatives": int(neighbors.sum()),
        "additional_paired_available_uninfluenced": int(no_influence.sum()),
        "neighbor_x_available": int(np.count_nonzero(ds[neighbors] & 1)),
        "neighbor_r_available": int(np.count_nonzero(ds[neighbors] & 2)),
        "neighbor_universal_exclusions": int(np.count_nonzero(ds[neighbors] & 8)),
        "reconstruction_quality_disposition": "existing donor ready; no failed-reconstruction cause; no new numerical acceptance tolerance",
        "consumer_eligibility": "unselected, not automatically excluded; preserve influence and response dependencies",
        "SCI_RTC_revision": "v0.1/r0.12 frozen 2026-08-21; unchanged normative core on WP-7.1 closure ancestry",
    }
    for name, mask in (
        ("neighbors", neighbors),
        ("direct_replacement", replaced),
        ("extra_uninfluenced", no_influence),
    ):
        err = dv[mask, 0] - rv[mask, 0]
        summary[name] = dict(
            rows=np.flatnonzero(mask).tolist(),
            count=int(mask.sum()),
            residual_RMS_x=float(np.sqrt(np.mean(err * err))),
            residual_max_abs_x=float(np.max(np.abs(err))),
            residual_mean_x=float(err.mean()),
            reference_std_x=float(np.std(rv[mask, 0])),
            universally_excluded=int(np.count_nonzero(ds[mask] & 8)),
        )
    same = cq & ((cs & 1) != 0) & ref_available
    assert np.array_equal(cv[same, 0], rv[same, 0])
    from matplotlib import pyplot as plt

    t = (np.arange(len(ds)) - 19340) * 0.008192
    span = np.abs(t) <= 4
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    ax[0].plot(t[span], rv[span, 0], color="0.6", label="Untouched filtered reference")
    ax[0].plot(t[span], dv[span, 0], color="#1b9e77", lw=0.8, label="Donor x")
    ax[0].plot(
        t[span], cv[span, 0], color="#7570b3", lw=0.8, label="Exclusion control x"
    )
    ax[0].set_ylabel("Original x units")
    ax[0].legend(fontsize=9)
    q = np.flatnonzero(neighbors)
    ax[1].plot(t[q], dv[q, 0] - rv[q, 0], ".", ms=3, color="#1b9e77")
    ax[1].axhline(0, color="0.5", lw=0.5)
    ax[1].set_ylabel("Neighbor x residual")
    ax[1].set_title(
        "334 original representatives: numerical error shown separately from availability"
    )
    for level, mask, color, label in (
        (3, extra, "#1b9e77", "Extra finite x: 337"),
        (2, neighbors, "#d95f02", "r unavailable, original x representative: 334"),
        (1, replaced, "red", "Direct replaced representative excluded: 1"),
        (0, no_influence, "#7570b3", "Additional paired available: 2"),
    ):
        q = np.flatnonzero(mask)
        ax[2].scatter(t[q], np.full(len(q), level), s=12, color=color, label=label)
    ax[2].set_yticks([])
    ax[2].set_xlabel("Seconds from declared single-occurrence contaminant")
    ax[2].legend(fontsize=8, loc="upper right")
    fig.suptitle(
        "Controlled donor test: x error, r availability and direct exclusion are distinct"
    )
    fig.savefig(output / "donor-neighbor-accounting.png", dpi=150)
    plt.close(fig)
    (output / "donor-neighbor-accounting.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predecessor", type=Path, required=True)
    p.add_argument("--geometry-evidence", type=Path, required=True)
    p.add_argument("--lowpass-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(exist_ok=False)
    previous = args.predecessor
    natural = previous / "natural-d92cd3a51"
    seals = [verify_seal(previous), verify_seal(args.geometry_evidence)]
    cfg = load(previous / "input.json")
    lowcfg = load(args.lowpass_run.parent / "lowpass-input.json")
    normalized = json.loads(json.dumps(lowcfg))
    for old, new in zip(cfg["detectors"], normalized["detectors"], strict=True):
        assert new["filter"] == "lowpass"
        new["filter"] = old["filter"]
    assert normalized == cfg, "only the diagnostic notch omission may differ"
    original = load(natural / "receipt.yaml")
    lowreceipt = load(args.lowpass_run / "receipt.yaml")
    for key in (
        "source_revision",
        "VAL_generation",
        "original_parent",
        "candidate_count",
        "event_count",
        "admitted_jump_groups",
        "native_time_sha256",
        "native_runs",
    ):
        assert original[key] == lowreceipt[key], key
    for f in (
        "events.yaml",
        "event-decisions.yaml",
        "support-decisions.yaml",
        "processing-scans.yaml",
        "original-psd.f64",
        "native-time.f64",
    ):
        assert digest(natural / f) == digest(args.lowpass_run / f), f
    assert original["original_pair_unchanged"] and lowreceipt["original_pair_unchanged"]
    geometry_root = args.geometry_evidence / "campaign-01"
    geomcfg = load(geometry_root / "269.json")
    for key in (
        "raw",
        "tune",
        "manifest",
        "telescope",
        "effective_config",
        "ast_acceptance",
    ):
        assert geomcfg[key] == cfg[key], key
    geometry = np.fromfile(geometry_root / "269/geometry.f64", "<f8").reshape(-1, 4)
    time = np.fromfile(natural / "native-time.f64", "<f8")
    assert np.array_equal(geometry[:, 0], time)
    n = len(time)
    speed = geometry[:, 1]
    physical = original["native_runs"]
    current = load(natural / "exclusion-control/apply-receipt.yaml")[
        "realized_detectors"
    ]
    lowplans = load(args.lowpass_run / "exclusion-control/apply-receipt.yaml")[
        "realized_detectors"
    ]
    scans = load(natural / "processing-scans.yaml")
    decisions = load(natural / "event-decisions.yaml")
    schedule = np.arange(n) % 2 == 0
    lp_half = len(cfg["fir"]) // 2
    notch_half = len(cfg["finite_notch"]) // 2
    stages = {name: [] for name in ("direct", "lowpass", "current")}
    expanded = {name: [] for name in ("lowpass", "current")}
    channel_results, checks, all_runs, cause_intervals = [], [], [], []
    diagnostic_chunk_delta = 0
    views = {}
    for entry, plan, lpplan in zip(cfg["detectors"], current, lowplans, strict=True):
        ch = entry["channel"]
        assert ch == plan["channel"] == lpplan["channel"]
        assert plan["admitted_runs"] == lpplan["admitted_runs"]
        assert (
            plan["sampling_speed_limit_arcsec_per_sec"]
            == lpplan["sampling_speed_limit_arcsec_per_sec"]
        )
        assert plan["lowpass_FIR"] == lpplan["lowpass_FIR"] == cfg["fir"]
        assert (plan["finite_notch_FIR"] or []) == (
            cfg["finite_notch"] if entry["filter"] == "w1-t3" else []
        )
        raw = np.fromfile(entry["samples"]["path"], "<f8").reshape(n, 4)
        assert digest(entry["samples"]["path"]) == entry["samples"]["sha256"]
        oldcauses = np.fromfile(geometry_root / f"{ch}/lowpass-causes.u8", "u1")
        oldcfg = load(geometry_root / f"{ch}.json")
        assert oldcfg["samples"] == entry["samples"]
        # Old sealed single-detector run had screening only (no scan binding).
        # Validate the screening population and union against today's plan.
        assert oldcfg["prior_population_record"]["scan_binding"] is None
        screen = oldcauses == 2
        assert intervals(screen) == [(151368, n)]
        jumps = []
        for event in decisions:
            if event["channel"] == ch and event["class"] == "admitted-level-shift":
                for scan in scans["scans"]:
                    for a, b in scan["science_native"]:
                        if a < event["affected"][1] and event["affected"][0] < b:
                            jumps.append((a, b))
        pending = [
            e["operation_unavailable"]
            for e in decisions
            if e["channel"] == ch
            and e["operation_unavailable"][1] > e["operation_unavailable"][0]
        ]
        masks = np.array(
            [
                ~((raw[:, 2] == 1) & (raw[:, 3] == 1)),
                screen,
                range_mask(n, jumps),
                range_mask(n, pending),
                range_mask(n, scans["native_outside_processing"]),
                ~np.isfinite(speed),
                np.isfinite(speed) & (speed < 1.0),
                np.isfinite(speed)
                & (speed > plan["sampling_speed_limit_arcsec_per_sec"]),
                np.isfinite(speed) & (speed * 1.05 > 235.0),
            ]
        )
        direct = support_bits(masks, 0, physical)
        # Reproduce exact runtime first-cause precedence from independent masks.
        expected = np.zeros(n, dtype=np.uint8)
        for mask, cause in (
            (masks[0], 1),
            (masks[1] | masks[2], 2),
            (masks[4], 11),
            (masks[3], 9),
            (masks[5], 3),
            (masks[6], 7),
            (masks[7], 8),
            (masks[8], 4),
        ):
            expected[(expected == 0) & mask] = cause
        inp = np.fromfile(natural / f"exclusion-control/{ch}-input-causes.u8", "u1")
        assert np.array_equal(expected, inp)
        assert np.array_equal(
            inp,
            np.fromfile(
                args.lowpass_run / f"exclusion-control/{ch}-input-causes.u8", "u1"
            ),
        )
        assert intervals(direct == 0) == list(map(tuple, plan["admitted_runs"]))
        h = lp_half + (notch_half if entry["filter"] == "w1-t3" else 0)
        lp = support_bits(masks, lp_half, physical)
        chain = support_bits(masks, h, physical)
        for label, bits, root, half in (
            ("lowpass", lp, args.lowpass_run, lp_half),
            ("current", chain, natural, h),
        ):
            state = np.fromfile(root / f"exclusion-control/{ch}-state.u8", "u1")
            rows = np.fromfile(root / f"exclusion-control/{ch}-rows.i64", "<i8")
            assert np.array_equal(bits == 0, (state & 3) == 3), (ch, label)
            assert np.array_equal(np.flatnonzero((bits == 0) & schedule), rows)
            assert np.array_equal(
                state, np.fromfile(root / f"donor-continuity/{ch}-state.u8", "u1")
            )
            stages[label].append(bits[schedule])
            expanded[label].append(bits[schedule & (direct == 0)])
            for a, b in plan["admitted_runs"]:
                first = a + half
                first += first % 2
                predicted = max(0, (b - half - 1 - first) // 2 + 1)
                actual = int(np.count_nonzero((rows >= a) & (rows < b)))
                assert predicted == actual
                all_runs.append(
                    dict(
                        channel=ch,
                        stage=label,
                        first=a,
                        past_last=b,
                        length=b - a,
                        half_support=half,
                        predicted=predicted,
                        measured=actual,
                    )
                )
        stages["direct"].append(direct[schedule])
        # Masks-only diagnostic: do NOT execute a filter on newly included data.
        without_chunk = masks.copy()
        without_chunk[4] = False
        noc = support_bits(without_chunk, h, physical)
        delta = int(np.count_nonzero(schedule & (noc == 0) & (chain != 0)))
        diagnostic_chunk_delta += delta
        result = dict(
            channel=ch,
            selected_filter=entry["filter"],
            admitted_runs=len(plan["admitted_runs"]),
            scheduled=int(schedule.sum()),
            direct_eligible=int(np.count_nonzero(schedule & (direct == 0))),
            lowpass_available=int(np.count_nonzero(schedule & (lp == 0))),
            current_available=int(np.count_nonzero(schedule & (chain == 0))),
            processing_selection_only_expanded_loss=delta,
            diagnostic_processing_only_positions=np.flatnonzero(
                schedule & (noc == 0) & (chain != 0)
            ).tolist(),
            raw_direct_causes=counts(direct),
            scheduled_direct_causes=counts(direct[schedule]),
            speed_limit=plan["sampling_speed_limit_arcsec_per_sec"],
        )
        channel_results.append(result)
        for i, name in enumerate(NAMES[:-1]):
            for a, b in intervals(masks[i]):
                cause_intervals.append(
                    dict(
                        channel=ch,
                        cause=name,
                        first=a,
                        past_last=b,
                        first_unix_seconds=float(time[a]),
                        last_unix_seconds=float(time[b - 1]),
                    )
                )
        views[ch] = dict(
            masks=masks,
            direct=direct,
            lowpass=lp,
            current=chain,
            half=h,
            runs=plan["admitted_runs"],
        )
    aggregate = {
        name: counts(np.concatenate(values)) for name, values in stages.items()
    }
    expanded_counts = {
        name: counts(np.concatenate(values)) for name, values in expanded.items()
    }
    denom = int(schedule.sum()) * len(channel_results)
    summary = dict(
        schema="rtc-retention-audit-v1",
        pair_counted=True,
        nominal_cadence_seconds=original["native_integration_seconds"],
        source_revision=original["source_revision"],
        denominator_scheduled=denom,
        learning_bindings={
            "current": original["learning_binding"],
            "lowpass": lowreceipt["learning_binding"],
            "note": "Distinct exact configuration identities; original evidence arrays verified equal.",
        },
        denominator_native=n * len(channel_results),
        retained={name: denom - v["union"] for name, v in aggregate.items()},
        scheduled_loss=aggregate,
        expanded_loss_on_directly_eligible=expanded_counts,
        channels=channel_results,
        physical_runs=physical,
        lowpass_half_samples=lp_half,
        notch_half_samples=notch_half,
        masks_only_processing_selection_delta=diagnostic_chunk_delta,
        original_evidence_seals=seals,
        raw_speed_counts=dict(
            unavailable=int(np.count_nonzero(~np.isfinite(speed))),
            below_1=int(np.count_nonzero(speed < 1)),
            above_fixed_F2_limit=int(
                np.count_nonzero(
                    speed > current[0]["sampling_speed_limit_arcsec_per_sec"]
                )
            ),
        ),
    )
    assert summary["retained"]["current"] == 440542
    # Representative intervals selected by cause/domain, not by detector signal.
    examples = [
        (269, 19000, 22000, "Processing boundaries crossed without reset"),
        (330, 900, 2400, "Short eligible run and internal processing boundary"),
        (269, 2600, 3800, "Candidate guard overlaps motion expansion"),
        (402, 106800, 109150, "Full processing-scan jump exclusion"),
        (269, 150500, 151537, "Processing selection at observation end"),
    ]
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(examples), 1, figsize=(13, 15), constrained_layout=True
    )
    for ax, (ch, begin, end, title) in zip(axes, examples, strict=True):
        view = views[ch]
        x = time[begin:end] - time[0]
        ax.plot(x, speed[begin:end], color="0.35", lw=0.9, label="AST speed")
        ax.axhline(
            current[0]["sampling_speed_limit_arcsec_per_sec"],
            color="firebrick",
            ls=":",
            label="Fixed upper speed",
        )
        ax.set_ylabel("Speed (arcsec/s)")
        ax.set_ylim(-55, 235)
        for level, label, color in (
            (-12, "direct", "#238443"),
            (-27, "lowpass", "#3182bd"),
            (-42, "current", "#756bb1"),
        ):
            valid = view[label][begin:end] == 0
            ax.plot(
                x,
                np.where(valid, level, np.nan),
                color=color,
                lw=5,
                solid_capstyle="butt",
                label=label + " eligible support",
            )
        for scan in scans["scans"]:
            boundary = scan["science_native"][0][0]
            if begin < boundary < end:
                ax.axvline(time[boundary] - time[0], color="0.5", alpha=0.5, ls="--")
        for i, mask in enumerate(view["masks"]):
            for a, b in intervals(mask[begin:end]):
                if i in (2, 3, 4):
                    ax.axvspan(
                        x[a],
                        x[min(b, len(x) - 1)],
                        alpha=0.18,
                        color={2: "red", 3: "orange", 4: "black"}[i],
                    )
        ax.set_title(f"Channel {ch}: {title} — native [{begin},{end})")
        ax.set_xlabel(
            "Seconds from first native sample; dashed lines = processing-scan starts"
        )
        checks.append(
            dict(
                channel=ch,
                first=begin,
                past_last=end,
                title=title,
                lowpass_footprint=f"[q-{lp_half},q+{lp_half}] inclusive",
                current_footprint=f"[q-{view['half']},q+{view['half']}] inclusive",
                cause_intervals=[
                    v
                    for v in cause_intervals
                    if v["channel"] == ch
                    and v["first"] < end
                    and v["past_last"] > begin
                ],
                admitted_run_checks=[
                    v
                    for v in all_runs
                    if v["channel"] == ch
                    and v["first"] < end
                    and v["past_last"] > begin
                ],
            )
        )
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    fig.suptitle(
        "RTC support audit: identical original masks and timing; no excluded data used"
    )
    fig.savefig(args.output / "representative-intervals.png", dpi=150)
    plt.close(fig)
    summary["controlled_donor"] = donor_accounting(previous, args.output)
    for name, value in (
        ("summary", summary),
        ("representative-intervals", checks),
        ("all-run-footprint-checks", all_runs),
        ("raw-cause-intervals", cause_intervals),
    ):
        (args.output / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
    print(
        json.dumps(
            {
                "retained": summary["retained"],
                "denominator": denom,
                "processing_selection_delta": diagnostic_chunk_delta,
                "run_footprint_checks": len(all_runs),
            }
        )
    )


if __name__ == "__main__":
    main()
