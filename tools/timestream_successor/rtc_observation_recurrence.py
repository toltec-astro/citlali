"""Review recurrence in the sealed RTC census; never select or apply a cut.

Counts retain producer-group identity. Elapsed-time quarters describe where
measurements occurred; they are neither PCA scans nor independent events.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np

from rtc_disturbance_burden import (
    compressed_writer, digest, emit, intersect, measure, records, require,
    union, write_json,
)

CENSUS_SEAL = "e8156f63a96b21f045e571ff784d1c246994acd83f8e1cf0647e4c4ef33b95b4"
CENSUS_SOURCE = "5973de4eb696ad211be4eceeeabcc20e25435b6f"
BASE = "4eb5de86ce3f4fe071053e0c2381bf77fdc14e1c"
COUNTS = (2, 3, 5, 10)
RATES = (0.25, 0.5, 1.0, 2.0)
QUARTERS = (2, 3, 4)
MINUTE_US = 60_000_000


def key(row):
    return row["observation"], row["network"], row["detector"]


def verify_census(root):
    require(digest(root / "SHA256SUMS") == CENSUS_SEAL, "unrecognized census seal")
    count = 0
    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()), "manifest path outside census")
        require(digest(path) == expected, f"census changed: {name}")
        count += 1
    binding = json.loads((root / "run-binding.json").read_text())
    require(binding["candidate"] == CENSUS_SOURCE and count == 158,
            "wrong census generation")
    return count


def retained_markers(events):
    """One marker per original group, even for two retained coordinates."""
    result = defaultdict(list)
    seen = set()
    total = 0
    for e in events:
        require(e["apply_authorized"] is False and e["hard_event_accepted"] is False,
                "unexpected admitted event")
        identity = (*key(e), e["producer_group"])
        require(identity not in seen, "duplicate original group")
        seen.add(identity)
        starts = [a for c in e["coordinates"] for a, _ in c["transition_us"]]
        if starts:
            result[key(e)].append((min(starts), e["producer_group"]))
        total += 1
    return result, total


def describe(row, markers):
    exposure = row["durations_us"]["eligible_us"]
    support = union(row["eligible_intervals_us"])
    require(measure(support) == exposure, "eligible support mismatch")
    require(len(markers) == row["counts"].get("transition_groups", 0),
            "retained group reconciliation failed")
    require(exposure > 0 or not markers, "retained evidence without eligibility")
    n = len(markers)
    out = {k: row[k] for k in ("observation", "network", "detector", "occurrence",
                              "acquisition_scope", "array", "program_metadata")}
    out.update(eligible_us=exposure, counts=row["counts"],
               direct_us=row["durations_us"]["direct_us"],
               noise_screening_required_us=row["durations_us"]["noise_screening_required_us"],
               health_review_concern=row["health_review_concern"],
               transition_groups=n, group_markers_us=sorted(markers),
               groups_per_eligible_minute=None, elapsed_span_us=None,
               acquisition_or_eligibility_gap_us=None, occupied_quarters=None,
               quarter_group_counts=None, quarter_eligible_us=None,
               first_to_last_elapsed_fraction=None, marker_elapsed_fractions=[],
               full_scan_loss_us=None, additional_beyond_full_scan_us=None,
               runtime_apply=False)
    require(0 <= out["direct_us"] <= exposure, "direct support exceeds exposure")
    if not exposure:
        return out
    start, end = support[0][0], support[-1][1]
    span = end - start
    # Integer boundaries partition the same half-open elapsed span, including gaps.
    boundaries = [start + (i * span + 3) // 4 for i in range(5)]
    qc = [0] * 4
    for t, _ in markers:
        require(any(a <= t < b for a, b in support), "marker outside eligible support")
        qc[min(3, 4 * (t - start) // span)] += 1
    times = [t for t, _ in markers]
    out.update(groups_per_eligible_minute=n * MINUTE_US / exposure,
               elapsed_span_us=span, acquisition_or_eligibility_gap_us=span - exposure,
               occupied_quarters=sum(x > 0 for x in qc), quarter_group_counts=qc,
               quarter_eligible_us=[measure(intersect(support, [(a, b)])) if a < b else 0
                                    for a, b in zip(boundaries, boundaries[1:])],
               first_to_last_elapsed_fraction=(max(times) - min(times)) / span if times else None,
               marker_elapsed_fractions=[(t - start) / span for t in sorted(times)])
    require(sum(out["quarter_eligible_us"]) == exposure, "quarter exposure mismatch")
    return out


def cost(rows, population_us):
    total = sum(r["eligible_us"] for r in rows)
    direct = sum(r["direct_us"] for r in rows)
    return dict(detector_records=len(rows), full_observation_us=total,
                full_observation_percent=100 * total / population_us if population_us else None,
                available_direct_us=direct, additional_beyond_direct_us=total - direct,
                full_scan_loss_us=None, additional_beyond_full_scan_us=None)


def summarize(rows):
    eligible = [r for r in rows if r["eligible_us"]]
    exposure = sum(r["eligible_us"] for r in eligible)
    count_table = []
    for category in ("transition_groups", "finite_recovery_groups", "unresolved_groups"):
        for n in COUNTS:
            chosen = [r for r in eligible if r["counts"].get(category, 0) >= n]
            count_table.append(dict(category=category, minimum_groups=n, **cost(chosen, exposure)))
    sensitivity = []
    for n in COUNTS:
        for rate in RATES:
            for q in QUARTERS:
                chosen = [r for r in eligible if r["transition_groups"] >= n
                          and r["groups_per_eligible_minute"] >= rate
                          and r["occupied_quarters"] >= q]
                sensitivity.append(dict(minimum_groups=n, minimum_groups_per_minute=rate,
                                        minimum_occupied_quarters=q, **cost(chosen, exposure)))
    return dict(detector_records=len(rows), eligible_detector_records=len(eligible),
                eligible_us=exposure, transition_groups=sum(r["transition_groups"] for r in rows),
                count_sensitivity=count_table, joint_sensitivity=sensitivity,
                health_review=cost([r for r in eligible if r["health_review_concern"]], exposure),
                transition_quarter_histogram=dict(sorted(Counter(
                    r["occupied_quarters"] for r in eligible if r["transition_groups"] >= 3).items())))


def audit_scan_inputs(metadata_root, observations):
    """Read exact existing V2 timing metadata. Never reconstruct alignment."""
    import yaml
    from netCDF4 import Dataset

    entries = []
    for obs in sorted(observations):
        if obs in (152385, 152386, 152387):
            run = metadata_root / "oof/refactor/1146+399/reduced/redu00"
        elif obs in (152390, 152392):
            run = metadata_root / "science/refactor/NGC4449/reduced/redu03"
        else:
            run = metadata_root / "science/pointings/reduced/redu00"
        folder = run / str(obs)
        files = []
        raw = folder / "raw_timestream_provenance.yaml"
        require(raw.is_file(), f"missing known provenance {raw}")
        with raw.open() as stream:
            doc = yaml.safe_load(stream)
        native = doc.get("realized", {}).get("native_cohort_provenance", {})
        require(native.get("available") is False,
                "native provenance state changed: reassess scan binding")
        for name in ("raw_timestream_provenance.yaml", "timestream_output_provenance.yaml"):
            p = folder / name
            require(p.is_file(), f"missing known metadata {p}")
            files.append(dict(path=str(p), sha256=digest(p)))
        products = []
        for p in sorted((folder / "raw").glob("*.nc")):
            if not p.name.endswith(("_rtc_timestream.nc", "_ptc_timestream.nc",
                                    "_rtcdiag.nc", "_ptcdiag.nc")):
                continue
            h = digest(p)
            with Dataset(p) as nc:
                selected = {}
                for name in ("obsnum", "VERSION", "scan_indices", "raw_scan_indices",
                             "output_scan_index", "scan_duration_s", "SAMPRATE", "RTC_SAMPRATE"):
                    if name not in nc.variables:
                        continue
                    v = nc[name]
                    values = v[...]
                    selected[name] = dict(values=values.tolist(),
                                          attributes={a: str(v.getncattr(a)) for a in v.ncattrs()})
            products.append(dict(path=str(p), sha256=h, timing_metadata=selected))
        entries.append(dict(observation=obs, inspected_run=str(run), files=files,
                            software_identity=doc.get("software_identity"),
                            canonical_run_identity=doc.get("canonical_run_identity"),
                            native_cohort_provenance=native, products=products,
                            native_to_pca_scan_relation=None,
                            cause="native cohort relation not published; output indices are not native indices"))
    logs = []
    for relative in ("oof/refactor/1146+399/log/reduce.log",
                     "science/refactor/NGC4449/log/reduce.log"):
        p = metadata_root / relative
        lines = p.read_text().splitlines()
        examples = []
        for i, line in enumerate(lines):
            if "scan_indices (4," in line:
                examples.append(dict(line=i + 1, text=lines[i:i + 5]))
                if len(examples) == 2:
                    break
        logs.append(dict(path=str(p), sha256=digest(p), examples=examples,
                         disposition="truncated matrix examples; not an exact native scan relation"))
    return dict(observations=entries, logs=logs, full_scan_cost_available=False,
                missing_binding=["exact native input/axis and acquisition-run identities",
                                 "exact requested processing generation and complete existing PCA scan support",
                                 "per-network native-to-processing-axis relation, validity and timing uncertainty"],
                scientific_decision_missing=False,
                policy="reuse every intersected existing scan; do not redefine scans")


def plots(output, rows, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    retained = [r for r in rows if r["transition_groups"] >= 3]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for goal, color in (("Oof", "#a56c24"), ("Pointing", "#367caf"), ("Science", "#c34755")):
        rr = [r for r in retained if r["program_metadata"]["goal"] == goal]
        axes[0].scatter([r["transition_groups"] for r in rr],
                        [r["groups_per_eligible_minute"] for r in rr],
                        s=16, alpha=.5, label=goal, color=color)
    axes[0].set(xscale="log", yscale="log", xlabel="Retained original groups (at least 3)",
                ylabel="Groups per eligible detector-minute", title="Equal counts can mean very different rates")
    axes[0].legend()
    grid = np.array([[next(s["full_observation_percent"] for s in summary["joint_sensitivity"]
                          if s["minimum_groups"] == 3 and s["minimum_groups_per_minute"] == rate
                          and s["minimum_occupied_quarters"] == q) for q in QUARTERS] for rate in RATES])
    im = axes[1].imshow(grid, cmap="Blues", aspect="auto", vmin=0)
    axes[1].set(xticks=range(3), xticklabels=QUARTERS, yticks=range(4), yticklabels=RATES,
                xlabel="Minimum occupied elapsed-time quarters",
                ylabel="Minimum groups per eligible minute",
                title="Whole-observation cost: at least 3 groups")
    for i in range(4):
        for j in range(3):
            axes[1].text(j, i, f"{grid[i,j]:.3f}%", ha="center", va="center",
                         color="white" if grid[i,j] > .55 * grid.max() else "black")
    fig.colorbar(im, ax=axes[1], label="Percent of all initial detector-time")
    fig.suptitle("Recurrence sensitivity — no exclusion policy selected")
    fig.savefig(output / "recurrence-cost.png", dpi=160)
    plt.close(fig)

    # Fixed deterministic examples of concentrated and distributed evidence.
    selected = []
    for q in (1, 2, 3, 4):
        pool = sorted((r for r in retained if r["occupied_quarters"] == q),
                      key=lambda r: (-r["groups_per_eligible_minute"], key(r)))
        selected.extend(pool[:2])
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for i, r in enumerate(selected):
        ax.plot(r["marker_elapsed_fractions"], [i] * r["transition_groups"], "|", ms=15, color="#296a87")
    ax.set(yticks=range(len(selected)), yticklabels=[
        f"{r['observation']} / nw {r['network']} / ch {r['detector']}  |  "
        f"{r['transition_groups']} groups, {r['eligible_us']/MINUTE_US:.2f} min"
        for r in selected], xlim=(-.02, 1.02), xlabel="Fraction of elapsed native observation span",
        title="Examples by time spread: each mark is one original retained group")
    for v in (.25, .5, .75):
        ax.axvline(v, color=".75", lw=1, ls="--")
    ax.invert_yaxis()
    fig.savefig(output / "recurrence-examples.png", dpi=160)
    plt.close(fig)
    write_json(output / "example-selection.json", [dict(key=list(key(r)),
               occupied_quarters=r["occupied_quarters"]) for r in selected])


def report(output, s, scan):
    lines = ["# RTC observation recurrence assessment", "",
             "Exploratory census accounting; no detector exclusion or runtime policy selected.", "",
             f"Same {s['eligible_detector_records']:,} initially eligible detector records in 13 observations; "
             f"{s['eligible_us']/3.6e9:,.3f} detector-hours. Each detector is counted anew in each observation.", "",
             "## Count-only sensitivity", "",
             "| Retained groups at least | Detector records | Whole-observation cost |",
             "| --- | ---: | ---: |"]
    for x in s["count_sensitivity"]:
        if x["category"] == "transition_groups":
            lines.append(f"| {x['minimum_groups']} | {x['detector_records']:,} | {x['full_observation_percent']:.4f}% |")
    lines += ["", "![Rate and exposure sensitivity](recurrence-cost.png)", "",
              "![Selected temporal patterns](recurrence-examples.png)", "",
              "Counts preserve original producer groups and count x/r together once. They are conditional "
              "measurement evidence, not independently accepted physical jumps. No regrouping or additional "
              "event-finder pass was performed. Source protection and spectral context remain unavailable.", "",
              "Rates divide by initial eligible detector-time. Four equal elapsed-time quarters describe "
              "spread; a group is placed by its earliest retained-bound start. These bins are not PCA scans, "
              "physical event partitions, or independent-trial counts. Gaps contribute no exposure. "
              "Per-record quarter exposures and counts are exported to make gaps and duration visible.", "",
              "The grids (counts 2/3/5/10; rates 0.25/0.5/1/2 per minute; occupied quarters 2/3/4) "
              "are exploratory sensitivity settings, not recommended or selected scientific thresholds. "
              "Costs always use the fixed initial paired producer/VAL denominator, not successful-fit exposure.", "",
              "## Other concerns remain distinct", "",
              f"Existing health review concerns: {s['health_review']['detector_records']} records, "
              f"{s['health_review']['full_observation_percent']:.5f}% of initial detector-time. "
              "The approved health rule remains review-only. Recovery and unresolved count sensitivities "
              "are in summary.json, separate from retained jumps. Unresolved groups are not confirmed jumps. "
              "An observation-scoped use exclusion would not rewrite Tune/APT or label hardware permanently bad.", "",
              "## Existing-scan prerequisite", "",
              "Full-scan exclusion and additional loss beyond that treatment remain **unavailable**. "
              "All 13 inspected observation provenance records declare native-cohort provenance unavailable. "
              "The selected V2 pointing files publish output-timebase scan indices; OOF diagnostic files publish "
              "scan identifiers/durations. The inspected NGC4449 generation has no RTC/PTC timestream or "
              "diagnostic NetCDF files. Saved reduction-log matrices contain ellipses and do not supply "
              "the complete per-network relation. No scan boundaries or offsets are interpolated from logs.", "",
              "The missing prerequisite is an exact binding, not a new scientific scan-selection decision: "
              + "; ".join(scan["missing_binding"]) + ". The settled rule still selects every existing scan "
              "intersected by an accepted conservative transition bound. A metadata export from the owning "
              "alignment/scan stage is the narrow way to supply it; no new scan definition is needed.", "",
              "`additional_beyond_direct_us` measures additional exposure beyond available direct supports, "
              "not beyond scan-by-scan exclusions. All actual scan-cost fields remain null.", "",
              "## Reproduction and boundaries", "",
              "run-binding.json records exact source, environment, command and input seal. "
              "summary.json, by-program.json and by-observation.json contain the complete sensitivity grids; "
              "detector-recurrence.jsonl.gz contains exact occurrence references and per-record descriptors. "
              "scan-binding-audit.json binds the inspected metadata/files. SHA256SUMS seals this output.", "",
              "This engineering consumer reads preserved runtime Learn evidence and Consider dispositions. "
              "It does not implement runtime Learn, select a runtime Consider plan or execute Apply. "
              "No map products, spectral measurements, new observations, science correction, flags, "
              "filtering, Unity operation or GitHub push occurs.", ""]
    (output / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    repo = Path(__file__).resolve().parents[2]
    git = lambda *a: subprocess.check_output(["git", "-C", str(repo), *a], text=True).strip()
    require(not git("status", "--porcelain"), "run requires clean exact source")
    require(not args.output.exists(), "output must be new")
    verified = verify_census(args.census)
    markers, group_total = retained_markers(records(args.census / "evidence-intervals.jsonl.gz"))
    rows, seen, occurrences = [], set(), set()
    for d in records(args.census / "detector-accounting.jsonl.gz"):
        k = key(d)
        require(k not in seen and d["occurrence"] not in occurrences, "duplicate detector occurrence")
        require(d["acquisition_scope"][0] == d["observation"], "observation identity mismatch")
        seen.add(k)
        occurrences.add(d["occurrence"])
        rows.append(describe(d, markers.get(k, [])))
    require(set(markers) <= seen, "unmatched event detector")
    summary = summarize(rows)
    original = json.loads((args.census / "summary.json").read_text())["totals"]
    require(len(rows) == original["detector_streams"] == 71734, "detector population changed")
    require(summary["eligible_us"] == original["eligible_us"] == 17216191725568, "denominator changed")
    require(summary["eligible_detector_records"] == original["eligible_detector_streams"] == 70024,
            "eligibility population changed")
    require(summary["transition_groups"] == original["transition_groups"] == 8389
            and group_total == original["candidate_groups"] == 242014, "event population changed")
    scan = audit_scan_inputs(args.metadata_root, {r["observation"] for r in rows})
    args.output.mkdir(parents=True)
    with compressed_writer(args.output / "detector-recurrence.jsonl.gz") as stream:
        for r in rows:
            emit(stream, r)
    write_json(args.output / "summary.json", summary)
    for filename, get in (("by-program.json", lambda r: r["program_metadata"]["goal"]),
                          ("by-observation.json", lambda r: str(r["observation"]))):
        groups = defaultdict(list)
        for r in rows:
            groups[get(r)].append(r)
        summaries = {k: summarize(v) for k, v in sorted(groups.items())}
        require(sum(v["eligible_us"] for v in summaries.values()) == summary["eligible_us"],
                "hierarchical exposure mismatch")
        write_json(args.output / filename, summaries)
    write_json(args.output / "scan-binding-audit.json", scan)
    plots(args.output, rows, summary)
    report(args.output, summary, scan)
    # Recheck the sealed generation after consuming it.
    verify_census(args.census)
    write_json(args.output / "run-binding.json", dict(
        source=git("rev-parse", "HEAD"), tree=git("rev-parse", "HEAD^{tree}"), literal_base=BASE,
        command=sys.argv, python=sys.version, platform=platform.platform(),
        census=str(args.census), census_source=CENSUS_SOURCE, census_seal=CENSUS_SEAL,
        verified_input_files=verified, tool_sha256=digest(__file__),
        runtime_apply=False, scientific_policy_selected=False))
    write_json(args.output / "final-status.json", dict(
        accounting="PASS", scan_cost="UNAVAILABLE: missing exact relation",
        elapsed_seconds=time.perf_counter() - start,
        maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        environment="local macOS Python; no C++ or Spack build", unexpected_errors=0))
    (args.output / "SHA256SUMS").write_text("".join(
        f"{digest(p)}  {p.name}\n" for p in sorted(args.output.iterdir()) if p.is_file()))
    print(json.dumps(dict(output=str(args.output), **summary["health_review"])))


if __name__ == "__main__":
    main()
