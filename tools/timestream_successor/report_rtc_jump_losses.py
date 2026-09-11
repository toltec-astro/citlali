"""Render a compact loss diagnosis from audited saved results and targeted replays."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


NAMES = ("Original", "Single reassessment", "Known-support diagnostic")
VERSIONS = ("original", "current", "truth_fit")
COLORS = ("#286da8", "#d76c18", "#259567")


def expand(sides):
    return [np.array([i for a,b in side for i in range(a,b)],dtype=int) for side in sides]


def detail(d, path):
    s=np.array(d["samples"],float);g=d["groups"][0];t=s[:,1]-d["truth_begin"]
    window=(t>=-2.15)&(t<=2.15)
    fig,axes=plt.subplots(4,2,figsize=(15,13),sharex=True,gridspec_kw={"height_ratios":[1,1,1,.75]})
    for c,coord in enumerate(g["coordinates"]):
        y=s[:,4+2*c];background=s[:,8+c]
        low,high=y[window].min(),y[window].max();pad=max((high-low)*.08,np.finfo(float).eps)
        for k,name in enumerate(VERSIONS):
            ax=axes[k,c];v=coord[name];color=COLORS[k]
            ax.plot(t[window],background[window],color=".7",lw=.7,label="Uninjected background")
            ax.plot(t[window],y[window],color=".25",lw=.6,label="Injected original samples")
            mask=g[("original_exclusions","inferred_reassessment_exclusions","truth_exclusions")[k]]
            for a,b in mask:
                a=max(0,a);b=min(len(s),b)
                if b>a and s[b-1,3]-d["truth_begin"]>=-2.15 and s[a,2]-d["truth_begin"]<=2.15:
                    ax.axvspan(s[a,2]-d["truth_begin"],s[b-1,3]-d["truth_begin"],color=color,alpha=.12)
            rows=expand(v["primary_rows"]);short=expand(v["short_rows"])
            fit=v["primary_offset"]
            for side,rs in enumerate(rows):
                rs=rs[window[rs]]
                ax.scatter(t[rs],y[rs],s=8,color=color,alpha=.6,
                           label=("Primary fit samples" if fit["available"] else "Primary eligible only") if side==0 else None)
            if fit["available"]:
                u=(s[:,1]-g["origin"])/g["time_scale"]
                pre=np.polynomial.polynomial.polyval(u,fit["coefficients"]);post=pre+fit["offset"]
                ax.plot(t[window],pre[window],color=color,lw=1.35,label="Fitted pre state")
                ax.plot(t[window],post[window],color=color,lw=1.35,ls="--",label="Fitted post state")
                for side,rs in enumerate(rows):
                    rs=rs[window[rs]]
                    residual=(y[rs]-(pre[rs]+side*fit["offset"]))/fit["scale"]
                    axes[3,c].scatter(t[rs],residual,s=7,alpha=.45,color=color,label=NAMES[k] if side==0 else None)
            for side,rs in enumerate(short):
                rs=rs[window[rs]]
                ax.scatter(t[rs],np.full(len(rs),low-pad*.4),s=8,marker="|",color="#8c5cad",
                           label=("Short fit support" if v["short_offset"]["available"] else "Short eligible only") if side==0 else None)
            tr=v["actual_transition"]
            if tr["available"]:
                style="-" if v["retained"] else ":"
                ax.plot(np.array([tr["begin"],tr["end"]])-d["truth_begin"],[high+pad*.4]*2,
                        color=color,lw=3,ls=style,label="Retained bound" if v["retained"] else "Bound; final rule unresolved")
                for confirmation in tr["confirmations"]:
                    if confirmation["begin"] is not None:
                        ax.plot(np.array([confirmation["begin"],confirmation["end"]])-d["truth_begin"],
                                [high+pad*.2]*2,color=color,lw=1)
            elif v["ungated_transition_probe"]["available"]:
                tr=v["ungated_transition_probe"]
                ax.plot(np.array([tr["begin"],tr["end"]])-d["truth_begin"],[high+pad*.4]*2,
                        color=".4",lw=2,ls=":",label="Ungated probe; not retained")
            if d["truth_end"]>d["truth_begin"]:
                ax.axvspan(0,d["truth_end"]-d["truth_begin"],color="#00a65a",alpha=.22)
            else:
                ax.axvline(0,color="#00a65a",lw=1.2,label="Known instantaneous jump")
            # Show every grouped candidate endpoint; their span is not truth.
            for member in g["candidate_members"]:
                if member["coordinate"]==coord["coordinate"]:
                    x=s[member["later"],1]-d["truth_begin"]
                    if -2.15<=x<=2.15:ax.axvline(x,color=".45",lw=.5,ls=":",alpha=.45)
            primary_count="/".join(str(len(z)) for z in rows);short_count="/".join(str(len(z)) for z in short)
            stage={"remaining_paired_overlap":"support conflict","fit_availability":"fit unavailable",
                   "transition_measurement":"transition unavailable","confirmed_recovery":"recovered",
                   "original_amplitude":"amplitude below cut"}.get(v["first_decisive_stage"],v["first_decisive_stage"])
            offset=lambda f: f'{f["offset"]:.3g}' if f["available"] else 'unavailable'
            ax.set_title(f'{coord["coordinate"]} · {NAMES[k]} · {stage}\nprimary pre/post {primary_count}; short {short_count}\nA₂={offset(v["primary_offset"])}; A₁={offset(v["short_offset"])}',fontsize=9)
            ax.set_ylim(low-pad,high+pad);ax.set_xlim(-2.15,2.15);ax.grid(alpha=.13)
            ax.ticklabel_format(axis="y",style="sci",scilimits=(-3,3))
            if c==0:ax.set_ylabel("Original coordinate units")
            if k==0:ax.legend(fontsize=6.5,loc="best",ncol=2)
        axes[3,c].axhline(0,color=".3",lw=.6);axes[3,c].axhline(4,color=".5",lw=.5,ls=":");axes[3,c].axhline(-4,color=".5",lw=.5,ls=":")
        axes[3,c].grid(alpha=.15);axes[3,c].set_xlabel("Seconds relative to known injected onset")
        axes[3,c].set_ylabel("Fit residual / frozen scale");axes[3,c].legend(fontsize=7,loc="best")
    identity=d["background_identity"].split(":sha256:")[0]
    fig.suptitle(f'{identity} · {d["trial"].replace("_"," ")}\nAdded offsets: x={d["injected_offset"][0]:.4g}, r={d["injected_offset"][1]:.4g}\nShading: fitting exclusions; green: known truth. All retention remains diagnostic.',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.935));fig.savefig(path,dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("census",type=Path);p.add_argument("truth",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--source",default="uncommitted diagnostic worktree; pilot only")
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    census=json.loads(args.census.read_text());truth=json.loads((args.truth/"truth-analysis.json").read_text());docs=json.loads((args.truth/"replays.json").read_text())
    metrics=truth["records"]
    selection=[("synthetic-cubic-noise-v1","sharp_step","synthetic-sharp.png"),
        ("152385-04","sharp_step","lost-sharp.png"),("152385-04","sharp_plus_neighbor_spike","lost-sharp-neighbor.png"),
        ("152430-08","sharp_plus_neighbor_spike","consistency-loss.png")]
    for identity,trial,name in selection:
        d=next(d for d in docs if d["background_identity"].startswith(identity) and d["trial"]==trial);detail(d,out/name)
    fields=["background","trial","event","coordinate","version","forced_location","candidate_detected",
        "end_to_end_retained","diagnostic_retained","offset_fit_available","primary_available","short_available",
        "injected_offset","fitted_offset","fitted_short_offset","matched_uninjected_offset","raw_fit_minus_injected",
        "raw_fit_minus_injected_sigma","recovered_injection_error","recovered_injection_error_sigma","first_decisive_stage",
        "boundary_available","begin_error_samples","end_error_samples","begin_error_seconds","end_error_seconds"]
    with (out/"all-injection-comparisons.tsv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,delimiter="\t");writer.writeheader()
        for r in metrics:
            row={k:r[k] for k in fields if k in r}
            boundary=r["actual_transition"]
            row["boundary_available"]=boundary["available"]
            for units in ("samples","seconds"):
                errors=boundary.get("boundary_error_"+units,[None,None])
                for side,value in zip(("begin","end"),errors):row[side+"_error_"+units]=value
            writer.writerow(row)
    def link(name,label=None):return f'[{label or name}]({out/name})'
    def metric(background,trial,coordinate,version):return next(r for r in metrics if r["background"]==background and r["trial"]==trial and r["coordinate"]==coordinate and r["version"]==version)
    def fmt(v):return "unavailable" if v is None else f"{v:.4g}"
    t=census["totals"];reasons=census["mutually_exclusive_coordinate_terminal_reasons"]
    lines=["# Why jump measurements are lost", "",
        "**The targeted tests identify avoidable information loss: later candidate edges can stretch a sharp jump’s transition bound across stable post-jump data. The reassessment then removes that data and makes its own short fit unavailable.** This is not evidence that every unresolved census measurement should be retained.","",
        "The candidate method is unchanged. The 143-file census was read from saved results; only the small fixed injection catalog was replayed. Models, contexts, frozen scales, thresholds and the one-additional-pass limit remain fixed.","",
        "## Saved census, reconciled", "",
        f'Of **{t["original_coordinate_measurements"]:,} original x/r measurements**, **7,809** remain without refitting, **1,263** remain after refitting and **2,758** remain unresolved. The actual request count is **3,226 paired refits**, touching **4,021** original coordinate measurements. Another **28** partner-coordinate measurements become available; they are not part of the retention numerator.',"",
        "| Mutually exclusive terminal reason among unresolved coordinates | Count |", "|---|---:|"]
    labels={"insufficient_fitting_support":"Insufficient fitting support","fit_failure":"Numerical fit failure",
        "short_long_offset_inconsistent":"Short/long offset disagreement","short_long_sign_inconsistent":"Short/long sign disagreement",
        "primary_amplitude_recheck_failed":"Primary amplitude recheck below threshold","short_amplitude_recheck_failed":"Short amplitude below threshold",
        "confirmed_recovery":"Confirmed recovery","transition_unavailable":"Transition measurement unavailable","remaining_overlap":"Remaining paired support overlap"}
    for key,label in labels.items():lines.append(f'| {label} | {reasons.get(key,0):,} |')
    lines += ["", "Within the combined fit-availability gate, insufficient support takes reporting precedence when a numerical failure also occurs. Additional evaluated conditions remain recorded: short post-event support is insufficient in 1,992 unresolved coordinates; primary post support in 964. There are 202 confirmed-recovery observations among unresolved coordinates, but only two have recovery as their first terminal cause. **Recovery changes the persistent-jump interpretation; it does not establish that no disturbance occurred.** Unevaluated defaults are excluded from these counts.","",
        "The 8,980 original paired candidate groups split into 5,754 retained without a refit, 942 retaining all original coordinate measurements after a refit, 57 losing only part of their original coordinate evidence, and 2,227 losing all original coordinate measurements. After the 28 new partner measurements are included, 6,758 groups retain at least one bound. These are unique candidate groups, not accepted physical-event identities or detector–scan exposures.","",
        "Losses occur in **1,039 observation/network/channel combinations**. The largest contributes 53 losses; the ten largest contribute 434 (15.7%). Observations 152390 and 152392 contribute 767 and 889 losses, respectively; they also contain most original measurements (2,765 and 4,960). Their loss fractions are 27.7% and 17.9%. The census contains 17 channel streams with a saved observation-time health concern; none supplied an original measured bracket, so none contributes to this lost-measurement population. This does not certify the remaining channels as healthy.","",
        f'Full disjoint counts, additional conditions, identities and observation/detector distributions: [census-losses.json]({args.census.resolve()}).',"",
        "## Failed sharp steps: where the loss occurs", "",
        "On real background **152385 / network 4 / channel 61**, a one-cell-averaged instantaneous injected step is grouped with later candidate edges roughly 0.8 seconds afterward. The transition learner excludes the span between the earliest and latest member endpoints from stable-state confirmation. It therefore reports cells **[1220,1324)**, a **0.851968-second** bound, for truth lying at the midpoint of cell 1221.","",
        "The paired reassessment mask becomes [1214,1330). Short-fit post-event support falls **104 → 20 samples** (89 → 20 with the nearby injected spike), below the unchanged minimum of 64. That is the first decisive failure. The 2σ agreement test is not reached. A known-support fit retains **111 post samples** (96 with the neighboring spike), and its x/r offsets pass the existing amplitude/sign/agreement checks.","",
        "| Case / coordinate | Original offset | Reassessed offset | Known-support offset | Added offset | First current failure |", "|---|---:|---:|---:|---:|---|"]
    for bg,trial,title in [(1,"sharp_step","152385 sharp"),(1,"sharp_plus_neighbor_spike","152385 sharp + spike"),(2,"sharp_plus_neighbor_spike","152430 sharp + spike")]:
        for c in ("x","r"):
            a,b,o=[metric(bg,trial,c,v) for v in VERSIONS]
            lines.append(f'| {title} / {c} | {fmt(a["fitted_offset"])} | {fmt(b["fitted_offset"])} | {fmt(o["fitted_offset"])} | {fmt(a["injected_offset"])} | {b["first_decisive_stage"]} |')
    lines += ["", "An available primary offset is shown even when the short fit or a later rule fails; amplitude summaries are not restricted to survivors. Values retain each coordinate’s original units.","",
        "The known-support fit does **not** silently bypass remaining decisions. On the failed 152385 sharp cases, the unchanged grouping-based transition rule still returns the same wide bound. That bound overlaps the correctly retained fitting samples, so final retention remains unresolved. This separates model capability from the transition-support/decision problem.","",
        "A different failure appears in **152430 / network 8 / channel 253, x, sharp plus neighboring spike**: the refitted long/short offsets disagree by slightly more than the fixed 2σ_delta tolerance. The x offset also differs substantially from the added jump; the same-support uninjected fit exposes a pre-existing background contribution. The r measurement survives. These results do not justify relaxing the consistency tolerance or assuming that a real background is event-free.","",
        "Four explanatory figures show injected samples and their uninjected backgrounds, all fitting masks, actual primary-fit samples, available or eligible-only short support, both fitted states, bounds, confirmations and residuals:","",
        f'- {link("lost-sharp.png","Lost sharp step: long exclusion removes the short fit’s support")}',
        f'- {link("lost-sharp-neighbor.png","Same failed background with the nearby injected spike")}',
        f'- {link("consistency-loss.png","A distinct loss at the offset-consistency decision")}',
        f'- {link("synthetic-sharp.png","Matched successful synthetic sharp step")}',"",
        "## Truth errors and denominators", "",
        "All **12 injected jumps / 24 coordinate cases** remain in the denominator. Eighteen coordinate cases have a candidate at the injected support. The original method retains **16/24**; the current reassessment retains **9/24**. The six missed coordinate cases are never recovered into these end-to-end counts by supplying their location.","",
        "The following table keeps x and r separate and shows their paired consequence. Y means a bound is retained under that version’s rules; N means unavailable or withheld. Known-support results are diagnostic only. A missed candidate stays N in every end-to-end column even when the supplied-location probe succeeds.","",
        "| Background / injected jump | Candidate x/r | Original x/r | Reassessed x/r | Known-support diagnostic x/r |", "|---|---|---|---|---|"]
    for bg in range(3):
        for trial in ("sharp_step","finite_3_cells","finite_12_cells","sharp_plus_neighbor_spike"):
            rows=[metric(bg,trial,c,"original") for c in ("x","r")]
            pair=lambda key,version: "/".join("Y" if metric(bg,trial,c,version)[key] else "N" for c in ("x","r"))
            identity=rows[0]["background_identity"].split(":sha256:")[0]
            lines.append(f'| {identity} / {trial.replace("_"," ")} | {pair("candidate_detected","original")} | {pair("end_to_end_retained","original")} | {pair("end_to_end_retained","current")} | {pair("end_to_end_retained","truth_fit")} |')
    lines += ["",
        "All 18 candidate-associated coordinate cases have a numerical primary offset fit, including those later withheld. Both the raw difference from the added offset and the **incremental error**—injected fit minus an uninjected fit on identical samples, basis and frozen scale, minus the added offset—are reported. The raw difference can include real background structure or genuine pre-existing events. The incremental comparison isolates recovery of the artificial addition; it does not certify the background model or estimate total parameter uncertainty.","",
        "Boundary error is signed reported minus true boundary, in native integration-cell positions and elapsed seconds. For a sharp midpoint truth, half-cell errors are expected; affected-cell coverage is recorded separately. The failed 152385 sharp bound has errors **−1.5 and +102.5 cells**, approximately **−12.288 and +839.680 ms**. Reassessment produces no requested bound after its fit gate fails; an explicitly ungated probe reproduces the wide bound and is not counted as retained.","",
        "| Twelve-cell ramp background | Actual duration (ms) | Added x offset | Added r offset | End-to-end detection | Supplied-location current bound |", "|---|---:|---:|---:|---|---|"]
    for bg in range(3):
        d=next(d for d in docs if d["trial"]=="finite_12_cells" and d["background_identity"].startswith(("synthetic","152385","152430")[bg]))
        retained=[metric(bg,"finite_12_cells",c,"current")["diagnostic_retained"] for c in ("x","r")]
        lines.append(f'| {d["background_identity"].split(":sha256:")[0]} | {1000*d["truth_duration_seconds"]:.6f} | {fmt(d["injected_offset"][0])} | {fmt(d["injected_offset"][1])} | Missed in x and r | x={retained[0]}, r={retained[1]} |')
    lines += ["", "The supplied-location probe is downstream-only. The same functions can measure the synthetic and 152385 ramps once a location is supplied; on 152430 the new bound still overlaps fitting support. None changes the three end-to-end candidate misses.","",
        "The catalog also retains all corresponding unmodified real backgrounds and spike-only controls. Two fixed synthetic finite pulses (3 and 12 cells) supplement the synthetic unmodified/spike controls. The spike and both finite pulses generate candidates, while all four synthetic control cases produce **zero retained persistent-jump measurements in eight coordinate cases**; recovery is confirmed in both coordinates of each. The corresponding real unmodified/spike-only comparisons also produce no retained jump at the tested location. Unrelated real-background candidates are not called false positives or assumed absent.","",
        f'Every x/r/version result, including missing fits, is in {link("all-injection-comparisons.tsv")}. Complete boundary errors, availability, paired consequences and first causes: [truth-analysis.json]({(args.truth/"truth-analysis.json").resolve()}).',"",
        "## The tolerance and the smallest next change", "",
        "The [owner’s recorded 2026-09-10 rule](/private/tmp/citlali-timestream-successor-rtc-event-background-001/handoff/TIMESTREAM_SUCCESSOR_RTC_EVENT_BACKGROUND_001_2026-09-08.md:697) intentionally uses the same candidate-block **sigma_delta = 1.4826 × MAD(adjacent differences)** for the 5σ amplitude cut and 2σ short/long agreement. It is explicitly an empirical comparison tolerance, not the standard error of either fitted offset. The primary and short pre-event residual scales are separate frozen inputs to their robust fits. The [reassessment comparison](/private/tmp/citlali-timestream-successor-rtc-event-background-001/include/citlali/core/pipeline/timestream_rtc_jump_reassessment.h:265) implements that recorded distinction.","",
        "Freezing sample noise does not establish parameter uncertainty: regression support/design and shared-data correlations also matter. The current code supplies no offset covariance or calibrated uncertainty for the revised estimates. This is the distinction illustrated by [NIST’s least-squares discussion](https://www.itl.nist.gov/div898/handbook/pmd/section4/pmd431.htm); it does not authorize substituting an independent/white-noise standard error for this robust, data-selected fit.","",
        "**Recommend one next change for owner consideration: separate the onset jump’s transition support from later candidate edges already grouped with it.** Keep those later edges visible and masked as separate context; do not force the whole intervening stable plateau into the onset transition merely because membership spans it. Preserve the current fitter, thresholds and one-pass limit. This diagnosis has not implemented that change, selected a new split/merge classifier, or shown that all 2,758 withheld measurements are safe to keep.","",
        "## Verification and scope", "",
        f'Diagnostic source: `{args.source}`. Control source: `aaa86ea1b006cb11aa740adce2429375b13e5026`, closure `8a66bc0203383995a2e6bdcf4261245003a20237`. All 18 earlier injection outputs reproduce byte-for-byte. Independent audits verify {truth["audits"]["sample_value_checks"]:,} sample values, {truth["audits"]["fit_loss_audits"]} available fit losses and {truth["audits"]["explicit_anchor_parity_checks"]} explicit-anchor/production comparisons. The independent arithmetic formula check differs by at most one floating-point ULP on a few ramp samples; this is recorded separately from exact legacy reproduction.',"",
        "The explicit-anchor adapter is test-only and cannot create or modify immutable RTC producer evidence. Original runtime Learn/Consider boundaries remain intact; no runtime Apply plan or flags are produced. Local cached-dependency evidence is supplemental, not Unity/Spack or production qualification. Final gate and independent exact-SHA review dispositions belong in the external completion receipt; this generated report alone does not establish closure.",""]
    (out/"README.md").write_text("\n".join(lines))
    (out/"plot-selection.json").write_text(json.dumps(selection,indent=2)+"\n")
    print(json.dumps(dict(report=str(out/"README.md"),plots=len(selection),comparison_rows=len(metrics)),indent=2))


if __name__ == "__main__":
    main()
