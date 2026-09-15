#!/usr/bin/env python3
"""Nine fixed coefficient screens; bounded original-pair RTC diagnostics only."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import fft

from run_rtc_notch_recovery import binding, digest, finite_trial, write_json

IDS = [269, 402, 296, 300, 366, 438, 406, 307, 193, 231, 226, 330]
WIDTHS = (0.5, 1.0, 2.0)
DURATIONS = (1, 3, 6)
COHERENT_SEAL = "221ed79ddff18ea1c89b0232410c8e87902401d02b240dca5fc363e393a5dd22"
POPULATION_SEAL = "066e5a494e6d1056461c9f2394cb6c401dfb168f99f34ab1841b7f08df5f7d83"


def sealed_reader(root, expected):
    manifest = root / "EVIDENCE_SHA256SUMS"
    if digest(manifest) != expected:
        raise ValueError("incompatible prior evidence manifest")
    entries = dict(
        line.split("  ", 1)[::-1] for line in manifest.read_text().splitlines()
    )
    used = []

    def read(relative):
        p = root / relative
        b = binding(p)
        if entries.get(relative, entries.get("./" + relative)) != b["sha256"]:
            raise ValueError(f"changed sealed input: {relative}")
        used.append(b)
        return p

    return read, used


def centered_band(f, condition, center):
    """Connected frequency interval containing center; display, not admission."""
    i = int(np.argmin(abs(f - center)))
    if not condition[i]:
        return None
    a = b = i
    while a and condition[a - 1]:
        a -= 1
    while b + 1 < len(f) and condition[b + 1]:
        b += 1
    return [float(f[a]), float(f[b])]


def feature_edges(cfg):
    feature = next(
        x for x in cfg["audit_record"]["features_x"] if x["family"] == "around11"
    )
    # The previous five-bin descriptor supplies centers and full bin span.
    half_bin = (feature["bin_span_hz"] - feature["last_hz"] + feature["first_hz"]) / 2
    return [feature["first_hz"] - half_bin, feature["last_hz"] + half_bin]


def screen(a):
    a.output.mkdir(parents=True, exist_ok=False)
    old, old_used = sealed_reader(a.prior, COHERENT_SEAL)
    pop, pop_used = sealed_reader(a.population, POPULATION_SEAL)
    cfg = json.loads(pop("replay-plans/valuable.json").read_text())
    dt, center = cfg["measured_interval"], cfg["notch_hz"]
    fir = np.array(cfg["fir"])
    reference = next(x for x in cfg["trials"] if x["id"] == "finite-6s")
    if not np.array_equal(
        finite_trial(dt, center, 6, fir)["finite_notch"], reference["finite_notch"]
    ):
        raise ValueError("previous narrow reference no longer reproduces exactly")
    receipt = json.loads(old("native-full/receipt.json").read_text())
    n = receipt["rows"]
    line = np.fromfile(old("real-01/known-added-line-selected.f64"), "<f8").reshape(
        n, 12, 2
    )
    if not np.isfinite(line).all():
        raise ValueError("nonfinite known numerical line probe")
    # Coefficient prediction of this exact constructed waveform, on its original
    # native axis. Zero extension is for the finite numerical probe's DFT only;
    # no real-data measurement or runtime padding is performed by this screen.
    nfft = fft.next_fast_len(n + 1039)
    ff = np.fft.rfftfreq(nfft, dt)
    probe_power = abs(np.fft.rfft(line[:, :8], nfft, axis=0)) ** 2
    probe_power[1:] *= 2
    if nfft % 2 == 0:
        probe_power[-1] /= 2
    lp_power = abs(np.fft.rfft(fir, nfft)) ** 2
    edges = feature_edges(cfg)
    feature = (ff >= edges[0]) & (ff <= edges[1])
    # Preserve the existing original PSD; this bin-weighted prediction is
    # separately identified from finite-record, common-window measurements.
    psd = np.memmap(
        old("native-full/spectra.f64"),
        "<f8",
        mode="r",
        shape=(491, 2, 4, receipt["bins"]),
    )
    pf = np.fft.rfftfreq(receipt["fft_samples"], dt)
    pband = (pf >= edges[0]) & (pf <= edges[1])
    nresponse = 262144
    f = np.fft.rfftfreq(nresponse, dt)
    lp = abs(np.fft.rfft(fir, nresponse))
    candidates = []
    for width in WIDTHS:
        for seconds in DURATIONS:
            trial = finite_trial(dt, center, seconds, fir, full_width_hz=width)
            if width == 0.5 and seconds == 6:
                trial = dict(
                    reference
                )  # authoritative saved vector, not a regenerated replacement
            name = f"w{width:g}-t{seconds}"
            h = np.asarray(trial["finite_notch"])
            amplitude = abs(np.fft.rfft(h, nresponse))
            combined = amplitude * lp
            transfer = abs(np.fft.rfft(h, nfft)) ** 2 * lp_power
            known_ratio = np.sum(
                probe_power[feature] * transfer[feature, None, None], axis=0
            ) / np.sum(probe_power[feature] * lp_power[feature, None, None], axis=0)
            whole_ratio = np.sum(
                probe_power * transfer[:, None, None], axis=0
            ) / np.sum(probe_power * lp_power[:, None, None], axis=0)
            hp = np.interp(pf, f, combined) ** 2
            lpp = np.interp(pf, f, lp) ** 2
            prediction = []
            for d in IDS[:8]:
                prediction.append(
                    [
                        float(
                            np.sum(psd[d, c, 0, pband] * hp[pband])
                            / np.sum(psd[d, c, 0, pband] * lpp[pband])
                        )
                        for c in range(2)
                    ]
                )
            affected = centered_band(f, abs(amplitude - 1) > 0.01, center)
            deep = centered_band(f, amplitude <= 0.1, center)
            cfg_trial = dict(trial, id=name)
            write_json(a.output / (name + ".json"), cfg_trial)
            np.column_stack([f, amplitude, combined]).astype("<f8").tofile(
                a.output / (name + "-response.f64")
            )
            candidates.append(
                dict(
                    id=name,
                    trial=binding(a.output / (name + ".json")),
                    full_requested_width_hz=width,
                    requested_cutoffs_hz=[center - width / 2, center + width / 2],
                    taps=len(h),
                    actual_notch_span_seconds=(len(h) - 1) * dt,
                    combined_half_seconds=(len(h) // 2 + len(fir) // 2) * dt,
                    deep20dB_band_hz=deep,
                    deep40dB_band_hz=centered_band(f, amplitude <= 0.01, center),
                    central_1percent_disturbance_hz=affected,
                    transition20dB_to_1percent_hz=None
                    if deep is None
                    else [[affected[0], deep[0]], [deep[1], affected[1]]],
                    maximum_amplitude_disturbance_outside_fixed_broad_band=float(
                        np.max(abs(amplitude[(f < 9.75) | (f > 12.25)] - 1))
                    ),
                    known_line_feature_power_ratio_to_LPF=known_ratio.tolist(),
                    known_line_whole_waveform_energy_ratio_to_LPF=whole_ratio.tolist(),
                    original_PSD_bin_prediction_ratio_to_LPF=prediction,
                    original_PSD_prediction_scope="fixed original five-bin descriptor; includes continuum and source; not measured finite-record suppression",
                    response=binding(a.output / (name + "-response.f64")),
                    scientific_admission=False,
                )
            )
    record = dict(
        candidates=candidates,
        fixed_center_hz=center,
        fixed_feature_edges_hz=edges,
        fixed_broad_diagnostic_hz=[9.75, 12.25],
        interval_seconds=dt,
        nominal_interval=cfg["nominal_interval"],
        original_psd_fft_samples=receipt["fft_samples"],
        known_probe_DFT_samples=nfft,
        known_probe="prior constructed common-envelope waveform, not independent physical line truth; no continuum subtraction",
        response_resolution_hz=float(f[1]),
        response_columns=["frequency_hz", "notch_magnitude", "combined_magnitude"],
        bandwidth_definitions="requested full cutoff separation; connected20/40dB attenuated regions and1percent amplitude disturbance are display measurements, never admission thresholds",
        reference_coefficients_exact=True,
        maximum_new_replays=2,
        line_excess_available=False,
        line_excess_reason="no admitted clean-noise/continuum reference",
        bindings=old_used + pop_used,
        production=False,
    )
    write_json(a.output / "screen.json", record)
    plot_screen(a.output, record)
    for c in candidates:
        print(
            c["id"],
            "half_s",
            round(c["combined_half_seconds"], 4),
            "known_x_ratio",
            np.round(
                np.array(c["known_line_feature_power_ratio_to_LPF"])[:, 0], 6
            ).tolist(),
            "deep20",
            c["deep20dB_band_hz"],
        )


def plot_screen(out, record):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for c in record["candidates"]:
        j = WIDTHS.index(c["full_requested_width_hz"])
        cfg = json.loads(Path(c["trial"]["path"]).read_text())
        response = np.fromfile(c["response"]["path"], "<f8").reshape(-1, 3)
        label = f"{cfg['requested_span_seconds']} s notch; {c['combined_half_seconds']:.3f} s combined half"
        axes[0, j].plot(
            response[:, 0], 20 * np.log10(np.maximum(response[:, 2], 1e-8)), label=label
        )
        axes[0, j].set(
            xlim=(7, 15),
            ylim=(-90, 1),
            title=f"{c['full_requested_width_hz']:g} Hz full design width",
            xlabel="Native frequency (Hz)",
            ylabel="Combined magnitude (dB)",
        )
        h = np.array(cfg["finite_notch"])
        h[len(h) // 2] -= 1
        axes[1, j].plot(
            (np.arange(len(h)) - len(h) // 2) * record["interval_seconds"],
            h,
            label=label,
        )
        axes[1, j].set(
            xlim=(-3.1, 3.1),
            xlabel="Native time offset (s)",
            ylabel="Notch impulse minus unit impulse",
        )
    for j in range(3):
        axes[0, j].axvspan(*record["fixed_feature_edges_hz"], alpha=0.1, color="grey")
        axes[0, j].legend(fontsize=7)
    fig.suptitle(
        "Fixed nine-design screen — same Hann family, center, unit DC and low-pass/F2\nShading: unchanged original five-bin feature; no treatment admission"
    )
    fig.tight_layout()
    fig.savefig(out / "frequency-and-impulse.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    screen(parser.parse_args())


if __name__ == "__main__":
    main()
