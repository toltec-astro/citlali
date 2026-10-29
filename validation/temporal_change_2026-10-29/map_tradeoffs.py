"""Toy fixed-operator map transfer, never a Citlali reduction or real observation.

Coordinates are beam FWHM; brightness is arbitrary. No harm threshold is chosen.
The affected cohort has identical coherent contamination and a fixed weight share.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from temporal import multiply


def setup():
    grid = np.linspace(-4, 4, 41)
    xx, yy = np.meshgrid(grid, grid)
    ids = np.arange(xx.size).reshape(xx.shape)
    row = ids.copy()
    row[1::2] = row[1::2, ::-1]
    column = ids.T.copy()
    column[1::2] = column[1::2, ::-1]
    scan = np.concatenate([row.ravel(), column.ravel(), row.ravel()[::-1],
                           column.ravel()[::-1]])
    t = np.linspace(0, 1, len(scan))
    atmosphere = (np.sin(2*np.pi*(3.3*t+.18)) + .5*np.sin(2*np.pi*(11.7*t+.1))
                  + .3*np.cos(2*np.pi*1.1*t))
    atmosphere *= 30/np.std(atmosphere)
    source = 20*np.exp(-4*np.log(2)*(xx*xx+yy*yy))
    off = (xx.ravel()[scan]**2 + yy.ravel()[scan]**2) > 1.5**2
    # Fixed atmospheric regression basis, source-protected fitting, all-row apply.
    columns = np.column_stack((np.ones(len(scan)), atmosphere))
    u, singular, vt = np.linalg.svd(columns[off], full_matrices=False)
    inverse = multiply(vt.T, u.T/singular[:, None])
    return {"x": xx, "y": yy, "scan": scan, "time": t, "atmosphere": atmosphere,
            "source": source, "off": off, "columns": columns, "inverse": inverse}


def clean(s, values):
    one = values.ndim == 1
    values = values[:, None] if one else values
    output = values - multiply(s["columns"], multiply(s["inverse"], values[s["off"]]))
    return output[:, 0] if one else output


def project(s, values):
    # Every pixel has four hits; same support for all experiments.
    return (np.bincount(s["scan"], weights=values, minlength=s["x"].size)/4).reshape(s["x"].shape)


def fit_source(s, image):
    mask = s["x"]**2+s["y"]**2 <= 1.5**2
    x, y, data = s["x"][mask], s["y"][mask], image[mask]
    def model(p):
        amplitude, cx, cy, width, offset = p
        return amplitude*np.exp(-4*np.log(2)*((x-cx)**2+(y-cy)**2)/width**2)+offset
    result = least_squares(lambda p: model(p)-data, [20, 0, 0, 1, 0],
                           bounds=([0, -1, -1, .2, -np.inf], [np.inf, 1, 1, 3, np.inf]),
                           xtol=1e-11, ftol=1e-11, gtol=1e-11)
    return result.x, bool(result.success and not np.any(result.active_mask))


def impact(s, delta, share, pattern, coupling):
    t = s["time"]
    if pattern == "drift":
        g = delta*t
    else:
        g = delta*(t >= float(pattern.split("_")[1]))
    contaminant = g*s["atmosphere"]
    if coupling == "all_signal":
        contaminant = contaminant+g*s["source"].ravel()[s["scan"]]
    elif coupling != "atmospheric_only":
        raise ValueError("unknown coupling")
    mapped = share*project(s, clean(s, contaminant))
    fit, ok = fit_source(s, s["source"]+mapped)
    off = s["x"]**2+s["y"]**2 > 1.5**2
    return {"gain_change": delta, "affected_weight_share": share, "pattern": pattern,
            "coupling": coupling, "background_rms": float(np.std(mapped[off])),
            "background_peak_abs": float(np.max(np.abs(mapped[off]))),
            "amplitude_bias_percent": float(100*(fit[0]/20-1)),
            "centroid_shift_fwhm": float(np.hypot(fit[1], fit[2])),
            "width_bias_percent": float(100*(fit[3]-1)), "fit_interior_success": ok}, mapped


def noise_benchmark(s, draws=256):
    rng = np.random.default_rng(90210)
    off = s["x"]**2+s["y"]**2 > 1.5**2
    metrics = []
    for _ in range(draws):
        # Four independent time samples/pixel: uncleaned map sigma = 1.
        image = project(s, clean(s, rng.normal(0, 2, len(s["scan"]))))
        fit, ok = fit_source(s, s["source"]+image)
        if not ok:
            raise ArithmeticError("toy noise fit failed")
        metrics.append([np.std(image[off]), fit[0], fit[1], fit[2], fit[3]])
    a = np.array(metrics)
    return {"realizations": draws, "background_rms_median": float(np.median(a[:, 0])),
            "amplitude_sigma_percent": float(100*np.std(a[:, 1], ddof=1)/20),
            "centroid_x_sigma_fwhm": float(np.std(a[:, 2], ddof=1)),
            "centroid_y_sigma_fwhm": float(np.std(a[:, 3], ddof=1)),
            "width_sigma_percent": float(100*np.std(a[:, 4], ddof=1))}


def render(s, records, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for share in (.001, .01, .05, .2):
        selected = [r for r in records if r["affected_weight_share"] == share
                    and r["pattern"] == "step_0.5" and r["coupling"] == "atmospheric_only"]
        axes[0, 0].plot([100*r["gain_change"] for r in selected],
                        [r["background_rms"] for r in selected], "o-", label=f"{100*share:g}% weight")
    axes[0, 0].set(xlabel="Response step (%)", ylabel="Structured background RMS (brightness units)",
                   title="Atmospheric contamination: weight matters")
    axes[0, 0].legend(fontsize=8)
    for coupling, style in [("atmospheric_only", "-"), ("all_signal", "--")]:
        selected = [r for r in records if r["affected_weight_share"] == .2
                    and r["pattern"] == "step_0.5" and r["coupling"] == coupling]
        axes[0, 1].plot([100*r["gain_change"] for r in selected],
                        [r["amplitude_bias_percent"] for r in selected], "o"+style,
                        label=coupling.replace("_", " "))
    axes[0, 1].set(xlabel="Response step (%)", ylabel="Fitted source-amplitude bias (%)",
                   title="20% affected weight: source gain is a separate risk")
    axes[0, 1].legend(fontsize=8)
    for ax, share in zip(axes[1], [.01, .2]):
        _, image = impact(s, .3, share, "step_0.5", "atmospheric_only")
        im = ax.imshow(image, origin="lower", extent=(-4, 4, -4, 4), cmap="RdBu_r", vmin=-2, vmax=2)
        ax.set(xlabel="Beam FWHM", ylabel="Beam FWHM",
               title=f"30% step; {100*share:g}% affected weight")
        fig.colorbar(im, ax=ax, label="Contamination (brightness units)")
    fig.suptitle("Synthetic fixed-operator tradeoffs — not TolTEC maps or acceptance limits", fontsize=13)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    s = setup()
    records = [impact(s, change, share, pattern, coupling)[0]
               for change in (.01, .05, .1, .3, .8)
               for share in (.001, .01, .05, .2)
               for pattern in ("step_0.25", "step_0.5", "step_0.75", "drift")
               for coupling in ("atmospheric_only", "all_signal")]
    benchmark = noise_benchmark(s)
    for r in records:
        r["background_rms_over_noise"] = r["background_rms"]/benchmark["background_rms_median"]
        r["amplitude_bias_over_noise_sigma"] = abs(r["amplitude_bias_percent"])/benchmark["amplitude_sigma_percent"]
        # Same artifact repeats, independent noise averages: conditional scaling.
        r["coadd16_background_over_noise_if_coherent"] = 4*r["background_rms_over_noise"]
        # Removing cohort changes independent inverse-variance noise only.
        r["noise_increase_if_dropped_percent"] = 100*(1/np.sqrt(1-r["affected_weight_share"])-1)
    result = {"scope": "synthetic fixed operator; no ALS/FRUIT or telescope prediction",
              "optical_cut": .1, "harm_threshold": None, "beam_fwhm": 1,
              "source_peak": 20, "atmosphere_rms": 30, "passes": 4,
              "noise_benchmark": benchmark, "cases": records}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    render(s, records, args.figure)
