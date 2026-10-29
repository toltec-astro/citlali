"""Configured AltAz POINT feedback using the existing POINT and BASE owners.

The source selector is conditional on the initial map's frozen C0 covariance
and BASE scale. These are selection diagnostics, never detector noise weights
or an endpoint uncertainty model. Each returned model replaces its predecessor.
Native timestream processing and exact subtraction/restoration live elsewhere.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
from astropy.io import fits

from inspect_forward_map import digest, read_yaml, require
from point_additive import NoiseRequest, learn
from point_additive_diagnostics import save_json
from point_empirical import patch, source_patch
from point_empirical_io import exact_header, write_fits
from point_expected import candidate, measure_expected
from point_fruit_trajectory import candidate_guard


FRAME = dict(
    ASTFRAME="delta-source-AltAz",
    CTYPE1="AZOFFSET",
    CTYPE2="ELOFFSET",
    CUNIT1="arcsec",
    CUNIT2="arcsec",
)
GRID_KEYS = tuple(FRAME) + (
    "CRPIX1",
    "CRPIX2",
    "CRVAL1",
    "CRVAL2",
    "CDELT1",
    "CDELT2",
    "NOMBFWHM",
    "ARRAY",
)
METHOD = "configured-point-base-feedback-v1"
SCOPE = (
    "Frozen initial-map C0 and BASE are conditional source-selection "
    "models; not current-pass detector noise weights, endpoint uncertainty, "
    "or a convergence criterion."
)


@dataclass(frozen=True)
class FeedbackContext:
    """Observation/array-local frozen selector, owned by one coordinator run.

    Arrays are read-only. Their pixel axes are the dynamic 40 arcsecond source
    crop; the internal learner header uses degrees only as a unit adapter.
    The full native AltAz grid identity remains explicit and unchanged.
    """

    grid: tuple
    full_shape: tuple
    half: int
    header: tuple
    beam: float
    scale: np.ndarray
    psd: np.ndarray
    base_scale: np.ndarray
    noise_valid: np.ndarray
    initial_map_sha256: str
    observation: str
    frozen_health_identity: object


def _frozen(values):
    result = np.array(values, copy=True)
    result.flags.writeable = False
    return result


def _read_map(path):
    """Validate the inputs used by the selector, without claiming MAP closure."""
    path = Path(path)
    bundle = read_yaml(path.parent / "bundle.yaml")
    require(
        bundle["schema"] == "citlali-forward-map-bundle-v1"
        and bundle["outcome"] == "success"
        and bundle["signal_unit"] == "mJy/beam",
        "feedback MAP bundle differs",
    )
    require(
        isinstance(bundle.get("observation"), str)
        and re.fullmatch(r"obs:\d+_\d+_\d+", bundle["observation"]) is not None
        and bundle["files"].get(path.name) == digest(path),
        "feedback MAP observation or payload identity differs",
    )
    require(
        "frozen_health_identity" not in bundle
        or (
            isinstance(bundle["frozen_health_identity"], str)
            and bool(bundle["frozen_health_identity"])
        ),
        "feedback frozen health identity invalid",
    )
    with fits.open(path, memmap=False) as hdus:
        header = hdus[0].header.copy()
        require(
            all(header.get(key) == value for key, value in FRAME.items()),
            "configured feedback requires source-offset AltAz arcsecond axes",
        )
        require(
            header.get("ARRAY") in ("a1100", "a1400", "a2000"),
            "configured feedback requires an explicit TolTEC array",
        )
        require(
            header["ARRAY"] == bundle["array"]
            and header["NOMBFWHM"] == bundle["nominal_beam_fwhm_arcsec"]
            and all(
                bundle["WCS"].get(key) == header[key]
                for key in GRID_KEYS
                if key not in ("ARRAY", "NOMBFWHM")
            ),
            "feedback MAP bundle array or grid differs",
        )
        require(
            all(
                np.isfinite(header[key]) for key in GRID_KEYS if key not in FRAME and key != "ARRAY"
            ),
            "configured feedback requires a finite grid and beam",
        )
        require(
            header["CRVAL1"] == header["CRVAL2"] == 0.0
            and header["CDELT1"] < 0.0 < header["CDELT2"]
            and header["NOMBFWHM"] > 0.0,
            "configured feedback grid differs",
        )
        require(
            header.get("BUNIT") == "mJy/beam",
            "configured feedback requires calibrated nominal-beam brightness",
        )
        names = ("PRIMARY", "SCI_SUPPORT", "NORMALIZATION")
        require(
            all(sum(h.name == name for h in hdus) == 1 for name in names),
            "configured feedback requires unique signal/support/normalization planes",
        )
        data = {}
        shape = hdus[0].data.shape
        require(len(shape) == 2 and min(shape) > 0, "configured feedback map shape")
        for name in names:
            hdu = hdus[name]
            require(
                hdu.data.shape == shape
                and all(hdu.header.get(key) == header[key] for key in GRID_KEYS),
                "configured feedback plane geometry differs",
            )
            require(
                hdu.verify_checksum() == 1 and hdu.verify_datasum() == 1,
                "configured feedback input checksum missing or invalid",
            )
            data[name] = np.array(hdu.data, copy=True)
    support = data["SCI_SUPPORT"]
    require(np.isin(support, (0, 1)).all(), "configured feedback nonbinary support")
    support = support.astype(bool)
    signal, coverage = data["PRIMARY"], data["NORMALIZATION"]
    require(
        np.isfinite(signal[support]).all() and np.isnan(signal[~support]).all(),
        "configured feedback signal missingness differs",
    )
    require(
        np.isfinite(coverage).all() and (coverage >= 0).all() and (coverage[support] > 0).all(),
        "configured feedback invalid normalization",
    )
    # Existing empirical owners use degree increments to express arcsecond
    # offsets. This local conversion changes units, never the AltAz frame.
    learner_header = header.copy()
    for axis in (1, 2):
        learner_header[f"CDELT{axis}"] /= 3600.0
        learner_header[f"CRVAL{axis}"] /= 3600.0
        learner_header[f"CUNIT{axis}"] = "deg"
    return signal, support, coverage, header, learner_header, bundle


def _guard_frame(guard):
    guard.pop("expected_origin_sky", None)
    guard.update(frame="delta-source-AltAz", centroid_axes=["azimuth_offset", "elevation_offset"])
    return guard


def _unavailable(cause):
    from point_expected import METHOD as point_method

    return dict(
        method=point_method,
        status="unavailable",
        accepted=False,
        complete=False,
        cause=str(cause),
        message=str(cause),
        accepted_parameters=None,
        pointing_correction=None,
        accepted_centroid_tangent_arcsec=None,
        expected_origin_tangent_arcsec=[0.0, 0.0],
        sigma_A=None,
        SNR_A=None,
        diagnostic=None,
        optimizer_calls=0,
    )


def _publish(
    path, destination, native_header, guard, selected, *, verbose, initial_map_sha256, bundle
):
    destination = Path(destination)
    require(not destination.exists(), "feedback destination must be fresh")
    destination.mkdir(parents=True, exist_ok=False)
    record = dict(
        schema=METHOD,
        status="stopped",
        cause=guard.get("cause"),
        guard=_guard_frame(guard),
        candidate_summary=None,
        model_path=None,
        model_sha256=None,
        map_path=str(Path(path).resolve()),
        map_sha256=digest(path),
        initial_map_sha256=initial_map_sha256,
        observation=bundle["observation"],
        frozen_health_identity=bundle.get("frozen_health_identity"),
        input_bundle_sha256=digest(Path(path).parent / "bundle.yaml"),
        array=native_header["ARRAY"],
        frame="delta-source-AltAz",
        coordinate_units="arcsec",
        signal_units="mJy/nominal beam",
        selection_scope=SCOPE,
        feedback_domain_radius_arcsec=40.0,
        replacement=True,
        gain=1.0,
        threshold=3.0,
        noise_weight_relearning_owned_by="native current-residual PTC",
    )
    if guard["accepted"] and selected is not None:
        record["candidate_summary"] = selected["summary"]
        try:
            # Preserve the accepted BASE support rules without weakening an
            # empty, boundary-touching, or unavailable candidate to continue.
            candidate_guard(
                {
                    "CANDIDATE": selected["candidate"],
                    "ADMITTED": selected["selected"],
                    "VALID": selected["valid"],
                },
                selected["header"],
            )
        except ValueError as exc:
            record["cause"] = "feedback candidate unavailable: " + str(exc)
        else:
            header = native_header.copy()
            for key in ("CRPIX1", "CRPIX2"):
                header[key] = selected["header"][key]
            header = exact_header(header)
            planes = [
                ("CANDIDATE", selected["candidate"], "mJy/beam"),
                ("ADMITTED", selected["selected"].astype("u1"), "1"),
                ("VALID", selected["valid"].astype("u1"), "1"),
            ]
            if verbose:
                planes.append(("RATIO", selected["z"], "1"))
            model = destination / "candidate.fits"
            write_fits(model, header, planes, companion="CONFIGURED-POINT-BASE")
            record.update(
                status="ready",
                cause=None,
                model_path=str(model.resolve()),
                model_sha256=digest(model),
            )
    save_json(destination / "feedback.json", record)
    return record


def initialize(map_path, destination, verbose=False):
    """Learn one initial conditional selector; return ``(context, record)``.

    A failed/unavailable source produces a stopped record and no usable model.
    Software, I/O, frame and identity failures propagate. No reductions run.
    """
    require(not Path(destination).exists(), "feedback destination must be fresh")
    signal, support, coverage, native, header, bundle = _read_map(map_path)
    initial_sha = digest(map_path)
    try:
        s, valid, _, stamp_header, half = source_patch(signal, support, coverage, header)
    except ValueError as exc:
        record = _publish(
            map_path,
            destination,
            native,
            _unavailable(exc),
            None,
            verbose=verbose,
            initial_map_sha256=initial_sha,
            bundle=bundle,
        )
        return None, record
    fixed = SimpleNamespace(header=stamp_header, beam=float(native["NOMBFWHM"]))
    models = {}
    row, column = int(round(header["CRPIX2"] - 1)), int(round(header["CRPIX1"] - 1))

    def covariance(parameters):
        for name, mask, amplitude_only in (("C0", 4.0, False), ("BASE", 3.0, True)):
            models[name] = learn(
                signal,
                coverage,
                support,
                header,
                parameters,
                NoiseRequest(mask_fwhm=mask, high_coverage_usable=True),
                stamp_size=max(49, max(s.shape)),
                amplitude_only=amplitude_only,
            )
        return patch(models["C0"]["scale"], row, column, half), models["C0"]["psd"]

    guard = measure_expected(s, valid, fixed, None, None, initialize_covariance=covariance)
    context, selected = None, None
    if guard["accepted"]:
        context = FeedbackContext(
            grid=tuple((key, native[key]) for key in GRID_KEYS),
            full_shape=signal.shape,
            half=half,
            header=tuple((key, stamp_header[key]) for key in GRID_KEYS),
            beam=fixed.beam,
            scale=_frozen(patch(models["C0"]["scale"], row, column, half)),
            psd=_frozen(models["C0"]["psd"]),
            base_scale=_frozen(patch(models["BASE"]["scale"], row, column, half)),
            noise_valid=_frozen(patch(models["BASE"]["valid"], row, column, half).astype(bool)),
            initial_map_sha256=initial_sha,
            observation=bundle["observation"],
            frozen_health_identity=bundle.get("frozen_health_identity"),
        )
        fixed.scale, fixed.noise_valid = context.base_scale, context.noise_valid
        selected = candidate(s, valid, fixed, guard)
        selected["header"] = stamp_header
    record = _publish(
        map_path,
        destination,
        native,
        guard,
        selected,
        verbose=verbose,
        initial_map_sha256=initial_sha,
        bundle=bundle,
    )
    if verbose:
        for name, model in models.items():
            save_json(Path(destination) / (name + ".json"), model["summary"])
            np.savez_compressed(
                Path(destination) / (name + ".npz"),
                **{
                    key: model[key]
                    for key in (
                        "scale",
                        "valid",
                        "coverage_extrapolated",
                        "source_mask",
                        "background_mask",
                    )
                },
                **(
                    {key: model[key] for key in ("psd", "empirical", "pairs")}
                    if name == "C0"
                    else {}
                ),
            )
    return context if record["status"] == "ready" else None, record


def update(map_path, context, destination, verbose=False):
    """Infer a complete replacement on a new total map with frozen selection.

    Normalization and scientific support may change after each fresh weighted
    cleaner. The observation/array grid cannot change. No old model is added.
    """
    require(isinstance(context, FeedbackContext), "feedback context unavailable")
    require(not Path(destination).exists(), "feedback destination must be fresh")
    signal, support, coverage, native, header, bundle = _read_map(map_path)
    require(
        bundle["observation"] == context.observation
        and bundle.get("frozen_health_identity") == context.frozen_health_identity,
        "feedback observation or frozen health identity differs",
    )
    require(
        tuple((key, native[key]) for key in GRID_KEYS) == context.grid
        and signal.shape == context.full_shape,
        "feedback frozen grid or array differs",
    )
    s, valid, _, stamp_header, half = source_patch(signal, support, coverage, header)
    require(
        half == context.half
        and tuple((key, stamp_header[key]) for key in GRID_KEYS) == context.header,
        "feedback source crop differs",
    )
    fixed = SimpleNamespace(
        header=stamp_header,
        beam=context.beam,
        scale=context.base_scale,
        noise_valid=context.noise_valid,
    )
    guard = measure_expected(s, valid, fixed, context.scale, context.psd)
    selected = candidate(s, valid, fixed, guard) if guard["accepted"] else None
    if selected is not None:
        selected["header"] = stamp_header
    return _publish(
        map_path,
        destination,
        native,
        guard,
        selected,
        verbose=verbose,
        initial_map_sha256=context.initial_map_sha256,
        bundle=bundle,
    )
