"""Strict application boundary for the opt-in, all-array POINT interface.

TolTECA owns YAML merging and input preparation. This module resolves the
resulting request once, without opening raw data or changing the supplied dict.
The explicitly selected migration replaces the old algorithm controls, rather
than pretending that their names retain their former scientific meaning.
"""

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any

from configured_point_runtime import data_root


ARRAYS = ("a1100", "a1400", "a2000")
SCHEMA = "citlali-point-v1"
MIGRATION = "replace-legacy-point-controls-v1"
SCHEMA_V2 = "citlali-point-v2"
MIGRATION_V2 = "replace-legacy-point-controls-v2"
WEIGHTED_CLEANING = "noise-weighted-observed-als-v1"
RTC_PROFILE = "existing-array-lowpass-v1"
DISTURBANCE_POLICY = "repeated-level-shifts-unassessable-remainder-v1"
REPO = data_root().parent


@dataclass(frozen=True)
class PointRequest:
    effective_config: dict[str, Any]
    output_dir: Path
    workers: int
    verbose: bool
    retain_intermediates: bool
    fit_diagnostics: bool
    cleaning_method: str
    ranks: dict[str, int]
    pixel_size_arcsec: float
    coverage_cut: float
    rtc_profiles: dict[str, Path]
    arrays: tuple[str, ...]
    migration_dispositions: tuple[dict[str, Any], ...]
    weighted: bool = False
    min_relative_response: float | None = None
    feedback_enabled: bool = False
    feedback_iterations: int = 0


def _mapping(value, path, allowed=None, required=()):
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError(f"{path} must be a mapping with string keys")
    if allowed is not None:
        unknown = set(value) - set(allowed)
        if unknown:
            raise ValueError(f"unsupported {path} fields: {', '.join(sorted(unknown))}")
    missing = set(required) - set(value)
    if missing:
        raise ValueError(f"missing {path} fields: {', '.join(sorted(missing))}")
    return value


def _bool(value, path):
    if type(value) is not bool:
        raise ValueError(f"{path} must be a boolean")
    return value


def _integer(value, path, maximum=None):
    if type(value) is not int or value < 1 or (maximum is not None and value > maximum):
        suffix = f" between 1 and {maximum}" if maximum is not None else " greater than zero"
        raise ValueError(f"{path} must be an integer{suffix}")
    return value


def _number(value, path, *, minimum, maximum=None, strict_minimum=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{path} must be a finite number")
    if (
        value < minimum
        or (strict_minimum and value == minimum)
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"{path} is outside its supported range")
    return float(value)


def _path(value, path, launch_dir):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a nonempty path string")
    p = Path(value).expanduser()
    return (p if p.is_absolute() else launch_dir / p).resolve()


def _enum(value, expected, path):
    if value != expected or type(value) is not str:
        raise ValueError(f"{path} supports only {expected!r}")
    return value


def _inputs(config, launch_dir):
    """Resolve ordinary prepared paths; acquisition/APT validation comes next."""
    inputs = config.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 1:
        raise ValueError("configured POINT requires exactly one prepared input observation")
    item = _mapping(inputs[0], "inputs[0]", required=("meta", "data_items", "cal_items"))
    meta = _mapping(item["meta"], "inputs[0].meta", required=("name",))
    name = meta["name"]
    if (
        not isinstance(name, str)
        or len(name.split("_")) != 3
        or not all(part.isdecimal() for part in name.split("_"))
    ):
        raise ValueError("inputs[0].meta.name must be observation_subobservation_scan")
    observation = int(name.split("_")[0])
    if not 1 <= observation <= 999999:
        raise ValueError(
            "observation must be between 1 and 999999 for TolTECA's six-digit product discovery"
        )
    # This is the native POINT acquisition domain, not an observation-number
    # experiment. Other scan shapes need their own supported native admission.
    if tuple(map(int, name.split("_")[1:])) != (0, 2):
        raise ValueError("configured POINT currently supports subobservation 0, scan 2")
    for group in ("data_items", "cal_items"):
        if not isinstance(item[group], list) or not item[group]:
            raise ValueError(f"inputs[0].{group} must be a nonempty list")
        for index, entry in enumerate(item[group]):
            _mapping(entry, f"inputs[0].{group}[{index}]")
            if "filepath" in entry:
                entry["filepath"] = str(
                    _path(entry["filepath"], f"inputs[0].{group}[{index}].filepath", launch_dir)
                )


def _timing(config):
    entries = config.get("interface_sync_offset")
    if not isinstance(entries, list):
        raise ValueError("interface_sync_offset must be an explicit sequence")
    offsets = {}
    for entry in entries:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError("interface_sync_offset entries must contain one interface")
        name, value = next(iter(entry.items()))
        if not isinstance(name, str):
            raise ValueError("interface_sync_offset interface must be a string")
        if not name.startswith("toltec"):
            continue
        if not re.fullmatch(r"toltec(?:[0-9]|1[0-2])", name) or name in offsets:
            raise ValueError("interface_sync_offset network is invalid or duplicated")
        offsets[name] = _number(value, f"interface_sync_offset.{name}", minimum=0, maximum=0)
    for item in config["inputs"][0]["data_items"]:
        name = item.get("meta", {}).get("interface", "")
        if isinstance(name, str) and name.startswith("toltec") and name not in offsets:
            raise ValueError(f"interface_sync_offset is missing the supplied network {name}")


def _calibration(config):
    """Adapt the known constant-offset sentinel, without changing calibration."""
    items = config["inputs"][0]["cal_items"]
    supported = {"array_prop_table", "astrometry", "photometry", "flxscale_correction"}
    seen = {}
    dispositions = []
    for index, item in enumerate(items):
        kind = item.get("type")
        if not isinstance(kind, str) or kind not in supported:
            raise ValueError(f"unsupported inputs[0].cal_items type: {kind!r}")
        if kind in seen:
            raise ValueError(f"duplicate inputs[0].cal_items {kind}")
        seen[kind] = index
        path = f"inputs[0].cal_items[{index}]"
        if kind == "flxscale_correction":
            _mapping(item, path, ("type", "value", "meta"), ("value",))
            _number(item["value"], path + ".flxscale_correction.value", minimum=1, maximum=1)
            dispositions.append(
                dict(
                    path=path,
                    disposition="identity-only",
                    replacement="flxscale_correction=1 is inert; native CAL consumes matched APT flxscale once; additional scaling is not implemented",
                )
            )
        elif kind == "photometry":
            dispositions.append(
                dict(
                    path=path,
                    disposition="metadata-only",
                    replacement="beammap source photometry is preserved metadata; POINT CAL uses the matched APT, not these source fluxes",
                )
            )
    for kind in ("array_prop_table", "astrometry"):
        if kind not in seen:
            raise ValueError(f"configured POINT requires one {kind} calibration item")

    item_index = seen["astrometry"]
    item = items[item_index]
    path = f"inputs[0].cal_items[{item_index}].pointing_offsets"
    offsets = item.get("pointing_offsets")
    if not isinstance(offsets, list) or not offsets:
        raise ValueError(f"{path} must contain explicit az and alt axes")
    axes = {}
    epoch_entries = []
    for index, entry in enumerate(offsets):
        _mapping(entry, f"{path}[{index}]")
        if "modified_julian_date" in entry:
            _mapping(entry, f"{path}[{index}]", ("modified_julian_date",))
            epoch_entries.append((index, entry["modified_julian_date"]))
        else:
            _mapping(
                entry,
                f"{path}[{index}]",
                ("axes_name", "value_arcsec"),
                ("axes_name", "value_arcsec"),
            )
            axis = entry["axes_name"]
            if not isinstance(axis, str) or axis not in ("az", "alt") or axis in axes:
                raise ValueError(f"{path} axes must be az and alt exactly once")
            values = entry["value_arcsec"]
            if (
                not isinstance(values, list)
                or len(values) not in (1, 2)
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
            ):
                raise ValueError(f"{path}.{axis} requires one or two finite numeric values")
            axes[axis] = values
    if set(axes) != {"az", "alt"} or len(axes["az"]) != len(axes["alt"]):
        raise ValueError(f"{path} requires equal az and alt value counts")
    if len(epoch_entries) > 1:
        raise ValueError(f"{path}.modified_julian_date is duplicated")
    if not epoch_entries:
        if len(axes["az"]) != 1:
            raise ValueError(f"{path} requires actual epochs for two-value offsets")
        return tuple(dispositions)

    index, epochs = epoch_entries[0]
    if (
        not isinstance(epochs, list)
        or len(epochs) not in (1, 2)
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in epochs)
    ):
        raise ValueError(
            f"{path}.modified_julian_date requires finite numeric epochs or the nonpositive sentinel"
        )
    if all(v <= 0 for v in epochs):
        if len(axes["az"]) != 1:
            raise ValueError(
                f"{path} nonpositive epoch sentinel requires one constant value per axis"
            )
        # Existing pointing_offsets_config_read.h normalizes this sentinel
        # to a constant observation model. The native CAL boundary expresses
        # that same model by absent epochs and binds its support to TEL bounds.
        # Never turn a real observation epoch into an invented endpoint.
        item["pointing_offsets"] = offsets[:index] + offsets[index + 1 :]
        dispositions.append(
            dict(
                path=f"{path}[{index}].modified_julian_date",
                disposition="normalized",
                original=list(epochs),
                replacement="nonpositive epoch sentinel omitted for explicit constant az/alt; native CAL binds telescope time support",
                authority="include/citlali/core/pipeline/pointing_offsets_config_read.h",
            )
        )
    elif len(epochs) != 2 or any(v <= 0 for v in epochs) or epochs[1] <= epochs[0]:
        raise ValueError(
            f"{path}.modified_julian_date requires two increasing positive actual epochs"
        )
    return tuple(dispositions)


def load_request(config: dict, launch_dir: Path) -> PointRequest:
    """Return the requested POINT plan and a one-way native compatibility copy.

    ``launch_dir`` is the TolTECA job working directory, not the location of its
    generated YAML. No new input root is introduced. File identity, complete
    array membership and writable fresh outputs are checked by preparation.
    """
    _mapping(config, "configuration")
    launch_dir = Path(launch_dir).resolve()
    successor = _mapping(
        config.get("successor"),
        "successor",
        required=("schema", "migration", "cleaning", "rtc", "map", "feedback"),
    )
    schema = successor["schema"]
    if type(schema) is not str or schema not in (SCHEMA, SCHEMA_V2):
        raise ValueError(f"successor.schema supports only {SCHEMA!r} or {SCHEMA_V2!r}")
    weighted = schema == SCHEMA_V2
    migration = MIGRATION_V2 if weighted else MIGRATION
    _enum(successor["migration"], migration, "successor.migration")
    _mapping(
        successor,
        "successor",
        allowed=(
            "schema",
            "migration",
            "arrays",
            "cleaning",
            "rtc",
            "map",
            "feedback",
            "products",
            *(("detector_response",) if weighted else ()),
        ),
    )
    arrays = successor.get("arrays", list(ARRAYS))
    if (
        not isinstance(arrays, list)
        or any(type(a) is not str for a in arrays)
        or len(arrays) != 3
        or set(arrays) != set(ARRAYS)
    ):
        raise ValueError("successor.arrays must contain a1100, a1400 and a2000 exactly once")

    cleaning = _mapping(
        successor["cleaning"], "successor.cleaning", ("method", "ranks"), ("method", "ranks")
    )
    # PTC alone implements other methods; the connected MAP consumer does not
    # admit them. Expose only a method that reaches the requested final product.
    method = _enum(
        cleaning["method"],
        WEIGHTED_CLEANING if weighted else "observed-als",
        "successor.cleaning.method",
    )
    rank_values = _mapping(cleaning["ranks"], "successor.cleaning.ranks", ARRAYS, ARRAYS)
    ranks = {a: _integer(rank_values[a], f"successor.cleaning.ranks.{a}") for a in ARRAYS}
    rtc = _mapping(
        successor["rtc"], "successor.rtc", ("profile", "disturbance_exclusion"), ("profile",)
    )
    _enum(rtc["profile"], RTC_PROFILE, "successor.rtc.profile")
    disturbance = rtc.get("disturbance_exclusion", "disabled")
    if disturbance not in ("disabled", DISTURBANCE_POLICY):
        raise ValueError("unsupported successor.rtc.disturbance_exclusion")
    rtc_profiles = {
        a: REPO / "data/development/bright_point_base_v1" / f"{a}-rtc.json" for a in ARRAYS
    }
    for path in rtc_profiles.values():
        if not path.is_file():
            raise ValueError(f"installed RTC profile is missing: {path}")

    mapping = _mapping(
        successor["map"],
        "successor.map",
        ("method", "frame", "pixel_size_arcsec", "coverage_cut"),
        ("method", "frame", "pixel_size_arcsec", "coverage_cut"),
    )
    _enum(mapping["method"], "containing-pixel", "successor.map.method")
    _enum(mapping["frame"], "delta-source-AltAz", "successor.map.frame")
    pixel = _number(
        mapping["pixel_size_arcsec"],
        "successor.map.pixel_size_arcsec",
        minimum=0,
        strict_minimum=True,
    )
    coverage = _number(mapping["coverage_cut"], "successor.map.coverage_cut", minimum=0, maximum=1)
    if weighted:
        response = _mapping(
            successor.get("detector_response", {}),
            "successor.detector_response",
            ("min_relative_response",),
        )
        # A fraction of the separately normalized same-array response. One is
        # permitted as an explicit owner request; zero, booleans and nonfinite
        # values are not disable switches. Never tune this from realized counts.
        min_response = _number(
            response.get("min_relative_response", 0.10),
            "successor.detector_response.min_relative_response",
            minimum=0,
            maximum=1,
            strict_minimum=True,
        )
        feedback = _mapping(successor["feedback"], "successor.feedback", ("enabled", "max_iters"))
        feedback_enabled = _bool(feedback.get("enabled", True), "successor.feedback.enabled")
        feedback_iterations = _integer(
            feedback.get("max_iters", 10), "successor.feedback.max_iters"
        )
    else:
        min_response = None
        feedback = _mapping(successor["feedback"], "successor.feedback", ("enabled",), ("enabled",))
        if _bool(feedback["enabled"], "successor.feedback.enabled"):
            raise ValueError(
                "successor.feedback.enabled=true is not implemented in citlali-point-v1"
            )
        feedback_enabled, feedback_iterations = False, 0
    products = _mapping(
        successor.get("products", {}),
        "successor.products",
        ("fits", "fit_diagnostics", "retain_intermediates"),
    )
    if not _bool(products.get("fits", True), "successor.products.fits"):
        raise ValueError("successor.products.fits is required for all three arrays")
    fit_diagnostics = _bool(
        products.get("fit_diagnostics", True), "successor.products.fit_diagnostics"
    )
    retain = _bool(
        products.get("retain_intermediates", False), "successor.products.retain_intermediates"
    )

    runtime = _mapping(
        config.get("runtime"), "runtime", required=("reduction_type", "output_dir", "n_threads")
    )
    _enum(runtime["reduction_type"], "pointing", "runtime.reduction_type")
    workers = _integer(runtime["n_threads"], "runtime.n_threads", maximum=12)
    verbose = _bool(runtime.get("verbose", False), "runtime.verbose")
    output_dir = _path(runtime["output_dir"], "runtime.output_dir", launch_dir)
    effective = deepcopy(config)
    _inputs(effective, launch_dir)
    _timing(effective)
    calibration_dispositions = _calibration(effective)
    kids = _mapping(effective.get("kids"), "kids", required=("fitter", "solver"))
    fitter = _mapping(kids["fitter"], "kids.fitter", required=("modelspec",))
    _enum(fitter["modelspec"], "gainlintrend", "kids.fitter.modelspec")
    solver = _mapping(kids["solver"], "kids.solver", required=("fitreportdir",))
    kids["solver"] = dict(
        fitreportdir=str(_path(solver["fitreportdir"], "kids.solver.fitreportdir", launch_dir)),
        parallel_policy="seq",
        extra_output=False,
    )

    # Input/calibration/timing/KIDs provenance remains ordinary project data.
    # Everything below is explicitly replaced by the selected successor profile.
    # Do not keep inactive old fields in the effective copy: they could falsely
    # suggest that requested legacy JINC, flags, weights or errors were applied.
    replacements = {
        "timestream": "native paired xs; existing RTC profile and validity policy with no spatial source mask; native CAL/APT and atmosphere policy; fixed native processing partition; observed-entry network cleaning; no feedback or polarimetry",
        "mapmaking": "containing-pixel map in delta-source-AltAz; uniform admitted coefficients; configured pixels and coverage",
        "noise_maps": "not produced; map normalization is not inverse-variance weight",
        "post_processing": "optional POINT-EXPECTED qualified fit diagnostics; no legacy source finder or map filter",
        "coadd": "one-observation POINT invocation; no coadd",
        "pointing": "existing POINT-EXPECTED source analysis; no legacy pointing correction publication",
        "source": "POINT observation semantics; legacy map-regime switch retired",
        "wiener_filter": "not executed",
        "beammap": "outside POINT scope; not executed",
    }
    if weighted:
        replacements.update(
            timestream="native paired xs; existing RTC profile and observation-local disturbance policy; immutable CAL/APT parent; source-protected optical eligibility frozen before noise-weighted observed-entry network cleaning; configured fresh-residual feedback; no legacy FRUIT or polarimetry",
            mapmaking="containing-pixel map in delta-source-AltAz; empirical inverse-noise coefficients from max(reference variance, remaining variance); configured pixels and coverage; normalization is not map precision",
            noise_maps="not produced; empirical detector coefficients do not establish full cleaned-data covariance or map uncertainty",
            post_processing="POINT-EXPECTED and BASE conditional source selection for requested feedback; optional qualified fit diagnostics; no legacy source finder or map filter",
        )
    dispositions = tuple(
        dict(
            path=key,
            disposition="replaced" if key in config else "profile-defined",
            replacement=value,
        )
        for key, value in replacements.items()
    ) + (
        dict(
            path="kids.solver",
            disposition="replaced",
            replacement="configured KMP directory retained; bound tune file, serial native KIDs solve and no extra producer outputs",
        ),
    )
    dispositions += calibration_dispositions
    for key in replacements:
        effective.pop(key, None)
    effective["timestream"] = {
        "enabled": True,
        "type": "xs",
        "polarimetry": {"enabled": False},
        "fruit_loops": {"enabled": False},
        "chunking": {"chunk_mode": "duration", "value": 10, "force_chunking": True},
        # These recover the accepted processing partition/context, not the
        # separately bound per-array RTC filter coefficients or factors.
        "raw_time_chunk": {
            "IIR_filter": {"enabled": False},
            "filter": {"enabled": True, "n_terms": 32},
        },
        "processed_time_chunk": {
            "clean": {
                "enabled": True,
                "grouping": ["nw"],
                "mask_radius_arcsec": 0.0,
                "tau": 0.0,
                "standard_pca": {
                    "enabled": True,
                    "stddev_limit": 0.0,
                    "n_eig_to_cut": {a: [ranks[a]] for a in ARRAYS},
                },
            }
        },
    }
    effective["mapmaking"] = dict(
        enabled=True,
        method="containing-pixel",
        pixel_axes="altaz",
        pixel_size_arcsec=pixel,
        coverage_cut=coverage,
        cunit="mJy/beam",
    )
    effective["coadd"] = {"enabled": False}
    effective["noise_maps"] = {"enabled": False}
    # This is a one-way compatibility carrier, not executable legacy FRUIT.
    # V2 stops upstream at CAL and its own native cycle consumes the separately
    # requested weights/ranks/feedback; v1 PTC reads the standard_pca rank path.
    effective["runtime"] = dict(
        reduction_type="pointing",
        output_dir=str(output_dir),
        n_threads=workers,
        verbose=verbose,
        interp_over_gaps=True,
    )
    effective_successor: dict[str, Any] = dict(
        schema=schema,
        migration=migration,
        arrays=list(ARRAYS),
        cleaning=dict(method=method, ranks=dict(ranks)),
        rtc=dict(profile=RTC_PROFILE, disturbance_exclusion=disturbance),
        map=dict(
            method="containing-pixel",
            frame="delta-source-AltAz",
            pixel_size_arcsec=pixel,
            coverage_cut=coverage,
        ),
        feedback=dict(enabled=feedback_enabled),
        products=dict(fits=True, fit_diagnostics=fit_diagnostics, retain_intermediates=retain),
    )
    effective["successor"] = effective_successor
    if weighted:
        effective_successor["detector_response"] = dict(min_relative_response=min_response)
        effective_successor["feedback"]["max_iters"] = feedback_iterations
        effective["successor_effective"] = dict(
            schema="citlali-point-v2-effective-semantics-v1",
            optical_eligibility="observation-local preclean CAL decision, frozen before all cleaner passes; no APT mutation or response-gain rescaling",
            cleaning_weight="1 / source-protected empirical reference residual variance",
            map_weight="1 / max(reference residual variance, remaining cleaned residual variance)",
            variance_units="(mJy/nominal-beam)^2",
            coefficient_units="(mJy/nominal-beam)^-2",
            normalization="sum of empirical analysis coefficients; not map precision or significance",
            missing_estimate="unavailable; no unity coefficient or arbitrary high-weight cap",
            feedback_owner="configured successor cycle; legacy timestream.fruit_loops disabled in upstream compatibility adapter",
            feedback_budget="fixed number of fresh immutable-CAL feedback applications after the weighted forward map; not convergence",
            source_selection="existing POINT-EXPECTED and BASE, conditional on initial-map selection covariance; not full adaptive or endpoint uncertainty",
            limitations=[
                "Empirical residual variance is not an independently calibrated detector or map covariance.",
                "The response threshold is an owner-selected policy, not a calibrated observation-wide false-rejection rate.",
                "Corrected-frame feedback and weighting require owner-run all-array validation; no production qualification is inferred.",
            ],
        )
    return PointRequest(
        effective,
        output_dir,
        workers,
        verbose,
        retain,
        fit_diagnostics,
        method,
        ranks,
        pixel,
        coverage,
        rtc_profiles,
        ARRAYS,
        dispositions,
        weighted,
        min_response,
        feedback_enabled,
        feedback_iterations,
    )
