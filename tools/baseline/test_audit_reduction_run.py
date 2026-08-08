from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import yaml

from tools.baseline import audit_reduction_run as audit


def valid_processed_document() -> dict:
    requested = {
        "fruit_loops": {
            "enabled": False,
            "save_all_iters": False,
            "max_iters": 1,
        },
        "processed_time_chunk": {
            "clean": {"active": "standard_pca"},
            "weighting": {
                "source_mask_radius_arcsec": 30,
                "validation": {"enabled": True},
                "busy_row_suppression": {"enabled": False},
            },
            "flagging": {
                "second_pass_local": {
                    "enabled": True,
                    "source_protection": {
                        "enabled": True,
                        "active": False,
                    },
                }
            },
        },
    }
    effective = {
        "fruit_loops": {
            "enabled": False,
            "save_all_iters": True,
            "max_iters": 1,
        },
        "processed_time_chunk": {
            "clean": {"active": "standard_pca"},
            "weighting": {
                "source_mask_radius_arcsec": 30,
                "validation": {"enabled": True},
                "busy_row_suppression": {"enabled": False},
            },
            "flagging": {
                "second_pass_local": {
                    "enabled": True,
                    "source_protection": {
                        "enabled": True,
                        "active": True,
                    },
                }
            },
        },
    }
    return {
        "schema_version": "citlali-processed-timestream-provenance-v1",
        "initialized": True,
        "requested": requested,
        "effective": {
            "config": effective,
            "resolutions": {
                "cleaner_mode": {
                    "available": True,
                    "value": {"effective": "standard_pca"},
                },
                "weighting_source_mask": {
                    "available": True,
                    "value": {
                        "requested_present": True,
                        "requested": 30,
                        "effective": 30,
                        "inherited_from_cleaning": False,
                    },
                },
                "weighting_dependencies": {
                    "available": True,
                    "value": {
                        "validation_forced_by_weighting_type": False,
                        "busy_row_disabled_without_second_pass": False,
                    },
                },
                "fruit_loop_iterations": {
                    "available": True,
                    "value": {
                        "effective_max_iters": 1,
                        "effective_save_all_iters": True,
                        "forced_single_iteration_while_disabled": True,
                    },
                },
                "fruit_loop_interpolation": {
                    "available": True,
                    "value": {"effective": "jinc"},
                },
            },
        },
        "realized": {
            "source_protection": {
                "available": True,
                "value": {
                    "processed_activation_requested": True,
                    "source_aware_reduction": True,
                    "processed_active": True,
                },
            },
            "fruit_loop_iterations_completed": {
                "available": True,
                "value": 1,
            },
            "fruit_loops_converged": {
                "available": True,
                "value": False,
            },
        },
    }


def valid_processed_v2_document() -> dict:
    document = valid_processed_document()
    document["schema_version"] = "citlali-processed-timestream-provenance-v2"
    document["requested"]["fruit_loops"]["restart_path"] = None
    document["effective"]["config"]["fruit_loops"]["restart_path"] = None
    document["effective"]["resolutions"]["fruit_loop_restart"] = {
        "available": False,
    }
    return document


def valid_raw_document() -> dict:
    return {
        "schema_version": "citlali-raw-timestream-provenance-v1",
        "initialized": True,
        "requested": {},
        "effective": {"config": {}, "resolutions": {}},
        "observation": {
            "available": True,
            "value": {
                "native_sample_rate_hz": {"available": True, "value": 100.0},
                "effective_sample_rate_hz": {"available": True, "value": 50.0},
                "downsample_factor": {"available": True, "value": 2},
                "filter_edge_guard_samples": {"available": True, "value": 4},
                "filter_outer_context_samples": {"available": True, "value": 8},
                "filter_edge_guard_parity_deferred": False,
                "source_protection_active": {"available": True, "value": False},
                "extinction_active": {"available": True, "value": False},
                "extinction_model": {"available": True, "value": "N/A"},
            },
        },
        "realized": {
            "execution_completed": True,
            "completed_scan_count": {"available": True, "value": 4},
            "flagged_sample_count": {"available": False},
            "dynamic_notch_count": {"available": False},
            "required_timestream_write_count": {
                "available": True,
                "value": 13,
            },
        },
    }


def valid_raw_v2_document() -> dict:
    document = valid_raw_document()
    document["schema_version"] = "citlali-raw-timestream-provenance-v2"
    interface_sync = {
        "unit": "s",
        "offsets": {
            **{f"toltec{index}": 0.0 for index in range(13)},
            "hwpr": 0.0,
        },
    }
    document["requested"]["interface_sync_offset"] = interface_sync
    document["effective"]["config"]["interface_sync_offset"] = {
        "unit": interface_sync["unit"],
        "offsets": dict(interface_sync["offsets"]),
    }
    return document


def valid_raw_v3_document() -> dict:
    document = valid_raw_v2_document()
    document["schema_version"] = "citlali-raw-timestream-provenance-v3"
    document["requested"]["calibration"] = {
        "reference_spectral_index_alpha": {
            "available": False,
        },
    }
    document["effective"]["config"]["calibration"] = {
        "reference_spectral_index_alpha": 0.0,
        "reference_spectral_index_default_applied": True,
    }
    calibration_values = {
        "tau225": 0.12,
        "reference_spectral_index_alpha": 0.0,
        "reference_spectral_index_default_applied": True,
        "atmosphere_operator_id":
            "am12_fixed_djf25_piecewise_linear_los_tau_v1",
        "atmosphere_operator_contract_sha256":
            "7a064ff768a3de4f427f1338d94ef6cb9026d248f3c3c816fc3dfc96d156e36a",
        "atmosphere_node_table_sha256":
            "fd688a4cd3f46585b08631bc63a562aed482feb9b24ec9ee0071b70db7eb8a5f",
        "passband_set_id":
            "toltec-passband-set-v1:sha256:5e6f38f14bcae93a29ffe8362c52b15209f51aee4e48373b23aaa5ec2f8a6433",
        "reference_profile_id":
            "LMT_DJF_25.amc:sha256:aeeeeb48bef422f2d9392b5d7a3d62ab1887fd9e7c10322d5246d914841ba866",
        "calibration_quality_regime": "science_qualification_regime",
        "calibration_valid": True,
        "calibration_validity_reason": "valid",
    }
    observation = document["observation"]["value"]
    observation["extinction_active"] = {"available": True, "value": True}
    observation["extinction_model"] = {
        "available": True,
        "value": calibration_values["atmosphere_operator_id"],
    }
    for name, value in calibration_values.items():
        observation[name] = {"available": True, "value": value}
        document["realized"][name] = {"available": True, "value": value}
    return document


def valid_output_document() -> dict:
    return {
        "schema_version": "citlali-timestream-output-provenance-v1",
        "requested": {},
        "effective": {},
        "realized": {"n_scans": 4},
    }


def valid_mapmaking_document() -> dict:
    requested = {
        "enabled": True,
        "grouping": "auto",
        "cunit": "mJy/beam",
    }
    effective = {
        "enabled": True,
        "grouping": "array",
        "cunit": "mJy/beam",
        "method": "jinc",
    }
    return {
        "schema_version": "citlali-mapmaking-provenance-v2",
        "initialized": True,
        "requested": requested,
        "effective": {
            "config": effective,
            "resolution": {
                "reduction_type": "science",
                "requested_grouping": "auto",
                "effective_grouping": "array",
                "automatic_grouping_resolved": True,
                "detector_grouping_fell_back_to_array": False,
                "requested_unit": "mJy/beam",
                "effective_unit": "mJy/beam",
                "uncalibrated_unit_substituted": False,
            },
        },
        "observations": [
            {
                "observation_index": 0,
                "obsnum": 152389,
                "map_count": 3,
                "effective_pixel_size_rad": 9.696273622e-6,
                "required_map_write_count": 3,
                "outputs_completed": True,
            },
            {
                "observation_index": 1,
                "obsnum": 152390,
                "map_count": 3,
                "effective_pixel_size_rad": 9.696273622e-6,
                "required_map_write_count": 3,
                "outputs_completed": True,
            },
        ],
        "coadd": {
            "available": True,
            "map_count": 3,
            "required_map_write_count": 6,
            "outputs_completed": True,
        },
        "realized": {
            "reduction_completed": True,
            "mapmaking_executed": True,
            "completed_observation_count": {"available": True, "value": 2},
            "completed_coadd_count": {"available": True, "value": 1},
        },
    }


def valid_mapmaking_v1_document() -> dict:
    document = valid_mapmaking_document()
    document["schema_version"] = "citlali-mapmaking-provenance-v1"
    document["observation"] = {"available": False}
    document.pop("observations")
    document.pop("coadd")
    document["realized"]["completed_observation_count"] = {
        "available": False
    }
    document["realized"]["completed_coadd_count"] = {"available": False}
    return document


def valid_coadd_document(enabled: bool = True) -> dict:
    return {
        "schema_version": "citlali-coadd-provenance-v1",
        "initialized": True,
        "requested": {"enabled": enabled},
        "effective": {
            "config": {"enabled": enabled},
            "resolution": {
                "mapmaking_enabled": True,
                "requested_enabled": enabled,
                "effective_enabled": enabled,
                "disabled_by_mapmaking": False,
            },
        },
        "realized": {
            "reduction_completed": True,
            "coadd_executed": enabled,
            "map_count": (
                {"available": True, "value": 3}
                if enabled
                else {"available": False}
            ),
            "required_map_write_count": (
                {"available": True, "value": 6}
                if enabled
                else {"available": False}
            ),
            "outputs_completed": enabled,
        },
    }


def valid_noise_document(enabled: bool = True) -> dict:
    requested_count = 10
    effective_count = requested_count if enabled else 0
    requested = {
        "enabled": enabled,
        "n_noise_maps": requested_count,
        "randomize_dets": False,
        "write_realizations": False,
        "products": {
            "enabled": False,
            "apply_empirical_weights": True,
        },
    }
    effective = {
        **requested,
        "n_noise_maps": effective_count,
        "products": dict(requested["products"]),
    }
    expected_values = {
        "noise_maps_per_scientific_map": effective_count,
        "observation_scientific_map_count": 6 if enabled else 0,
        "observation_noise_realization_count": 60 if enabled else 0,
        "coadd_scientific_map_count": 3 if enabled else 0,
        "coadd_noise_realization_count": 30 if enabled else 0,
        "total_noise_realization_count": 90 if enabled else 0,
        "empirical_product_map_count": 0,
        "realization_image_write_count": 0,
    }
    realized = {
        "reduction_completed": True,
        "generation_executed": enabled,
        **{
            name: {"available": True, "value": value}
            for name, value in expected_values.items()
        },
        "actual_completion_valid": True,
        "completed_count_matches_effective": True,
        "uncertainty_use_valid": enabled and effective_count >= 2,
        "completion_basis": (
            "observed_successful_publication_lifecycle"
            if enabled else "effective_disabled_zero_work"
        ),
        "outputs_completed": True,
    }
    return {
        "schema_version": "citlali-noise-products-provenance-v1",
        "initialized": True,
        "requested": requested,
        "effective": {
            "config": effective,
            "resolution": {
                "mapmaking_enabled": True,
                "requested_enabled": enabled,
                "effective_enabled": enabled,
                "disabled_by_mapmaking": False,
                "requested_n_noise_maps": requested_count,
                "effective_n_noise_maps": effective_count,
                "count_zeroed_while_disabled": not enabled,
                "randomization": {
                    "engine": "boost::random::mt19937",
                    "seed": 5489,
                    "seed_policy": "fixed_internal_default",
                    "generator_scope": "reduction_pipeline_invocation",
                },
            },
        },
        "expected": {"initialized": True, **expected_values},
        "realized": realized,
        "package": {
            "package_id": "citlali-noise-products",
            "provenance_id": "noise_products_provenance.yaml",
            "product_contract_version": "SCI-NOI-002-v1",
            "authority": "package_sidecar",
            "detached_product_status": "unverified_out_of_contract",
            "product_contract_inventory": [
                {
                    "product_identity": identity,
                    "product_version": "SCI-NOI-002-v1",
                    "semantic_digest": (
                        audit.noise_product_semantic_digest(identity)
                    ),
                    "digest_kind": "semantic_contract_sha256",
                    "scope": scope,
                    "restriction": restriction,
                }
                for identity, (scope, restriction) in sorted(
                    audit.NOISE_PRODUCT_CONTRACTS.items()
                )
            ],
            "member_files": [],
            "member_count": 0,
            "member_inventory_digest": audit.noise_member_inventory_digest_v2([]),
            "member_inventory_digest_kind": "sha256",
            "member_inventory_preimage_encoding": (
                "canonical_length_prefixed_member_records_v2"
            ),
            "publication_state": "complete",
            "complete": True,
        },
    }


def refresh_noise_member_inventory(redu: Path, document: dict) -> None:
    for member in document["package"]["member_files"]:
        identity = member["member_product_identity"]
        path = redu / identity
        digest = audit.sha256_file(path)
        size = path.stat().st_size
        member["sha256"] = digest
        member["size_bytes"] = size
    document["package"]["member_inventory_digest"] = (
        audit.noise_member_inventory_digest_v2(
            document["package"]["member_files"]
        )
    )


def valid_noise_fits_join(
    identity: str,
    scope: str,
    validity: str,
    restriction: str,
) -> dict[str, str]:
    return {
        "NOIPKG": "citlali-noise-products",
        "NOIPROV": "noise_products_provenance.yaml",
        "NOIPRID": identity,
        "NOIPVER": "SCI-NOI-002-v1",
        "NOIDGST": audit.noise_product_semantic_digest(identity),
        "NOIDGKND": "semantic_contract_sha256",
        "NOISCOPE": scope,
        "NOIVALID": validity,
        "NOIRESTR": restriction,
        "NOIMISS": "nonfinite_unavailable",
    }


def write_valid_noise_fits(
    path: Path, logical_maps: tuple[str, ...] = ("I",)
) -> list[str]:
    import numpy as np
    from astropy.io import fits

    joins = []
    for logical_map in logical_maps:
        joins.extend(
            (
                (
                    f"weight_formal_{logical_map}",
                    valid_noise_fits_join(
                        "formal_nonprecision_coefficient_snapshot",
                        "raw_map_pixel",
                        "available",
                        "nonprecision_snapshot_not_inverse_variance",
                    ),
                ),
                (
                    f"noise_variance_{logical_map}",
                    valid_noise_fits_join(
                        "conditional_finite_stack_scatter",
                        "raw_map_pixel",
                        "conditional_descriptive",
                        (
                            "retained_legacy_name_not_physical_noise_"
                            "variance_or_covariance"
                        ),
                    ),
                ),
            )
        )
    images = []
    for extname, join in joins:
        image = fits.ImageHDU(
            np.ones((1, 1), dtype=float), name=extname
        )
        for key, value in join.items():
            image.header[key] = value
        images.append(image)
    fits.HDUList([fits.PrimaryHDU(), *images]).writeto(path, overwrite=True)
    return sorted({join["NOIPRID"] for _, join in joins})


def write_valid_noise_realization_fits(
    path: Path, logical_maps: tuple[str, ...] = ("I",), count: int = 2
) -> list[str]:
    import numpy as np
    from astropy.io import fits

    identity = "source_imprinted_current_realization"
    images = []
    for logical_map in logical_maps:
        stokes_separator = logical_map.rfind("_")
        for ordinal in range(count):
            extname = (
                f"signal_{ordinal}_{logical_map}"
                if stokes_separator < 0
                else "signal_"
                + logical_map[:stokes_separator]
                + f"_{ordinal}"
                + logical_map[stokes_separator:]
            )
            image = fits.ImageHDU(
                np.ones((1, 1), dtype=float), name=extname
            )
            join = valid_noise_fits_join(
                identity,
                f"realization_map_index_{ordinal}",
                "conditional_design_member",
                "source_imprinted_current_not_physical_noise_repeat",
            )
            for key, value in join.items():
                image.header[key] = value
            images.append(image)
    fits.HDUList([fits.PrimaryHDU(), *images]).writeto(path, overwrite=True)
    return [identity]


def write_valid_scaled_coadd_fits(path: Path) -> list[str]:
    import numpy as np
    from astropy.io import fits

    identity = "global_nonprecision_scaled_coefficient"
    image = fits.ImageHDU(np.ones((1, 1), dtype=float), name="weight_I")
    join = valid_noise_fits_join(
        identity,
        "raw_map_pixel",
        "available",
        "existing_use_only_nonprecision_not_precision",
    )
    for key, value in join.items():
        image.header[key] = value
    fits.HDUList([fits.PrimaryHDU(), image]).writeto(path, overwrite=True)
    return [identity]


def write_valid_source_ecsv(path: Path) -> str:
    from astropy.table import Table

    source_identity = "fitted_amplitude_over_full_map_rms_ratio"
    source_table = Table({"array": [0], "sig2noise": [1.0]})
    source_table.meta["noise_product_contract"] = {
        "package_id": "citlali-noise-products",
        "provenance_id": "noise_products_provenance.yaml",
        "column": "sig2noise",
        "product_identity": source_identity,
        "product_version": "SCI-NOI-002-v1",
        "semantic_digest": audit.noise_product_semantic_digest(
            source_identity
        ),
        "digest_kind": "semantic_contract_sha256",
        "missingness": "nonfinite_unavailable",
        "scope": "source_table_row",
        "validity": "finite_amplitude_and_finite_positive_full_map_rms",
        "restriction": "legacy_alias_deprecated_not_significance",
    }
    source_table.write(path, format="ascii.ecsv", overwrite=True)
    return source_identity


def valid_noise_netcdf_contracts() -> dict[str, tuple[str, str, str]]:
    return {
        "map_noise_weight_median_ratio": (
            "global_nonprecision_scale_diagnostic",
            "available_when_finite_positive_calibration_support_exists",
            "engineering_scale_diagnostic_not_precision_or_significance",
        ),
        "map_noise_weight_scale": (
            "global_nonprecision_scale_diagnostic",
            "available_when_finite_positive_median_ratio_exists",
            "nonprecision_scale_not_inverse_variance_or_precision",
        ),
        "map_noise_products_s2n_sigma": (
            "pooled_stack_scale_diagnostic",
            "available_when_finite_pooled_stack_scale_exists",
            "engineering_scale_diagnostic_not_calibrated_significance",
        ),
    }


def write_valid_noise_netcdf(path: Path) -> list[str]:
    from netCDF4 import Dataset

    variables = valid_noise_netcdf_contracts()
    with Dataset(path, "w") as dataset:
        dataset.createDimension("n_maps", 1)
        for name, (product_identity, validity, restriction) in (
            variables.items()
        ):
            variable = dataset.createVariable(name, "f8", ("n_maps",))
            variable.comment = (
                "fixture; "
                + audit.noise_netcdf_join_record(
                    name, product_identity, validity, restriction
                )
            )
            variable[:] = [1.0]
    return sorted({contract[0] for contract in variables.values()})


def enable_single_map_empirical_noise(document: dict) -> None:
    document["requested"]["products"]["enabled"] = True
    document["effective"]["config"]["products"]["enabled"] = True
    counts = {
        "observation_scientific_map_count": 1,
        "observation_noise_realization_count": 10,
        "coadd_scientific_map_count": 0,
        "coadd_noise_realization_count": 0,
        "total_noise_realization_count": 10,
        "empirical_product_map_count": 1,
    }
    for name, value in counts.items():
        document["expected"][name] = value
        document["realized"][name]["value"] = value


def configure_enabled_noise_counts(
    document: dict,
    *,
    observation_maps: int,
    coadd_maps: int,
    empirical_maps: int,
    realization_writes: int,
    n_noise_maps: int = 2,
) -> None:
    document["requested"]["enabled"] = True
    document["requested"]["n_noise_maps"] = n_noise_maps
    document["requested"]["write_realizations"] = True
    document["requested"]["products"]["enabled"] = True
    effective = document["effective"]["config"]
    effective["enabled"] = True
    effective["n_noise_maps"] = n_noise_maps
    effective["write_realizations"] = True
    effective["products"]["enabled"] = True
    resolution = document["effective"]["resolution"]
    resolution["requested_enabled"] = True
    resolution["effective_enabled"] = True
    resolution["requested_n_noise_maps"] = n_noise_maps
    resolution["effective_n_noise_maps"] = n_noise_maps
    resolution["count_zeroed_while_disabled"] = False
    counts = {
        "noise_maps_per_scientific_map": n_noise_maps,
        "observation_scientific_map_count": observation_maps,
        "observation_noise_realization_count": (
            observation_maps * n_noise_maps
        ),
        "coadd_scientific_map_count": coadd_maps,
        "coadd_noise_realization_count": coadd_maps * n_noise_maps,
        "total_noise_realization_count": (
            (observation_maps + coadd_maps) * n_noise_maps
        ),
        "empirical_product_map_count": empirical_maps,
        "realization_image_write_count": realization_writes,
    }
    for name, value in counts.items():
        document["expected"][name] = value
        document["realized"][name]["value"] = value
    document["realized"]["generation_executed"] = True
    document["realized"]["uncertainty_use_valid"] = True


def add_valid_noise_member_inventory(redu: Path, document: dict) -> list[Path]:
    enable_single_map_empirical_noise(document)
    fits_path = redu / "map.fits"
    ecsv_path = redu / "sources.ecsv"
    netcdf_path = redu / "noise.nc"
    fits_identities = write_valid_noise_fits(fits_path)
    source_identity = write_valid_source_ecsv(ecsv_path)
    netcdf_identities = write_valid_noise_netcdf(netcdf_path)

    identities = {
        "map.fits": fits_identities,
        "noise.nc": netcdf_identities,
        "sources.ecsv": [source_identity],
    }
    kinds = {".fits": "fits", ".nc": "netcdf", ".ecsv": "ecsv"}
    members = []
    paths = sorted((fits_path, netcdf_path, ecsv_path), key=lambda path: path.name)
    for path in paths:
        members.append(
            {
                "member_product_identity": path.name,
                "member_kind": kinds[path.suffix],
                "joined_product_identities": identities[path.name],
                "digest_kind": "file_sha256",
                "detached_status": (
                    "unverified_out_of_contract_without_package"
                ),
            }
        )
    document["package"]["member_files"] = members
    document["package"]["member_count"] = len(members)
    refresh_noise_member_inventory(redu, document)
    return paths


def set_noise_member_inventory(
    redu: Path,
    document: dict,
    entries: list[tuple[Path, str, list[str]]],
) -> None:
    document["package"]["member_files"] = [
        {
            "member_product_identity": path.name,
            "member_kind": kind,
            "joined_product_identities": identities,
            "digest_kind": "file_sha256",
            "detached_status": (
                "unverified_out_of_contract_without_package"
            ),
        }
        for path, kind, identities in sorted(
            entries, key=lambda entry: entry[0].name
        )
    ]
    document["package"]["member_count"] = len(entries)
    refresh_noise_member_inventory(redu, document)


def valid_pointing_document(enabled: bool = True) -> dict:
    requested = {
        "source_strategy": "standard",
        "fit_gaussian": True,
        "fruitloops_center_mode": "auto",
        "header_max_radius_arcsec": 0.0,
        "header_require_coverage": True,
    }
    effective = {
        **requested,
        "fit_gaussian": enabled,
        "header_max_radius_arcsec": 30.0,
    }
    observations = [
        {
            "observation_index": index,
            "obsnum": obsnum,
            "map_count": 3,
            "raw_fit_attempt_count": 3,
            "raw_valid_fit_count": valid_count,
            "raw_fit_results_recorded": True,
            "filtered_fit_attempt_count": 3,
            "filtered_valid_fit_count": valid_count,
            "filtered_fit_results_recorded": True,
            "outputs_completed": True,
        }
        for index, (obsnum, valid_count) in enumerate(
            (("152389", 2), ("152390", 3))
        )
    ] if enabled else []
    return {
        "schema_version": "citlali-pointing-provenance-v2",
        "initialized": True,
        "requested": requested,
        "effective": {
            "config": effective,
            "resolution": {
                "mapmaking_enabled": enabled,
                "map_filter_enabled": True,
                "coadd_enabled": False,
                "fit_output_path_available": enabled,
                "explicit_request": {
                    "source_strategy": False,
                    "fit_gaussian": False,
                    "fruitloops_center_mode": False,
                    "header_max_radius_arcsec": False,
                    "header_require_coverage": False,
                },
                "fit_disabled_by_mapmaking": not enabled,
                "fit_disabled_by_output_policy": False,
                "default_header_max_radius_arcsec": 30.0,
                "header_max_radius_defaulted": True,
            },
        },
        "observations": observations,
        "realized": {
            "reduction_completed": True,
            "pointing_executed": enabled,
            "completed_observation_count": len(observations),
            "scientific_map_count": 3 * len(observations),
            "raw_fit_attempt_count": 3 * len(observations),
            "raw_valid_fit_count": 5 if enabled else 0,
            "filtered_fit_attempt_count": 3 * len(observations),
            "filtered_valid_fit_count": 5 if enabled else 0,
            "outputs_completed": True,
        },
    }


def valid_pointing_v1_document() -> dict[str, object]:
    document = valid_pointing_document()
    document["schema_version"] = "citlali-pointing-provenance-v1"
    observations = document["observations"]
    assert isinstance(observations, list)
    for observation in observations:
        assert isinstance(observation, dict)
        observation["fit_attempt_count"] = observation.pop(
            "raw_fit_attempt_count"
        )
        observation["valid_fit_count"] = observation.pop(
            "raw_valid_fit_count"
        )
        observation["fit_results_recorded"] = observation.pop(
            "raw_fit_results_recorded"
        )
        observation.pop("filtered_fit_attempt_count")
        observation.pop("filtered_valid_fit_count")
        observation.pop("filtered_fit_results_recorded")
    realized = document["realized"]
    assert isinstance(realized, dict)
    realized["fit_attempt_count"] = realized.pop(
        "raw_fit_attempt_count"
    )
    realized["valid_fit_count"] = realized.pop("raw_valid_fit_count")
    realized.pop("filtered_fit_attempt_count")
    realized.pop("filtered_valid_fit_count")
    return document


def valid_beammap_document(
    enabled: bool = True, detector_tod_enabled: bool = True,
) -> dict[str, object]:
    observations = []
    if enabled:
        iterations = []
        phases = ("locator", "measurement_start", "measurement")
        for index, phase in enumerate(phases):
            terminal = index == len(phases) - 1
            iterations.append(
                {
                    "iteration_index": index,
                    "phase": phase,
                    "active_map_count": 5234,
                    "mapmaking_pass_count": 1,
                    "source_aware_rtc_rerun": {
                        "available": True,
                        "value": index == 1,
                    },
                    "fitting_completed": True,
                    "newly_converged_map_count": 0,
                    "total_converged_map_count": 0,
                    "termination_reason": (
                        "maximum_iterations" if terminal else "none"
                    ),
                    "completed": True,
                }
            )
        observations.append(
            {
                "observation_index": 0,
                "obsnum": 148670,
                "source_identity_authority": "telescope_data",
                "photometry": {
                    "calibrator_flux_authority": "tolproj",
                    "flux_input_path": "beammap_source.fluxes",
                    "required_flux_policy": "fail_reduction",
                    "fluxes": [
                        {
                            "array_name": "a1100",
                            "value_mJy": 1000.0,
                            "uncertainty_mJy": 10.0,
                        },
                        {
                            "array_name": "a1400",
                            "value_mJy": 900.0,
                            "uncertainty_mJy": 9.0,
                        },
                        {
                            "array_name": "a2000",
                            "value_mJy": 800.0,
                            "uncertainty_mJy": 8.0,
                        },
                    ],
                },
                "detector_count": 5234,
                "map_count": 5234,
                "scan_count": 198,
                "iterations": iterations,
                "terminal_iteration": {"available": True, "value": 2},
                "termination_reason": "maximum_iterations",
                "detector_tod": {
                    "required": detector_tod_enabled,
                    "completed_write_count": (
                        1 if detector_tod_enabled else 0
                    ),
                    "output_iteration": {
                        "available": detector_tod_enabled,
                        **({"value": 2} if detector_tod_enabled else {}),
                    },
                    "detector_count": {
                        "available": detector_tod_enabled,
                        **(
                            {"value": 5234}
                            if detector_tod_enabled
                            else {}
                        ),
                    },
                    "slot_count": {
                        "available": detector_tod_enabled,
                        **({"value": 20} if detector_tod_enabled else {}),
                    },
                    "maximum_sample_count": {
                        "available": detector_tod_enabled,
                        **({"value": 788} if detector_tod_enabled else {}),
                    },
                },
                "outputs_completed": True,
            }
        )
    return {
        "schema_version": "citlali-beammap-provenance-v2",
        "initialized": True,
        "requested": {
            "iter_max": 3,
            "detector_tod_output": {"enabled": detector_tod_enabled},
        },
        "effective": {
            "config": {
                "iter_max": 3 if enabled else 1,
                "detector_tod_output": {"enabled": detector_tod_enabled},
            },
            "resolution": {
                "mapmaking_enabled": enabled,
                "requested_max_iterations": 3,
                "effective_max_iterations": 3 if enabled else 1,
            },
        },
        "observations": observations,
        "realized": {
            "reduction_completed": True,
            "beammap_executed": enabled,
            "completed_observation_count": {
                "available": True,
                "value": len(observations),
            },
            "completed_iteration_count": 3 if enabled else 0,
            "outputs_completed": True,
        },
    }


def valid_beammap_mapmaking_document() -> dict[str, object]:
    document = valid_mapmaking_document()
    document["requested"]["grouping"] = "detector"
    document["effective"]["config"]["grouping"] = "detector"
    resolution = document["effective"]["resolution"]
    resolution.update(
        {
            "reduction_type": "beammap",
            "requested_grouping": "detector",
            "effective_grouping": "detector",
            "automatic_grouping_resolved": False,
        }
    )
    document["observations"] = [
        {
            "observation_index": 0,
            "obsnum": 148670,
            "map_count": 5234,
            "effective_pixel_size_rad": 4.848136811e-6,
            "required_map_write_count": 5234,
            "outputs_completed": True,
        }
    ]
    document["coadd"] = {"available": False}
    document["realized"]["completed_observation_count"] = {
        "available": True,
        "value": 1,
    }
    document["realized"]["completed_coadd_count"] = {
        "available": True,
        "value": 0,
    }
    return document


def valid_post_processing_document(
    reduction_type: str = "science",
) -> dict[str, object]:
    pointing = reduction_type == "pointing"
    beammap = reduction_type == "beammap"
    coadd = reduction_type == "science"
    observation_contexts = 2 if pointing else 0
    observation_maps = 6 if pointing else 0
    coadd_contexts = 1 if coadd else 0
    coadd_maps = 3 if coadd else 0

    def map_context(contexts: int, maps: int, sources: int) -> dict:
        return {
            "filter_context_count": contexts,
            "filtered_map_count": maps,
            "source_finding_context_count": contexts,
            "detected_source_count": sources,
            "source_table_write_count": contexts,
            "source_table_row_count": sources,
            "catalog_fits": {
                "context_count": contexts,
                "attempt_count": sources,
                "valid_count": max(0, sources - 1),
            },
        }

    requested = {
        "map_filtering": {"enabled": not beammap},
        "source_finding": {"enabled": not beammap},
        "source_fitting": {"active": False},
    }
    effective = {
        "map_filtering": {"enabled": not beammap},
        "source_finding": {"enabled": not beammap},
        "source_fitting": {"active": True},
    }
    return {
        "schema_version": "citlali-post-processing-provenance-v1",
        "initialized": True,
        "requested": requested,
        "effective": {
            "values": effective,
            "resolution": {
                "reduction_type": reduction_type,
                "mapmaking_enabled": True,
                "coadd_enabled": coadd,
                "map_filtering_requested": not beammap,
                "map_filtering_effective": not beammap,
                "map_filtering_disabled_by_mapmaking": False,
                "source_finding_requested": not beammap,
                "source_finding_effective": not beammap,
                "source_finding_disabled_by_mapmaking": False,
                "source_fitting_required_by_reduction": pointing or beammap,
                "source_fitting_required_by_map_filtering": not beammap,
                "source_fitting_required_by_source_finding": not beammap,
                "source_fitting_effective": True,
                "source_fitting_disabled_by_mapmaking": False,
            },
        },
        "realized": {
            "reduction_completed": True,
            "observation": map_context(
                observation_contexts, observation_maps,
                5 if pointing else 0,
            ),
            "coadd": map_context(
                coadd_contexts, coadd_maps, 4 if coadd else 0,
            ),
            "pointing_fits": {
                "raw": {
                    "context_count": 2 if pointing else 0,
                    "attempt_count": 6 if pointing else 0,
                    "valid_count": 5 if pointing else 0,
                },
                "filtered": {
                    "context_count": 2 if pointing else 0,
                    "attempt_count": 6 if pointing else 0,
                    "valid_count": 5 if pointing else 0,
                },
            },
            "beammap_fits": {
                "context_count": 3 if beammap else 0,
                "attempt_count": 15407 if beammap else 0,
                "valid_count": 15407 if beammap else 0,
            },
            "outputs_completed": True,
        },
    }


def valid_kids_external_document() -> dict:
    values = {
        "fitter": {
            "modelspec": "gainlintrend",
            "weight_window": {"type": "lorentz", "fwhm_Hz": 15000.0},
        },
        "solver": {
            "fitreportdir": "/tmp/fits",
            "parallel_policy": "seq",
            "extra_output": False,
        },
    }
    return {
        "schema_version": "citlali-kids-external-provenance-v1",
        "initialized": True,
        "authority": "kidscpp",
        "config_schema": "citlali-kidscpp-bridge-v1",
        "data_schema": "toltec.1",
        "dependency": {"name": "kidscpp", "version": "04088da"},
        "supported_tod_types": ["xs", "rs", "is", "qs"],
        "selected_tod_type": "xs",
        "requested": {
            "values": values,
            "solver_extra_output_present": True,
        },
        "effective": {
            "values": values,
            "resolution": {"solver_extra_output_forced_disabled": False},
        },
    }


def valid_polarimetry_document() -> dict:
    config = {
        "enabled": False,
        "grouping": "fg",
        "ignore_hwpr": "auto",
    }
    return {
        "schema_version": "citlali-polarimetry-provenance-v1",
        "initialized": True,
        "capability": {
            "status": "planned-unavailable",
            "enabled_supported": False,
            "reason": "no approved contract or enabled reference dataset",
            "exit_condition": "approve the contract and pass validation",
        },
        "requested": config,
        "effective": {
            "config": config,
            "capability_resolution": {
                "enabled_capability_available": False,
                "requested_enabled": False,
                "request_accepted": True,
                "disabled_by_capability": False,
            },
        },
        "realized": {
            "reduction_completed": True,
            "polarimetry_executed": False,
            "hwpr_loaded": False,
        },
    }


def valid_astrometry_document() -> dict:
    config = {
        "pointing_offsets": {
            "enabled": True,
            "az_arcsec": [1.0, 2.0],
            "alt_arcsec": [3.0, 4.0],
            "modified_julian_date": [60000.0, 60001.0],
        }
    }
    return {
        "schema_version": "citlali-astrometry-provenance-v1",
        "initialized": True,
        "authority": {
            "calibration_selection": "tolteca",
            "application": "citlali",
            "support_origin_metadata_available": False,
            "configured_values_origin": "upstream-unspecified",
        },
        "identity": {
            "axes": ["az", "alt"],
            "offset_unit": "arcsec",
            "time_support": "modified-julian-date",
            "algorithm": "legacy-citlali-constant-or-linear-v1",
        },
        "contract": {
            "upstream_selection_owner": "tolteca",
            "one_configured_value": "constant",
            "two_values_without_positive_mjd_pair": "observation-span-linear",
            "two_values_with_positive_mjd_pair": "explicit-mjd-linear",
            "explicit_mjd_requires_observation_bracketing": True,
            "extrapolation": "forbidden",
        },
        "expected_observation_count": 1,
        "observations": [
            {
                "observation_index": 0,
                "obsnum": 152389,
                "requested": config,
                "effective": {
                    "config": config,
                    "resolution": {
                        "application_mode": "explicit-mjd-linear",
                        "explicit_mjd_support": True,
                    },
                },
                "realized": {
                    "installation_count": 2,
                    "application_count": 2,
                    "telescope_sample_count": 303,
                },
            }
        ],
        "reduction_completed": True,
    }


def write_valid_config_source_manifest(redu: Path) -> None:
    source = redu / "70_reduce.yaml"
    merged = redu / "citlali_merged_config.yaml"
    source.write_text("value: 1\n", encoding="utf-8")
    merged.write_text("value: 1\n", encoding="utf-8")
    document = {
        "schema_version": "citlali-config-source-manifest-v1",
        "merge_authority": "citlali_cli",
        "merge_semantics": "ordered_later_sources_override",
        "upstream": {
            "authority": "tolteca",
            "ordered_sources_provided": False,
        },
        "sources": [
            {
                "precedence": 0,
                "role": "citlali_cli_config",
                "source_path": "/upstream/70_reduce.yaml",
                "copied_filename": source.name,
                "size_bytes": source.stat().st_size,
                "sha256": audit.sha256_file(source),
            }
        ],
        "merged": {
            "snapshot_filename": merged.name,
            "serialization": "yaml_cpp_dump",
            "size_bytes": merged.stat().st_size,
            "sha256": audit.sha256_file(merged),
        },
    }
    (redu / "config_source_manifest.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )


class ProvenanceAuditTest(unittest.TestCase):
    def test_accepts_required_runtime_v2_provenance(self) -> None:
        document = {
            "schema_version": "citlali-runtime-provenance-v2",
            "initialized": True,
            "requested": {"n_threads": 6},
            "effective": {"values": {"n_threads": 6}},
            "realized": {"threads": {"omp": 6}},
        }
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            (redu / "runtime_provenance.yaml").write_text(
                yaml.safe_dump(document, sort_keys=False),
                encoding="utf-8",
            )

            record = audit.audit_provenance_sidecars(
                redu, require_runtime=True
            )["runtime"]

            self.assertTrue(record["required"])
            self.assertTrue(record["valid"])

    def test_rejects_missing_required_runtime_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            record = audit.audit_provenance_sidecars(
                Path(directory), require_runtime=True
            )["runtime"]

            self.assertTrue(record["required"])
            self.assertFalse(record["valid"])

    def test_required_runtime_provenance_rejects_v1(self) -> None:
        document = {
            "schema_version": "citlali-runtime-provenance-v1",
            "initialized": True,
            "requested": {"n_threads": 6},
            "effective": {"values": {"n_threads": 6}},
            "realized": {"threads": {"omp": 6}},
        }
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            (redu / "runtime_provenance.yaml").write_text(
                yaml.safe_dump(document, sort_keys=False),
                encoding="utf-8",
            )

            record = audit.audit_provenance_sidecars(
                redu, require_runtime=True
            )["runtime"]

            self.assertFalse(record["schema_ok"])
            self.assertFalse(record["valid"])

    def test_accepts_complete_astrometry_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            (redu / "astrometry_provenance.yaml").write_text(
                yaml.safe_dump(valid_astrometry_document(), sort_keys=False),
                encoding="utf-8",
            )

            record = audit.audit_provenance_sidecars(
                redu, require_astrometry=True
            )["astrometry"]

            self.assertTrue(record["present"])
            self.assertTrue(record["required"])
            self.assertTrue(record["valid"])

    def test_rejects_inconsistent_astrometry_application_mode(self) -> None:
        document = valid_astrometry_document()
        document["observations"][0]["effective"]["resolution"][
            "application_mode"
        ] = "constant"

        self.assertIn(
            "astrometry observation 0 application mode is inconsistent",
            audit.astrometry_provenance_semantic_errors(document),
        )

    def test_rejects_missing_required_astrometry_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            record = audit.audit_provenance_sidecars(
                Path(directory), require_astrometry=True
            )["astrometry"]

            self.assertFalse(record["valid"])

    def test_accepts_disabled_polarimetry_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            (redu / "polarimetry_provenance.yaml").write_text(
                yaml.safe_dump(
                    valid_polarimetry_document(), sort_keys=False
                ),
                encoding="utf-8",
            )

            record = audit.audit_provenance_sidecars(
                redu, require_polarimetry=True
            )["polarimetry"]

            self.assertTrue(record["present"])
            self.assertTrue(record["required"])
            self.assertTrue(record["valid"])

    def test_rejects_executed_unavailable_polarimetry(self) -> None:
        document = valid_polarimetry_document()
        document["realized"]["polarimetry_executed"] = True

        self.assertIn(
            "unavailable polarimetry was executed",
            audit.polarimetry_provenance_semantic_errors(document),
        )

    def test_accepts_required_external_config_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            (redu / "kids_external_provenance.yaml").write_text(
                yaml.safe_dump(
                    valid_kids_external_document(), sort_keys=False
                ),
                encoding="utf-8",
            )
            write_valid_config_source_manifest(redu)

            records = audit.audit_provenance_sidecars(
                redu,
                require_kids_external=True,
                require_config_source_manifest=True,
            )

            self.assertTrue(records["kids_external"]["valid"])
            self.assertTrue(records["config_source_manifest"]["valid"])

    def test_rejects_config_source_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            write_valid_config_source_manifest(redu)
            (redu / "70_reduce.yaml").write_text(
                "value: changed\n", encoding="utf-8"
            )

            record = audit.audit_provenance_sidecars(
                redu, require_config_source_manifest=True
            )["config_source_manifest"]

            self.assertFalse(record["valid"])
            self.assertIn(
                "config source 0 copied file SHA-256 differs",
                record["files"][0]["semantic_errors"],
            )

    def test_rejects_incomplete_kids_tod_contract(self) -> None:
        document = valid_kids_external_document()
        document["supported_tod_types"] = ["xs"]

        self.assertIn(
            "KIDs supported TOD types must be xs, rs, is, qs",
            audit.kids_external_provenance_semantic_errors(document),
        )

    def test_accepts_complete_beammap_provenance_and_cross_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            documents = {
                "mapmaking_provenance.yaml": (
                    valid_beammap_mapmaking_document()
                ),
                "beammap_provenance.yaml": valid_beammap_document(),
                "post_processing_provenance.yaml": (
                    valid_post_processing_document("beammap")
                ),
            }
            for filename, document in documents.items():
                (redu / filename).write_text(
                    yaml.safe_dump(document, sort_keys=False),
                    encoding="utf-8",
                )

            records = audit.audit_provenance_sidecars(
                redu,
                require_mapmaking=True,
                require_post_processing=True,
                require_beammap=True,
            )

            self.assertTrue(records["mapmaking"]["valid"])
            self.assertTrue(records["post_processing"]["valid"])
            self.assertTrue(records["beammap"]["present"])
            self.assertTrue(records["beammap"]["required"])
            self.assertTrue(records["beammap"]["valid"])

    def test_rejects_missing_required_beammap_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records = audit.audit_provenance_sidecars(
                Path(directory), require_beammap=True
            )

            self.assertFalse(records["beammap"]["valid"])
            self.assertFalse(audit.provenance_ok({"provenance": records}))

    def test_accepts_historical_beammap_v1_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            document = valid_beammap_document()
            document["schema_version"] = "citlali-beammap-provenance-v1"
            document["observations"][0].pop("source_identity_authority")
            document["observations"][0].pop("photometry")
            (redu / "beammap_provenance.yaml").write_text(
                yaml.safe_dump(document, sort_keys=False),
                encoding="utf-8",
            )

            beammap = audit.audit_provenance_sidecars(
                redu, require_beammap=True
            )["beammap"]

            self.assertTrue(beammap["valid"])
            self.assertEqual(
                beammap["schema_version"],
                "citlali-beammap-provenance-v1",
            )

    def test_rejects_invalid_beammap_v2_photometry_flux(self) -> None:
        document = valid_beammap_document()
        document["observations"][0]["photometry"]["fluxes"][0][
            "value_mJy"
        ] = 0.0

        self.assertIn(
            "beammap observation 0 photometry flux 0 value must be positive and finite",
            audit.beammap_provenance_semantic_errors(document),
        )

    def test_rejects_incomplete_beammap_detector_tod(self) -> None:
        document = valid_beammap_document()
        document["observations"][0]["detector_tod"][
            "completed_write_count"
        ] = 0

        self.assertIn(
            "beammap observation 0 detector-TOD write cardinality is inconsistent",
            audit.beammap_provenance_semantic_errors(document),
        )

    def test_rejects_beammap_post_processing_iteration_drift(self) -> None:
        post_processing = valid_post_processing_document("beammap")
        post_processing["realized"]["beammap_fits"]["context_count"] = 2

        self.assertEqual(
            audit.beammap_post_processing_cross_check_errors(
                valid_beammap_document(), post_processing
            ),
            [
                "beammap iteration count differs from post-processing fit contexts"
            ],
        )

    def test_accepts_complete_post_processing_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            (redu / "post_processing_provenance.yaml").write_text(
                yaml.safe_dump(
                    valid_post_processing_document(), sort_keys=False
                ),
                encoding="utf-8",
            )

            post_processing = audit.audit_provenance_sidecars(
                redu, require_post_processing=True
            )["post_processing"]

            self.assertTrue(post_processing["present"])
            self.assertTrue(post_processing["required"])
            self.assertTrue(post_processing["valid"])

    def test_rejects_post_processing_source_row_mismatch(self) -> None:
        document = valid_post_processing_document()
        document["realized"]["coadd"]["source_table_row_count"] = 3

        self.assertIn(
            "coadd source-table row count is inconsistent",
            audit.post_processing_provenance_semantic_errors(document),
        )

    def test_cross_checks_post_processing_pointing_and_mapmaking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            mapmaking = valid_mapmaking_document()
            mapmaking["effective"]["resolution"]["reduction_type"] = (
                "pointing"
            )
            mapmaking["coadd"] = {"available": False}
            mapmaking["realized"]["completed_coadd_count"] = {
                "available": True,
                "value": 0,
            }
            documents = {
                "mapmaking_provenance.yaml": mapmaking,
                "pointing_provenance.yaml": valid_pointing_document(),
                "post_processing_provenance.yaml": (
                    valid_post_processing_document("pointing")
                ),
            }
            for filename, document in documents.items():
                (redu / filename).write_text(
                    yaml.safe_dump(document, sort_keys=False),
                    encoding="utf-8",
                )

            records = audit.audit_provenance_sidecars(
                redu,
                require_mapmaking=True,
                require_pointing=True,
                require_post_processing=True,
            )

            self.assertTrue(records["mapmaking"]["valid"])
            self.assertTrue(records["pointing"]["valid"])
            self.assertTrue(records["post_processing"]["valid"])

    def test_rejects_post_processing_pointing_cross_check_drift(self) -> None:
        post_processing = valid_post_processing_document("pointing")
        post_processing["realized"]["pointing_fits"]["raw"][
            "valid_count"
        ] = 4

        self.assertEqual(
            audit.post_processing_pointing_cross_check_errors(
                post_processing, valid_pointing_document()
            ),
            ["post-processing raw fits differ from pointing provenance"],
        )

    def test_accepts_complete_mapmaking_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            (redu / "mapmaking_provenance.yaml").write_text(
                yaml.safe_dump(valid_mapmaking_document(), sort_keys=False),
                encoding="utf-8",
            )

            mapmaking = audit.audit_provenance_sidecars(
                redu, require_mapmaking=True
            )["mapmaking"]

            self.assertTrue(mapmaking["present"])
            self.assertTrue(mapmaking["required"])
            self.assertTrue(mapmaking["valid"])
            self.assertEqual(len(mapmaking["sha256"]), 64)

    def test_accepts_historical_mapmaking_v1_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            (redu / "mapmaking_provenance.yaml").write_text(
                yaml.safe_dump(
                    valid_mapmaking_v1_document(), sort_keys=False
                ),
                encoding="utf-8",
            )

            mapmaking = audit.audit_provenance_sidecars(
                redu, require_mapmaking=True
            )["mapmaking"]

            self.assertTrue(mapmaking["valid"])
            self.assertEqual(
                mapmaking["schema_version"],
                "citlali-mapmaking-provenance-v1",
            )

    def test_rejects_missing_required_mapmaking_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records = audit.audit_provenance_sidecars(
                Path(directory), require_mapmaking=True
            )

            self.assertFalse(records["mapmaking"]["valid"])
            self.assertFalse(audit.provenance_ok({"provenance": records}))

    def test_rejects_inconsistent_mapmaking_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            document = valid_mapmaking_document()
            document["effective"]["config"]["grouping"] = "detector"
            (redu / "mapmaking_provenance.yaml").write_text(
                yaml.safe_dump(document, sort_keys=False),
                encoding="utf-8",
            )

            mapmaking = audit.audit_provenance_sidecars(
                redu, require_mapmaking=True
            )["mapmaking"]

            self.assertFalse(mapmaking["valid"])
            self.assertEqual(
                mapmaking["files"][0]["semantic_errors"],
                [
                    "grouping resolution does not match effective config",
                ],
            )

    def test_rejects_incomplete_mapmaking_reduction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            document = valid_mapmaking_document()
            document["realized"]["reduction_completed"] = False
            (redu / "mapmaking_provenance.yaml").write_text(
                yaml.safe_dump(document, sort_keys=False),
                encoding="utf-8",
            )

            mapmaking = audit.audit_provenance_sidecars(
                redu, require_mapmaking=True
            )["mapmaking"]

            self.assertFalse(mapmaking["valid"])
            self.assertEqual(
                mapmaking["files"][0]["semantic_errors"],
                ["mapmaking reduction is not complete"],
            )

    def test_rejects_inconsistent_mapmaking_cardinality(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            document = valid_mapmaking_document()
            document["realized"]["completed_observation_count"]["value"] = 1
            (redu / "mapmaking_provenance.yaml").write_text(
                yaml.safe_dump(document, sort_keys=False),
                encoding="utf-8",
            )

            mapmaking = audit.audit_provenance_sidecars(
                redu, require_mapmaking=True
            )["mapmaking"]

            self.assertFalse(mapmaking["valid"])
            self.assertIn(
                "completed observation count does not match observations",
                mapmaking["files"][0]["semantic_errors"],
            )

    def test_accepts_digit_string_mapmaking_obsnums(self) -> None:
        document = valid_mapmaking_document()
        document["observations"][0]["obsnum"] = "00152389"

        self.assertEqual(
            audit.mapmaking_cardinality_semantic_errors(document), []
        )

    def test_rejects_duplicate_mapmaking_numeric_identity(self) -> None:
        document = valid_mapmaking_document()
        document["observations"][1]["obsnum"] = "0152389"

        self.assertIn(
            "duplicate mapmaking obsnum: 152389",
            audit.mapmaking_cardinality_semantic_errors(document),
        )

    def test_accepts_complete_coadd_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            (redu / "coadd_provenance.yaml").write_text(
                yaml.safe_dump(valid_coadd_document(), sort_keys=False),
                encoding="utf-8",
            )

            coadd = audit.audit_provenance_sidecars(
                redu, require_coadd=True
            )["coadd"]

            self.assertTrue(coadd["present"])
            self.assertTrue(coadd["required"])
            self.assertTrue(coadd["valid"])

    def test_accepts_effectively_disabled_coadd_provenance(self) -> None:
        document = valid_coadd_document(enabled=False)

        self.assertEqual(
            audit.coadd_provenance_semantic_errors(document), []
        )

    def test_accepts_complete_noise_products_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            (redu / "noise_products_provenance.yaml").write_text(
                yaml.safe_dump(valid_noise_document(), sort_keys=False),
                encoding="utf-8",
            )

            noise = audit.audit_provenance_sidecars(
                redu, require_noise_products=True
            )["noise_products"]

            self.assertTrue(noise["present"])
            self.assertTrue(noise["required"])
            self.assertTrue(noise["valid"])

    def test_accepts_explicit_fits_ecsv_netcdf_noise_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            document = valid_noise_document()
            add_valid_noise_member_inventory(redu, document)
            (redu / "noise_products_provenance.yaml").write_text(
                yaml.safe_dump(document, sort_keys=False),
                encoding="utf-8",
            )

            noise = audit.audit_provenance_sidecars(
                redu, require_noise_products=True
            )["noise_products"]

            self.assertTrue(noise["valid"], noise.get("semantic_errors"))

    def test_successor_coadd_packages_preserve_realizations_only_or_scaled(
        self,
    ) -> None:
        for scaled in (False, True):
            with self.subTest(scaled=scaled), tempfile.TemporaryDirectory() as directory:
                redu = Path(directory)
                document = valid_noise_document()
                configure_enabled_noise_counts(
                    document,
                    observation_maps=0,
                    coadd_maps=1,
                    empirical_maps=0,
                    realization_writes=2,
                )
                document["requested"]["products"][
                    "apply_empirical_weights"
                ] = scaled
                document["effective"]["config"]["products"][
                    "apply_empirical_weights"
                ] = scaled
                realization_path = redu / "coadd_noise.fits"
                realization_identities = write_valid_noise_realization_fits(
                    realization_path
                )
                entries = [
                    (realization_path, "fits", realization_identities)
                ]
                if scaled:
                    scaled_path = redu / "coadd_map.fits"
                    scaled_identities = write_valid_scaled_coadd_fits(
                        scaled_path
                    )
                    entries.append((scaled_path, "fits", scaled_identities))
                set_noise_member_inventory(redu, document, entries)

                self.assertEqual(
                    audit.noise_provenance_semantic_errors(document), []
                )
                self.assertEqual(
                    audit.noise_package_integrity_errors(
                        document,
                        redu / "noise_products_provenance.yaml",
                    ),
                    [],
                )

                falsely_counted = valid_noise_document()
                configure_enabled_noise_counts(
                    falsely_counted,
                    observation_maps=0,
                    coadd_maps=1,
                    empirical_maps=1,
                    realization_writes=2,
                )
                set_noise_member_inventory(redu, falsely_counted, entries)
                semantic_errors = audit.noise_provenance_semantic_errors(
                    falsely_counted
                )
                self.assertIn(
                    "empirical products exist without observation maps",
                    semantic_errors,
                )
                package_errors = audit.noise_package_integrity_errors(
                    falsely_counted,
                    redu / "noise_products_provenance.yaml",
                )
                self.assertIn(
                    "noise package empirical FITS inventory does not match "
                    "observed empirical product maps",
                    package_errors,
                )

    def test_split_beammap_package_shapes_reconcile_logical_maps(self) -> None:
        layouts = (
            (("det_0_I",), ("det_1_I",)),
            (("det_0_I", "det_1_I"),),
        )
        for layout_index, layout in enumerate(layouts):
            with self.subTest(layout=layout_index), tempfile.TemporaryDirectory() as directory:
                from astropy.io import fits

                redu = Path(directory)
                document = valid_noise_document()
                configure_enabled_noise_counts(
                    document,
                    observation_maps=2,
                    coadd_maps=0,
                    empirical_maps=2,
                    realization_writes=4,
                )
                entries: list[tuple[Path, str, list[str]]] = []
                for file_index, logical_maps in enumerate(layout):
                    data_path = redu / f"array{file_index}_map.fits"
                    realization_path = (
                        redu / f"array{file_index}_noise.fits"
                    )
                    entries.extend(
                        (
                            (
                                data_path,
                                "fits",
                                write_valid_noise_fits(
                                    data_path, logical_maps
                                ),
                            ),
                            (
                                realization_path,
                                "fits",
                                write_valid_noise_realization_fits(
                                    realization_path, logical_maps
                                ),
                            ),
                        )
                    )
                empty_path = redu / "unused_array_map.fits"
                fits.HDUList([fits.PrimaryHDU()]).writeto(empty_path)
                set_noise_member_inventory(redu, document, entries)

                self.assertEqual(
                    audit.noise_provenance_semantic_errors(document), []
                )
                self.assertEqual(
                    audit.noise_package_integrity_errors(
                        document,
                        redu / "noise_products_provenance.yaml",
                    ),
                    [],
                )
                self.assertNotIn(
                    empty_path.name,
                    {
                        member["member_product_identity"]
                        for member in document["package"]["member_files"]
                    },
                )

                admitted_empty = valid_noise_document()
                configure_enabled_noise_counts(
                    admitted_empty,
                    observation_maps=2,
                    coadd_maps=0,
                    empirical_maps=2,
                    realization_writes=4,
                )
                set_noise_member_inventory(
                    redu,
                    admitted_empty,
                    [*entries, (empty_path, "fits", [])],
                )
                empty_errors = audit.noise_package_integrity_errors(
                    admitted_empty,
                    redu / "noise_products_provenance.yaml",
                )
                self.assertTrue(
                    any(
                        "admitted FITS member has no noise-product join"
                        in error
                        for error in empty_errors
                    ),
                    empty_errors,
                )

                document["realized"]["empirical_product_map_count"][
                    "value"
                ] = 1
                count_errors = audit.noise_package_integrity_errors(
                    document,
                    redu / "noise_products_provenance.yaml",
                )
                self.assertIn(
                    "noise package empirical FITS inventory does not match "
                    "observed empirical product maps",
                    count_errors,
                )

    def test_disabled_noise_allows_only_non_stack_source_ecsv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            document = valid_noise_document(enabled=False)
            ecsv_path = redu / "sources.ecsv"
            source_identity = write_valid_source_ecsv(ecsv_path)
            document["package"]["member_files"] = [
                {
                    "member_product_identity": ecsv_path.name,
                    "member_kind": "ecsv",
                    "joined_product_identities": [source_identity],
                    "digest_kind": "file_sha256",
                    "detached_status": (
                        "unverified_out_of_contract_without_package"
                    ),
                }
            ]
            document["package"]["member_count"] = 1
            refresh_noise_member_inventory(redu, document)

            self.assertEqual(
                audit.noise_package_integrity_errors(
                    document, redu / "noise_products_provenance.yaml"
                ),
                [],
            )

            fits_path = redu / "map.fits"
            fits_identities = write_valid_noise_fits(fits_path)
            document["package"]["member_files"] = [
                {
                    "member_product_identity": fits_path.name,
                    "member_kind": "fits",
                    "joined_product_identities": fits_identities,
                    "digest_kind": "file_sha256",
                    "detached_status": (
                        "unverified_out_of_contract_without_package"
                    ),
                }
            ]
            refresh_noise_member_inventory(redu, document)
            errors = audit.noise_package_integrity_errors(
                document, redu / "noise_products_provenance.yaml"
            )
            self.assertIn(
                "disabled noise package contains a stack-derived member",
                errors,
            )

            netcdf_path = redu / "noise.nc"
            netcdf_identities = write_valid_noise_netcdf(netcdf_path)
            document["package"]["member_files"] = [
                {
                    "member_product_identity": netcdf_path.name,
                    "member_kind": "netcdf",
                    "joined_product_identities": netcdf_identities,
                    "digest_kind": "file_sha256",
                    "detached_status": (
                        "unverified_out_of_contract_without_package"
                    ),
                }
            ]
            refresh_noise_member_inventory(redu, document)
            errors = audit.noise_package_integrity_errors(
                document, redu / "noise_products_provenance.yaml"
            )
            self.assertIn(
                "disabled noise package contains a stack-derived member",
                errors,
            )

    def test_enabled_noise_reconciles_empirical_identity_cardinality(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            missing = valid_noise_document()
            enable_single_map_empirical_noise(missing)
            errors = audit.noise_package_integrity_errors(
                missing, redu / "noise_products_provenance.yaml"
            )
            self.assertIn(
                "noise package empirical FITS inventory does not match "
                "observed empirical product maps",
                errors,
            )

            extra = valid_noise_document()
            paths = add_valid_noise_member_inventory(redu, extra)
            extra_fits = redu / "map_extra.fits"
            extra_identities = write_valid_noise_fits(extra_fits)
            extra["package"]["member_files"].append(
                {
                    "member_product_identity": extra_fits.name,
                    "member_kind": "fits",
                    "joined_product_identities": extra_identities,
                    "digest_kind": "file_sha256",
                    "detached_status": (
                        "unverified_out_of_contract_without_package"
                    ),
                }
            )
            extra["package"]["member_files"].sort(
                key=lambda member: member["member_product_identity"]
            )
            extra["package"]["member_count"] += 1
            refresh_noise_member_inventory(redu, extra)
            errors = audit.noise_package_integrity_errors(
                extra, redu / "noise_products_provenance.yaml"
            )
            self.assertIn(
                "noise package empirical FITS inventory does not match "
                "observed empirical product maps",
                errors,
            )

            inconsistent = valid_noise_document()
            add_valid_noise_member_inventory(redu, inconsistent)
            inconsistent["realized"]["empirical_product_map_count"][
                "value"
            ] = 2
            errors = audit.noise_package_integrity_errors(
                inconsistent, redu / "noise_products_provenance.yaml"
            )
            self.assertIn(
                "noise package empirical FITS inventory does not match "
                "observed empirical product maps",
                errors,
            )
            self.assertTrue(all(path.exists() for path in paths))

    def test_rejects_duplicate_non_realization_fits_identity(self) -> None:
        from astropy.io import fits

        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            document = valid_noise_document()
            paths = add_valid_noise_member_inventory(redu, document)
            with fits.open(paths[0], mode="update", memmap=False) as hdus:
                hdus.append(hdus[1].copy())
            refresh_noise_member_inventory(redu, document)

            errors = audit.noise_package_integrity_errors(
                document, redu / "noise_products_provenance.yaml"
            )
            self.assertTrue(
                any(
                    "duplicate non-realization FITS noise-product identity"
                    in error
                    for error in errors
                ),
                errors,
            )

    def test_reconciles_repeated_fits_identities_by_logical_map(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split_array.fits"
            write_valid_noise_fits(path, ("det_0_I", "det_1_I"))

            identities, realization_count, empirical_count, stack = (
                audit.fits_noise_member_joins(path)
            )

            self.assertEqual(
                identities,
                [
                    "conditional_finite_stack_scatter",
                    "formal_nonprecision_coefficient_snapshot",
                ],
            )
            self.assertEqual(realization_count, 0)
            self.assertEqual(empirical_count, 2)
            self.assertTrue(stack)

    def test_realization_scopes_restart_only_across_logical_maps(self) -> None:
        import numpy as np
        from astropy.io import fits

        identity = "source_imprinted_current_realization"
        images = []
        for detector in range(2):
            for ordinal in range(2):
                image = fits.ImageHDU(
                    np.ones((1, 1), dtype=float),
                    name=f"signal_det_{detector}_{ordinal}_I",
                )
                values = valid_noise_fits_join(
                    identity,
                    f"realization_map_index_{ordinal}",
                    "conditional_design_member",
                    "source_imprinted_current_not_physical_noise_repeat",
                )
                for key, value in values.items():
                    image.header[key] = value
                images.append(image)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split_realizations.fits"
            fits.HDUList([fits.PrimaryHDU(), *images]).writeto(path)
            _, realization_count, empirical_count, _ = (
                audit.fits_noise_member_joins(path)
            )
            self.assertEqual(realization_count, 4)
            self.assertEqual(empirical_count, 0)

            with fits.open(path, mode="update", memmap=False) as hdus:
                hdus.append(hdus[1].copy())
            with self.assertRaises(ValueError):
                audit.fits_noise_member_joins(path)

    def test_counts_repeated_realization_identity_hdus(self) -> None:
        import numpy as np
        from astropy.io import fits

        identity = "source_imprinted_current_realization"
        images = []
        for ordinal in range(2):
            image = fits.ImageHDU(
                np.ones((1, 1), dtype=float),
                name=f"signal_{ordinal}_I",
            )
            values = {
                "NOIPKG": "citlali-noise-products",
                "NOIPROV": "noise_products_provenance.yaml",
                "NOIPRID": identity,
                "NOIPVER": "SCI-NOI-002-v1",
                "NOIDGST": audit.noise_product_semantic_digest(identity),
                "NOIDGKND": "semantic_contract_sha256",
                "NOISCOPE": f"realization_map_index_{ordinal}",
                "NOIVALID": "conditional_design_member",
                "NOIRESTR": (
                    "source_imprinted_current_not_physical_noise_repeat"
                ),
                "NOIMISS": "nonfinite_unavailable",
            }
            for key, value in values.items():
                image.header[key] = value
            images.append(image)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "realizations.fits"
            fits.HDUList([fits.PrimaryHDU(), *images]).writeto(path)

            identities, realization_count, empirical_count, stack = (
                audit.fits_noise_member_joins(path)
            )

            self.assertEqual(identities, [identity])
            self.assertEqual(realization_count, 2)
            self.assertEqual(empirical_count, 0)
            self.assertTrue(stack)

    def test_fits_noise_join_rejects_wrong_or_missing_exact_fields(self) -> None:
        from astropy.io import fits

        wrong_values = {
            "EXTNAME": "weight_formal_other",
            "NOIDGKND": "file_sha256",
            "NOIMISS": "zero_filled",
            "NOISCOPE": "other_scope",
            "NOIVALID": "other_validity",
            "NOIRESTR": "other_restriction",
            "NOIDGST": "sha256:incorrect",
            "NOIPRID": "source_imprinted_current_realization",
            "NOIPVER": "SCI-NOI-002-v0",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.fits"
            for key, wrong_value in wrong_values.items():
                with self.subTest(key=key, case="missing"):
                    write_valid_noise_fits(path)
                    with fits.open(path, mode="update", memmap=False) as hdus:
                        del hdus[1].header[key]
                    with self.assertRaises(ValueError):
                        audit.fits_noise_member_joins(path)
                with self.subTest(key=key, case="wrong"):
                    write_valid_noise_fits(path)
                    with fits.open(path, mode="update", memmap=False) as hdus:
                        hdus[1].header[key] = wrong_value
                    with self.assertRaises(ValueError):
                        audit.fits_noise_member_joins(path)

    def test_ecsv_noise_join_requires_exact_structured_column_binding(self) -> None:
        from astropy.table import Table

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sources.ecsv"
            write_valid_source_ecsv(path)
            audit.ecsv_noise_member_joins(path)

            table = Table.read(path, format="ascii.ecsv")
            table.meta["noise_product_contract"]["column"] = "amp"
            table.write(path, format="ascii.ecsv", overwrite=True)
            with self.assertRaises(ValueError):
                audit.ecsv_noise_member_joins(path)

            write_valid_source_ecsv(path)
            table = Table.read(path, format="ascii.ecsv")
            del table.meta["noise_product_contract"]["digest_kind"]
            table.write(path, format="ascii.ecsv", overwrite=True)
            with self.assertRaises(ValueError):
                audit.ecsv_noise_member_joins(path)

            write_valid_source_ecsv(path)
            table = Table.read(path, format="ascii.ecsv")
            del table.meta["noise_product_contract"]["missingness"]
            table.write(path, format="ascii.ecsv", overwrite=True)
            with self.assertRaises(ValueError):
                audit.ecsv_noise_member_joins(path)

            for wrong_missingness in ("", "zero_filled"):
                with self.subTest(missingness=wrong_missingness):
                    write_valid_source_ecsv(path)
                    table = Table.read(path, format="ascii.ecsv")
                    table.meta["noise_product_contract"]["missingness"] = (
                        wrong_missingness
                    )
                    table.write(path, format="ascii.ecsv", overwrite=True)
                    with self.assertRaises(ValueError):
                        audit.ecsv_noise_member_joins(path)

            write_valid_source_ecsv(path)
            lines = path.read_text(encoding="utf-8").splitlines(True)
            missingness_line = next(
                line for line in lines if "missingness:" in line
            )
            duplicate_at = lines.index(missingness_line)
            lines.insert(duplicate_at + 1, missingness_line)
            path.write_text("".join(lines), encoding="utf-8")
            with self.assertRaises(ValueError):
                audit.ecsv_noise_member_joins(path)

            write_valid_source_ecsv(path)
            table = Table.read(path, format="ascii.ecsv")
            table.remove_column("sig2noise")
            table.write(path, format="ascii.ecsv", overwrite=True)
            with self.assertRaises(ValueError):
                audit.ecsv_noise_member_joins(path)

    def test_netcdf_noise_join_rejects_missing_duplicate_and_swapped_fields(
        self,
    ) -> None:
        from netCDF4 import Dataset

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "noise.nc"
            write_valid_noise_netcdf(path)
            audit.netcdf_noise_member_joins(path)

            def update_comment(transform) -> None:
                with Dataset(path, "a") as dataset:
                    variable = dataset.variables[
                        "map_noise_weight_median_ratio"
                    ]
                    variable.comment = transform(str(variable.comment))

            update_comment(
                lambda comment: "|".join(
                    token for token in comment.split("|")
                    if not token.startswith("semantic_digest=")
                )
            )
            with self.assertRaises(ValueError):
                audit.netcdf_noise_member_joins(path)

            write_valid_noise_netcdf(path)
            update_comment(
                lambda comment: "|".join(
                    token for token in comment.split("|")
                    if not token.startswith("missingness=")
                )
            )
            with self.assertRaises(ValueError):
                audit.netcdf_noise_member_joins(path)

            for wrong_missingness in ("", "zero_filled"):
                with self.subTest(missingness=wrong_missingness):
                    write_valid_noise_netcdf(path)
                    update_comment(
                        lambda comment: comment.replace(
                            "missingness=nonfinite_unavailable",
                            f"missingness={wrong_missingness}",
                        )
                    )
                    with self.assertRaises(ValueError):
                        audit.netcdf_noise_member_joins(path)

            write_valid_noise_netcdf(path)
            update_comment(
                lambda comment: comment
                + "|missingness=nonfinite_unavailable"
            )
            with self.assertRaises(ValueError):
                audit.netcdf_noise_member_joins(path)

            write_valid_noise_netcdf(path)
            update_comment(lambda comment: comment + "|scope=map_summary")
            with self.assertRaises(ValueError):
                audit.netcdf_noise_member_joins(path)

            write_valid_noise_netcdf(path)
            update_comment(
                lambda comment: comment.replace(
                    "variable=map_noise_weight_median_ratio",
                    "variable=map_noise_weight_scale",
                )
            )
            with self.assertRaises(ValueError):
                audit.netcdf_noise_member_joins(path)

    def test_rejects_tampered_explicit_noise_inventory_member(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            document = valid_noise_document()
            paths = add_valid_noise_member_inventory(redu, document)
            (redu / "noise_products_provenance.yaml").write_text(
                yaml.safe_dump(document, sort_keys=False),
                encoding="utf-8",
            )
            paths[-1].write_text(
                paths[-1].read_text(encoding="utf-8") + "\n# tamper\n",
                encoding="utf-8",
            )

            noise = audit.audit_provenance_sidecars(
                redu, require_noise_products=True
            )["noise_products"]

            self.assertFalse(noise["valid"])
            self.assertIn(
                "noise package member 2 SHA-256 is inconsistent",
                noise["files"][0]["semantic_errors"],
            )

    def test_rejects_symlinked_explicit_noise_inventory_member(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            document = valid_noise_document()
            paths = add_valid_noise_member_inventory(redu, document)
            target = redu / "unadmitted-map-target.fits"
            paths[0].rename(target)
            paths[0].symlink_to(target.name)

            errors = audit.noise_package_integrity_errors(
                document, redu / "noise_products_provenance.yaml"
            )

            self.assertIn("noise package member 0 is a symlink", errors)

    def test_rejects_intermediate_symlink_noise_inventory_member(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            real_dir = redu / "real"
            real_dir.mkdir()
            target = real_dir / "sources.ecsv"
            source_identity = write_valid_source_ecsv(target)
            alias = redu / "alias"
            alias.symlink_to(real_dir, target_is_directory=True)
            document = valid_noise_document(enabled=False)
            document["package"]["member_files"] = [
                {
                    "member_product_identity": "alias/sources.ecsv",
                    "member_kind": "ecsv",
                    "joined_product_identities": [source_identity],
                    "digest_kind": "file_sha256",
                    "detached_status": (
                        "unverified_out_of_contract_without_package"
                    ),
                }
            ]
            document["package"]["member_count"] = 1
            refresh_noise_member_inventory(redu, document)

            errors = audit.noise_package_integrity_errors(
                document, redu / "noise_products_provenance.yaml"
            )

            self.assertTrue(
                any("has a symlink path component" in error for error in errors),
                errors,
            )

    def test_rejects_alternate_lexical_noise_member_spelling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            real_dir = redu / "real"
            real_dir.mkdir()
            target = real_dir / "sources.ecsv"
            source_identity = write_valid_source_ecsv(target)
            document = valid_noise_document(enabled=False)
            document["package"]["member_files"] = [
                {
                    "member_product_identity": "real/../real/sources.ecsv",
                    "member_kind": "ecsv",
                    "joined_product_identities": [source_identity],
                    "sha256": audit.sha256_file(target),
                    "size_bytes": target.stat().st_size,
                    "digest_kind": "file_sha256",
                    "detached_status": (
                        "unverified_out_of_contract_without_package"
                    ),
                }
            ]
            document["package"]["member_count"] = 1
            document["package"]["member_inventory_digest"] = (
                audit.noise_member_inventory_digest_v2(
                    document["package"]["member_files"]
                )
            )

            errors = audit.noise_package_integrity_errors(
                document, redu / "noise_products_provenance.yaml"
            )

            self.assertIn(
                "noise package member 0 path is not normalized and relative",
                errors,
            )

    def test_noise_inventory_v2_is_injective_for_newline_names(self) -> None:
        newline = [
            {
                "member_product_identity": "line\nbreak.fits",
                "sha256": "sha256:aaaaaaaa",
                "size_bytes": 7,
            }
        ]
        plain = [
            {
                "member_product_identity": "linebreak.fits",
                "sha256": "sha256:aaaaaaaa",
                "size_bytes": 7,
            }
        ]

        self.assertNotEqual(
            audit.noise_member_inventory_preimage_v2(newline),
            audit.noise_member_inventory_preimage_v2(plain),
        )
        self.assertNotEqual(
            audit.noise_member_inventory_digest_v2(newline),
            audit.noise_member_inventory_digest_v2(plain),
        )
        self.assertEqual(
            audit.noise_member_inventory_digest_v2(newline),
            "sha256:9fa4aab8f2b41bb83a412019cf8ac158dbf332905cdd0ee711075158ea00863e",
        )

    def test_rejects_noncanonical_noise_inventory_with_matching_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            document = valid_noise_document()
            add_valid_noise_member_inventory(redu, document)
            document["package"]["member_files"].reverse()

            preimage = bytearray(b"citlali-noise-member-inventory-v2|")

            def append_field(value: str) -> None:
                encoded = value.encode("utf-8")
                preimage.extend(str(len(encoded)).encode("ascii"))
                preimage.extend(b":")
                preimage.extend(encoded)

            append_field(str(len(document["package"]["member_files"])))
            for member in document["package"]["member_files"]:
                append_field(member["member_product_identity"])
                append_field(member["sha256"])
                append_field(str(member["size_bytes"]))
            document["package"]["member_inventory_digest"] = (
                "sha256:" + hashlib.sha256(preimage).hexdigest()
            )

            errors = audit.noise_package_integrity_errors(
                document, redu / "noise_products_provenance.yaml"
            )

            self.assertTrue(
                any(
                    "not in canonical lexical order" in error
                    for error in errors
                ),
                errors,
            )

    def test_rejects_partial_fits_noise_product_join(self) -> None:
        from astropy.io import fits

        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            document = valid_noise_document()
            paths = add_valid_noise_member_inventory(redu, document)
            with fits.open(paths[0], mode="update", memmap=False) as hdus:
                del hdus[1].header["NOIRESTR"]
            refresh_noise_member_inventory(redu, document)

            errors = audit.noise_package_integrity_errors(
                document, redu / "noise_products_provenance.yaml"
            )

            self.assertTrue(
                any("partial FITS noise-product join" in error
                    for error in errors),
                errors,
            )

    def test_rejects_member_without_package_contract_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            document = valid_noise_document()
            add_valid_noise_member_inventory(redu, document)
            contracts = document["package"]["product_contract_inventory"]
            document["package"]["product_contract_inventory"] = [
                contract for contract in contracts
                if contract["product_identity"] != (
                    "conditional_finite_stack_scatter"
                )
            ]

            errors = audit.noise_package_integrity_errors(
                document, redu / "noise_products_provenance.yaml"
            )

            self.assertIn(
                "noise package product-contract inventory is incomplete",
                errors,
            )
            self.assertIn(
                "noise package member 0 identity is absent from package "
                "contract inventory",
                errors,
            )

    def test_accepts_effectively_disabled_noise_products(self) -> None:
        self.assertEqual(
            audit.noise_provenance_semantic_errors(
                valid_noise_document(enabled=False)
            ),
            [],
        )

    def test_rejects_disabled_noise_without_available_zero_counters(self) -> None:
        document = valid_noise_document(enabled=False)
        document["realized"]["total_noise_realization_count"] = {
            "available": False
        }

        self.assertIn(
            "total_noise_realization_count is unavailable",
            audit.noise_provenance_semantic_errors(document),
        )

    def test_rejects_disabled_noise_incomplete_or_wrong_basis(self) -> None:
        document = valid_noise_document(enabled=False)
        document["realized"]["outputs_completed"] = False
        document["realized"]["completion_basis"] = "pipeline_return"

        errors = audit.noise_provenance_semantic_errors(document)
        self.assertIn(
            "disabled noise-products zero-work completion is incomplete",
            errors,
        )
        self.assertIn(
            "disabled noise-products completion basis is inconsistent",
            errors,
        )

    def test_rejects_enabled_zero_noise_count(self) -> None:
        document = valid_noise_document()
        document["requested"]["n_noise_maps"] = 0
        document["effective"]["config"]["n_noise_maps"] = 0
        document["effective"]["resolution"]["requested_n_noise_maps"] = 0
        document["effective"]["resolution"]["effective_n_noise_maps"] = 0
        document["expected"]["noise_maps_per_scientific_map"] = 0
        for name in (
            "observation_noise_realization_count",
            "coadd_noise_realization_count",
            "total_noise_realization_count",
        ):
            document["expected"][name] = 0
            document["realized"][name]["value"] = 0
        document["realized"]["noise_maps_per_scientific_map"]["value"] = 0

        self.assertIn(
            "enabled noise requested count must be positive",
            audit.noise_provenance_semantic_errors(document),
        )
        self.assertIn(
            "enabled noise effective count must be positive",
            audit.noise_provenance_semantic_errors(document),
        )

    def test_rejects_requested_enabled_zero_when_mapmaking_disabled(
        self,
    ) -> None:
        document = valid_noise_document(enabled=False)
        document["requested"]["enabled"] = True
        document["requested"]["n_noise_maps"] = 0
        resolution = document["effective"]["resolution"]
        resolution["mapmaking_enabled"] = False
        resolution["requested_enabled"] = True
        resolution["disabled_by_mapmaking"] = True
        resolution["requested_n_noise_maps"] = 0
        resolution["count_zeroed_while_disabled"] = False

        errors = audit.noise_provenance_semantic_errors(document)

        self.assertIn(
            "enabled noise requested count must be positive", errors
        )
        self.assertNotIn(
            "enabled noise effective count must be positive", errors
        )

    def test_accepts_requested_disabled_zero_noise_count(self) -> None:
        document = valid_noise_document(enabled=False)
        document["requested"]["n_noise_maps"] = 0
        resolution = document["effective"]["resolution"]
        resolution["requested_n_noise_maps"] = 0
        resolution["count_zeroed_while_disabled"] = False

        self.assertEqual(
            audit.noise_provenance_semantic_errors(document), []
        )

    def test_rejects_missing_required_noise_products_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records = audit.audit_provenance_sidecars(
                Path(directory), require_noise_products=True
            )

            self.assertFalse(records["noise_products"]["valid"])

    def test_rejects_inconsistent_noise_activation(self) -> None:
        document = valid_noise_document()
        document["effective"]["config"]["enabled"] = False

        self.assertIn(
            "noise effective activation resolution is inconsistent",
            audit.noise_provenance_semantic_errors(document),
        )

    def test_rejects_inconsistent_noise_realization_count(self) -> None:
        document = valid_noise_document()
        document["realized"]["total_noise_realization_count"]["value"] = 89

        self.assertIn(
            "noise observed total_noise_realization_count differs from plan-derived expected count",
            audit.noise_provenance_semantic_errors(document),
        )

    def test_rejects_noise_rng_identity_drift(self) -> None:
        document = valid_noise_document()
        document["effective"]["resolution"]["randomization"]["seed"] = 1

        self.assertIn(
            "noise randomization identity is inconsistent",
            audit.noise_provenance_semantic_errors(document),
        )

    def test_rejects_noise_mapmaking_cardinality_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            mapmaking = valid_mapmaking_document()
            noise = valid_noise_document()
            noise["expected"]["coadd_scientific_map_count"] = 6
            noise["expected"]["coadd_noise_realization_count"] = 60
            noise["expected"]["total_noise_realization_count"] = 120
            noise["realized"]["coadd_scientific_map_count"]["value"] = 6
            noise["realized"]["coadd_noise_realization_count"]["value"] = 60
            noise["realized"]["total_noise_realization_count"]["value"] = 120
            (redu / "mapmaking_provenance.yaml").write_text(
                yaml.safe_dump(mapmaking, sort_keys=False),
                encoding="utf-8",
            )
            (redu / "noise_products_provenance.yaml").write_text(
                yaml.safe_dump(noise, sort_keys=False),
                encoding="utf-8",
            )

            records = audit.audit_provenance_sidecars(
                redu,
                require_mapmaking=True,
                require_noise_products=True,
            )

            self.assertTrue(records["mapmaking"]["valid"])
            self.assertFalse(records["noise_products"]["valid"])
            self.assertEqual(
                records["noise_products"]["cross_check_errors"],
                [
                    "noise expected coadd map count differs from mapmaking provenance"
                ],
            )

    def test_accepts_complete_pointing_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            (redu / "pointing_provenance.yaml").write_text(
                yaml.safe_dump(
                    valid_pointing_document(), sort_keys=False
                ),
                encoding="utf-8",
            )

            pointing = audit.audit_provenance_sidecars(
                redu, require_pointing=True
            )["pointing"]

            self.assertTrue(pointing["present"])
            self.assertTrue(pointing["required"])
            self.assertTrue(pointing["valid"])

    def test_accepts_historical_pointing_v1_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            (redu / "pointing_provenance.yaml").write_text(
                yaml.safe_dump(
                    valid_pointing_v1_document(), sort_keys=False
                ),
                encoding="utf-8",
            )

            pointing = audit.audit_provenance_sidecars(
                redu, require_pointing=True
            )["pointing"]

            self.assertTrue(pointing["valid"])
            self.assertEqual(
                pointing["schema_version"],
                "citlali-pointing-provenance-v1",
            )

    def test_accepts_effectively_disabled_pointing(self) -> None:
        self.assertEqual(
            audit.pointing_provenance_semantic_errors(
                valid_pointing_document(enabled=False)
            ),
            [],
        )

    def test_accepts_pointing_fit_without_filtered_or_coadd_outputs(self) -> None:
        document = valid_pointing_document()
        resolution = document["effective"]["resolution"]
        resolution["map_filter_enabled"] = False
        resolution["coadd_enabled"] = True
        for observation in document["observations"]:
            observation["filtered_fit_attempt_count"] = 0
            observation["filtered_valid_fit_count"] = 0
            observation["filtered_fit_results_recorded"] = False
        document["realized"]["filtered_fit_attempt_count"] = 0
        document["realized"]["filtered_valid_fit_count"] = 0

        self.assertEqual(
            audit.pointing_provenance_semantic_errors(document),
            [],
        )

    def test_rejects_pointing_fit_disabled_by_map_filter_policy(self) -> None:
        document = valid_pointing_document()
        document["effective"]["config"]["fit_gaussian"] = False
        resolution = document["effective"]["resolution"]
        resolution["map_filter_enabled"] = False
        resolution["fit_output_path_available"] = False
        resolution["fit_disabled_by_output_policy"] = True

        self.assertIn(
            "pointing fit activation does not follow mapmaking policy",
            audit.pointing_provenance_semantic_errors(document),
        )

    def test_rejects_missing_required_pointing_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records = audit.audit_provenance_sidecars(
                Path(directory), require_pointing=True
            )

            self.assertFalse(records["pointing"]["valid"])

    def test_rejects_inconsistent_pointing_fit_count(self) -> None:
        document = valid_pointing_document()
        document["observations"][0]["raw_fit_attempt_count"] = 2

        self.assertIn(
            "pointing observation 0 raw_fit attempts are inconsistent",
            audit.pointing_provenance_semantic_errors(document),
        )

    def test_rejects_pointing_mapmaking_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            mapmaking = valid_mapmaking_document()
            pointing = valid_pointing_document()
            pointing["observations"][1]["obsnum"] = "152391"
            (redu / "mapmaking_provenance.yaml").write_text(
                yaml.safe_dump(mapmaking, sort_keys=False),
                encoding="utf-8",
            )
            (redu / "pointing_provenance.yaml").write_text(
                yaml.safe_dump(pointing, sort_keys=False),
                encoding="utf-8",
            )

            records = audit.audit_provenance_sidecars(
                redu,
                require_mapmaking=True,
                require_pointing=True,
            )

            self.assertTrue(records["mapmaking"]["valid"])
            self.assertFalse(records["pointing"]["valid"])
            self.assertEqual(
                records["pointing"]["cross_check_errors"],
                [
                    "pointing observation 1 differs from mapmaking provenance"
                ],
            )

    def test_rejects_inconsistent_coadd_activation(self) -> None:
        document = valid_coadd_document()
        document["effective"]["config"]["enabled"] = False

        self.assertIn(
            "coadd effective resolution is inconsistent",
            audit.coadd_provenance_semantic_errors(document),
        )

    def test_rejects_non_boolean_coadd_resolution(self) -> None:
        document = valid_coadd_document()
        document["effective"]["resolution"]["mapmaking_enabled"] = 1

        self.assertIn(
            "coadd resolution values must be boolean",
            audit.coadd_provenance_semantic_errors(document),
        )

    def test_rejects_coadd_mapmaking_cardinality_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            mapmaking = valid_mapmaking_document()
            coadd = valid_coadd_document()
            coadd["realized"]["map_count"]["value"] = 6
            (redu / "mapmaking_provenance.yaml").write_text(
                yaml.safe_dump(mapmaking, sort_keys=False),
                encoding="utf-8",
            )
            (redu / "coadd_provenance.yaml").write_text(
                yaml.safe_dump(coadd, sort_keys=False),
                encoding="utf-8",
            )

            records = audit.audit_provenance_sidecars(
                redu, require_mapmaking=True, require_coadd=True
            )

            self.assertTrue(records["mapmaking"]["valid"])
            self.assertFalse(records["coadd"]["valid"])
            self.assertEqual(
                records["coadd"]["cross_check_errors"],
                ["coadd cardinality differs from mapmaking provenance"],
            )

    def test_accepts_raw_provenance_for_every_observation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            for obsnum in ("1", "2"):
                observation = redu / obsnum
                observation.mkdir()
                (observation / "raw_timestream_provenance.yaml").write_text(
                    yaml.safe_dump(valid_raw_document(), sort_keys=False),
                    encoding="utf-8",
                )
                (observation / "timestream_output_provenance.yaml").write_text(
                    yaml.safe_dump(valid_output_document(), sort_keys=False),
                    encoding="utf-8",
                )

            raw = audit.audit_provenance_sidecars(
                redu, require_raw=True
            )["raw_timestream"]

            self.assertTrue(raw["required"])
            self.assertTrue(raw["valid"])
            self.assertEqual(raw["count"], 2)
            self.assertTrue(raw["observation_coverage_ok"])

    def test_accepts_finite_interface_sync_provenance_v2(self) -> None:
        document = valid_raw_v2_document()

        self.assertEqual(
            audit.raw_provenance_semantic_errors(document), []
        )

        document["effective"]["config"]["interface_sync_offset"][
            "offsets"
        ]["toltec12"] = float("nan")
        self.assertIn(
            "effective interface-sync offset toltec12 is not finite",
            audit.raw_provenance_semantic_errors(document),
        )

    def test_accepts_complete_sci_cal_001_raw_provenance_v3(self) -> None:
        document = valid_raw_v3_document()

        self.assertEqual(
            audit.raw_provenance_semantic_errors(document), []
        )

    def test_rejects_sci_cal_001_identity_alpha_and_regime_tampering(self) -> None:
        document = valid_raw_v3_document()
        document["realized"]["atmosphere_node_table_sha256"]["value"] = "bad"
        document["effective"]["config"]["calibration"][
            "reference_spectral_index_alpha"
        ] = 1.0
        document["observation"]["value"]["tau225"]["value"] = 0.2
        document["realized"]["tau225"]["value"] = 0.2

        errors = audit.raw_provenance_semantic_errors(document)

        self.assertIn(
            "realized calibration atmosphere_node_table_sha256 is not approved",
            errors,
        )
        self.assertIn("effective calibration alpha is unsupported", errors)
        self.assertIn("calibration quality regime is inconsistent", errors)

    def test_rejects_incomplete_raw_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            document = valid_raw_document()
            document["realized"]["execution_completed"] = False
            document["realized"]["completed_scan_count"] = {
                "available": False
            }
            (redu / "raw_timestream_provenance.yaml").write_text(
                yaml.safe_dump(document, sort_keys=False),
                encoding="utf-8",
            )
            (redu / "timestream_output_provenance.yaml").write_text(
                yaml.safe_dump(valid_output_document(), sort_keys=False),
                encoding="utf-8",
            )

            raw = audit.audit_provenance_sidecars(
                redu, require_raw=True
            )["raw_timestream"]

            self.assertFalse(raw["valid"])
            self.assertEqual(
                raw["files"][0]["semantic_errors"],
                [
                    "raw observation execution is not complete",
                    "realized completed_scan_count is unavailable",
                ],
            )

    def test_rejects_missing_required_raw_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records = audit.audit_provenance_sidecars(
                Path(directory), require_raw=True
            )

            self.assertFalse(records["raw_timestream"]["valid"])

    def test_rejects_missing_raw_observation_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            for obsnum in ("1", "2"):
                observation = redu / obsnum
                observation.mkdir()
                (observation / "timestream_output_provenance.yaml").write_text(
                    yaml.safe_dump(valid_output_document(), sort_keys=False),
                    encoding="utf-8",
                )
            (redu / "1" / "raw_timestream_provenance.yaml").write_text(
                yaml.safe_dump(valid_raw_document(), sort_keys=False),
                encoding="utf-8",
            )

            raw = audit.audit_provenance_sidecars(
                redu, require_raw=True
            )["raw_timestream"]

            self.assertFalse(raw["valid"])
            self.assertFalse(raw["observation_coverage_ok"])
            self.assertEqual(raw["missing_observation_dirs"], [str(redu / "2")])

    def test_rejects_raw_scan_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            (redu / "raw_timestream_provenance.yaml").write_text(
                yaml.safe_dump(valid_raw_document(), sort_keys=False),
                encoding="utf-8",
            )
            output = valid_output_document()
            output["realized"]["n_scans"] = 5
            (redu / "timestream_output_provenance.yaml").write_text(
                yaml.safe_dump(output, sort_keys=False),
                encoding="utf-8",
            )

            raw = audit.audit_provenance_sidecars(
                redu, require_raw=True
            )["raw_timestream"]

            self.assertFalse(raw["valid"])
            self.assertEqual(
                raw["files"][0]["semantic_errors"],
                [
                    "completed scan count does not match "
                    "timestream-output provenance"
                ],
            )

    def test_rejects_inconsistent_raw_sample_rates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            document = valid_raw_document()
            document["observation"]["value"]["effective_sample_rate_hz"][
                "value"
            ] = 40.0
            (redu / "raw_timestream_provenance.yaml").write_text(
                yaml.safe_dump(document, sort_keys=False),
                encoding="utf-8",
            )
            (redu / "timestream_output_provenance.yaml").write_text(
                yaml.safe_dump(valid_output_document(), sort_keys=False),
                encoding="utf-8",
            )

            raw = audit.audit_provenance_sidecars(
                redu, require_raw=True
            )["raw_timestream"]

            self.assertFalse(raw["valid"])
            self.assertIn(
                "effective sample rate does not match native "
                "rate/downsample factor",
                raw["files"][0]["semantic_errors"],
            )

    def test_accepts_complete_processed_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            (redu / "processed_timestream_provenance.yaml").write_text(
                yaml.safe_dump(valid_processed_document(), sort_keys=False),
                encoding="utf-8",
            )

            records = audit.audit_provenance_sidecars(
                redu, require_processed=True
            )

            processed = records["processed_timestream"]
            self.assertTrue(processed["present"])
            self.assertTrue(processed["required"])
            self.assertTrue(processed["valid"])
            self.assertEqual(len(processed["sha256"]), 64)

    def test_accepts_complete_processed_v2_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            (redu / "processed_timestream_provenance.yaml").write_text(
                yaml.safe_dump(valid_processed_v2_document(), sort_keys=False),
                encoding="utf-8",
            )

            processed = audit.audit_provenance_sidecars(
                redu, require_processed=True
            )["processed_timestream"]

            self.assertTrue(processed["valid"])
            self.assertEqual(
                processed["schema_version"],
                "citlali-processed-timestream-provenance-v2",
            )

    def test_accepts_consistent_processed_v2_restart_resolution(self) -> None:
        document = valid_processed_v2_document()
        restart_path = "/data/prior/redu04"
        document["requested"]["fruit_loops"]["restart_path"] = restart_path
        document["effective"]["config"]["fruit_loops"][
            "restart_path"
        ] = restart_path
        document["effective"]["resolutions"]["fruit_loop_restart"] = {
            "available": True,
            "value": {
                "source_reduction_dir": restart_path,
                "checkpoint_path": restart_path
                + "/citlali_restart_checkpoint.nc",
                "creator_version": "test",
                "completed_iteration": 4,
                "next_iteration": 5,
                "effective_sample_mask_intervals": 10,
                "effective_detector_penalties": 2,
            },
        }

        self.assertEqual(
            audit.processed_provenance_semantic_errors(document), []
        )

        document["effective"]["resolutions"]["fruit_loop_restart"][
            "value"
        ]["next_iteration"] = 6
        self.assertIn(
            "fruit-loop restart iteration identity is inconsistent",
            audit.processed_provenance_semantic_errors(document),
        )

    def test_rejects_missing_processed_sections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            document = valid_processed_document()
            del document["effective"]["resolutions"]
            (redu / "processed_timestream_provenance.yaml").write_text(
                yaml.safe_dump(document, sort_keys=False),
                encoding="utf-8",
            )

            processed = audit.audit_provenance_sidecars(
                redu, require_processed=True
            )["processed_timestream"]

            self.assertFalse(processed["valid"])
            self.assertEqual(
                processed["missing_paths"],
                ["effective.resolutions"],
            )

    def test_rejects_missing_required_processed_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)

            records = audit.audit_provenance_sidecars(
                redu, require_processed=True
            )

            self.assertFalse(records["processed_timestream"]["valid"])
            self.assertFalse(audit.provenance_ok({"provenance": records}))

    def test_rejects_inconsistent_processed_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            document = valid_processed_document()
            document["effective"]["config"]["fruit_loops"]["max_iters"] = 2
            (redu / "processed_timestream_provenance.yaml").write_text(
                yaml.safe_dump(document, sort_keys=False),
                encoding="utf-8",
            )

            processed = audit.audit_provenance_sidecars(
                redu, require_processed=True
            )["processed_timestream"]

            self.assertFalse(processed["valid"])
            self.assertEqual(
                processed["files"][0]["semantic_errors"],
                ["iteration resolution does not match effective max_iters"],
            )

    def test_validates_every_observation_provenance_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            for obsnum, schema in (
                ("1", "citlali-timestream-output-provenance-v1"),
                ("2", "wrong-schema"),
            ):
                observation = redu / obsnum
                observation.mkdir()
                (observation / "timestream_output_provenance.yaml").write_text(
                    f"""\
schema_version: {schema}
requested: {{}}
effective: {{}}
realized: {{}}
""",
                    encoding="utf-8",
                )

            output = audit.audit_provenance_sidecars(redu)[
                "timestream_output"
            ]

            self.assertEqual(output["count"], 2)
            self.assertTrue(output["cardinality_ok"])
            self.assertFalse(output["valid"])
            self.assertEqual(len(output["files"]), 2)


if __name__ == "__main__":
    unittest.main()
