from __future__ import annotations

import copy
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
    sources = {
        **{f"toltec{index}": "schema_default_zero" for index in range(13)},
        "hwpr": "schema_default_zero",
    }
    sources["toltec0"] = "configured_zero"
    document["requested"]["interface_sync_offset"]["sources"] = dict(
        sources
    )
    document["effective"]["config"]["interface_sync_offset"][
        "sources"
    ] = dict(sources)

    lifecycle = []
    for interface_id in (
        *(f"toltec{index}" for index in range(13)),
        "hwpr",
        "lmt",
    ):
        configured = interface_id != "lmt"
        requested = (
            document["requested"]["interface_sync_offset"]["offsets"][
                interface_id
            ]
            if configured
            else 0.0
        )
        source = sources[interface_id] if configured else "schema_default_zero"
        resolved = interface_id in {"toltec0", "lmt"}
        lifecycle.append(
            {
                "interface_id": interface_id,
                "requested_sec": requested,
                "effective_sec": requested,
                "observation_resolved_sec": requested if resolved else 0.0,
                "realized_sec": requested if resolved else 0.0,
                "source": source,
                "sign": "positive_add",
                "reference": "detector_clock",
                "unit": "s",
                "application_stage": "before_ordering_slotting_and_gaps",
                "uncertainty": "unavailable",
                "availability": (
                    "observation_resolved" if resolved else "not_applicable"
                ),
                "applied_exactly_once": resolved,
            }
        )
    document["observation"]["value"]["interface_offsets"] = lifecycle
    return document


def valid_output_document() -> dict:
    return {
        "schema_version": "citlali-timestream-output-provenance-v1",
        "requested": {},
        "effective": {},
        "realized": {"n_scans": 4},
    }


def valid_output_v2_document() -> dict:
    return {
        "schema_version": "citlali-timestream-output-provenance-v2",
        "requested": {},
        "effective": {
            "raw_time_chunk": {"enabled": True, "mode": "full"},
            "processed_time_chunk": {"enabled": True, "mode": "full"},
        },
        "realized": {
            "evidence_stage": "observation_setup_plan",
            "execution_completed": False,
            "n_scans": 1,
            "sci_align_scan_plan": {
                "identity": "zero_based_stable_processing_record_id",
                "interval_convention": "half_open_start_stop",
                "physical_identity": (
                    "zero_based_physical_window_id_when_authority_available"
                ),
                "policy": "fixed_count_balanced_v1",
                "requested_value": 1.0,
                "effective_duration_sec": 0.065536,
                "observation_sample_count": 8,
                "physical_identity_count": 0,
                "identity_count": 1,
                "compatibility_admitted_count": 1,
                "compatibility_ordinal_to_stable_id": [0],
                "physical_records": [],
                "records": [
                    {
                        "stable_id": 0,
                        "status": "usable",
                        "physical_id": None,
                        "identity_authority": (
                            "requested_processing_chunk_under_"
                            "continuous_observation_no_physical_scan_authority"
                        ),
                        "processing": {"start": 0, "stop": 8},
                        "science": {"start": 0, "stop": 8},
                        "context": {"start": 0, "stop": 8},
                        "compatibility_science": {"start": 0, "stop": 8},
                        "compatibility_context": {"start": 0, "stop": 8},
                        "legacy_processing_admitted": True,
                        "compatibility_ordinal": 0,
                    }
                ],
            },
            "sci_align_alignment": {
                "initialized": True,
                "representation": (
                    "compact_generative_grid_plus_exception_runs_v1"
                ),
                "dense_mapping_persisted": False,
                "field_registry_version": "sci-align-active-field-registry-v2",
                "grid": {
                    "phase_sec": 1_000.0,
                    "cadence_sec": 0.008192,
                    "exclusive_half_cell_sec": 0.004096,
                    "assignment_operator": "floor_q_plus_half_v1",
                    "physical_timestamp_semantics": "unavailable",
                },
                "hwpr": {
                    "policy": "bounded_nonpolarimetric_optional_hwpr_v1",
                    "observation_resolved": True,
                    "producer_input_present": False,
                    "aligned_angle_available": False,
                    "intensity_eligible": True,
                    "polarization_eligible": False,
                    "availability_reason": (
                        "producer_input_absent_optional_nonfatal"
                    ),
                    "physical_timestamp_semantics": (
                        "unavailable_no_producer_integration_event_authority"
                    ),
                    "demodulation_semantics": (
                        "unavailable_not_authorized_by_bounded_profile"
                    ),
                    "dense_angle_mapping_persisted": False,
                },
                "telescope": {},
                "support": {
                    "nominal_common_axis_slot_count": 8,
                    "guarded_original_interface_slot_count": 0,
                    "gap_policy_eligible_original_interface_slot_count": 6,
                },
                "interfaces": [
                    {"interface_id": "toltec0", "roach_index": 0}
                ],
                "exception_run_contract": {
                    "source_slot_identity": (
                        "zero_based_observation_common_axis_slot"
                    ),
                    "continuity_action_stage": (
                        "candidate_only_chunk_plan_controls_permission"
                    ),
                    "continuity_weight_rule": {
                        "operator": "linear_slot_coordinate_weights_v1",
                        "coordinate_basis": (
                            "observation_common_axis_slot_coordinates"
                        ),
                        "target_domain": (
                            "exception_start_inclusive_stop_exclusive"
                        ),
                        "left_source_weight": (
                            "(right_source_slot-target_slot)/"
                            "(right_source_slot-left_source_slot)"
                        ),
                        "right_source_weight": (
                            "(target_slot-left_source_slot)/"
                            "(right_source_slot-left_source_slot)"
                        ),
                        "normalization": (
                            "left_source_weight_plus_right_source_weight_"
                            "equals_one"
                        ),
                        "dense_source_weights_persisted": False,
                    },
                },
                "exception_runs": [
                    {
                        "interface_id": "toltec0",
                        "field_id": "detector_acquisition",
                        "start": 3,
                        "stop": 5,
                        "interval_convention": "half_open_start_stop",
                        "origin": "native_detector_gap",
                        "validity": "unavailable_original",
                        "action": "bounded_continuity_candidate",
                        "reason": "bounded_by_acquired_originals",
                        "source_slot_identity": (
                            "zero_based_observation_common_axis_slot"
                        ),
                        "source_slots_available": True,
                        "left_source_slot": 2,
                        "right_source_slot": 5,
                    }
                ],
                "processing_support_plan": {
                    "observation_resolved": True,
                    "evidence_stage": (
                        "observation_resolved_planned_processing"
                    ),
                    "execution_realized": False,
                    "realization_semantics": (
                        "plan_only_no_execution_outcome_claim"
                    ),
                    "interval_convention": "half_open_start_stop",
                    "signal_domain": "xs",
                    "count_scope": (
                        "planned_occurrences_across_admitted_scan_contexts"
                    ),
                    "gap_admission_contract": {
                        "support_reference": (
                            "sci_align_scan_plan.records[stable_scan_id]."
                            "compatibility_science"
                        ),
                        "window_relationship": (
                            "compatibility_science_is_a_half_open_subset_of_"
                            "compatibility_context"
                        ),
                        "cumulative_missing_count_scope": (
                            "stable_record_science_window_only"
                        ),
                        "longest_missing_run_count_scope": (
                            "stable_record_science_window_only"
                        ),
                        "unusable_rule": (
                            "four_times_cumulative_or_longest_missing_"
                            "strictly_exceeds_science_window_size"
                        ),
                        "exact_quarter": "admitted",
                    },
                    "planned_action_support_reference": (
                        "chunk_dispositions[].context_expanded_support"
                    ),
                    "continuity_source_contract": (
                        "each_planned_continuity_run_is_a_subrange_of_one_"
                        "bounded_exception_run"
                    ),
                    "chunk_disposition_encoding": {
                        "representation": "sparse_exceptions_v1",
                        "key_order": (
                            "compatibility_ordinal_then_roach_index"
                        ),
                        "persisted_rows": (
                            "nondefault_scan_interface_dispositions_only"
                        ),
                        "absent_default": {
                            "support": (
                                "all_acquired_original_zero_detector_gap"
                            ),
                            "cumulative_missing_count": 0,
                            "longest_missing_run_count": 0,
                            "gap_policy_eligible_original_within_science": True,
                            "full_network_unusable": False,
                            "continuity_surrogate_permitted": (
                                "signal_domain_is_xs"
                            ),
                            "planned_actions": "none",
                        },
                    },
                    "planned_occurrence_counts": {
                        "continuity_surrogate_missing": 2,
                        "unavailable_missing": 0,
                        "guarded_original": 0,
                        "full_network_unusable_original": 0,
                    },
                    "chunk_dispositions": [
                        {
                            "stable_scan_id": 0,
                            "compatibility_ordinal": 0,
                            "interface_id": "toltec0",
                            "roach_index": 0,
                            "context": {"start": 0, "stop": 8},
                            "cumulative_missing_count": 2,
                            "longest_missing_run_count": 2,
                            "full_network_unusable": False,
                            "continuity_surrogate_permitted": True,
                            "planned_actions": {
                                "continuity_surrogate_missing": {
                                    "action": (
                                        "bounded_continuity_surrogate"
                                    ),
                                    "runs": [{"start": 3, "stop": 5}],
                                },
                                "unavailable_missing": {
                                    "action": "remain_unavailable",
                                    "runs": [],
                                },
                                "guarded_original": {
                                    "action": (
                                        "guard_original_processing_sample"
                                    ),
                                    "runs": [],
                                },
                            },
                        }
                    ],
                },
                "availability": {},
            },
            "raw_time_chunk": {
                "n_output_scans": 1,
                "scan_to_output": [0],
                "selected_output_windows": [
                    {
                        "stable_processing_record_id": 0,
                        "compatibility_ordinal": 0,
                        "output_row": 0,
                        "output_interval": {"start": 0, "stop": 8},
                        "interval_convention": "half_open_start_stop",
                        "interval_authority": "science_inner",
                    }
                ],
            },
            "processed_time_chunk": {
                "n_output_scans": 1,
                "scan_to_output": [0],
                "selected_output_windows": [
                    {
                        "stable_processing_record_id": 0,
                        "compatibility_ordinal": 0,
                        "output_row": 0,
                        "output_interval": {"start": 0, "stop": 8},
                        "interval_convention": "half_open_start_stop",
                        "interval_authority": "science_inner",
                    }
                ],
            },
        },
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
    unavailable = {"available": False}
    realized = {
        "reduction_completed": True,
        "generation_executed": enabled,
        "noise_maps_per_scientific_map": (
            {"available": True, "value": effective_count}
            if enabled else dict(unavailable)
        ),
        "observation_scientific_map_count": (
            {"available": True, "value": 6}
            if enabled else dict(unavailable)
        ),
        "observation_noise_realization_count": (
            {"available": True, "value": 60}
            if enabled else dict(unavailable)
        ),
        "coadd_scientific_map_count": (
            {"available": True, "value": 3}
            if enabled else dict(unavailable)
        ),
        "coadd_noise_realization_count": (
            {"available": True, "value": 30}
            if enabled else dict(unavailable)
        ),
        "total_noise_realization_count": (
            {"available": True, "value": 90}
            if enabled else dict(unavailable)
        ),
        "empirical_product_map_count": (
            {"available": True, "value": 0}
            if enabled else dict(unavailable)
        ),
        "realization_image_write_count": (
            {"available": True, "value": 0}
            if enabled else dict(unavailable)
        ),
        "outputs_completed": enabled,
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
        "realized": realized,
    }


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

    def test_accepts_effectively_disabled_noise_products(self) -> None:
        self.assertEqual(
            audit.noise_provenance_semantic_errors(
                valid_noise_document(enabled=False)
            ),
            [],
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
            "noise total realization count is inconsistent",
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
                ["noise coadd map count differs from mapmaking provenance"],
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

    def test_accepts_complete_interface_offset_lifecycle_v3(self) -> None:
        document = valid_raw_v3_document()

        self.assertEqual(audit.raw_provenance_semantic_errors(document), [])

        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
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

            self.assertTrue(raw["valid"])
            self.assertEqual(
                raw["schema_version"],
                "citlali-raw-timestream-provenance-v3",
            )

    def test_rejects_invalid_interface_offset_lifecycle_v3(self) -> None:
        field_cases = (
            (
                "sign",
                "negative_add",
                "interface-offset lifecycle toltec0 sign must be positive_add",
            ),
            (
                "unit",
                "ms",
                "interface-offset lifecycle toltec0 unit must be s",
            ),
            (
                "reference",
                "telescope_clock",
                "interface-offset lifecycle toltec0 reference must be "
                "detector_clock",
            ),
            (
                "application_stage",
                "after_slotting",
                "interface-offset lifecycle toltec0 application_stage must be "
                "before_ordering_slotting_and_gaps",
            ),
            (
                "source",
                "schema_default_zero",
                "interface-offset lifecycle toltec0 source does not match "
                "effective interface-sync source",
            ),
            (
                "availability",
                "available",
                "interface-offset lifecycle toltec0 availability is invalid",
            ),
            (
                "applied_exactly_once",
                False,
                "interface-offset lifecycle toltec0 resolved offset was not "
                "applied exactly once",
            ),
        )
        for field, invalid_value, expected_error in field_cases:
            with self.subTest(field=field):
                document = copy.deepcopy(valid_raw_v3_document())
                document["observation"]["value"]["interface_offsets"][0][
                    field
                ] = invalid_value
                self.assertIn(
                    expected_error,
                    audit.raw_provenance_semantic_errors(document),
                )

        duplicate = valid_raw_v3_document()
        duplicate["observation"]["value"]["interface_offsets"][0][
            "interface_id"
        ] = "toltec1"
        duplicate_errors = audit.raw_provenance_semantic_errors(duplicate)
        self.assertIn(
            "observation interface-offset lifecycle identity 'toltec1' is "
            "invalid or duplicated",
            duplicate_errors,
        )
        self.assertIn(
            "observation interface-offset lifecycle identities are incomplete",
            duplicate_errors,
        )

        unresolved = valid_raw_v3_document()
        unresolved_record = unresolved["observation"]["value"][
            "interface_offsets"
        ][0]
        unresolved_record["availability"] = "unavailable_authority"
        unresolved_record["applied_exactly_once"] = False
        self.assertIn(
            "interface-offset lifecycle toltec0 has unavailable authority in "
            "completed execution",
            audit.raw_provenance_semantic_errors(unresolved),
        )

    def test_accepts_timestream_output_provenance_v2(self) -> None:
        document = valid_output_v2_document()
        self.assertEqual(
            audit.timestream_output_provenance_semantic_errors(document), []
        )

        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            (redu / "timestream_output_provenance.yaml").write_text(
                yaml.safe_dump(document, sort_keys=False),
                encoding="utf-8",
            )

            output = audit.audit_provenance_sidecars(redu)[
                "timestream_output"
            ]

            self.assertTrue(output["valid"])
            self.assertEqual(
                output["schema_version"],
                "citlali-timestream-output-provenance-v2",
            )

    def test_accepts_completed_timestream_output_provenance_v2(self) -> None:
        document = valid_output_v2_document()
        document["realized"].update(
            {
                "evidence_stage": "observation_execution_completed",
                "execution_completed": True,
            }
        )
        processing = document["realized"]["sci_align_alignment"][
            "processing_support_plan"
        ]
        processing.update(
            {
                "evidence_stage": (
                    "observation_execution_completed_compact_result"
                ),
                "execution_realized": True,
                "realization_semantics": (
                    "required_processing_and_outputs_completed_"
                    "compact_plan_result"
                ),
            }
        )
        self.assertEqual(
            audit.timestream_output_provenance_semantic_errors(document), []
        )

    def test_rejects_inconsistent_timestream_output_provenance_v2(
        self,
    ) -> None:
        document = valid_output_v2_document()
        document["realized"]["sci_align_scan_plan"]["records"][0][
            "stable_id"
        ] = 1
        document["realized"]["sci_align_alignment"]["grid"][
            "physical_timestamp_semantics"
        ] = "integration_centroid"

        errors = audit.timestream_output_provenance_semantic_errors(document)
        self.assertIn(
            "SCI-ALIGN scan stable identities are not contiguous from zero",
            errors,
        )
        self.assertIn(
            "SCI-ALIGN physical timestamp semantics are not unavailable",
            errors,
        )

        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            (redu / "timestream_output_provenance.yaml").write_text(
                yaml.safe_dump(document, sort_keys=False),
                encoding="utf-8",
            )
            output = audit.audit_provenance_sidecars(redu)[
                "timestream_output"
            ]
            self.assertFalse(output["valid"])

    def test_rejects_inconsistent_sci_align_output_windows_v2(self) -> None:
        wrong_authority = valid_output_v2_document()
        wrong_authority["realized"]["raw_time_chunk"][
            "selected_output_windows"
        ][0]["interval_authority"] = "context_outer"
        self.assertIn(
            "SCI-ALIGN raw_time_chunk output windows record 0 conflicts "
            "with realized selection and scan-plan support",
            audit.timestream_output_provenance_semantic_errors(
                wrong_authority
            ),
        )

        duplicate_row = valid_output_v2_document()
        duplicate_row["realized"]["processed_time_chunk"][
            "n_output_scans"
        ] = 2
        duplicate_row["realized"]["processed_time_chunk"][
            "scan_to_output"
        ] = [1]
        self.assertIn(
            "SCI-ALIGN processed_time_chunk output windows are not a "
            "complete output-row bijection",
            audit.timestream_output_provenance_semantic_errors(duplicate_row),
        )

        outer = valid_output_v2_document()
        outer["effective"]["raw_time_chunk"]["mode"] = "full_outer"
        outer["realized"]["raw_time_chunk"]["selected_output_windows"][0][
            "interval_authority"
        ] = "context_outer"
        self.assertEqual(
            audit.timestream_output_provenance_semantic_errors(outer), []
        )

    def test_retains_timestream_output_provenance_v1_compatibility(
        self,
    ) -> None:
        document = valid_output_document()
        self.assertEqual(
            audit.timestream_output_provenance_semantic_errors(document), []
        )

        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            (redu / "timestream_output_provenance.yaml").write_text(
                yaml.safe_dump(document, sort_keys=False),
                encoding="utf-8",
            )
            output = audit.audit_provenance_sidecars(redu)[
                "timestream_output"
            ]
            self.assertTrue(output["valid"])
            self.assertEqual(
                output["schema_version"],
                "citlali-timestream-output-provenance-v1",
            )

    def test_rejects_inconsistent_sci_align_hwpr_status_v2(self) -> None:
        missing = valid_output_v2_document()
        del missing["realized"]["sci_align_alignment"]["hwpr"][
            "physical_timestamp_semantics"
        ]
        self.assertIn(
            "SCI-ALIGN HWPR status fields are incomplete or non-compact",
            audit.timestream_output_provenance_semantic_errors(missing),
        )

        false_physical_claim = valid_output_v2_document()
        false_physical_claim["realized"]["sci_align_alignment"]["hwpr"][
            "physical_timestamp_semantics"
        ] = "integration_centroid"
        self.assertIn(
            "SCI-ALIGN HWPR status conflicts with the bounded "
            "nonpolarimetric contract",
            audit.timestream_output_provenance_semantic_errors(
                false_physical_claim
            ),
        )

        present = valid_output_v2_document()
        hwpr = present["realized"]["sci_align_alignment"]["hwpr"]
        hwpr["producer_input_present"] = True
        hwpr["availability_reason"] = (
            "producer_input_present_not_loaded_or_aligned"
        )
        self.assertEqual(
            audit.timestream_output_provenance_semantic_errors(present), []
        )

    def test_rejects_malformed_sci_align_exception_contract_v2(
        self,
    ) -> None:
        missing_contract_field = valid_output_v2_document()
        del missing_contract_field["realized"]["sci_align_alignment"][
            "exception_run_contract"
        ]["source_slot_identity"]
        self.assertIn(
            "SCI-ALIGN exception-run contract is missing, malformed, or dense",
            audit.timestream_output_provenance_semantic_errors(
                missing_contract_field
            ),
        )

        dense_weights = valid_output_v2_document()
        dense_weights["realized"]["sci_align_alignment"][
            "exception_run_contract"
        ]["continuity_weight_rule"]["dense_source_weights_persisted"] = True
        self.assertIn(
            "SCI-ALIGN exception-run contract is missing, malformed, or dense",
            audit.timestream_output_provenance_semantic_errors(dense_weights),
        )

        dense_exception = valid_output_v2_document()
        dense_exception["realized"]["sci_align_alignment"][
            "exception_runs"
        ][0]["source_weights"] = [2.0 / 3.0, 1.0 / 3.0]
        self.assertIn(
            "SCI-ALIGN exception run 0 fields are incomplete or non-compact",
            audit.timestream_output_provenance_semantic_errors(
                dense_exception
            ),
        )

    def test_rejects_inconsistent_sci_align_exception_endpoints_v2(
        self,
    ) -> None:
        document = valid_output_v2_document()
        alignment = document["realized"]["sci_align_alignment"]
        alignment["exception_runs"][0]["right_source_slot"] = 6
        self.assertIn(
            "SCI-ALIGN exception run 0 lacks exact bounded-continuity "
            "source endpoints",
            audit.timestream_output_provenance_semantic_errors(document),
        )

        adjacent = valid_output_v2_document()
        alignment = adjacent["realized"]["sci_align_alignment"]
        second = copy.deepcopy(alignment["exception_runs"][0])
        second.update(
            {
                "start": 5,
                "stop": 6,
                "left_source_slot": 4,
                "right_source_slot": 6,
            }
        )
        alignment["exception_runs"].append(second)
        self.assertIn(
            "SCI-ALIGN exception run 1 overlaps or is adjacent to its "
            "preceding compact run",
            audit.timestream_output_provenance_semantic_errors(adjacent),
        )

    def test_rejects_execution_or_dense_processing_claims_v2(self) -> None:
        realized = valid_output_v2_document()
        processing = realized["realized"]["sci_align_alignment"][
            "processing_support_plan"
        ]
        processing["execution_realized"] = True
        self.assertIn(
            "SCI-ALIGN processing-support realization conflicts with "
            "observation completion stage",
            audit.timestream_output_provenance_semantic_errors(realized),
        )

        outcome = valid_output_v2_document()
        processing = outcome["realized"]["sci_align_alignment"][
            "processing_support_plan"
        ]
        processing["execution_completed"] = True
        self.assertIn(
            "SCI-ALIGN processing-support plan fields are incomplete or "
            "non-compact",
            audit.timestream_output_provenance_semantic_errors(outcome),
        )

        dense = valid_output_v2_document()
        run = dense["realized"]["sci_align_alignment"][
            "processing_support_plan"
        ]["chunk_dispositions"][0]["planned_actions"][
            "continuity_surrogate_missing"
        ]["runs"][0]
        run["per_sample_values"] = [0.0, 0.0]
        self.assertIn(
            "SCI-ALIGN chunk disposition 0 continuity_surrogate_missing "
            "run 0 fields are incomplete or non-compact",
            audit.timestream_output_provenance_semantic_errors(dense),
        )

    def test_rejects_inconsistent_processing_support_plan_v2(self) -> None:
        count_mismatch = valid_output_v2_document()
        count_mismatch["realized"]["sci_align_alignment"][
            "processing_support_plan"
        ]["planned_occurrence_counts"]["continuity_surrogate_missing"] = 3
        self.assertIn(
            "SCI-ALIGN planned occurrence counts conflict with compact runs",
            audit.timestream_output_provenance_semantic_errors(count_mismatch),
        )

    def test_accepts_sparse_ordinary_default_and_mixed_interfaces_v2(
        self,
    ) -> None:
        ordinary = valid_output_v2_document()
        alignment = ordinary["realized"]["sci_align_alignment"]
        alignment["exception_runs"] = []
        processing = alignment["processing_support_plan"]
        processing["chunk_dispositions"] = []
        processing["planned_occurrence_counts"] = {
            "continuity_surrogate_missing": 0,
            "unavailable_missing": 0,
            "guarded_original": 0,
            "full_network_unusable_original": 0,
        }
        alignment["support"][
            "gap_policy_eligible_original_interface_slot_count"
        ] = 8
        self.assertEqual(
            audit.timestream_output_provenance_semantic_errors(ordinary), []
        )

        mixed = valid_output_v2_document()
        alignment = mixed["realized"]["sci_align_alignment"]
        alignment["interfaces"].append(
            {"interface_id": "toltec1", "roach_index": 1}
        )
        alignment["support"][
            "gap_policy_eligible_original_interface_slot_count"
        ] = 14
        self.assertEqual(
            audit.timestream_output_provenance_semantic_errors(mixed), []
        )

    def test_rejects_spurious_or_missing_sparse_dispositions_v2(self) -> None:
        spurious = valid_output_v2_document()
        alignment = spurious["realized"]["sci_align_alignment"]
        alignment["exception_runs"] = []
        alignment["support"][
            "gap_policy_eligible_original_interface_slot_count"
        ] = 8
        processing = alignment["processing_support_plan"]
        processing["planned_occurrence_counts"] = {
            "continuity_surrogate_missing": 0,
            "unavailable_missing": 0,
            "guarded_original": 0,
            "full_network_unusable_original": 0,
        }
        disposition = processing["chunk_dispositions"][0]
        disposition["cumulative_missing_count"] = 0
        disposition["longest_missing_run_count"] = 0
        for action in disposition["planned_actions"].values():
            action["runs"] = []
        self.assertIn(
            "SCI-ALIGN chunk disposition 0 persists a spurious ordinary "
            "default row",
            audit.timestream_output_provenance_semantic_errors(spurious),
        )

        missing = valid_output_v2_document()
        alignment = missing["realized"]["sci_align_alignment"]
        processing = alignment["processing_support_plan"]
        processing["chunk_dispositions"] = []
        processing["planned_occurrence_counts"][
            "continuity_surrogate_missing"
        ] = 0
        self.assertIn(
            "SCI-ALIGN sparse processing plan omits a nondefault "
            "scan/interface disposition 0/toltec0",
            audit.timestream_output_provenance_semantic_errors(missing),
        )

    def test_sparse_dispositions_use_compatibility_roach_order_v2(self) -> None:
        document = valid_output_v2_document()
        alignment = document["realized"]["sci_align_alignment"]
        alignment["interfaces"].append(
            {"interface_id": "toltec1", "roach_index": 1}
        )
        second_exception = copy.deepcopy(alignment["exception_runs"][0])
        second_exception["interface_id"] = "toltec1"
        alignment["exception_runs"].append(second_exception)
        processing = alignment["processing_support_plan"]
        second_disposition = copy.deepcopy(processing["chunk_dispositions"][0])
        second_disposition["interface_id"] = "toltec1"
        second_disposition["roach_index"] = 1
        processing["chunk_dispositions"].append(second_disposition)
        processing["planned_occurrence_counts"][
            "continuity_surrogate_missing"
        ] = 4
        alignment["support"][
            "gap_policy_eligible_original_interface_slot_count"
        ] = 12
        self.assertEqual(
            audit.timestream_output_provenance_semantic_errors(document), []
        )

        processing["chunk_dispositions"].reverse()
        self.assertIn(
            "SCI-ALIGN chunk disposition 1 is not in deterministic "
            "compatibility/roach order",
            audit.timestream_output_provenance_semantic_errors(document),
        )

    def test_sparse_contract_uses_compatibility_science_authority_v2(
        self,
    ) -> None:
        document = valid_output_v2_document()
        record = document["realized"]["sci_align_scan_plan"]["records"][0]
        record["science"] = {"start": 1, "stop": 7}
        self.assertEqual(
            audit.timestream_output_provenance_semantic_errors(document), []
        )

        shifted = valid_output_v2_document()
        disposition = shifted["realized"]["sci_align_alignment"][
            "processing_support_plan"
        ]["chunk_dispositions"][0]
        disposition["planned_actions"]["continuity_surrogate_missing"][
            "runs"
        ] = [{"start": 2, "stop": 5}]
        self.assertIn(
            "SCI-ALIGN chunk disposition 0 planned runs do not partition "
            "support",
            audit.timestream_output_provenance_semantic_errors(shifted),
        )

        unavailable = valid_output_v2_document()
        alignment = unavailable["realized"]["sci_align_alignment"]
        processing = alignment["processing_support_plan"]
        disposition = processing["chunk_dispositions"][0]
        disposition["planned_actions"]["continuity_surrogate_missing"][
            "runs"
        ] = []
        disposition["planned_actions"]["unavailable_missing"]["runs"] = [
            {"start": 3, "stop": 5}
        ]
        processing["planned_occurrence_counts"][
            "continuity_surrogate_missing"
        ] = 0
        processing["planned_occurrence_counts"]["unavailable_missing"] = 2
        self.assertIn(
            "SCI-ALIGN chunk disposition 0 marks bounded continuity support "
            "unavailable",
            audit.timestream_output_provenance_semantic_errors(unavailable),
        )

    def test_rejects_altered_sparse_disposition_encoding_v2(self) -> None:
        for mutation in ("missing", "altered", "extra"):
            with self.subTest(mutation=mutation):
                document = valid_output_v2_document()
                processing = document["realized"]["sci_align_alignment"][
                    "processing_support_plan"
                ]
                encoding = processing["chunk_disposition_encoding"]
                if mutation == "missing":
                    del encoding["absent_default"][
                        "gap_policy_eligible_original_within_science"
                    ]
                elif mutation == "altered":
                    encoding["key_order"] = "interface_then_scan"
                else:
                    encoding["dense_rows"] = True
                self.assertIn(
                    "SCI-ALIGN sparse chunk-disposition encoding is invalid "
                    "or incomplete",
                    audit.timestream_output_provenance_semantic_errors(
                        document
                    ),
                )

    def test_science_window_controls_gap_admission_not_context_v2(
        self,
    ) -> None:
        document = valid_output_v2_document()
        plan = document["realized"]["sci_align_scan_plan"]
        plan["observation_sample_count"] = 16
        record = plan["records"][0]
        record["processing"] = {"start": 4, "stop": 12}
        record["science"] = {"start": 4, "stop": 12}
        record["context"] = {"start": 0, "stop": 16}
        record["compatibility_science"] = {"start": 4, "stop": 12}
        record["compatibility_context"] = {"start": 0, "stop": 16}

        alignment = document["realized"]["sci_align_alignment"]
        alignment["support"]["nominal_common_axis_slot_count"] = 16
        alignment["support"][
            "gap_policy_eligible_original_interface_slot_count"
        ] = 6
        exception = alignment["exception_runs"][0]
        exception["start"] = 4
        exception["stop"] = 6
        exception["left_source_slot"] = 3
        exception["right_source_slot"] = 6
        left_edge = copy.deepcopy(exception)
        left_edge.update(
            {
                "start": 0,
                "stop": 2,
                "action": "none",
                "source_slots_available": False,
                "left_source_slot": -1,
                "right_source_slot": -1,
            }
        )
        right_edge = copy.deepcopy(left_edge)
        right_edge.update({"start": 14, "stop": 16})
        alignment["exception_runs"] = [left_edge, exception, right_edge]
        processing = alignment["processing_support_plan"]
        processing["planned_occurrence_counts"][
            "continuity_surrogate_missing"
        ] = 2
        processing["planned_occurrence_counts"]["unavailable_missing"] = 4
        disposition = processing["chunk_dispositions"][0]
        disposition["context"] = {"start": 0, "stop": 16}
        disposition["planned_actions"]["continuity_surrogate_missing"][
            "runs"
        ] = [{"start": 4, "stop": 6}]
        disposition["planned_actions"]["unavailable_missing"]["runs"] = [
            {"start": 0, "stop": 2},
            {"start": 14, "stop": 16},
        ]

        for stream_name in ("raw_time_chunk", "processed_time_chunk"):
            document["realized"][stream_name]["selected_output_windows"][0][
                "output_interval"
            ] = {"start": 4, "stop": 12}

        self.assertEqual(
            audit.timestream_output_provenance_semantic_errors(document), []
        )

        context_threshold = copy.deepcopy(document)
        context_threshold["realized"]["sci_align_alignment"][
            "processing_support_plan"
        ]["chunk_dispositions"][0]["full_network_unusable"] = True
        self.assertIn(
            "SCI-ALIGN chunk disposition 0 full-network usability is "
            "inconsistent",
            audit.timestream_output_provenance_semantic_errors(
                context_threshold
            ),
        )

        context_mismatch = valid_output_v2_document()
        disposition = context_mismatch["realized"]["sci_align_alignment"][
            "processing_support_plan"
        ]["chunk_dispositions"][0]
        disposition["context"]["stop"] = 7
        self.assertIn(
            "SCI-ALIGN chunk disposition 0 context conflicts with its scan plan",
            audit.timestream_output_provenance_semantic_errors(
                context_mismatch
            ),
        )

        missing_source = valid_output_v2_document()
        missing_source["realized"]["sci_align_alignment"][
            "exception_runs"
        ] = []
        self.assertIn(
            "SCI-ALIGN chunk disposition 0 continuity run has no unique "
            "compact source exception",
            audit.timestream_output_provenance_semantic_errors(
                missing_source
            ),
        )

        wrong_domain = valid_output_v2_document()
        wrong_domain["realized"]["sci_align_alignment"][
            "processing_support_plan"
        ]["signal_domain"] = "rs"
        self.assertIn(
            "SCI-ALIGN chunk disposition 0 continuity permission is "
            "inconsistent",
            audit.timestream_output_provenance_semantic_errors(wrong_domain),
        )

    def test_rejects_missing_or_nondeterministic_processing_plan_v2(
        self,
    ) -> None:
        unresolved = valid_output_v2_document()
        processing = unresolved["realized"]["sci_align_alignment"][
            "processing_support_plan"
        ]
        processing["observation_resolved"] = False
        processing["evidence_stage"] = "not_observation_resolved"
        processing["signal_domain"] = ""
        processing["chunk_dispositions"] = []
        processing["planned_occurrence_counts"] = {
            "continuity_surrogate_missing": 0,
            "unavailable_missing": 0,
            "guarded_original": 0,
            "full_network_unusable_original": 0,
        }
        self.assertIn(
            "admitted SCI-ALIGN scans have no observation-resolved "
            "processing plan",
            audit.timestream_output_provenance_semantic_errors(unresolved),
        )

        duplicate = valid_output_v2_document()
        processing = duplicate["realized"]["sci_align_alignment"][
            "processing_support_plan"
        ]
        processing["chunk_dispositions"].append(
            copy.deepcopy(processing["chunk_dispositions"][0])
        )
        self.assertIn(
            "SCI-ALIGN chunk disposition 1 duplicates a scan/interface "
            "identity",
            audit.timestream_output_provenance_semantic_errors(duplicate),
        )

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
