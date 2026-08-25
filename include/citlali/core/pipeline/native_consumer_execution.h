#pragma once

// Private application orchestration for the explicitly activated compact-v2
// Science/Pointing consumer. The Stage 3--6 adapters retain all authority;
// this layer only supplies run/group-local views to the established RTC/PTC
// numerical bodies and prepares the immutable native map input.

#include <citlali/core/mapmaking/science_map_contract.h>
#include <citlali/core/engine/learning.h>
#include <citlali/core/pipeline/mapmaking_dispatch.h>
#include <citlali/core/pipeline/native_cohort_product_provenance_v3.h>
#include <citlali/core/pipeline/native_consumer_execution_policy.h>
#include <citlali/core/pipeline/native_detector_run_fcf_contract.h>
#include <citlali/core/pipeline/native_scan_runtime_state.h>
#include <citlali/core/pipeline/downsample_config.h>
#include <citlali/core/pipeline/jinc_processing_provenance.h>
#include <citlali/core/pipeline/timestream_run_context.h>

#include <netcdf>

#include <citlali/core/timestream/ptc/clean.h>
#include <citlali/core/timestream/rtc/despike.h>
#include <citlali/core/timestream/timestream.h>

#include <Eigen/Core>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace citlali::pipeline {

inline constexpr NativeDetectorFlagBits native_rtc_processing_flag_bit_v2 =
    native_cohort_rtc_processing_flag_bit_v3;
inline constexpr NativeDetectorFlagBits native_learned_rtc_flag_bit_v2 =
    native_cohort_learned_rtc_exclusion_bit_v3;
inline constexpr NativeDetectorFlagBits
    native_duplicate_tone_exclusion_bit_v2 =
        native_cohort_duplicate_tone_exclusion_bit_v3;

template <class FlagMatrix>
void reconcile_native_rtc_detector_result(
    const Eigen::MatrixXd &measured, Eigen::Index detector,
    const std::optional<std::int64_t> &apt_flag,
    Eigen::MatrixXd &processed, FlagMatrix &flags, double &fcf) {
    if (detector < 0 || detector >= measured.cols() ||
        processed.rows() != measured.rows() ||
        processed.cols() != measured.cols() ||
        flags.rows() != measured.rows() ||
        flags.cols() != measured.cols()) {
        throw std::logic_error(
            "native RTC detector reconciliation has unequal shape");
    }
    const bool apt_excluded = !apt_flag.has_value() || *apt_flag != 0;
    if (!apt_excluded) {
        if (!std::isfinite(fcf) ||
            !processed.col(detector).array().isFinite().all()) {
            throw std::logic_error(
                "native RTC returned a nonfinite result for an eligible detector");
        }
        return;
    }

    flags.col(detector).setConstant(true);
    for (Eigen::Index row = 0; row < processed.rows(); ++row) {
        if (!std::isfinite(processed(row, detector))) {
            const auto preserved = measured(row, detector);
            if (!std::isfinite(preserved)) {
                throw std::logic_error(
                    "native RTC excluded detector lacks a finite measured value");
            }
            processed(row, detector) = preserved;
        }
    }
    if (!std::isfinite(fcf)) fcf = 1.0;
}

inline void add_native_jinc_trace_count(
    std::size_t &total, std::size_t value, const char *field) {
    if (value > std::numeric_limits<std::size_t>::max() - total) {
        throw std::overflow_error(std::string{"native JINC "} + field +
                                  " count overflow");
    }
    total += value;
}

inline std::string native_jinc_trace_facts_digest(
    std::string_view identity,
    const std::vector<std::pair<std::string, std::string>> &facts) {
    return mapmaking::jinc_realization_identity_digest(
        std::string{identity}, facts);
}

inline std::string native_jinc_processing_scan_trace_digest_v2(
    const mapmaking::JincProcessingScanTrace &trace) {
    const std::vector<std::pair<std::string, std::string>> facts{
        {"detector_count", std::to_string(trace.detector_count)},
        {"detector_sample_count",
         std::to_string(trace.detector_sample_count)},
        {"rtc_flagged_sample_count",
         std::to_string(trace.rtc_flagged_sample_count)},
        {"ptc_flagged_sample_count",
         std::to_string(trace.ptc_flagged_sample_count)},
        {"apt_flagged_detector_count",
         std::to_string(trace.apt_flagged_detector_count)},
        {"rtc_source_masked_sample_count",
         std::to_string(trace.rtc_source_masked_sample_count)},
        {"ptc_mean_masked_sample_count",
         std::to_string(trace.ptc_mean_masked_sample_count)},
        {"pca_solve_count", std::to_string(trace.pca_solve_count)},
        {"configured_notch_applied_count",
         std::to_string(trace.configured_notch_applied_count)},
        {"fixed_notch_count", std::to_string(trace.fixed_notch_count)},
        {"dynamic_notch_count", std::to_string(trace.dynamic_notch_count)},
        {"detector_notch_count",
         std::to_string(trace.detector_notch_count)},
        {"rtc_flags_digest", trace.rtc_flags_digest},
        {"ptc_flags_digest", trace.ptc_flags_digest},
        {"apt_flags_digest", trace.apt_flags_digest},
        {"map_indices_digest", trace.map_indices_digest},
        {"ptc_signal_digest", trace.ptc_signal_digest},
        {"ptc_kernel_digest", trace.ptc_kernel_digest},
        {"rtc_source_mask_digest", trace.rtc_source_mask_digest},
        {"rtc_notch_operators_digest",
         trace.rtc_notch_operators_digest},
        {"ptc_mean_mask_digest", trace.ptc_mean_mask_digest},
        {"pca_realization_digest", trace.pca_realization_digest},
    };
    for (const auto &[name, value] : facts) {
        if (name.ends_with("digest") &&
            (value.empty() || value == "unavailable")) {
            throw std::logic_error(
                "native JINC scan trace is incomplete");
        }
    }
    return native_jinc_trace_facts_digest(
        "native-jinc-processing-scan-trace-v2", facts);
}

template <class Engine>
void publish_native_jinc_processing_trace_if_active(
    Engine &engine, Eigen::Index scan_index,
    const mapmaking::JincProcessingScanTrace &trace) {
    if (!jinc_processing_provenance_active(engine)) {
        return;
    }
    (void)native_jinc_processing_scan_trace_digest_v2(trace);
    auto &products = engine.omb.jinc_products;
    std::scoped_lock<std::mutex> lock(*products.processing_trace_mutex);
    products.processing_scan_traces[scan_index] = trace;
}

template <class Engine>
NativeCohortMapPublicationRequestV3
make_native_map_publication_request_v3(
    Engine &engine, citlali::config::MapMethod method, bool enabled,
    const NativeScanRuntimeState &runtime,
    const Eigen::VectorXd &eligible_weights) {
    if (!enabled) return {};
    NativeCohortMapPublicationRequestV3 result;
    result.mapmaking_enabled = true;
    result.method = std::string{citlali::config::to_string(method)};
    if (eligible_weights.size() != static_cast<Eigen::Index>(
            runtime.mapping_handle()->detector_count()) ||
        !eligible_weights.array().isFinite().all() ||
        (eligible_weights.array() < 0.0).any()) {
        throw std::logic_error(
            "native map occurrence lacks exact finite detector weights");
    }
    result.eligible_weight_digest =
        mapmaking::jinc_matrix_digest(eligible_weights);
    result.positive_weight_detector_count = static_cast<std::size_t>(
        (eligible_weights.array() > 0.0).count());
    result.zero_weight_detector_count = static_cast<std::size_t>(
        (eligible_weights.array() == 0.0).count());
    result.zero_weight_detector_columns.reserve(
        result.zero_weight_detector_count);
    for (Eigen::Index detector = 0;
         detector < eligible_weights.size(); ++detector) {
        if (eligible_weights(detector) == 0.0) {
            result.zero_weight_detector_columns.push_back(detector);
        }
    }
    result.learned_map_zero_weight_detector_columns =
        runtime.learned_map_zero_weight_detector_columns;
    if (!runtime.noise_assignment) {
        throw std::logic_error(
            "native map occurrence lacks noise-assignment realization state");
    }
    result.noise_assignment = *runtime.noise_assignment;
    if (!runtime.fruit_loop_feedback) {
        throw std::logic_error(
            "native map occurrence lacks fruit-loop realization state");
    }
    result.fruit_loop_feedback = *runtime.fruit_loop_feedback;
    if (citlali::config::is_naive_map_method(method)) {
        const auto &identity = engine.omb.science_products.bundle_identity;
        if (!identity) {
            throw std::logic_error(
                "native naive map occurrence lacks its admitted product identity");
        }
        result.product_identity_digest =
            mapmaking::science_map_bundle_identity_digest(*identity);
        result.product_occurrence =
            "urn:citlali:science-map-bundle:" +
            result.product_identity_digest;
        return result;
    }
    if (!citlali::config::is_jinc_map_method(method) ||
        !runtime.jinc_processing_trace) {
        throw std::logic_error(
            "native map occurrence supports only traced naive or JINC products");
    }
    const auto &provenance = engine.omb.jinc_products.provenance;
    if (!engine.omb.jinc_products.initialized || !provenance.available ||
        !provenance.processing_configuration_bound ||
        provenance.effective_digest.empty() ||
        provenance.processing_configuration_identity.empty() ||
        provenance.processing_configuration_identity == "unavailable") {
        throw std::logic_error(
            "native JINC map occurrence lacks its admitted product configuration");
    }
    result.product_identity_digest = provenance.effective_digest;
    result.product_occurrence =
        "urn:citlali:jinc-map-configuration:" +
        result.product_identity_digest;
    result.jinc_processing_configuration_digest =
        provenance.processing_configuration_identity;
    return result;
}

template <class Calib>
Calib make_native_detector_subset_calibration(
    const Calib &source,
    const std::vector<TimestreamDetectorColumn> &columns) {
    if (source.n_dets <= 0 || columns.empty()) {
        throw std::invalid_argument(
            "native numerical calibration subset is empty");
    }
    Calib result = source;
    for (auto &[name, values] : result.apt) {
        const auto found = source.apt.find(name);
        if (found == source.apt.end()) {
            throw std::logic_error(
                "native numerical calibration APT changed during staging");
        }
        if (found->second.size() == source.n_dets) {
            values.resize(static_cast<Eigen::Index>(columns.size()));
            for (std::size_t local = 0; local < columns.size(); ++local) {
                const auto column = columns[local];
                if (column < 0 || column >= source.n_dets) {
                    throw std::out_of_range(
                        "native numerical detector column is out of range");
                }
                values(static_cast<Eigen::Index>(local)) =
                    found->second(column);
            }
        }
    }
    if (source.flux_conversion_factor.size() == source.n_dets) {
        result.flux_conversion_factor.resize(
            static_cast<Eigen::Index>(columns.size()));
        for (std::size_t local = 0; local < columns.size(); ++local) {
            result.flux_conversion_factor(
                static_cast<Eigen::Index>(local)) =
                source.flux_conversion_factor(columns[local]);
        }
    }
    result.setup();
    return result;
}

template <class Engine>
NativeRtcDispatchResult run_native_rtc_numerical_bodies(
    Engine &engine, NativeScanRuntimeState &runtime) {
    require_supported_native_consumer_execution(engine);
    const int factor = should_run_downsample(engine)
        ? downsample_factor(engine) : 1;
    NativeDetectorRunFcfContract fcf_contract(
        runtime.mapping_handle()->detector_count(),
        raw_time_chunk_config(engine).extinction_correction_enabled);
    mapmaking::JincProcessingScanTrace trace;
    std::vector<std::pair<std::string, std::string>> source_mask_facts;
    std::vector<std::pair<std::string, std::string>> notch_facts;
    const auto &raw = raw_time_chunk_config(engine);
    const bool run_cross_network_observers =
        raw.flagging.impulsive_coincidence.enabled ||
        raw.coherent_iq_mode_observer.enabled;
    const bool run_global_rtc_body =
        run_cross_network_observers || raw.line_audit.enabled ||
        raw.altaz_destripe.enabled ||
        raw.flagging.lower_tod_inv_var_factor != 0.0 ||
        raw.flagging.upper_tod_inv_var_factor != 0.0;
    if (run_global_rtc_body) {
        engine.rtcproc.clear_cached_diagnostics(
            runtime.mapping_handle()->scope().scan_index);
    }

    const auto run_numerical_body =
        [&](const NativeRtcRunInput &run) {
            auto local_calib = make_native_detector_subset_calibration(
                engine.calib, run.detector_columns);
            auto local_telescope = engine.telescope;
            auto local_rtc = engine.rtcproc;
            local_rtc.run_downsample = false;
            if (!run_global_rtc_body) {
                local_rtc.impulsive_coincidence.enabled = false;
                local_rtc.coherent_iq_mode_observer_enabled = false;
            }

            timestream::TCData<timestream::TCDataKind::RTC,
                               Eigen::MatrixXd> input;
            timestream::TCData<timestream::TCDataKind::PTC,
                               Eigen::MatrixXd> output;
            input.scans.data = run.measured_values;
            input.flags.data.resize(
                run.input_flag_bits.rows(), run.input_flag_bits.cols());
            for (Eigen::Index row = 0; row < input.flags.data.rows(); ++row) {
                for (Eigen::Index column = 0;
                     column < input.flags.data.cols(); ++column) {
                    input.flags.data(row, column) =
                        run.input_flag_bits(row, column) != 0;
                }
            }
            input.scan_indices.data.resize(4);
            input.scan_indices.data <<
                static_cast<Eigen::Index>(run.selected_first_common_slot),
                static_cast<Eigen::Index>(
                    run.selected_past_last_common_slot - 1),
                static_cast<Eigen::Index>(run.first_common_slot),
                static_cast<Eigen::Index>(run.past_last_common_slot - 1);
            input.index.data = runtime.mapping_handle()->scope().scan_index;

            const auto &pointing = runtime.mapping_handle()
                ->carriers_handle()->pointing_handle()->network(
                    run.run.network_id);
            const auto first = pointing.local_row(run.run.first_native_row);
            const auto rows = static_cast<Eigen::Index>(run.run.row_count());
            for (const auto &[name, values] : pointing.telescope_data()) {
                input.tel_data.data[name] = values.segment(first, rows);
            }
            for (const auto &[axis, values] :
                 pointing.pointing_offsets_arcsec()) {
                input.pointing_offsets_arcsec.data[axis] =
                    values.segment(first, rows);
            }

            engine.apply_learned_detector_exclusions(
                input, local_calib, "pre_rtc_detector_exclusion", true,
                false, true, true);
            const auto &raw_source_protection =
                raw_time_chunk_config(engine).despike.source_protection;
            engine.apply_learned_sample_masks(
                input, local_calib, true, "pre_rtc",
                raw_source_protection.active,
                raw_source_protection.radius_arcsec,
                static_cast<long long>(run.first_common_slot));
            NativeDetectorFlagBitsMatrix effective_input_flag_bits =
                run.input_flag_bits;
            for (Eigen::Index row = 0;
                 row < input.flags.data.rows(); ++row) {
                for (Eigen::Index column = 0;
                     column < input.flags.data.cols(); ++column) {
                    if (input.flags.data(row, column) &&
                        effective_input_flag_bits(row, column) == 0) {
                        effective_input_flag_bits(row, column) |=
                            native_learned_rtc_flag_bit_v2;
                    }
                }
            }

            (void)local_rtc.run(
                input, output, local_calib, local_telescope,
                engine.omb.pixel_size_rad,
                active_map_grouping_name(engine));
            local_rtc.remove_flagged_dets(output, local_calib.apt);
            if (raw.flagging.lower_tod_inv_var_factor != 0.0 ||
                raw.flagging.upper_tod_inv_var_factor != 0.0) {
                auto outlier_calib = local_rtc.remove_bad_dets(
                    output, local_calib,
                    active_map_grouping_name(engine));
                (void)outlier_calib;
            }
            const auto fact_prefix =
                "segment_" + std::to_string(run.segment_ordinal) +
                "_network_" + std::to_string(run.run.network_id) +
                "_first_row_" +
                std::to_string(run.run.first_native_row) + "_";
            const auto source =
                local_rtc.snapshot_source_protection_diag_summary(
                    input.index.data);
            source_mask_facts.emplace_back(
                fact_prefix + "enabled", source.enabled ? "true" : "false");
            source_mask_facts.emplace_back(
                fact_prefix + "protected_samples",
                std::to_string(source.enabled ? source.protected_samples : 0));
            source_mask_facts.emplace_back(
                fact_prefix + "mask_digest",
                source.enabled ? source.mask_digest : "disabled");
            if (source.enabled &&
                (source.protected_samples < 0 ||
                 source.mask_digest == "unavailable")) {
                throw std::logic_error(
                    "native RTC source-protection realization is incomplete");
            }
            add_native_jinc_trace_count(
                trace.rtc_source_masked_sample_count,
                source.enabled
                    ? static_cast<std::size_t>(source.protected_samples)
                    : 0U,
                "RTC source-mask");
            auto detector_learning =
                local_rtc.snapshot_detector_diag_summary(input.index.data);
            for (auto &summary : detector_learning) {
                const auto add_common_slot_offset = [&](auto &event) {
                    if (!event.valid()) return;
                    event.sample +=
                        static_cast<int>(run.first_common_slot);
                    event.start_sample +=
                        static_cast<int>(run.first_common_slot);
                    event.end_sample +=
                        static_cast<int>(run.first_common_slot);
                };
                add_common_slot_offset(summary.local_raw_event);
                add_common_slot_offset(summary.local_delta_event);
            }
            engine.collect_rtc_learning_diagnostics(
                input, output, local_calib, detector_learning);
            if (source.enabled) {
                ReductionLearningState::SourceProtectionSummary summary;
                summary.obsnum = engine.observation_identity.obsnum;
                summary.producer = "rtc_despike";
                summary.mode = "map_center_radius";
                summary.iter = engine.iteration.fruit_iter;
                summary.scan = static_cast<int>(input.index.data);
                summary.protected_samples = source.protected_samples;
                summary.total_samples = source.total_samples;
                summary.radius_arcsec = source.radius_arcsec;
                engine.learning.record_source_protection_summary(
                    std::move(summary));
            }
            const auto notches = local_rtc.snapshot_notch_operator_summary(
                input.index.data);
            notch_facts.emplace_back(
                fact_prefix + "operator_count",
                std::to_string(notches.size()));
            for (std::size_t index = 0; index < notches.size(); ++index) {
                const auto &notch = notches[index];
                const auto prefix = fact_prefix + "operator_" +
                    std::to_string(index) + "_";
                notch_facts.emplace_back(prefix + "stage", notch.stage);
                notch_facts.emplace_back(
                    prefix + "detector_index",
                    std::to_string(notch.detector_index));
                notch_facts.emplace_back(
                    prefix + "center_hz",
                    mapmaking::jinc_double_hex(notch.center_hz));
                notch_facts.emplace_back(
                    prefix + "width_hz",
                    mapmaking::jinc_double_hex(notch.width_hz));
                notch_facts.emplace_back(
                    prefix + "zero_phase",
                    notch.zero_phase ? "true" : "false");
                if (notch.stage == "configured_tod") {
                    ++trace.configured_notch_applied_count;
                }
                else if (notch.stage == "line_audit_fixed_pre") {
                    ++trace.fixed_notch_count;
                }
                else if (notch.stage == "line_audit_shared_pre" ||
                         notch.stage == "line_audit_shared_post") {
                    ++trace.dynamic_notch_count;
                }
                else if (notch.stage == "line_audit_detector_post") {
                    ++trace.detector_notch_count;
                }
                else {
                    throw std::logic_error(
                        "native RTC notch realization has an unknown stage");
                }
            }
            const auto selected_rows = static_cast<Eigen::Index>(
                run.selected_row_count());
            const auto selected_offset = static_cast<Eigen::Index>(
                run.selected_row_offset());
            const Eigen::MatrixXd selected_measured =
                run.measured_values.middleRows(
                    selected_offset, selected_rows);
            if (output.scans.data.rows() != selected_rows ||
                output.scans.data.cols() != run.measured_values.cols() ||
                output.flags.data.rows() != selected_rows ||
                output.flags.data.cols() != run.input_flag_bits.cols() ||
                output.fcf.data.size() != output.scans.data.cols()) {
                throw std::logic_error(
                    "established native RTC body returned an unequal run-local shape");
            }
            const bool kernel_expected =
                raw_time_chunk_config(engine).kernel.enabled;
            if (kernel_expected != (output.kernel.data.size() != 0) ||
                (kernel_expected &&
                 (output.kernel.data.rows() != output.scans.data.rows() ||
                  output.kernel.data.cols() != output.scans.data.cols() ||
                  !output.kernel.data.array().isFinite().all()))) {
                throw std::logic_error(
                    "established native RTC body returned incomplete run-local kernel support");
            }
            for (std::size_t local = 0;
                 local < run.detector_columns.size(); ++local) {
                const auto detector = run.detector_columns[local];
                auto value = output.fcf.data(
                    static_cast<Eigen::Index>(local));
                reconcile_native_rtc_detector_result(
                    selected_measured,
                    static_cast<Eigen::Index>(local),
                    runtime.mapping_handle()->binding(detector).apt_flag,
                    output.scans.data, output.flags.data, value);
                output.fcf.data(static_cast<Eigen::Index>(local)) = value;
            }
            fcf_contract.observe(
                run.detector_columns, output.fcf.data,
                static_cast<std::size_t>(output.scans.data.rows()));

            NativeDetectorFlagBitsMatrix flags =
                effective_input_flag_bits.middleRows(
                    selected_offset, selected_rows);
            for (Eigen::Index row = 0; row < flags.rows(); ++row) {
                for (Eigen::Index column = 0; column < flags.cols();
                     ++column) {
                    if (output.flags.data(row, column) &&
                        flags(row, column) == 0) {
                        flags(row, column) |=
                            native_rtc_processing_flag_bit_v2;
                    }
                }
            }
            return NativeRtcProcessedRun{
                std::move(output.scans.data), std::move(flags),
                output.kernel.data.size() == 0
                    ? std::optional<Eigen::MatrixXd>{}
                    : std::optional<Eigen::MatrixXd>{
                          std::move(output.kernel.data)}};
        };

    const NativeRtcCohortNumericalBody cohort_numerical_body =
        run_global_rtc_body
            ? NativeRtcCohortNumericalBody{
                  [&](const NativeRtcCohortInput &cohort) {
                      const auto &alignment = *runtime.mapping_handle()
                                                   ->carriers_handle()
                                                   ->alignment_handle();
                      const auto network_id =
                          alignment.participant_network_ids().front();
                      const auto first_native_row = alignment.association(
                          network_id, cohort.common_slots.front()).native_row;
                      const auto past_last_native_row = alignment.association(
                          network_id, cohort.common_slots.back()).native_row + 1;
                      if (past_last_native_row - first_native_row !=
                          static_cast<TimestreamNativeRow>(
                              cohort.common_slots.size())) {
                          throw std::logic_error(
                              "native global RTC cohort reference is not contiguous");
                      }
                      std::vector<TimestreamDetectorColumn> detectors(
                          runtime.mapping_handle()->detector_count());
                      for (std::size_t detector = 0;
                           detector < detectors.size(); ++detector) {
                          detectors[detector] =
                              static_cast<TimestreamDetectorColumn>(detector);
                      }
                      std::stable_sort(
                          detectors.begin(), detectors.end(),
                          [&](const auto lhs, const auto rhs) {
                              const auto lhs_nw = runtime.mapping_handle()
                                  ->binding(lhs).network_id;
                              const auto rhs_nw = runtime.mapping_handle()
                                  ->binding(rhs).network_id;
                              return lhs_nw != rhs_nw
                                  ? lhs_nw < rhs_nw : lhs < rhs;
                          });
                      NativeRtcRunInput input;
                      input.segment_ordinal = cohort.segment_ordinal;
                      input.first_common_slot = cohort.common_slots.front();
                      input.past_last_common_slot =
                          cohort.common_slots.back() + 1;
                      input.selected_first_common_slot =
                          cohort.selected_first_common_slot;
                      input.selected_past_last_common_slot =
                          cohort.selected_past_last_common_slot;
                      input.run = NativeContiguousRun{
                          network_id, first_native_row,
                          past_last_native_row, {}, {}};
                      input.common_slots = cohort.common_slots;
                      input.detector_columns = detectors;
                      input.measured_values.resize(
                          cohort.values.rows(), cohort.values.cols());
                      input.input_flag_bits.resize(
                          cohort.flag_bits.rows(), cohort.flag_bits.cols());
                      for (std::size_t local = 0;
                           local < detectors.size(); ++local) {
                          const auto detector = detectors[local];
                          input.measured_values.col(
                              static_cast<Eigen::Index>(local)) =
                              cohort.values.col(detector);
                          input.input_flag_bits.col(
                              static_cast<Eigen::Index>(local)) =
                              cohort.flag_bits.col(detector);
                      }
                      auto ordered = run_numerical_body(input);
                      NativeRtcProcessedRun result;
                      result.values.resize(
                          ordered.values.rows(), ordered.values.cols());
                      result.flag_bits.resize(
                          ordered.flag_bits.rows(), ordered.flag_bits.cols());
                      if (ordered.kernel_values) {
                          result.kernel_values.emplace(
                              ordered.kernel_values->rows(),
                              ordered.kernel_values->cols());
                      }
                      for (std::size_t local = 0;
                           local < detectors.size(); ++local) {
                          const auto detector = detectors[local];
                          result.values.col(detector) = ordered.values.col(
                              static_cast<Eigen::Index>(local));
                          result.flag_bits.col(detector) =
                              ordered.flag_bits.col(
                                  static_cast<Eigen::Index>(local));
                          if (ordered.kernel_values) {
                              result.kernel_values->col(detector) =
                                  ordered.kernel_values->col(
                                      static_cast<Eigen::Index>(local));
                          }
                      }
                      return result;
                  }}
            : NativeRtcCohortNumericalBody{};
    auto result = dispatch_native_rtc_runs(
        *runtime.mapping_handle(),
        {factor, false, runtime.selected_first_common_slot(),
         runtime.selected_past_last_common_slot()},
        run_numerical_body, cohort_numerical_body);
    auto fcf = fcf_contract.finish();
    const auto cohorts = detail::make_native_ptc_rtc_cohort_segments(
        *runtime.mapping_handle(), result);
    Eigen::Index rtc_rows = 0;
    for (const auto &cohort : cohorts) {
        rtc_rows += static_cast<Eigen::Index>(cohort.rows.size());
    }
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> rtc_flags(
        rtc_rows,
        static_cast<Eigen::Index>(runtime.mapping_handle()->detector_count()));
    rtc_flags.setConstant(true);
    Eigen::Index row = 0;
    for (const auto &cohort : cohorts) {
        for (const auto &cohort_row : cohort.rows) {
            for (std::size_t detector = 0;
                 detector < cohort_row.detector_samples.size(); ++detector) {
                const auto &sample = cohort_row.detector_samples[detector];
                if (!sample) {
                    throw std::logic_error(
                        "native RTC trace contains an absent complete-cohort cell");
                }
                rtc_flags(row, static_cast<Eigen::Index>(detector)) =
                    sample->delivered_flag_bits != 0;
            }
            ++row;
        }
    }
    trace.detector_count = runtime.mapping_handle()->detector_count();
    trace.detector_sample_count = static_cast<std::size_t>(rtc_flags.size());
    trace.rtc_flagged_sample_count =
        static_cast<std::size_t>(rtc_flags.array().count());
    trace.rtc_flags_digest = mapmaking::jinc_matrix_digest(rtc_flags);
    trace.rtc_source_mask_digest = native_jinc_trace_facts_digest(
        "native-rtc-source-mask-realization-v2", source_mask_facts);
    trace.rtc_notch_operators_digest = native_jinc_trace_facts_digest(
        "native-rtc-notch-operator-realization-v2", notch_facts);
    runtime.jinc_processing_trace = std::move(trace);
    runtime.fcf = std::move(fcf);
    return result;
}

template <class Engine>
Eigen::VectorXI populate_native_ptc_group_pointing(
    Engine &engine, const NativePtcGroupWorkingSet &group,
    timestream::TCData<timestream::TCDataKind::PTC,
                       Eigen::MatrixXd> &data,
    const Eigen::VectorXI &map_indices,
    const NativeMeasuredDetectorScan &mapping) {
    const auto detector_count = static_cast<Eigen::Index>(
        engine.calib.n_dets);
    if (map_indices.size() != detector_count ||
        data.scans.data.rows() != group.slot_count() ||
        data.scans.data.cols() != group.detector_count()) {
        throw std::logic_error(
            "native fruit-loop group pointing has unequal shape");
    }
    const auto require_apt = [&](const char *name) -> const auto & {
        const auto found = engine.calib.apt.find(name);
        if (found == engine.calib.apt.end() ||
            found->second.size() != detector_count) {
            throw std::logic_error(
                std::string{"native fruit-loop projection lacks exact "} +
                name + " inventory");
        }
        return found->second;
    };
    const auto &x_t = require_apt("x_t");
    const auto &y_t = require_apt("y_t");
    Eigen::VectorXI local_map_indices(group.detector_count());
    auto &latitudes = data.pointing.data["lat"];
    auto &longitudes = data.pointing.data["lon"];
    latitudes.resize(group.slot_count(), group.detector_count());
    longitudes.resize(group.slot_count(), group.detector_count());
    const auto &native_pointing =
        *mapping.carriers_handle()->pointing_handle();
    for (Eigen::Index local = 0; local < group.detector_count(); ++local) {
        const auto detector = group.detector_columns().at(
            static_cast<std::size_t>(local));
        const auto &binding = mapping.binding(detector);
        const NativeScienceDetectorProjection projection{
            detector, binding.output_uid, binding.array,
            binding.network_id, binding.apt_flag, map_indices(detector),
            native_science_projection_detail::resolve_detector_offset_arcsec(
                x_t(detector), binding.apt_flag),
            native_science_projection_detail::resolve_detector_offset_arcsec(
                y_t(detector), binding.apt_flag)};
        local_map_indices(local) = map_indices(detector);
        for (Eigen::Index row = 0; row < group.slot_count(); ++row) {
            const auto &cell = group.cell(row, local);
            if (!cell.identity) {
                throw std::logic_error(
                    "native fruit-loop group cell lacks exact identity");
            }
            const auto [latitude, longitude] =
                native_science_projection_detail::project_native_pointing(
                    native_pointing.network(binding.network_id),
                    *cell.identity, projection,
                    engine.telescope.pixel_axes,
                    active_map_grouping_name(engine));
            latitudes(row, local) = latitude;
            longitudes(row, local) = longitude;
        }
    }
    return local_map_indices;
}

template <class Engine>
NativePtcPreparedOperation run_native_ptc_numerical_bodies(
    Engine &engine, NativeScanRuntimeState &runtime,
    const NativeRtcDispatchResult &rtc,
    const Eigen::VectorXI &map_indices,
    std::size_t &fruit_loop_subtraction_sample_count) {
    if (!runtime.jinc_processing_trace) {
        throw std::logic_error(
            "native PTC execution lacks its staged RTC realization trace");
    }
    auto &trace = *runtime.jinc_processing_trace;
    std::vector<std::pair<std::string, std::string>> mean_facts;
    std::vector<std::pair<std::string, std::string>> pca_facts;
    std::vector<std::pair<std::string, std::string>> second_pass_facts{
        {"enabled", engine.ptcproc.second_pass_local.enabled
                        ? "true" : "false"}};
    std::string grouping = "detector";
    if (engine.ptcproc.run_clean) {
        if (engine.ptcproc.cleaner.grouping.size() != 1) {
            throw std::logic_error(
                "native consumer requires exactly one established PTC grouping");
        }
        grouping = engine.ptcproc.cleaner.grouping.front();
    }
    const auto normalized =
        timestream::Cleaner::normalize_group_name(grouping);
    NativePtcCohortRequest request{
        normalized, FinitePcaPlaceholder::checked(0.0), {}, {}, false,
        false};
    const auto duplicate = engine.calib.apt.find("duplicate_tone");
    if (duplicate == engine.calib.apt.end() ||
        duplicate->second.size() !=
            static_cast<Eigen::Index>(
                runtime.mapping_handle()->detector_count())) {
        throw std::logic_error(
            "native PTC lacks exact duplicate-tone detector state");
    }
    for (const auto &run : rtc.runs) {
        for (const auto &support : run.support) {
            for (const auto detector : support.detector_columns) {
                if (duplicate->second(detector) == 0.0) continue;
                const auto &binding =
                    runtime.mapping_handle()->binding(detector);
                if (!binding.apt_flag.has_value() ||
                    *binding.apt_flag != 0) {
                    continue;
                }
                const NativeDetectorSampleKey key{
                    support.selected_anchor.key(), detector};
                if (!request.operation_exclusion_bits
                         .emplace(
                             key,
                             native_duplicate_tone_exclusion_bit_v2)
                         .second) {
                    throw std::logic_error(
                        "native duplicate-tone exclusion repeats a detector sample");
                }
            }
        }
    }
    request.corr_grouping_enabled =
        engine.ptcproc.cleaner.corr_grouping.enabled;
    request.requires_second_pass_window =
        engine.ptcproc.second_pass_local.enabled;
    request.optional_modes.null_model_active_for_operation =
        engine.ptcproc.cleaner.null_model.enabled &&
        engine.ptcproc.cleaner.null_model_enabled_for_group(normalized);
    request.optional_modes.adaptive_selector_active_for_operation =
        engine.ptcproc.cleaner.adaptive_selector.enabled &&
        engine.ptcproc.cleaner.adaptive_selector_enabled_for_group(
            normalized) && normalized != "corr_nw";
    request.optional_modes.marchenko_pastur_active_for_operation =
        engine.ptcproc.cleaner.marchenko_pastur.enabled &&
        engine.ptcproc.cleaner.marchenko_pastur_enabled_for_group(
            normalized);
    request.optional_modes.marchenko_pastur_band_requested =
        engine.ptcproc.cleaner.marchenko_pastur.band_low_Hz > 0.0 ||
        engine.ptcproc.cleaner.marchenko_pastur.band_high_Hz > 0.0;

    NativePtcCorrGroupingBody corr_body;
    if (normalized == "corr_nw" && request.corr_grouping_enabled) {
        corr_body = NativePtcCorrGroupingBody{
            [&](const NativePtcGroupWorkingSet &group) {
                return engine.ptcproc.cleaner.get_corr_groups(
                    group.values(), group.exclusion_flags(),
                    group.apt_exclusion_flags()).groups;
            }};
    }
    auto prepared = prepare_native_ptc_cohorts(
        runtime.ledger(), rtc, request, corr_body);
    std::vector<Eigen::Index> flag_rows_per_segment(
        prepared.segment_count(), -1);
    for (const auto &group : prepared.groups()) {
        auto &rows = flag_rows_per_segment.at(group.segment_ordinal());
        if (rows < 0) rows = group.slot_count();
        if (rows != group.slot_count()) {
            throw std::logic_error(
                "native PTC flag segment row counts are unequal");
        }
    }
    std::vector<Eigen::Index> flag_offsets(
        flag_rows_per_segment.size(), 0);
    Eigen::Index flag_total_rows = 0;
    for (std::size_t segment = 0;
         segment < flag_rows_per_segment.size(); ++segment) {
        if (flag_rows_per_segment[segment] <= 0) {
            throw std::logic_error(
                "native PTC flag segment inventory is incomplete");
        }
        flag_offsets[segment] = flag_total_rows;
        flag_total_rows += flag_rows_per_segment[segment];
    }
    const auto fruit_weight_policy = fruit_loop_weight_policy(engine);
    fruit_loop_subtraction_sample_count = 0;
    auto processed = run_native_ptc_groups(
        prepared, [&](const NativePtcGroupWorkingSet &group) {
            auto local_calib = make_native_detector_subset_calibration(
                engine.calib, group.detector_columns());
            auto local_ptc = engine.ptcproc;
            local_ptc.cleaner.grouping = {"all"};
            local_ptc.mask_radius_arcsec = 0.0;
            timestream::TCData<timestream::TCDataKind::PTC,
                               Eigen::MatrixXd> data;
            data.scans.data = group.values();
            data.flags.data = group.exclusion_flags();
            if (group.kernel_values()) {
                data.kernel.data = *group.kernel_values();
            }
            data.index.data = runtime.mapping_handle()->scope().scan_index;
            const auto segment_offset =
                flag_offsets.at(group.segment_ordinal());
            const auto &source_protection =
                processed_time_chunk_config(engine)
                    .flagging.second_pass_local.source_protection;
            engine.apply_learned_sample_masks(
                data, local_calib, false, "pre_ptc",
                source_protection.active,
                source_protection.radius_arcsec,
                static_cast<long long>(segment_offset));
            engine.apply_learned_detector_exclusions(
                data, local_calib, "pre_ptc_detector_exclusion", false,
                true, true, true);
            const auto preclean_flags = data.flags.data;
            if (fruit_weight_policy.use_noise_weights) {
                auto local_map_indices =
                    populate_native_ptc_group_pointing(
                        engine, group, data, map_indices,
                        *runtime.mapping_handle());
                long long applied = 0;
                local_ptc.template map_to_tod<
                    timestream::TCProc::SourceType::NegativeMap>(
                    local_ptc.tod_mb, data, local_calib,
                    local_map_indices, engine.telescope.pixel_axes,
                    active_map_grouping_name(engine), &applied);
                if (applied < 0 ||
                    static_cast<std::size_t>(applied) >
                        std::numeric_limits<std::size_t>::max() -
                            fruit_loop_subtraction_sample_count) {
                    throw std::overflow_error(
                        "native fruit-loop subtraction count overflow");
                }
                fruit_loop_subtraction_sample_count +=
                    static_cast<std::size_t>(applied);
            }
            if (group.role() != NativePtcGroupRole::pca_clean) {
                return NativePtcNumericalResult{
                    std::move(data.scans.data),
                    data.kernel.data.size() == 0
                        ? std::optional<Eigen::MatrixXd>{}
                        : std::optional<Eigen::MatrixXd>{
                              std::move(data.kernel.data)},
                    preclean_flags,
                    std::move(data.flags.data)};
            }
            local_ptc.run(
                data, data, local_calib,
                engine.telescope.pixel_axes,
                active_map_grouping_name(engine));
            const auto prefix =
                "segment_" + std::to_string(group.segment_ordinal()) +
                "_group_" + std::to_string(group.group_key()) +
                "_subgroup_" + std::to_string(group.subgroup_index()) +
                "_";
            const auto mean = local_ptc.snapshot_mean_realization_summary(
                data.index.data);
            if (!mean || !mean->mean_subtracted ||
                mean->mask_digest == "unavailable") {
                throw std::logic_error(
                    "native PTC mean realization is incomplete");
            }
            mean_facts.emplace_back(
                prefix + "source_mask_applied",
                mean->source_mask_applied ? "true" : "false");
            mean_facts.emplace_back(
                prefix + "masked_sample_count",
                std::to_string(mean->masked_sample_count));
            mean_facts.emplace_back(
                prefix + "mask_digest", mean->mask_digest);
            add_native_jinc_trace_count(
                trace.ptc_mean_masked_sample_count,
                mean->masked_sample_count, "PTC mean-mask");

            const auto pca = local_ptc.snapshot_pca_realization_summary(
                data.index.data);
            for (std::size_t index = 0; index < pca.size(); ++index) {
                const auto &entry = pca[index];
                const auto pca_prefix = prefix + "solve_" +
                    std::to_string(index) + "_";
                pca_facts.emplace_back(
                    pca_prefix + "grouping", entry.grouping);
                pca_facts.emplace_back(
                    pca_prefix + "group_key",
                    std::to_string(entry.group_key));
                pca_facts.emplace_back(
                    pca_prefix + "array_index",
                    std::to_string(entry.array_index));
                pca_facts.emplace_back(
                    pca_prefix + "configured_cut",
                    std::to_string(entry.configured_cut));
                pca_facts.emplace_back(
                    pca_prefix + "applied_cut",
                    std::to_string(entry.applied_cut));
                pca_facts.emplace_back(
                    pca_prefix + "forced_limit_index",
                    std::to_string(entry.forced_limit_index));
                pca_facts.emplace_back(
                    pca_prefix + "eigenvalue_digest",
                    entry.eigenvalue_digest);
                pca_facts.emplace_back(
                    pca_prefix + "eigenvector_digest",
                    entry.eigenvector_digest);
            }
            add_native_jinc_trace_count(
                trace.pca_solve_count, pca.size(), "PTC PCA solve");
            auto second_pass =
                local_ptc.snapshot_second_pass_summary(data.index.data);
            for (auto &summary : second_pass) {
                if (summary.top_candidate_cluster_sample !=
                    timestream::kTransientFillInt) {
                    summary.top_candidate_cluster_sample +=
                        static_cast<int>(segment_offset);
                }
                if (summary.top_event.valid()) {
                    summary.top_event.sample +=
                        static_cast<int>(segment_offset);
                    summary.top_event.start_sample +=
                        static_cast<int>(segment_offset);
                    summary.top_event.end_sample +=
                        static_cast<int>(segment_offset);
                }
                for (auto &event : summary.candidate_events) {
                    event.sample += static_cast<int>(segment_offset);
                    event.start_sample +=
                        static_cast<int>(segment_offset);
                    event.end_sample += static_cast<int>(segment_offset);
                    event.cluster_sample +=
                        static_cast<int>(segment_offset);
                }
            }
            engine.collect_ptc_learning_diagnostics(
                data, local_calib, second_pass, {});
            second_pass_facts.emplace_back(
                prefix + "summary_count",
                std::to_string(second_pass.size()));
            for (std::size_t index = 0;
                 index < second_pass.size(); ++index) {
                const auto &summary = second_pass[index];
                const auto summary_prefix = prefix + "summary_" +
                    std::to_string(index) + "_";
                second_pass_facts.emplace_back(
                    summary_prefix + "network",
                    std::to_string(summary.nw));
                second_pass_facts.emplace_back(
                    summary_prefix + "accepted_clusters",
                    std::to_string(summary.n_accepted_clusters));
                second_pass_facts.emplace_back(
                    summary_prefix + "accepted_events",
                    std::to_string(summary.n_accepted_events));
                second_pass_facts.emplace_back(
                    summary_prefix + "rejected_events",
                    std::to_string(summary.n_rejected_events));
                second_pass_facts.emplace_back(
                    summary_prefix + "newly_flagged_fraction",
                    mapmaking::jinc_double_hex(
                        summary.newly_flagged_fraction));
                std::size_t accepted_index = 0;
                for (const auto &event : summary.candidate_events) {
                    if (!event.accepted) continue;
                    const auto event_prefix = summary_prefix +
                        "accepted_event_" +
                        std::to_string(accepted_index++) + "_";
                    second_pass_facts.emplace_back(
                        event_prefix + "uid", std::to_string(event.uid));
                    second_pass_facts.emplace_back(
                        event_prefix + "kind", std::to_string(event.kind));
                    second_pass_facts.emplace_back(
                        event_prefix + "start_sample",
                        std::to_string(event.start_sample));
                    second_pass_facts.emplace_back(
                        event_prefix + "end_sample",
                        std::to_string(event.end_sample));
                    second_pass_facts.emplace_back(
                        event_prefix + "score",
                        mapmaking::jinc_double_hex(event.score));
                }
                second_pass_facts.emplace_back(
                    summary_prefix + "accepted_event_record_count",
                    std::to_string(accepted_index));
            }
            return NativePtcNumericalResult{
                std::move(data.scans.data),
                data.kernel.data.size() == 0
                    ? std::optional<Eigen::MatrixXd>{}
                    : std::optional<Eigen::MatrixXd>{
                          std::move(data.kernel.data)},
                preclean_flags,
                std::move(data.flags.data)};
        }, true);
    scatter_native_ptc_results_transactionally(
        runtime.ledger(), prepared, processed);
    mean_facts.emplace_back(
        "operation_sequence", std::to_string(prepared.operation().sequence));
    pca_facts.emplace_back(
        "operation_sequence", std::to_string(prepared.operation().sequence));
    pca_facts.emplace_back(
        "solve_count", std::to_string(trace.pca_solve_count));
    pca_facts.emplace_back(
        "second_pass_realization_digest",
        native_jinc_trace_facts_digest(
            "native-ptc-second-pass-realization-v1",
            second_pass_facts));
    trace.ptc_mean_mask_digest = native_jinc_trace_facts_digest(
        "native-ptc-mean-mask-realization-v2", mean_facts);
    trace.pca_realization_digest = native_jinc_trace_facts_digest(
        "native-ptc-pca-realization-v2", pca_facts);
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> ptc_preclean_flags(
        flag_total_rows,
        static_cast<Eigen::Index>(
            runtime.mapping_handle()->detector_count()));
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> ptc_flags(
        flag_total_rows,
        static_cast<Eigen::Index>(
            runtime.mapping_handle()->detector_count()));
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> flag_seen(
        ptc_flags.rows(), ptc_flags.cols());
    ptc_preclean_flags.setConstant(true);
    ptc_flags.setConstant(true);
    flag_seen.setConstant(false);
    for (std::size_t index = 0;
         index < processed.groups().size(); ++index) {
        const auto &source = prepared.groups().at(index);
        const auto &result = processed.groups().at(index);
        if (!result.exclusion_flags()) {
            throw std::logic_error(
                "native PTC output flag inventory is absent");
        }
        if (!result.preclean_exclusion_flags()) {
            throw std::logic_error(
                "native PTC pre-clean flag inventory is absent");
        }
        for (Eigen::Index local = 0;
             local < source.detector_count(); ++local) {
            const auto detector = source.detector_columns().at(
                static_cast<std::size_t>(local));
            for (Eigen::Index row = 0; row < source.slot_count(); ++row) {
                const auto output_row =
                    flag_offsets.at(source.segment_ordinal()) + row;
                if (flag_seen(output_row, detector)) {
                    throw std::logic_error(
                        "native PTC flag destination is duplicated");
                }
                flag_seen(output_row, detector) = true;
                ptc_preclean_flags(output_row, detector) =
                    (*result.preclean_exclusion_flags())(row, local);
                ptc_flags(output_row, detector) =
                    (*result.exclusion_flags())(row, local);
            }
        }
    }
    if (!flag_seen.array().all()) {
        throw std::logic_error(
            "native PTC flag projection is incomplete");
    }
    runtime.ptc_preclean_flags = std::move(ptc_preclean_flags);
    runtime.ptc_flags = std::move(ptc_flags);
    const bool has_kernel = std::any_of(
        processed.groups().begin(), processed.groups().end(),
        [](const auto &group) { return group.kernel_values().has_value(); });
    if (has_kernel) {
        std::vector<Eigen::Index> rows_per_segment(
            prepared.segment_count(), -1);
        for (const auto &group : prepared.groups()) {
            auto &rows = rows_per_segment.at(group.segment_ordinal());
            if (rows < 0) rows = group.slot_count();
            if (rows != group.slot_count()) {
                throw std::logic_error(
                    "native PTC kernel segment row counts are unequal");
            }
        }
        std::vector<Eigen::Index> offsets(rows_per_segment.size(), 0);
        Eigen::Index total_rows = 0;
        for (std::size_t segment = 0; segment < rows_per_segment.size();
             ++segment) {
            if (rows_per_segment[segment] <= 0) {
                throw std::logic_error(
                    "native PTC kernel segment inventory is incomplete");
            }
            offsets[segment] = total_rows;
            total_rows += rows_per_segment[segment];
        }
        Eigen::MatrixXd kernel(
            total_rows,
            static_cast<Eigen::Index>(
                runtime.mapping_handle()->detector_count()));
        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> seen(
            kernel.rows(), kernel.cols());
        kernel.setConstant(std::numeric_limits<double>::quiet_NaN());
        seen.setConstant(false);
        for (std::size_t index = 0;
             index < processed.groups().size(); ++index) {
            const auto &source = prepared.groups().at(index);
            const auto &result = processed.groups().at(index);
            if (!result.kernel_values()) {
                throw std::logic_error(
                    "native PTC kernel inventory is partial");
            }
            for (Eigen::Index local = 0;
                 local < source.detector_count(); ++local) {
                const auto detector = source.detector_columns().at(
                    static_cast<std::size_t>(local));
                for (Eigen::Index row = 0; row < source.slot_count();
                     ++row) {
                    const auto output_row =
                        offsets.at(source.segment_ordinal()) + row;
                    if (seen(output_row, detector)) {
                        throw std::logic_error(
                            "native PTC kernel destination is duplicated");
                    }
                    seen(output_row, detector) = true;
                    kernel(output_row, detector) =
                        (*result.kernel_values())(row, local);
                }
            }
        }
        if (!(seen.array().all()) ||
            !kernel.array().isFinite().all()) {
            throw std::logic_error(
                "native PTC kernel projection is incomplete or nonfinite");
        }
        runtime.kernel = std::move(kernel);
    }
    else {
        runtime.kernel.reset();
    }
    return prepared;
}

inline NativeScienceProjectionRequest native_projection_request(
    const NativeMeasuredDetectorScan &mapping,
    const Eigen::VectorXI &map_indices,
    const std::map<std::string, Eigen::VectorXd> &apt,
    std::string pixel_axes, std::string map_grouping) {
    if (map_indices.size() !=
            static_cast<Eigen::Index>(mapping.detector_count())) {
        throw std::logic_error(
            "native map-index authority has unequal detector cardinality");
    }
    const auto x = apt.find("x_t");
    const auto y = apt.find("y_t");
    if (x == apt.end() || y == apt.end() ||
        x->second.size() != map_indices.size() ||
        y->second.size() != map_indices.size()) {
        throw std::logic_error(
            "native map projection lacks exact detector offsets");
    }
    NativeScienceProjectionRequest request;
    request.pixel_axes = std::move(pixel_axes);
    request.map_grouping = std::move(map_grouping);
    request.detectors.reserve(mapping.detector_count());
    for (std::size_t detector = 0; detector < mapping.detector_count();
         ++detector) {
        const auto column = static_cast<TimestreamDetectorColumn>(detector);
        const auto &binding = mapping.binding(column);
        const auto az_offset =
            native_science_projection_detail::resolve_detector_offset_arcsec(
                x->second(column), binding.apt_flag);
        const auto el_offset =
            native_science_projection_detail::resolve_detector_offset_arcsec(
                y->second(column), binding.apt_flag);
        request.detectors.push_back({
            column, binding.output_uid, binding.array,
            binding.network_id, binding.apt_flag,
            map_indices(column), az_offset, el_offset});
    }
    return request;
}

struct NativeConsumerPreparedMapScan {
    std::shared_ptr<NativeScanRuntimeState> runtime;
    timestream::TCData<timestream::TCDataKind::PTC, Eigen::MatrixXd>
        ptcdata;
    Eigen::VectorXI map_indices;
};

template <class Diagnostics>
void collect_native_consumer_scan_diagnostics(
    Diagnostics &diagnostics, NativeConsumerPreparedMapScan &prepared) {
    auto &ptcdata = prepared.ptcdata;
    const auto scan_index = ptcdata.index.data;
    if (ptcdata.scans.data.rows() <= 0 ||
        ptcdata.scans.data.cols() <= 0 ||
        ptcdata.flags.data.rows() != ptcdata.scans.data.rows() ||
        ptcdata.flags.data.cols() != ptcdata.scans.data.cols() ||
        ptcdata.weights.data.size() != ptcdata.scans.data.cols() ||
        scan_index < 0) {
        throw std::logic_error(
            "native consumer diagnostics require one complete PTC scan");
    }
    for (const auto &name : diagnostics.det_stats_header) {
        const auto found = diagnostics.stats.find(name);
        if (found == diagnostics.stats.end() ||
            found->second.rows() != ptcdata.scans.data.cols() ||
            scan_index >= found->second.cols()) {
            throw std::logic_error(
                "native consumer detector diagnostics are not initialized");
        }
    }
    diagnostics.calc_stats(ptcdata);
}

template <class Engine, class RtcData>
NativeConsumerPreparedMapScan prepare_native_consumer_map_scan(
    Engine &engine, const RtcData &source_rtc,
    const Eigen::VectorXI &map_indices) {
    if (!source_rtc.native_runtime) {
        throw std::logic_error(
            "native-required scan lacks its scan-owned runtime transaction");
    }
    auto runtime = source_rtc.native_runtime;
    runtime->rtc = run_native_rtc_numerical_bodies(engine, *runtime);
    std::size_t fruit_loop_subtraction_sample_count = 0;
    runtime->ptc_prepared = run_native_ptc_numerical_bodies(
        engine, *runtime, *runtime->rtc, map_indices,
        fruit_loop_subtraction_sample_count);
    runtime->science_projection = make_native_science_projection(
        runtime->ledger(), *runtime->ptc_prepared,
        native_projection_request(
            *runtime->mapping_handle(), map_indices, engine.calib.apt,
            engine.telescope.pixel_axes,
            active_map_grouping_name(engine)));

    NativeConsumerPreparedMapScan result;
    result.runtime = runtime;
    result.map_indices = map_indices;
    result.ptcdata.scans.data = runtime->science_projection->values();
    result.ptcdata.flags.data = runtime->science_projection->flags();
    if (runtime->ptc_flags) {
        if (runtime->ptc_flags->rows() !=
                result.ptcdata.flags.data.rows() ||
            runtime->ptc_flags->cols() !=
                result.ptcdata.flags.data.cols() ||
            (result.ptcdata.flags.data.array() &&
             !runtime->ptc_flags->array()).any()) {
            throw std::logic_error(
                "native PTC runtime flags are unequal or remove an exclusion");
        }
        result.ptcdata.flags.data = *runtime->ptc_flags;
    }
    if (runtime->kernel) result.ptcdata.kernel.data = *runtime->kernel;
    result.ptcdata.index.data = runtime->mapping_handle()->scope().scan_index;
    result.ptcdata.fcf.data = runtime->fcf;
    result.ptcdata.status.calibrated =
        raw_time_chunk_config(engine).flux_calibration_enabled;
    result.ptcdata.status.cleaned =
        processed_time_chunk_config(engine).clean.enabled;
    result.ptcdata.noise.data = source_rtc.noise.data;
    const auto &noise = noise_config(engine);
    result.runtime->noise_assignment =
        make_native_noise_assignment_summary_v3(
            result.ptcdata.noise.data, noise_maps_enabled(engine),
            noise.randomize_dets,
            noise_maps_enabled(engine)
                ? static_cast<std::size_t>(noise.n_noise_maps)
                : 0U,
            runtime->mapping_handle()->detector_count());
    result.ptcdata.map_indices.data = map_indices;

    const auto &fruit_config = fruit_loops_config(engine);
    const auto fruit_weight_policy = fruit_loop_weight_policy(engine);
    NativeFruitLoopFeedbackSummaryV3 fruit_summary;
    if (fruit_config.enabled) {
        fruit_summary.enabled = true;
        fruit_summary.source_model_available =
            fruit_weight_policy.use_noise_weights;
        fruit_summary.iteration = engine.iteration.fruit_iter;
        fruit_summary.interpolation_mode =
            engine.ptcproc.fruit_loops_interp_mode;
        fruit_summary.support_authority =
            native_fruit_loop_feedback_authority_v3;
        if (fruit_summary.source_model_available) {
            fruit_summary.model_map_count =
                engine.ptcproc.tod_mb.signal.size();
            fruit_summary.subtraction_sample_count =
                fruit_loop_subtraction_sample_count;
            fruit_summary.keep_source_subtracted_weights =
                fruit_weight_policy.keep_source_subtracted_weights;
        }
    }

    auto local_ptc = engine.ptcproc;
    local_ptc.source_mask_radius_arcsec = 0.0;
    const auto calculate_and_reset_weights = [&](bool noise_only) {
        local_ptc.calc_weights(
            result.ptcdata, engine.calib.apt, engine.telescope,
            noise_only);
        const auto authoritative_flags = result.ptcdata.flags.data;
        auto reset_calib = local_ptc.reset_weights(
            result.ptcdata, engine.calib,
            active_map_grouping_name(engine));
        (void)reset_calib;
        for (Eigen::Index detector = 0;
             detector < result.ptcdata.flags.data.cols(); ++detector) {
            if ((result.ptcdata.flags.data.col(detector).array() &&
                 !authoritative_flags.col(detector).array()).any()) {
                result.ptcdata.weights.data(detector) = 0.0;
            }
        }
        result.ptcdata.flags.data = authoritative_flags;
    };

    if (fruit_weight_policy.use_noise_weights) {
        calculate_and_reset_weights(true);
        if (mapmaking_enabled(engine) && noise_maps_enabled(engine)) {
            const auto noise_projection =
                runtime->science_projection->with_deterministic_state(
                    result.ptcdata.scans.data,
                    result.ptcdata.flags.data);
            bool run_omb = false;
            populate_naive_or_jinc_maps_native(
                mapmaking_config(engine).method, engine.naive_mm,
                engine.jinc_mm, result.ptcdata, engine.omb, engine.cmb,
                result.map_indices, engine.telescope.pixel_axes,
                engine.calib.apt, engine.telescope.d_fsmp, run_omb, true,
                noise_projection);
            fruit_summary.noise_map_pass_applied = true;
        }
        result.ptcdata.pointing.data["lat"] =
            runtime->science_projection->latitudes_rad();
        result.ptcdata.pointing.data["lon"] =
            runtime->science_projection->longitudes_rad();
        long long addback_samples = 0;
        local_ptc.template map_to_tod<
            timestream::TCProc::SourceType::Map>(
            local_ptc.tod_mb, result.ptcdata, engine.calib,
            result.map_indices, engine.telescope.pixel_axes,
            active_map_grouping_name(engine), &addback_samples);
        if (addback_samples < 0) {
            throw std::logic_error(
                "native fruit-loop add-back count is negative");
        }
        fruit_summary.addback_sample_count =
            static_cast<std::size_t>(addback_samples);
    }

    const auto &processed = processed_time_chunk_config(engine);
    if (processed.flagging.lower_tod_inv_var_factor != 0.0 ||
        processed.flagging.upper_tod_inv_var_factor != 0.0) {
        auto outlier_calib = local_ptc.remove_bad_dets(
            result.ptcdata, engine.calib,
            active_map_grouping_name(engine));
        (void)outlier_calib;
    }

    if (!fruit_weight_policy.keep_source_subtracted_weights) {
        calculate_and_reset_weights(false);
    }

    runtime->map_projection =
        runtime->science_projection->with_deterministic_state(
            result.ptcdata.scans.data, result.ptcdata.flags.data);
    fruit_summary.validate();
    runtime->fruit_loop_feedback = std::move(fruit_summary);

    if (!runtime->jinc_processing_trace) {
        throw std::logic_error(
            "native map preparation lacks its staged processing trace");
    }
    auto &trace = *runtime->jinc_processing_trace;
    trace.ptc_flagged_sample_count = static_cast<std::size_t>(
        result.ptcdata.flags.data.array().count());
    trace.ptc_flags_digest =
        mapmaking::jinc_matrix_digest(result.ptcdata.flags.data);
    trace.ptc_signal_digest =
        mapmaking::jinc_matrix_digest(result.ptcdata.scans.data);
    trace.ptc_kernel_digest =
        mapmaking::jinc_matrix_digest(result.ptcdata.kernel.data);
    trace.map_indices_digest = mapmaking::jinc_matrix_digest(map_indices);
    const auto apt_flag = engine.calib.apt.find("flag");
    if (apt_flag == engine.calib.apt.end() ||
        apt_flag->second.size() !=
            static_cast<Eigen::Index>(
                runtime->mapping_handle()->detector_count())) {
        throw std::logic_error(
            "native JINC trace lacks exact APT flag state");
    }
    trace.apt_flags_digest =
        mapmaking::jinc_matrix_digest(apt_flag->second);
    trace.apt_flagged_detector_count = static_cast<std::size_t>(
        (apt_flag->second.array() != 0.0).count());
    (void)native_jinc_processing_scan_trace_digest_v2(trace);

    const auto high_weight_summary =
        local_ptc.snapshot_high_weight_summary(
            result.ptcdata.index.data);
    auto learning_calib = engine.calib;
    engine.collect_ptc_learning_diagnostics(
        result.ptcdata, learning_calib, {}, high_weight_summary);

    const auto pre_map_flags = result.ptcdata.flags.data;
    const auto learned_map_columns =
        engine.apply_learned_mapmaking_detector_exclusions(
        result.ptcdata, learning_calib);
    runtime->learned_map_zero_weight_detector_columns.clear();
    for (const auto detector : learned_map_columns) {
        if (detector < 0 ||
            detector >= result.ptcdata.weights.data.size()) {
            throw std::logic_error(
                "learned pre-map detector exclusion is out of range");
        }
        result.ptcdata.weights.data(detector) = 0.0;
        runtime->learned_map_zero_weight_detector_columns.push_back(
            detector);
    }
    result.ptcdata.flags.data = pre_map_flags;
    return result;
}

}  // namespace citlali::pipeline
