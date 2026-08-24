#pragma once

// Private application orchestration for the explicitly activated compact-v2
// Science/Pointing consumer. The Stage 3--6 adapters retain all authority;
// this layer only supplies run/group-local views to the established RTC/PTC
// numerical bodies and prepares the immutable native map input.

#include <citlali/core/mapmaking/science_map_contract.h>
#include <citlali/core/pipeline/native_cohort_product_provenance_v2.h>
#include <citlali/core/pipeline/native_consumer_execution_policy.h>
#include <citlali/core/pipeline/native_scan_runtime_state.h>
#include <citlali/core/pipeline/downsample_config.h>
#include <citlali/core/pipeline/jinc_processing_provenance.h>

#include <netcdf>

#include <citlali/core/timestream/ptc/clean.h>
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
    NativeDetectorFlagBits{1} << 63U;
inline constexpr NativeDetectorFlagBits
    native_duplicate_tone_exclusion_bit_v2 =
        NativeDetectorFlagBits{1} << 62U;

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
NativeCohortMapPublicationRequestV2
make_native_map_publication_request_v2(
    Engine &engine, citlali::config::MapMethod method, bool enabled,
    const NativeScanRuntimeState &runtime,
    const Eigen::VectorXd &eligible_weights) {
    if (!enabled) return {};
    NativeCohortMapPublicationRequestV2 result;
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
    result.jinc_scan_trace_digest =
        native_jinc_processing_scan_trace_digest_v2(
            *runtime.jinc_processing_trace);
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
    Eigen::VectorXd fcf = Eigen::VectorXd::Constant(
        static_cast<Eigen::Index>(runtime.mapping_handle()->detector_count()),
        std::numeric_limits<double>::quiet_NaN());
    mapmaking::JincProcessingScanTrace trace;
    std::vector<std::pair<std::string, std::string>> source_mask_facts;
    std::vector<std::pair<std::string, std::string>> notch_facts;

    auto result = dispatch_native_rtc_runs(
        *runtime.mapping_handle(), {factor, false},
        [&](const NativeRtcRunInput &run) {
            auto local_calib = make_native_detector_subset_calibration(
                engine.calib, run.detector_columns);
            auto local_telescope = engine.telescope;
            auto local_rtc = engine.rtcproc;
            local_rtc.run_downsample = false;

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
            input.scan_indices.data << 0, input.scans.data.rows() - 1,
                0, input.scans.data.rows() - 1;
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

            (void)local_rtc.run(
                input, output, local_calib, local_telescope,
                engine.omb.pixel_size_rad,
                active_map_grouping_name(engine));
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
            if (output.scans.data.rows() != run.measured_values.rows() ||
                output.scans.data.cols() != run.measured_values.cols() ||
                output.flags.data.rows() != run.input_flag_bits.rows() ||
                output.flags.data.cols() != run.input_flag_bits.cols() ||
                output.fcf.data.size() != output.scans.data.cols()) {
                throw std::logic_error(
                    "established native RTC body returned an unequal run-local shape");
            }
            for (std::size_t local = 0;
                 local < run.detector_columns.size(); ++local) {
                const auto detector = run.detector_columns[local];
                auto value = output.fcf.data(
                    static_cast<Eigen::Index>(local));
                reconcile_native_rtc_detector_result(
                    run.measured_values,
                    static_cast<Eigen::Index>(local),
                    runtime.mapping_handle()->binding(detector).apt_flag,
                    output.scans.data, output.flags.data, value);
                output.fcf.data(static_cast<Eigen::Index>(local)) = value;
                auto &stored = fcf(detector);
                if (std::isfinite(stored) &&
                    std::bit_cast<std::uint64_t>(stored) !=
                        std::bit_cast<std::uint64_t>(value)) {
                    throw std::logic_error(
                        "native RTC detector FCF differs between contiguous runs");
                }
                stored = value;
            }

            NativeDetectorFlagBitsMatrix flags = run.input_flag_bits;
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
                std::move(output.scans.data), std::move(flags)};
        });
    if (!fcf.array().isFinite().all()) {
        throw std::logic_error(
            "native RTC dispatch did not realize every detector FCF");
    }
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
NativePtcPreparedOperation run_native_ptc_numerical_bodies(
    Engine &engine, NativeScanRuntimeState &runtime,
    const NativeRtcDispatchResult &rtc) {
    if (!runtime.jinc_processing_trace) {
        throw std::logic_error(
            "native PTC execution lacks its staged RTC realization trace");
    }
    auto &trace = *runtime.jinc_processing_trace;
    std::vector<std::pair<std::string, std::string>> mean_facts;
    std::vector<std::pair<std::string, std::string>> pca_facts;
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
    auto processed = run_native_ptc_groups(
        prepared, [&](const NativePtcGroupWorkingSet &group) {
            auto local_calib = make_native_detector_subset_calibration(
                engine.calib, group.detector_columns());
            auto local_ptc = engine.ptcproc;
            local_ptc.cleaner.grouping = {"all"};
            local_ptc.mask_radius_arcsec = 0.0;
            local_ptc.second_pass_local.enabled = false;
            timestream::TCData<timestream::TCDataKind::PTC,
                               Eigen::MatrixXd> data;
            data.scans.data = group.values();
            data.flags.data = group.exclusion_flags();
            data.index.data = runtime.mapping_handle()->scope().scan_index;
            local_ptc.run(
                data, data, local_calib, engine.telescope.pixel_axes,
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
            return data.scans.data;
        });
    scatter_native_ptc_results_transactionally(
        runtime.ledger(), prepared, processed);
    mean_facts.emplace_back(
        "operation_sequence", std::to_string(prepared.operation().sequence));
    pca_facts.emplace_back(
        "operation_sequence", std::to_string(prepared.operation().sequence));
    pca_facts.emplace_back(
        "solve_count", std::to_string(trace.pca_solve_count));
    trace.ptc_mean_mask_digest = native_jinc_trace_facts_digest(
        "native-ptc-mean-mask-realization-v2", mean_facts);
    trace.pca_realization_digest = native_jinc_trace_facts_digest(
        "native-ptc-pca-realization-v2", pca_facts);
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
    runtime->ptc_prepared = run_native_ptc_numerical_bodies(
        engine, *runtime, *runtime->rtc);
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
    result.ptcdata.index.data = runtime->mapping_handle()->scope().scan_index;
    result.ptcdata.fcf.data = runtime->fcf;
    result.ptcdata.status.calibrated =
        raw_time_chunk_config(engine).flux_calibration_enabled;
    result.ptcdata.status.cleaned =
        processed_time_chunk_config(engine).clean.enabled;
    result.ptcdata.noise.data = source_rtc.noise.data;
    result.ptcdata.map_indices.data = map_indices;

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

    auto local_ptc = engine.ptcproc;
    local_ptc.source_mask_radius_arcsec = 0.0;
    local_ptc.calc_weights(
        result.ptcdata, engine.calib.apt, engine.telescope);
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
    return result;
}

}  // namespace citlali::pipeline
