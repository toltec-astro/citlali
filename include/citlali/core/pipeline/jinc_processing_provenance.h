#pragma once

#include <citlali/core/mapmaking/jinc_contract.h>
#include <citlali/core/pipeline/raw_timestream_policy.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/runtime_policy.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace citlali::pipeline {

inline std::string jinc_processing_bool_text(bool value) {
    return value ? "true" : "false";
}

template <class Values>
std::string jinc_processing_vector_text(const Values &values) {
    std::ostringstream stream;
    for (const auto &value : values) {
        if (stream.tellp() > 0) {
            stream << ',';
        }
        if constexpr (std::is_floating_point_v<
                          std::remove_cv_t<std::remove_reference_t<
                              decltype(value)>>>) {
            stream << mapmaking::jinc_double_hex(value);
        }
        else {
            stream << value;
        }
    }
    return stream.str();
}

inline std::size_t jinc_processing_checked_count(Eigen::Index count,
                                                 const char *name) {
    if (count < 0) {
        throw std::logic_error(std::string{name} + " cannot be negative");
    }
    return static_cast<std::size_t>(count);
}

inline void jinc_processing_add_count(std::size_t &total,
                                      std::size_t value,
                                      const char *name) {
    if (value > std::numeric_limits<std::size_t>::max() - total) {
        throw std::overflow_error(std::string{name} + " overflow");
    }
    total += value;
}

template <class Engine>
inline constexpr bool jinc_processing_provenance_capable_v =
    requires(Engine &engine) {
        engine.omb.jinc_products;
        engine.raw_timestream_plan;
        engine.processed_timestream_plan;
        engine.rtcproc;
        engine.ptcproc.cleaner;
        engine.map_indices.n_maps;
        engine.telescope.d_fsmp;
        engine.telescope.scan_indices;
        engine.runtime_config_provenance;
        engine.typed_config.mapmaking;
    };

template <class Engine>
bool jinc_processing_provenance_active(const Engine &engine) {
    if constexpr (jinc_processing_provenance_capable_v<Engine>) {
        return mapmaking_config(engine).method ==
                   citlali::config::MapMethod::jinc &&
               engine.omb.jinc_products.initialized;
    }
    return false;
}

template <class Engine>
void bind_jinc_processing_configuration(Engine &engine) {
    if (!jinc_processing_provenance_active(engine)) {
        return;
    }
    auto &provenance = engine.omb.jinc_products.provenance;
    if (provenance.processing_configuration_bound) {
        throw std::logic_error(
            "JINC processing configuration is already bound");
    }
    const auto &raw_plan = raw_timestream_plan(engine);
    const auto &processed_plan = processed_timestream_plan(engine);
    if (!raw_plan.initialized || !processed_plan.initialized ||
        !raw_plan.observation.has_value()) {
        throw std::logic_error(
            "JINC processing configuration requires initialized observation plans");
    }

    const auto &raw = raw_plan.effective;
    const auto &processed =
        processed_plan.effective.processed_time_chunk;
    const auto &rtc = engine.rtcproc;
    const auto &ptc = engine.ptcproc;
    const auto &cleaner = ptc.cleaner;
    std::vector<double> configured_notch_widths_hz;
    configured_notch_widths_hz.reserve(rtc.filter.w0s.size());
    for (std::size_t index = 0; index < rtc.filter.w0s.size(); ++index) {
        if (index >= rtc.filter.qs.size() ||
            !std::isfinite(rtc.filter.qs[index]) ||
            rtc.filter.qs[index] <= 0.0) {
            throw std::logic_error(
                "JINC configured notch identity requires finite-positive Q values");
        }
        configured_notch_widths_hz.push_back(
            rtc.filter.w0s[index] / rtc.filter.qs[index]);
    }
    std::vector<std::pair<std::string, std::string>> facts{
        {"raw_timestream_active",
         jinc_processing_bool_text(timestream_processing_enabled(engine))},
        {"kernel_enabled", jinc_processing_bool_text(raw.kernel.enabled)},
        {"despike_enabled", jinc_processing_bool_text(raw.despike.enabled)},
        {"temporal_fir_enabled", jinc_processing_bool_text(raw.filter.enabled)},
        {"configured_notch_enabled",
         jinc_processing_bool_text(
             raw.filter.enabled && raw.filter.notch.enabled)},
        {"iir_highpass_enabled",
         jinc_processing_bool_text(raw.iir_filter.enabled)},
        {"downsample_enabled",
         jinc_processing_bool_text(raw.downsample.enabled)},
        {"ptc_clean_enabled",
         jinc_processing_bool_text(processed.clean.enabled)},
        {"ptc_mean_subtraction", "always-before-optional-pca-v1"},
        {"ptc_mean_source_mask_enabled",
         jinc_processing_bool_text(
             processed.clean.enabled && ptc.mask_radius_arcsec > 0.0)},
        {"ptc_mean_source_mask_radius_arcsec",
         mapmaking::jinc_double_hex(ptc.mask_radius_arcsec)},
        {"fir_coefficients_digest",
         mapmaking::jinc_matrix_digest(rtc.filter.filter)},
        {"configured_notch_centers_hz",
         jinc_processing_vector_text(rtc.filter.w0s)},
        {"configured_notch_widths_hz",
         jinc_processing_vector_text(configured_notch_widths_hz)},
        {"configured_notch_q", jinc_processing_vector_text(rtc.filter.qs)},
        {"configured_notch_count", std::to_string(rtc.filter.w0s.size())},
        {"line_audit_fixed_notch_enabled",
         jinc_processing_bool_text(rtc.line_audit.enabled &&
                                   rtc.line_audit.pre_filter_enabled &&
                                   rtc.line_audit.fixed_notch_enabled)},
        {"line_audit_fixed_notch_centers_hz",
         jinc_processing_vector_text(
             rtc.line_audit.fixed_notch_freqs_hz)},
        {"line_audit_fixed_notch_widths_hz",
         jinc_processing_vector_text(
             rtc.line_audit.fixed_notch_widths_hz)},
        {"iir_highpass_hz",
         mapmaking::jinc_double_hex(rtc.filter.iir_highpass_freq_Hz)},
        {"iir_highpass_order", std::to_string(rtc.filter.iir_highpass_order)},
        {"iir_highpass_zero_phase",
         jinc_processing_bool_text(rtc.filter.iir_highpass_zero_phase)},
        {"notch_zero_phase",
         jinc_processing_bool_text(rtc.filter.notch_zero_phase)},
        {"filter_edge_guard_enabled",
         jinc_processing_bool_text(rtc.filter_edge_guard.enabled)},
        {"filter_edge_guard_mode", rtc.filter_edge_guard.mode},
        {"filter_edge_guard_combine", rtc.filter_edge_guard.combine},
        {"filter_edge_guard_context_samples",
         std::to_string(rtc.filter_edge_guard.context_samples)},
        {"filter_edge_guard_samples",
         std::to_string(rtc.filter_edge_guard.guard_samples)},
        {"ptc_stddev_limit",
         mapmaking::jinc_double_hex(cleaner.stddev_limit)},
        {"ptc_tau", mapmaking::jinc_double_hex(cleaner.tau)},
        {"ptc_n_calc", std::to_string(cleaner.n_calc)},
        {"ptc_grouping", jinc_processing_vector_text(cleaner.grouping)},
        {"ptc_standard_pca_enabled",
         jinc_processing_bool_text(cleaner.standard_pca.enabled)},
        {"ptc_null_model_enabled",
         jinc_processing_bool_text(cleaner.null_model.enabled)},
        {"ptc_marchenko_pastur_enabled",
         jinc_processing_bool_text(cleaner.marchenko_pastur.enabled)},
        {"ptc_adaptive_selector_enabled",
         jinc_processing_bool_text(cleaner.adaptive_selector.enabled)},
        {"population_map_grouping",
         std::string{citlali::config::to_string(mapmaking_config(engine).grouping)}},
        {"population_map_method", "jinc"},
        {"population_map_count",
         std::to_string(engine.map_indices.n_maps)},
        {"population_planned_scan_count",
         std::to_string(engine.telescope.scan_indices.cols())},
    };
    for (const auto &[group, cuts] : cleaner.n_eig_to_cut) {
        facts.emplace_back(
            "ptc_configured_eigen_cuts_group_" + std::to_string(group),
            jinc_processing_vector_text(cuts));
    }
    if (!std::isfinite(engine.telescope.d_fsmp) ||
        engine.telescope.d_fsmp <= 0.0) {
        throw std::logic_error(
            "JINC coverage requires a finite positive realized sample frequency");
    }
    provenance.processing_configuration_facts = facts;
    provenance.processing_configuration_identity =
        mapmaking::jinc_realization_identity_digest(
            "actual-enabled-processing-operators-v3", facts);
    provenance.processing_configuration_bound = true;
    provenance.coverage_sample_frequency_identity =
        "effective-processed-timestream-sample-rate-telescope-d_fsmp-v1";
    provenance.coverage_sample_frequency_hz = engine.telescope.d_fsmp;
}

template <class Engine>
void bind_jinc_processing_configuration_if_available(Engine &engine) {
    if constexpr (jinc_processing_provenance_capable_v<Engine>) {
        bind_jinc_processing_configuration(engine);
    }
}

template <class Engine, class PtcData, class Apt, class MapIndices>
void record_jinc_rtc_scan_state_if_available(
    Engine &engine, const PtcData &ptcdata, const Apt &apt,
    const MapIndices &map_indices) {
    if (!jinc_processing_provenance_active(engine)) {
        return;
    }
    auto &products = engine.omb.jinc_products;
    std::scoped_lock<std::mutex> lock(*products.processing_trace_mutex);
    auto &trace = products.processing_scan_traces[ptcdata.index.data];
    trace.detector_count = jinc_processing_checked_count(
        ptcdata.scans.data.cols(), "JINC detector count");
    trace.detector_sample_count = jinc_processing_checked_count(
        ptcdata.flags.data.size(), "JINC detector-sample count");
    trace.rtc_flagged_sample_count = jinc_processing_checked_count(
        ptcdata.flags.data.array().count(), "JINC RTC flagged count");
    trace.rtc_flags_digest =
        mapmaking::jinc_matrix_digest(ptcdata.flags.data);
    trace.map_indices_digest = mapmaking::jinc_matrix_digest(map_indices);
    const auto apt_flag_it = apt.find("flag");
    if (apt_flag_it == apt.end()) {
        throw std::logic_error("JINC scan lacks actual APT flag state");
    }
    trace.apt_flags_digest =
        mapmaking::jinc_matrix_digest(apt_flag_it->second);
    trace.apt_flagged_detector_count = jinc_processing_checked_count(
        (apt_flag_it->second.array() != 0.0).count(),
        "JINC APT flagged detector count");

    const auto source =
        engine.rtcproc.snapshot_source_protection_diag_summary(
            ptcdata.index.data);
    if (source.enabled && source.protected_samples < 0) {
        throw std::logic_error(
            "JINC RTC source-mask count cannot be negative");
    }
    if (source.enabled && source.mask_digest == "unavailable") {
        throw std::logic_error(
            "JINC RTC source-mask identity is unavailable");
    }
    trace.rtc_source_masked_sample_count = source.enabled
        ? static_cast<std::size_t>(source.protected_samples)
        : 0U;
    trace.rtc_source_mask_digest =
        source.enabled ? source.mask_digest : "disabled";

    const auto notches = engine.rtcproc.snapshot_notch_operator_summary(
        ptcdata.index.data);
    trace.configured_notch_applied_count = 0;
    trace.fixed_notch_count = 0;
    trace.dynamic_notch_count = 0;
    trace.detector_notch_count = 0;
    std::vector<std::pair<std::string, std::string>> notch_facts{
        {"operator_count", std::to_string(notches.size())}};
    for (std::size_t index = 0; index < notches.size(); ++index) {
        const auto &notch = notches[index];
        const auto prefix = "operator_" + std::to_string(index) + "_";
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
            jinc_processing_bool_text(notch.zero_phase));
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
                "JINC RTC notch realization has an unknown stage");
        }
    }
    trace.rtc_notch_operators_digest =
        mapmaking::jinc_realization_identity_digest(
            "actual-rtc-notch-operators-v1", notch_facts);
}

template <class Engine, class PtcData, class Apt, class MapIndices>
void record_jinc_ptc_scan_state_if_available(
    Engine &engine, const PtcData &ptcdata, const Apt &apt,
    const MapIndices &map_indices) {
    if (!jinc_processing_provenance_active(engine)) {
        return;
    }
    auto &products = engine.omb.jinc_products;
    std::scoped_lock<std::mutex> lock(*products.processing_trace_mutex);
    const auto trace_it = products.processing_scan_traces.find(
        ptcdata.index.data);
    if (trace_it == products.processing_scan_traces.end()) {
        throw std::logic_error(
            "JINC PTC realization lacks the corresponding RTC scan trace");
    }
    auto &trace = trace_it->second;
    trace.ptc_flagged_sample_count = jinc_processing_checked_count(
        ptcdata.flags.data.array().count(), "JINC PTC flagged count");
    trace.ptc_flags_digest =
        mapmaking::jinc_matrix_digest(ptcdata.flags.data);
    trace.ptc_signal_digest =
        mapmaking::jinc_matrix_digest(ptcdata.scans.data);
    trace.ptc_kernel_digest =
        mapmaking::jinc_matrix_digest(ptcdata.kernel.data);
    trace.map_indices_digest = mapmaking::jinc_matrix_digest(map_indices);
    const auto apt_flag_it = apt.find("flag");
    if (apt_flag_it == apt.end()) {
        throw std::logic_error("JINC PTC scan lacks actual APT flag state");
    }
    trace.apt_flags_digest =
        mapmaking::jinc_matrix_digest(apt_flag_it->second);
    trace.apt_flagged_detector_count = jinc_processing_checked_count(
        (apt_flag_it->second.array() != 0.0).count(),
        "JINC APT flagged detector count");

    const auto mean = engine.ptcproc.snapshot_mean_realization_summary(
        ptcdata.index.data);
    if (!mean || !mean->mean_subtracted ||
        mean->mask_digest == "unavailable") {
        throw std::logic_error(
            "JINC PTC mean realization is incomplete");
    }
    trace.ptc_mean_masked_sample_count = mean->masked_sample_count;
    trace.ptc_mean_mask_digest = mean->mask_digest;
    const auto pca = engine.ptcproc.snapshot_pca_realization_summary(
        ptcdata.index.data);
    trace.pca_solve_count = pca.size();
    std::vector<std::pair<std::string, std::string>> pca_facts{
        {"enabled", jinc_processing_bool_text(engine.ptcproc.run_clean)},
        {"solve_count", std::to_string(pca.size())},
    };
    for (std::size_t index = 0; index < pca.size(); ++index) {
        const auto &entry = pca[index];
        const auto prefix = "solve_" + std::to_string(index) + "_";
        pca_facts.emplace_back(prefix + "grouping", entry.grouping);
        pca_facts.emplace_back(
            prefix + "group_key", std::to_string(entry.group_key));
        pca_facts.emplace_back(
            prefix + "array_index", std::to_string(entry.array_index));
        pca_facts.emplace_back(
            prefix + "configured_cut",
            std::to_string(entry.configured_cut));
        pca_facts.emplace_back(
            prefix + "applied_cut", std::to_string(entry.applied_cut));
        pca_facts.emplace_back(
            prefix + "forced_limit_index",
            std::to_string(entry.forced_limit_index));
        pca_facts.emplace_back(
            prefix + "eigenvalue_digest", entry.eigenvalue_digest);
        pca_facts.emplace_back(
            prefix + "eigenvector_digest", entry.eigenvector_digest);
    }
    trace.pca_realization_digest =
        mapmaking::jinc_realization_identity_digest(
            "actual-ptc-pca-realization-v2", pca_facts);
}

template <class Engine>
void bind_jinc_processing_realization(Engine &engine) {
    if (!jinc_processing_provenance_active(engine)) {
        return;
    }
    auto &products = engine.omb.jinc_products;
    auto &provenance = products.provenance;
    auto &raw_plan = raw_timestream_plan(engine);
    if (!provenance.processing_configuration_bound ||
        provenance.processing_realization_bound) {
        throw std::logic_error(
            "JINC processing realization requires exactly one completed configuration binding");
    }
    if (!raw_plan.realized.execution_completed ||
        !raw_plan.realized.completed_scan_count.has_value()) {
        throw std::logic_error(
            "JINC processing realization requires successful raw execution");
    }

    std::scoped_lock<std::mutex> lock(*products.processing_trace_mutex);
    if (products.processing_scan_traces.size() !=
        *raw_plan.realized.completed_scan_count) {
        throw std::logic_error(
            "JINC processing trace cardinality differs from completed scans");
    }
    std::size_t detector_samples = 0;
    std::size_t detector_scan_slots = 0;
    std::size_t rtc_flagged = 0;
    std::size_t ptc_flagged = 0;
    std::size_t apt_flagged = 0;
    std::size_t rtc_source_masked = 0;
    std::size_t ptc_mean_masked = 0;
    std::size_t pca_solves = 0;
    std::size_t configured_notches_applied = 0;
    std::size_t fixed_notches = 0;
    std::size_t dynamic_notches = 0;
    std::size_t detector_notches = 0;
    std::vector<std::pair<std::string, std::string>> trace_facts;
    std::vector<std::pair<std::string, std::string>> source_mask_facts;
    std::vector<std::pair<std::string, std::string>> notch_operator_facts;
    for (const auto &[scan, trace] : products.processing_scan_traces) {
        if (trace.rtc_flags_digest == "unavailable" ||
            trace.ptc_flags_digest == "unavailable" ||
            trace.apt_flags_digest == "unavailable" ||
            trace.map_indices_digest == "unavailable" ||
            trace.ptc_signal_digest == "unavailable" ||
            trace.ptc_kernel_digest == "unavailable" ||
            trace.rtc_source_mask_digest == "unavailable" ||
            trace.rtc_notch_operators_digest == "unavailable" ||
            trace.ptc_mean_mask_digest == "unavailable" ||
            trace.pca_realization_digest == "unavailable") {
            throw std::logic_error(
                "JINC successful processing cannot serialize unavailable scan realization");
        }
        jinc_processing_add_count(detector_samples, trace.detector_sample_count,
                                  "JINC detector-sample count");
        jinc_processing_add_count(detector_scan_slots, trace.detector_count,
                                  "JINC detector-scan slot count");
        jinc_processing_add_count(rtc_flagged, trace.rtc_flagged_sample_count,
                                  "JINC RTC flagged count");
        jinc_processing_add_count(ptc_flagged, trace.ptc_flagged_sample_count,
                                  "JINC PTC flagged count");
        jinc_processing_add_count(apt_flagged, trace.apt_flagged_detector_count,
                                  "JINC APT flagged count");
        jinc_processing_add_count(
            rtc_source_masked, trace.rtc_source_masked_sample_count,
            "JINC RTC source-mask count");
        jinc_processing_add_count(
            ptc_mean_masked, trace.ptc_mean_masked_sample_count,
            "JINC PTC mean-mask count");
        jinc_processing_add_count(pca_solves, trace.pca_solve_count,
                                  "JINC PCA solve count");
        jinc_processing_add_count(
            configured_notches_applied,
            trace.configured_notch_applied_count,
            "JINC configured-notch applied count");
        jinc_processing_add_count(fixed_notches, trace.fixed_notch_count,
                                  "JINC fixed-notch count");
        jinc_processing_add_count(dynamic_notches, trace.dynamic_notch_count,
                                  "JINC dynamic-notch count");
        jinc_processing_add_count(detector_notches, trace.detector_notch_count,
                                  "JINC detector-notch count");
        const auto prefix = "scan_" + std::to_string(scan) + "_";
        trace_facts.emplace_back(
            prefix + "detector_count", std::to_string(trace.detector_count));
        trace_facts.emplace_back(
            prefix + "detector_sample_count",
            std::to_string(trace.detector_sample_count));
        trace_facts.emplace_back(
            prefix + "rtc_flagged_sample_count",
            std::to_string(trace.rtc_flagged_sample_count));
        trace_facts.emplace_back(
            prefix + "ptc_flagged_sample_count",
            std::to_string(trace.ptc_flagged_sample_count));
        trace_facts.emplace_back(
            prefix + "apt_flagged_detector_count",
            std::to_string(trace.apt_flagged_detector_count));
        trace_facts.emplace_back(
            prefix + "rtc_source_masked_sample_count",
            std::to_string(trace.rtc_source_masked_sample_count));
        trace_facts.emplace_back(
            prefix + "ptc_mean_masked_sample_count",
            std::to_string(trace.ptc_mean_masked_sample_count));
        trace_facts.emplace_back(
            prefix + "pca_solve_count",
            std::to_string(trace.pca_solve_count));
        trace_facts.emplace_back(
            prefix + "configured_notch_applied_count",
            std::to_string(trace.configured_notch_applied_count));
        trace_facts.emplace_back(
            prefix + "fixed_notch_count",
            std::to_string(trace.fixed_notch_count));
        trace_facts.emplace_back(
            prefix + "dynamic_notch_count",
            std::to_string(trace.dynamic_notch_count));
        trace_facts.emplace_back(
            prefix + "detector_notch_count",
            std::to_string(trace.detector_notch_count));
        trace_facts.emplace_back(prefix + "rtc_flags", trace.rtc_flags_digest);
        trace_facts.emplace_back(prefix + "ptc_flags", trace.ptc_flags_digest);
        trace_facts.emplace_back(prefix + "apt_flags", trace.apt_flags_digest);
        trace_facts.emplace_back(prefix + "map_indices", trace.map_indices_digest);
        trace_facts.emplace_back(prefix + "ptc_signal", trace.ptc_signal_digest);
        trace_facts.emplace_back(prefix + "ptc_kernel", trace.ptc_kernel_digest);
        trace_facts.emplace_back(
            prefix + "rtc_source_mask", trace.rtc_source_mask_digest);
        trace_facts.emplace_back(
            prefix + "rtc_notch_operators",
            trace.rtc_notch_operators_digest);
        trace_facts.emplace_back(prefix + "ptc_mean_mask", trace.ptc_mean_mask_digest);
        trace_facts.emplace_back(prefix + "pca", trace.pca_realization_digest);
        source_mask_facts.emplace_back(
            prefix + "masked_sample_count",
            std::to_string(trace.rtc_source_masked_sample_count));
        source_mask_facts.emplace_back(
            prefix + "identity", trace.rtc_source_mask_digest);
        notch_operator_facts.emplace_back(
            prefix + "operator_count",
            std::to_string(trace.configured_notch_applied_count +
                           trace.fixed_notch_count +
                           trace.dynamic_notch_count +
                           trace.detector_notch_count));
        notch_operator_facts.emplace_back(
            prefix + "identity", trace.rtc_notch_operators_digest);
    }

    raw_plan.realized.flagged_sample_count = ptc_flagged;
    raw_plan.realized.dynamic_notch_count = dynamic_notches;
    provenance.kernel_template_identity =
        mapmaking::jinc_kernel_template_identity(
            engine.rtcproc.kernel, raw_kernel_enabled(engine));
    const auto trace_identity =
        mapmaking::jinc_realization_identity_digest(
            "actual-processing-scan-traces-v2", trace_facts);
    const auto source_mask_identity =
        mapmaking::jinc_realization_identity_digest(
            "actual-rtc-source-protection-masks-v1",
            source_mask_facts);
    const auto notch_operator_identity =
        mapmaking::jinc_realization_identity_digest(
            "actual-rtc-notch-operator-sequence-v1",
            notch_operator_facts);
    std::vector<std::pair<std::string, std::string>> facts{
        {"configuration_identity",
         provenance.processing_configuration_identity},
        {"raw_execution_completed", "true"},
        {"completed_scan_count",
         std::to_string(*raw_plan.realized.completed_scan_count)},
        {"detector_scan_slot_count", std::to_string(detector_scan_slots)},
        {"detector_sample_count", std::to_string(detector_samples)},
        {"rtc_flagged_sample_count", std::to_string(rtc_flagged)},
        {"ptc_flagged_sample_count", std::to_string(ptc_flagged)},
        {"apt_flagged_detector_count", std::to_string(apt_flagged)},
        {"rtc_source_masked_sample_count",
         std::to_string(rtc_source_masked)},
        {"ptc_mean_masked_sample_count",
         std::to_string(ptc_mean_masked)},
        {"pca_solve_count", std::to_string(pca_solves)},
        {"configured_notch_count",
         std::to_string(engine.rtcproc.filter.w0s.size())},
        {"configured_notch_applied_count",
         std::to_string(configured_notches_applied)},
        {"fixed_notch_count", std::to_string(fixed_notches)},
        {"dynamic_notch_count", std::to_string(dynamic_notches)},
        {"detector_notch_count", std::to_string(detector_notches)},
        {"rtc_source_mask_identity", source_mask_identity},
        {"rtc_notch_operator_identity", notch_operator_identity},
        {"filter_application_scan_count",
         std::to_string(*raw_plan.realized.completed_scan_count)},
        {"population_grouping",
         std::string{citlali::config::to_string(mapmaking_config(engine).grouping)}},
        {"population_outer_policy",
         runtime_parallel_policy_name(engine)},
        {"population_map_count", std::to_string(engine.map_indices.n_maps)},
        {"coverage_sample_frequency_basis",
         provenance.coverage_sample_frequency_identity},
        {"coverage_sample_frequency_hz",
         mapmaking::jinc_double_hex(
             provenance.coverage_sample_frequency_hz)},
        {"kernel_template_identity", provenance.kernel_template_identity},
        {"scan_trace_identity", trace_identity},
    };
    provenance.processing_realization_facts = facts;
    provenance.processing_realization_identity =
        mapmaking::jinc_realization_identity_digest(
            "actual-processing-realization-v4", facts);
    provenance.processing_realization_bound = true;
}

template <class Engine>
void bind_jinc_processing_realization_if_available(Engine &engine) {
    if constexpr (jinc_processing_provenance_capable_v<Engine>) {
        bind_jinc_processing_realization(engine);
    }
}

}  // namespace citlali::pipeline
