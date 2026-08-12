#pragma once

// Engine timestream config implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/downsample_config.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

inline double Engine::processed_time_chunk_fs_hz() const {
    double fs_hz = telescope.fsmp;
    if (citlali::pipeline::should_run_downsample(*this) &&
        citlali::pipeline::downsample_factor(*this) > 1) {
        fs_hz /= static_cast<double>(
            citlali::pipeline::downsample_factor(*this));
    }
    return fs_hz;
}

template <class calib_t>
Eigen::Index Engine::apply_model_protected_ptc_line_audit(
    TCData<timestream::TCDataKind::PTC, Eigen::MatrixXd> &ptcdata,
    calib_t &calib_for_scan,
    bool model_subtracted) {

    const auto &audit_policy =
        citlali::pipeline::raw_time_chunk_config(*this).line_audit;
    const auto &base_audit = rtcproc.line_audit;
    if (!audit_policy.enabled ||
        !audit_policy.ptc_model_protected_enabled) {
        return 0;
    }
    if (audit_policy.ptc_require_model_subtracted && !model_subtracted) {
        logger->debug(
            "skipping model-protected PTC line-audit notch pass for scan {} because no model was subtracted",
            ptcdata.index.data + 1);
        return 0;
    }
    if (!audit_policy.ptc_apply_fixed_notches &&
        !audit_policy.ptc_apply_shared_notches &&
        !audit_policy.ptc_apply_detector_notches) {
        return 0;
    }

    auto audit = base_audit;
    audit.pre_filter_enabled = false;
    audit.post_filter_enabled = false;
    audit.apply_shared_notches = audit_policy.ptc_apply_shared_notches;
    audit.post_filter_apply_detector_notches =
        audit_policy.ptc_apply_detector_notches;
    audit.fixed_notch_enabled = audit_policy.fixed_notch_enabled &&
        audit_policy.ptc_apply_fixed_notches;
    if (std::isfinite(audit_policy.ptc_line_min_hz)) {
        audit.line_min_hz = audit_policy.ptc_line_min_hz;
    }
    if (std::isfinite(audit_policy.ptc_line_max_hz)) {
        audit.line_max_hz = audit_policy.ptc_line_max_hz;
    }

    const double fs_hz = processed_time_chunk_fs_hz();
    if (!std::isfinite(fs_hz) || fs_hz <= 0.0) {
        logger->warn("skipping model-protected PTC line-audit notch pass; invalid fs_hz={}", fs_hz);
        return 0;
    }

    Eigen::Index total_notches = 0;
    Eigen::Index max_notches_per_timestream = 0;
    const Eigen::Index ptc_iteration =
        rtcproc.begin_ptc_response_iteration(ptcdata.index.data);
    timestream::RTCProc::RTCResponseApplicationContext application_context;
    application_context.phase = "ptc";
    application_context.stage = "model_protected";
    application_context.scan = ptcdata.index.data;
    application_context.ptc_iteration = ptc_iteration;
    application_context.model_subtracted = model_subtracted;

    if (audit.fixed_notch_enabled) {
        const Eigen::Index n_fixed_sections =
            rtcproc.count_rtc_line_audit_fixed_notches(fs_hz, audit);
        const auto n_fixed =
            rtcproc.apply_rtc_line_audit_fixed_notches(
                ptcdata, fs_hz, audit, application_context);
        total_notches += n_fixed;
        if (n_fixed > 0) {
            max_notches_per_timestream += n_fixed_sections;
        }
    }

    if (audit.apply_shared_notches) {
        const Eigen::Index n_iters = std::max<Eigen::Index>(
            1, audit_policy.ptc_apply_iterations);
        for (Eigen::Index iter = 0; iter < n_iters; ++iter) {
            rtcproc.capture_rtc_line_audit(
                ptcdata, calib_for_scan, 0, ptcdata.scans.data.rows(), audit, true);
            const auto n_shared =
                rtcproc.apply_rtc_line_audit_shared_notches(
                    ptcdata, fs_hz, audit, true, application_context);
            total_notches += n_shared;
            if (n_shared > 0) {
                max_notches_per_timestream += n_shared;
            }
            if (n_shared <= 0) {
                break;
            }
        }
    }

    if (audit.post_filter_apply_detector_notches) {
        const auto n_detector =
            rtcproc.apply_rtc_line_audit_detector_notches(
                ptcdata, fs_hz, audit, 0, ptcdata.scans.data.rows(),
                application_context);
        total_notches += n_detector;
        if (n_detector > 0) {
            if (audit.detector_notch_max_notches > 0) {
                max_notches_per_timestream +=
                    std::min<Eigen::Index>(audit.detector_notch_max_notches, n_detector);
            }
            else {
                max_notches_per_timestream += n_detector;
            }
        }
    }

    if (total_notches > 0) {
        ptcdata.status.tod_filtered = true;
        const auto &edge_guard =
            citlali::pipeline::raw_time_chunk_config(*this)
                .filter.edge_guard;
        if (edge_guard.enabled && edge_guard.apply_dynamic_notch &&
            max_notches_per_timestream > 0) {
            const double min_width_hz =
                std::min(audit.apply_min_width_hz, audit.detector_notch_min_width_hz);
            Eigen::Index guard_samples =
                max_notches_per_timestream *
                timestream::Filter::notch_settle_samples_for_width(
                    fs_hz, min_width_hz,
                    edge_guard.iir_settle_attenuation);
            guard_samples = std::max<Eigen::Index>(
                guard_samples, edge_guard.min_samples);
            guard_samples += edge_guard.extra_samples;
            if (edge_guard.max_samples > 0) {
                guard_samples = std::min<Eigen::Index>(
                    guard_samples, edge_guard.max_samples);
            }
            guard_samples = std::max<Eigen::Index>(0, guard_samples);
            if (guard_samples > 0) {
                rtcproc.apply_filter_edge_guard(ptcdata, 0, ptcdata.scans.data.rows(), guard_samples);
            }
        }
        logger->info(
            "model-protected PTC line-audit notch pass scan {}: total_notches={} fs_hz={} model_subtracted={} fixed={} shared={} detector={}",
            ptcdata.index.data + 1,
            total_notches,
            fs_hz,
            model_subtracted,
            audit.fixed_notch_enabled,
            audit.apply_shared_notches,
            audit.post_filter_apply_detector_notches);
    }

    return total_notches;
}
