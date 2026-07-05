#pragma once

// Engine timestream config implementation detail.
// Include this only after Engine has been declared.

inline double Engine::processed_time_chunk_fs_hz() const {
    double fs_hz = telescope.fsmp;
    if (rtcproc.run_downsample && rtcproc.downsampler.factor > 1) {
        fs_hz /= static_cast<double>(rtcproc.downsampler.factor);
    }
    return fs_hz;
}

template <class calib_t>
Eigen::Index Engine::apply_model_protected_ptc_line_audit(
    TCData<timestream::TCDataKind::PTC, Eigen::MatrixXd> &ptcdata,
    calib_t &calib_for_scan,
    bool model_subtracted) {

    const auto &base_audit = rtcproc.line_audit;
    if (!base_audit.enabled || !base_audit.ptc_model_protected_enabled) {
        return 0;
    }
    if (base_audit.ptc_require_model_subtracted && !model_subtracted) {
        logger->debug(
            "skipping model-protected PTC line-audit notch pass for scan {} because no model was subtracted",
            ptcdata.index.data + 1);
        return 0;
    }
    if (!base_audit.ptc_apply_fixed_notches &&
        !base_audit.ptc_apply_shared_notches &&
        !base_audit.ptc_apply_detector_notches) {
        return 0;
    }

    auto audit = base_audit;
    audit.pre_filter_enabled = false;
    audit.post_filter_enabled = false;
    audit.apply_shared_notches = audit.ptc_apply_shared_notches;
    audit.post_filter_apply_detector_notches = audit.ptc_apply_detector_notches;
    audit.fixed_notch_enabled = audit.fixed_notch_enabled && audit.ptc_apply_fixed_notches;
    if (std::isfinite(base_audit.ptc_line_min_hz)) {
        audit.line_min_hz = base_audit.ptc_line_min_hz;
    }
    if (std::isfinite(base_audit.ptc_line_max_hz)) {
        audit.line_max_hz = base_audit.ptc_line_max_hz;
    }

    const double fs_hz = processed_time_chunk_fs_hz();
    if (!std::isfinite(fs_hz) || fs_hz <= 0.0) {
        logger->warn("skipping model-protected PTC line-audit notch pass; invalid fs_hz={}", fs_hz);
        return 0;
    }

    Eigen::Index total_notches = 0;
    Eigen::Index max_notches_per_timestream = 0;

    if (audit.fixed_notch_enabled) {
        const Eigen::Index n_fixed_sections =
            rtcproc.count_rtc_line_audit_fixed_notches(fs_hz, audit);
        const auto n_fixed =
            rtcproc.apply_rtc_line_audit_fixed_notches(ptcdata, fs_hz, audit);
        total_notches += n_fixed;
        if (n_fixed > 0) {
            max_notches_per_timestream += n_fixed_sections;
        }
    }

    if (audit.apply_shared_notches) {
        const Eigen::Index n_iters = std::max<Eigen::Index>(1, audit.ptc_apply_iterations);
        for (Eigen::Index iter = 0; iter < n_iters; ++iter) {
            rtcproc.capture_rtc_line_audit(
                ptcdata, calib_for_scan, 0, ptcdata.scans.data.rows(), audit, true);
            const auto n_shared =
                rtcproc.apply_rtc_line_audit_shared_notches(ptcdata, fs_hz, audit, true);
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
                ptcdata, fs_hz, audit, 0, ptcdata.scans.data.rows());
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
        if (rtcproc.filter_edge_guard.enabled &&
            rtcproc.filter_edge_guard.apply_dynamic_notch &&
            max_notches_per_timestream > 0) {
            const double min_width_hz =
                std::min(audit.apply_min_width_hz, audit.detector_notch_min_width_hz);
            Eigen::Index guard_samples =
                max_notches_per_timestream *
                timestream::Filter::notch_settle_samples_for_width(
                    fs_hz, min_width_hz, rtcproc.filter_edge_guard.iir_settle_attenuation);
            guard_samples = std::max(guard_samples, rtcproc.filter_edge_guard.min_samples);
            guard_samples += rtcproc.filter_edge_guard.extra_samples;
            if (rtcproc.filter_edge_guard.max_samples > 0) {
                guard_samples = std::min(guard_samples, rtcproc.filter_edge_guard.max_samples);
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
