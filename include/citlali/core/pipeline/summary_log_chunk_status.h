#pragma once

// Included by summary_log.h inside namespace citlali::pipeline.

template <class Status>
void write_chunk_processing_status_summary(std::ostream &stream,
                                           const Status &status) {
    stream << "-Calibrated: " << status.calibrated << "\n";
    stream << "-Extinction Corrected: " << status.extinction_corrected << "\n";
    stream << "-Demodulated: " << status.demodulated << "\n";
    stream << "-Kernel Generated: " << status.kernel_generated << "\n";
    stream << "-Despiked: " << status.despiked << "\n";
    stream << "-TOD filtered: " << status.tod_filtered << "\n";
    stream << "-Downsampled: " << status.downsampled << "\n";
    stream << "-Cleaned: " << status.cleaned << "\n";
}

template <class RtcProc>
void write_chunk_tod_filter_summary(std::ostream &stream,
                                    const RtcProc &rtcproc,
                                    int outer_context_samples) {
    stream << "-TOD notch enabled: " << rtcproc.run_tod_notch << "\n";
    stream << "-TOD IIR highpass enabled: "
           << rtcproc.run_tod_iir_highpass << "\n";
    stream << "-TOD IIR highpass freq (Hz): "
           << rtcproc.filter.iir_highpass_freq_Hz << "\n";
    stream << "-TOD IIR highpass order: "
           << rtcproc.filter.iir_highpass_order << "\n";
    stream << "-TOD IIR highpass zero-phase: "
           << rtcproc.filter.iir_highpass_zero_phase << "\n";
    stream << "-TOD filter edge guard enabled: "
           << rtcproc.filter_edge_guard.enabled << "\n";
    stream << "-TOD filter edge guard context samples: "
           << rtcproc.filter_edge_guard.context_samples << "\n";
    stream << "-TOD filter edge guard samples per edge: "
           << rtcproc.filter_edge_guard.guard_samples << "\n";
    stream << "-TOD loaded outer context samples: "
           << outer_context_samples << "\n";
    stream << "-RTC detector notch context samples: "
           << rtcproc.line_audit.detector_notch_context_samples << "\n";
    stream << "-RTC fixed line-audit notch enabled: "
           << rtcproc.line_audit.fixed_notch_enabled << "\n";
    stream << "-RTC fixed line-audit notch count: "
           << rtcproc.line_audit.fixed_notch_freqs_hz.size() << "\n";
}

template <class LineAudit>
void write_chunk_ptc_model_line_audit_summary(std::ostream &stream,
                                              const LineAudit &line_audit) {
    stream << "-PTC model-protected line-audit notch enabled: "
           << line_audit.ptc_model_protected_enabled << "\n";
    stream << "-PTC model-protected line-audit require model: "
           << line_audit.ptc_require_model_subtracted << "\n";
    stream << "-PTC model-protected fixed/shared/detector notches: "
           << line_audit.ptc_apply_fixed_notches << "/"
           << line_audit.ptc_apply_shared_notches << "/"
           << line_audit.ptc_apply_detector_notches << "\n";
}

