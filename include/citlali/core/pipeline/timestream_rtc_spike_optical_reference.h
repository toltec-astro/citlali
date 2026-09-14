#pragma once

#include <citlali/core/pipeline/timestream_rtc_spike_learn.h>
#include <citlali/core/pipeline/ast_scan_motion_alignment.h>
#include <citlali/core/pipeline/timestream_rtc_optical_model.h>

#include <complex>
#include <numbers>

namespace citlali::pipeline {

// Retain the existing public spelling while sharing the accepted RTC model.
using RtcSpikeBeamArray = RtcOpticalArray;
inline double rtc_spike_reference_frequency_hz(RtcSpikeBeamArray array) {
    return rtc_optical_frequency_hz(array);
}

struct RtcSpikeLocalOpticalScale {
    // A mapped instantaneous AST speed is a local reference, NOT an upper
    // bound on velocity over an event. No hard spike test may use it as one.
    double speed_arcsec_per_sec = 0.0;
    double airy_fwhm_arcsec = 0.0;
    double temporal_optical_support_hz = 0.0;
    std::optional<double> crossing_fwhm_sec; // absent when stationary
    AstScanMotionMappedSupport motion_support;
};

// A small immutable reference product for protected-event assessment. No
// template mismatch, observed duration or hard-event predicate is defined here.
// The caller binds the array association explicitly; missing AST support stays
// missing. The original mapping/time authorities remain reachable by handle.
class RtcSpikeOpticalReference {
public:
    static std::shared_ptr<const RtcSpikeOpticalReference> bind(
        std::shared_ptr<const RtcSpikeEvidence> evidence,
        std::size_t candidate_index,
        std::shared_ptr<const AstScanMotionNetworkViews> motion,
        RtcSpikeBeamArray array, std::string array_association_authority) {
        (void)rtc_spike_reference_frequency_hz(array);
        if (!evidence || !motion || array_association_authority.empty() ||
            candidate_index >= evidence->candidates().size() ||
            motion->scope() != evidence->input_handle()->parent_handle()->scope())
            throw std::invalid_argument("RTC optical reference requires evidence, AST scope and array association");
        const auto &candidate = evidence->candidates()[candidate_index];
        const auto &block = evidence->blocks()[candidate.noise_block_index];
        const auto &network = evidence->input_handle()->network(block.network_id);
        if (array_association_authority != network.detector(block.detector_index).detector_association_record_id ||
            candidate.protection != RtcSpikeProtection::protected_source ||
            motion->network(block.network_id).network_timing_handle().get() !=
                network.occurrence_axis().native_timing_handle().get())
            throw std::invalid_argument("RTC optical reference requires protected candidate and exact ALIGN timing");
        return std::shared_ptr<const RtcSpikeOpticalReference>(new RtcSpikeOpticalReference{
            std::move(evidence), candidate_index, std::move(motion), array,
            std::move(array_association_authority)});
    }

    const auto &evidence_handle() const noexcept { return evidence_; }
    const auto &motion_handle() const noexcept { return motion_; }
    const auto &array_association_authority() const noexcept { return array_authority_; }
    std::size_t candidate_index() const noexcept { return candidate_index_; }
    RtcSpikeBeamArray array() const noexcept { return array_; }
    static constexpr std::string_view readout_model_id = RtcSpikeLearnPolicy::readout_assumption;
    static constexpr bool conditional_on_readout_assumption = true;

    std::optional<RtcSpikeLocalOpticalScale> local_scale(bool later_endpoint) const {
        const auto row = endpoint(later_endpoint);
        const auto &mapped = motion_->network(network_id());
        const auto speed = mapped.scalar_speed_arcsec_per_sec(row);
        const auto support = mapped.support(row);
        if (!speed || !support) return std::nullopt;
        const auto scale = rtc_optical_scale(array_, *speed);
        const double fwhm = scale.airy_fwhm_arcsec;
        const double band = scale.temporal_support_hz;
        if (!std::isfinite(band)) return std::nullopt;
        std::optional<double> crossing;
        if (*speed > 0.0) {
            crossing = fwhm / *speed;
            if (!std::isfinite(*crossing)) return std::nullopt;
        }
        return RtcSpikeLocalOpticalScale{*speed, fwhm, band, crossing, *support};
    }

    // Exact analytic boxcar average of exp(i*(2*pi*f*(t-t_event)+phase)).
    // Frequency is in Hz, phase in radians at the recorded native event time.
    // Any finite phase is allowed; it is never snapped to a sample boundary.
    // This transfer alone cannot establish an optical impossibility predicate.
    std::complex<double> sampled_harmonic(bool later_endpoint, double frequency_hz,
                                          double phase_radians) const {
        if (!std::isfinite(frequency_hz) || !std::isfinite(phase_radians))
            throw std::invalid_argument("RTC readout reference requires finite frequency and phase");
        const auto row = endpoint(later_endpoint);
        const auto &axis = evidence_->input_handle()->network(network_id()).occurrence_axis();
        const auto support = axis.occurrence(row).integration_support;
        const double time = axis.native_identity(row).reconstructed_time_unix_sec();
        const double width = support.end_unix_sec - support.begin_unix_sec;
        const double midpoint_offset = std::midpoint(support.begin_unix_sec, support.end_unix_sec) - time;
        const double z = std::numbers::pi * frequency_hz * width;
        const double phase = phase_radians + 2.0 * std::numbers::pi * frequency_hz * midpoint_offset;
        if (!std::isfinite(z) || !std::isfinite(phase))
            throw std::overflow_error("RTC readout reference arithmetic is nonfinite");
        const double gain = z == 0.0 ? 1.0 : std::sin(z) / z;
        return gain * std::complex<double>{std::cos(phase), std::sin(phase)};
    }

private:
    RtcSpikeOpticalReference(std::shared_ptr<const RtcSpikeEvidence> evidence,
        std::size_t candidate, std::shared_ptr<const AstScanMotionNetworkViews> motion,
        RtcSpikeBeamArray array, std::string authority)
        : evidence_{std::move(evidence)}, candidate_index_{candidate},
          motion_{std::move(motion)}, array_{array}, array_authority_{std::move(authority)} {}
    TimestreamNetworkId network_id() const {
        return evidence_->blocks()[evidence_->candidates()[candidate_index_].noise_block_index].network_id;
    }
    TimestreamNativeRow endpoint(bool later) const {
        const auto &candidate = evidence_->candidates()[candidate_index_];
        return later ? candidate.later_row : candidate.earlier_row;
    }
    std::shared_ptr<const RtcSpikeEvidence> evidence_;
    std::size_t candidate_index_;
    std::shared_ptr<const AstScanMotionNetworkViews> motion_;
    RtcSpikeBeamArray array_;
    std::string array_authority_;
};

} // namespace citlali::pipeline
