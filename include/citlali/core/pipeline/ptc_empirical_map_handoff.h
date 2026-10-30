#pragma once
#include <citlali/core/pipeline/ptc_inverse_noise.h>
#include <citlali/core/pipeline/ptc_map_handoff.h>

namespace citlali::pipeline {
struct PtcEmpiricalDetectorRecord {
    std::string detector;
    PtcDetectorNoiseEvidence noise;
};
struct EmpiricalCoefficientDecision {
    bool requested=false, structurally_bound=false, applicable=false, realized=false,
        available=false;
    bool pass() const noexcept {
        return sealed_pass_ && requested && structurally_bound && applicable && realized && available;
    }
private:
    friend class ValEmpiricalHandoffEvaluator;
    friend class PtcEmpiricalCoefficients;
    const void *coefficient_=nullptr;
    std::size_t time_=0, detector_=0;
    bool sealed_pass_=false;
};
// Separate owner-approved empirical family. Coefficients are derived from the
// frozen detector evidence, not accepted as arbitrary values and not published
// under the existing uniform scientific identity. One exact physical segment,
// CAL/health/cycle publication and ordered detector domain are bound together.
class PtcEmpiricalCoefficients {
public:
    static std::shared_ptr<const PtcEmpiricalCoefficients> publish(
        std::shared_ptr<const PtcMapPublication>,std::string coefficient_identity,
        std::string noise_identity,std::vector<PtcEmpiricalDetectorRecord>,
        std::string stored_support_reference,PointWeighting weighting=PointWeighting::inverse_noise);
    std::optional<double> value(const EmpiricalCoefficientDecision &,
        std::size_t time,std::size_t detector) const;
    const auto &parent() const noexcept {return parent_;}
    const auto &identity() const noexcept {return identity_;}
    const auto &noise_identity() const noexcept {return noise_identity_;}
    const auto &detectors() const noexcept {return detectors_;}
    static constexpr std::string_view family="OWNER-PTC:empirical-inverse-noise@2026-10-29-v1";
    static constexpr std::string_view control_family="OWNER-PTC:equal-weight-control@2026-10-30-v1";
    static constexpr std::string_view family_for(PointWeighting mode) {
        return mode==PointWeighting::equal_control?control_family:family;
    }
    static constexpr std::string_view unit="(mJy/nominal-beam)^-2";
    static constexpr std::string_view interpretation="empirical-analysis-coefficient-not-map-precision";
private:
    friend class ValEmpiricalHandoffEvaluator;
    std::shared_ptr<const PtcMapPublication> parent_;
    std::string identity_,noise_identity_;
    std::vector<PtcEmpiricalDetectorRecord> detectors_;
    std::vector<std::optional<double>> coefficients_;
};
class ValEmpiricalHandoffEvaluator {
public:
    static EmpiricalCoefficientDecision evaluate(const PtcEmpiricalCoefficients &,
        const std::shared_ptr<const PtcMapPublication> &exact,
        std::size_t time,std::size_t detector,bool requested);
};
} // namespace citlali::pipeline
