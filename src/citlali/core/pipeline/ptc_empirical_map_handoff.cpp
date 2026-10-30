#include <citlali/core/pipeline/ptc_empirical_map_handoff.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace citlali::pipeline {
std::shared_ptr<const PtcEmpiricalCoefficients> PtcEmpiricalCoefficients::publish(
    std::shared_ptr<const PtcMapPublication> parent,std::string identity,std::string noise_identity,
    std::vector<PtcEmpiricalDetectorRecord> detectors,std::string support,PointWeighting weighting) {
    if(weighting!=PointWeighting::inverse_noise && weighting!=PointWeighting::equal_control)
        throw std::invalid_argument("PTC unsupported coefficient weighting");
    if(!parent || identity.empty() || noise_identity.empty() || support!=parent->identity().support_reference ||
       detectors.size()!=parent->identity().detectors.size())
        throw std::invalid_argument("PTC empirical coefficient exact publication/support binding absent");
    auto out=std::shared_ptr<PtcEmpiricalCoefficients>(new PtcEmpiricalCoefficients);
    out->parent_=std::move(parent);out->identity_=std::move(identity);
    out->noise_identity_=std::move(noise_identity);out->detectors_=std::move(detectors);
    for(std::size_t d=0;d<out->detectors_.size();++d) {
        const auto &record=out->detectors_[d];const auto &e=record.noise;
        if(e.weighting!=weighting || record.detector!=out->parent_->identity().detectors[d] || e.reference_reason.empty() || e.remaining_reason.empty())
            throw std::invalid_argument("PTC empirical coefficient detector/evidence identity differs");
        if(e.eligible_count>out->parent_->identity().slots.size() || e.source_free_count>e.eligible_count ||
           e.reference_count>e.source_free_count || e.remaining_count>e.source_free_count ||
           e.reference_variance.has_value()!=(e.reference_reason=="available") ||
           e.remaining_variance.has_value()!=(e.remaining_reason=="available"))
            throw std::invalid_argument("PTC empirical coefficient support counts or availability reasons differ");
        if(e.reference_variance) {
            const double v=*e.reference_variance;
            if(!(v>0) || !std::isfinite(v) || e.reference_count<=2 ||
               !e.cleaning_coefficient || *e.cleaning_coefficient!=(weighting==PointWeighting::equal_control?1.:1/v) || !std::isfinite(1/v) || !(1/v>0))
                throw std::invalid_argument("PTC empirical coefficient invalid reference variance");
        } else if(e.cleaning_coefficient || e.remaining_variance || e.map_coefficient)
            throw std::invalid_argument("PTC empirical coefficient cannot replace missing reference noise");
        std::optional<double> coefficient;
        if(e.remaining_variance) {
            const double v=*e.remaining_variance;
            if(!e.reference_variance || !(v>=0) || !std::isfinite(v) || e.remaining_count<2)
                throw std::invalid_argument("PTC empirical coefficient invalid remaining variance");
            coefficient=weighting==PointWeighting::equal_control?1.:1/std::max(*e.reference_variance,v);
            if(!e.map_coefficient || *e.map_coefficient!=*coefficient || !std::isfinite(*coefficient) || !(*coefficient>0))
                throw std::invalid_argument("PTC empirical coefficient differs from conservative variance rule");
        } else if(e.map_coefficient)
            throw std::invalid_argument("PTC empirical coefficient cannot replace missing remaining noise");
        out->coefficients_.push_back(coefficient);
    }
    return out;
}
EmpiricalCoefficientDecision ValEmpiricalHandoffEvaluator::evaluate(const PtcEmpiricalCoefficients &g,
    const std::shared_ptr<const PtcMapPublication> &exact,std::size_t time,std::size_t detector,bool requested) {
    EmpiricalCoefficientDecision out;out.requested=requested;
    if(!requested || exact.get()!=g.parent_.get())return out;
    out.structurally_bound=true;out.applicable=g.parent_->member(time,detector);out.realized=true;
    out.available=g.coefficients_[detector].has_value();out.coefficient_=&g;out.time_=time;out.detector_=detector;
    out.sealed_pass_=out.applicable && out.available;return out;
}
std::optional<double> PtcEmpiricalCoefficients::value(const EmpiricalCoefficientDecision &decision,
    std::size_t time,std::size_t detector) const {
    if(decision.coefficient_!=this || decision.time_!=time || decision.detector_!=detector ||
       !decision.pass() || !parent_->member(time,detector))return {};
    return coefficients_[detector];
}
} // namespace citlali::pipeline
