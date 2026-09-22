#include <citlali/core/pipeline/ptc_map_handoff.h>
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace citlali::pipeline {
std::shared_ptr<const PtcMapPublication> PtcMapPublication::bind_verified_segment(
    PtcMapPublicationIdentity id,PtcMatrix values,PtcMask causes,bool converged) {
    for(const auto *s:{&id.publication,&id.application,&id.cal_parent,&id.rtc_grid,
                       &id.processing,&id.observation,&id.segment,&id.support_reference})
        if(s->empty())throw std::invalid_argument("PTC completed MAP publication binding absent");
    if(id.rank<=0 || id.network<0 || id.array<0 || id.array>2 || id.detectors.empty() || id.slots.empty() ||
       values.rows()!=static_cast<Eigen::Index>(id.slots.size()) || values.cols()!=static_cast<Eigen::Index>(id.detectors.size()) ||
       causes.rows()!=values.rows() || causes.cols()!=values.cols())
        throw std::invalid_argument("PTC MAP publication rank/domain mismatch");
    std::set<std::string> detectors;
    for(const auto &d:id.detectors)if(d.empty() || !detectors.insert(d).second)
        throw std::invalid_argument("PTC MAP detector identity absent or duplicate");
    if(id.slots.front()<0 || std::adjacent_find(id.slots.begin(),id.slots.end(),std::greater_equal<>())!=id.slots.end())
        throw std::invalid_argument("PTC MAP stable RTC slots invalid or duplicate");
    auto out=std::shared_ptr<PtcMapPublication>(new PtcMapPublication);
    for(Eigen::Index t=0;t<values.rows();++t)for(Eigen::Index d=0;d<values.cols();++d) {
        if(causes(t,d)&~std::uint8_t{31})throw std::invalid_argument("PTC MAP unknown output cause");
        if(causes(t,d)==0) {
            if(!converged || !std::isfinite(values(t,d)))throw std::invalid_argument("PTC MAP invalid retained output");
            ++out->retained_;
        }
    }
    out->identity_=std::move(id);out->values_=std::move(values);out->causes_=std::move(causes);return out;
}
bool PtcMapPublication::member(std::size_t time,std::size_t detector) const {
    if(time>=identity_.slots.size() || detector>=identity_.detectors.size())
        throw std::out_of_range("PTC MAP occurrence locator outside bound domain");
    return causes_(time,detector)==0;
}
UniformFact uniform_all(std::span<const UniformFact> facts) {
    auto result=UniformFact::true_value;
    for(auto f:facts) {
        if(f==UniformFact::false_value)return f;
        if(f==UniformFact::conflict)result=f;
        if(f==UniformFact::unknown && result==UniformFact::true_value)result=f;
    }
    return result;
}
bool UniformDecision::pass() const noexcept {
    return evaluated_pass_ && requested && structurally_bound && applicable && realized &&
        std::all_of(restrictions.begin(),restrictions.end(),[](auto f){return f==UniformFact::true_value;});
}
std::shared_ptr<const PtcUniformCoefficients> PtcUniformCoefficients::publish(
    std::shared_ptr<const PtcMapPublication> parent,UniformSelection selection,
    std::string identity,std::string stored_support) {
    if(!parent || selection.identity.empty() || identity.empty() || stored_support.empty())
        throw std::invalid_argument("PTC uniform structural binding absent");
    if(!parent->retained())return {}; // empty support creates no coefficient instance
    auto out=std::shared_ptr<PtcUniformCoefficients>(new PtcUniformCoefficients);
    out->parent_=std::move(parent);out->selection_=std::move(selection);
    out->identity_=std::move(identity);out->stored_support_=std::move(stored_support);
    out->restrictions_.fill(UniformFact::true_value);
    out->restrictions_[0]=out->selection_.conflicting?UniformFact::conflict:
        !out->selection_.family?UniformFact::unknown:
        *out->selection_.family==family?UniformFact::true_value:UniformFact::false_value;
    // Scalar is exactly 1, unit exactly 1, representation analytic mean-to-one.
    // The compact stored support is independently compared with the target.
    out->restrictions_[4]=out->stored_support_==out->parent_->identity().support_reference?
        UniformFact::true_value:UniformFact::false_value;
    return out;
}
UniformDecision ValUniformHandoffEvaluator::evaluate(const PtcUniformCoefficients &g,
    const std::shared_ptr<const PtcMapPublication> &exact,
    std::size_t time,std::size_t detector,bool requested) {
    UniformDecision out;out.restrictions.fill(UniformFact::unknown);out.requested=requested;
    if(!requested)return out;
    if(exact.get()!=g.parent_.get())return out; // malformed q,g pair is local, not shared corruption
    out.structurally_bound=true;out.applicable=g.parent_->member(time,detector);
    out.restrictions=g.restrictions_;out.realized=true;
    out.coefficient_=&g;out.time_=time;out.detector_=detector;
    out.evaluated_pass_=out.applicable && std::all_of(g.restrictions_.begin(),g.restrictions_.end(),
        [](auto f){return f==UniformFact::true_value;});return out;
}
std::optional<double> PtcUniformCoefficients::value(const UniformDecision &decision,
    std::size_t time,std::size_t detector) const {
    if(decision.coefficient_!=this || decision.time_!=time || decision.detector_!=detector ||
       !decision.pass() || !parent_->member(time,detector))return {};
    return 1.;
}
} // namespace citlali::pipeline
