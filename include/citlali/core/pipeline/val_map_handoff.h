#pragma once
#include <citlali/core/mapmaking/forward_map_admission.h>
namespace citlali::pipeline {
// Concrete VAL evaluator for the admitted saved ordinary-MAP route. The MAP
// policy owns predicates; this evaluator owns four-axis decision realization.
class ValMapAdmission {
public:
    static ValMapAdmission bind(std::shared_ptr<const PtcMapPublication> q,
        std::shared_ptr<const PtcUniformCoefficients> g,const mapmaking::MapCoordinateParent &ast) {
        return ValMapAdmission{mapmaking::ForwardMapAdmission::bind(std::move(q),std::move(g),ast)};
    }
    struct Decision {
        UniformDecision coefficient_qc;
        bool requested=false,applicable=false,eligible=false,realized=false;
        std::uint16_t cal_causes=0;std::uint8_t ptc_causes=0;
        bool pass() const {return sealed_pass_ && requested && applicable && eligible && realized;}
    private:
        friend class ValMapAdmission;
        bool sealed_pass_=false;
    };
    Decision at(std::size_t time,std::size_t detector,std::uint16_t cal_causes,bool requested=true) const {
        Decision d;d.requested=requested;
        if(!requested)return d;
        const auto facts=policy_.facts(time,detector,cal_causes,requested);
        d.coefficient_qc=facts.coefficient_qc;d.applicable=facts.applicable;
        d.cal_causes=facts.cal_causes;d.ptc_causes=facts.ptc_causes;
        d.eligible=facts.producer_eligible && d.coefficient_qc.pass();
        d.realized=true;d.sealed_pass_=d.applicable && d.eligible;return d;
    }
    static constexpr std::string_view registry="SCI-VAL_PROFILE_REGISTRY v0.1/r0.3-map-v0.2-r0.4-2026-09-08";
    static constexpr std::string_view registry_sha256="ce027c3a5c0e71291e9ea509c3a2d1e2d9971fac64763e2afc676241df7d442b";
    static constexpr std::string_view source_register="SCI-VAL_SOURCE_BINDING_REGISTER v0.1/r0.3-map-v0.2-r0.4-2026-09-08";
    static constexpr std::string_view source_register_sha256="a30c1b71ada4793b6ae805699f9335e9f912399e066a4af6bb68ab81456d4555";
private:
    explicit ValMapAdmission(mapmaking::ForwardMapAdmission policy):policy_(std::move(policy)){}
    mapmaking::ForwardMapAdmission policy_;
};
} // namespace citlali::pipeline
