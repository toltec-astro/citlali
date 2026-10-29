#pragma once
#include <citlali/core/mapmaking/forward_empirical_map_admission.h>

namespace citlali::pipeline {
class ValEmpiricalMapAdmission {
public:
    static ValEmpiricalMapAdmission bind(std::shared_ptr<const PtcMapPublication> q,
        std::shared_ptr<const PtcEmpiricalCoefficients> g,const mapmaking::MapCoordinateParent &ast) {
        return ValEmpiricalMapAdmission{mapmaking::ForwardEmpiricalMapAdmission::bind(std::move(q),std::move(g),ast)};
    }
    struct Decision {
        EmpiricalCoefficientDecision coefficient_qc;
        bool requested=false,applicable=false,eligible=false,realized=false;
        std::uint16_t cal_causes=0;std::uint8_t ptc_causes=0;
        bool pass() const {return sealed_pass_ && requested && applicable && eligible && realized;}
    private:
        friend class ValEmpiricalMapAdmission;
        bool sealed_pass_=false;
    };
    Decision at(std::size_t time,std::size_t detector,std::uint16_t cal_causes,bool requested=true) const {
        Decision d;d.requested=requested;if(!requested)return d;
        const auto facts=policy_.facts(time,detector,cal_causes,requested);
        d.coefficient_qc=facts.coefficient_qc;d.applicable=facts.applicable;
        d.cal_causes=facts.cal_causes;d.ptc_causes=facts.ptc_causes;
        d.eligible=facts.producer_eligible && d.coefficient_qc.pass();
        d.realized=true;d.sealed_pass_=d.applicable && d.eligible;return d;
    }
private:
    explicit ValEmpiricalMapAdmission(mapmaking::ForwardEmpiricalMapAdmission policy):policy_(std::move(policy)){}
    mapmaking::ForwardEmpiricalMapAdmission policy_;
};
} // namespace citlali::pipeline
