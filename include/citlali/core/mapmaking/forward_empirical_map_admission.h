#pragma once
#include <citlali/core/mapmaking/forward_map_admission.h>
#include <citlali/core/pipeline/ptc_empirical_map_handoff.h>

namespace citlali::mapmaking {
// The same exact occurrence and zero-CAL/PTC-cause requirements as ordinary
// MAP, with a separately named empirical coefficient family. Noise training
// masks do not participate in this occurrence admission.
class ForwardEmpiricalMapAdmission {
public:
    static ForwardEmpiricalMapAdmission bind(std::shared_ptr<const pipeline::PtcMapPublication> q,
        std::shared_ptr<const pipeline::PtcEmpiricalCoefficients> g,const MapCoordinateParent &ast) {
        if(!q || !g || g->parent().get()!=q.get() || ast.ptc_publication!=q->identity().publication ||
           ast.cal_parent!=q->identity().cal_parent || ast.rtc_grid!=q->identity().rtc_grid ||
           ast.detectors!=q->identity().detectors || ast.slots!=q->identity().slots)
            throw std::invalid_argument("MAP exact empirical PTC/coefficient/AST occurrence relation differs");
        return ForwardEmpiricalMapAdmission{std::move(q),std::move(g)};
    }
    struct Facts {
        pipeline::EmpiricalCoefficientDecision coefficient_qc;
        bool applicable=true,producer_eligible=false;
        std::uint16_t cal_causes=0;std::uint8_t ptc_causes=0;
    };
    Facts facts(std::size_t time,std::size_t detector,std::uint16_t cal_causes,bool requested) const {
        Facts d;d.cal_causes=cal_causes;
        d.coefficient_qc=pipeline::ValEmpiricalHandoffEvaluator::evaluate(*g_,q_,time,detector,requested);
        q_->member(time,detector);d.ptc_causes=q_->causes()(time,detector);
        d.producer_eligible=!d.cal_causes && !d.ptc_causes;return d;
    }
private:
    ForwardEmpiricalMapAdmission(std::shared_ptr<const pipeline::PtcMapPublication> q,
        std::shared_ptr<const pipeline::PtcEmpiricalCoefficients> g):q_(std::move(q)),g_(std::move(g)){}
    std::shared_ptr<const pipeline::PtcMapPublication> q_;
    std::shared_ptr<const pipeline::PtcEmpiricalCoefficients> g_;
};
} // namespace citlali::mapmaking
