#pragma once
#include <citlali/core/pipeline/ptc_map_handoff.h>
#include <citlali/core/pipeline/timestream_cal_pipeline.h>
namespace citlali::mapmaking {
struct MapCoordinateParent {
    std::string ptc_publication, cal_parent, rtc_grid;
    std::vector<std::string> detectors;
    std::vector<std::int64_t> slots;
};
// MAP-authored occurrence profile; structural binding precedes payload access.
// The saved adapter supplies CAL producer causes without relabeling them.
class ForwardMapAdmission {
public:
    static ForwardMapAdmission bind(std::shared_ptr<const pipeline::PtcMapPublication> q,
        std::shared_ptr<const pipeline::PtcUniformCoefficients> g,const MapCoordinateParent &ast) {
        if(!q || !g || g->parent().get()!=q.get() || ast.ptc_publication!=q->identity().publication ||
           ast.cal_parent!=q->identity().cal_parent || ast.rtc_grid!=q->identity().rtc_grid ||
           ast.detectors!=q->identity().detectors || ast.slots!=q->identity().slots)
            throw std::invalid_argument("MAP exact PTC/coefficient/AST parent or occurrence relation differs");
        return ForwardMapAdmission{std::move(q),std::move(g)};
    }
    struct Facts {
        pipeline::UniformDecision coefficient_qc;
        bool applicable=true, producer_eligible=false;
        std::uint16_t cal_causes=0;
        std::uint8_t ptc_causes=0;
    };
    Facts facts(std::size_t time,std::size_t detector,std::uint16_t cal_causes,bool requested) const {
        Facts d;d.cal_causes=cal_causes;
        d.coefficient_qc=pipeline::ValUniformHandoffEvaluator::evaluate(*g_,q_,time,detector,requested);
        q_->member(time,detector); // validate locator before matrix access
        d.ptc_causes=q_->causes()(time,detector);
        // CAL zero-cause facts include the direct-origin prohibition.
        // Replacement influence and engineering classification are non-veto.
        d.producer_eligible=!d.cal_causes && !d.ptc_causes;return d;
    }
    static constexpr std::string_view profile="SCI-MAP:map_upstream_admission@2";
    static constexpr std::string_view binding="SCI-MAP-v0.2-r0.4-2026-09-08";
private:
    ForwardMapAdmission(std::shared_ptr<const pipeline::PtcMapPublication> q,
        std::shared_ptr<const pipeline::PtcUniformCoefficients> g):q_(std::move(q)),g_(std::move(g)){}
    std::shared_ptr<const pipeline::PtcMapPublication> q_;
    std::shared_ptr<const pipeline::PtcUniformCoefficients> g_;
};
} // namespace citlali::mapmaking
