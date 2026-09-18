#pragma once
#include <citlali/core/pipeline/timestream_ptc_cal_source.h>
#include <citlali/core/pipeline/timestream_ptc_numerics.h>
#include <citlali/core/pipeline/timestream_processing_scan_native.h>
namespace citlali::pipeline {
// One network, one physical portion of an existing processing scan. Indices
// reference the retained CAL grid; no synthesized common grid or time collapse.
struct PtcGroupEvidence {
    std::uint64_t scan=0;
    TimestreamNetworkId network=0;
    RtcEventRange native;
    // Full requested output domain; fit_columns indexes detectors before fitting.
    std::vector<std::size_t> detectors, slots, fit_columns, eligible_per_detector;
    PtcPrepared input;
    PtcFit fit;
};
class PtcEvidence {
public:
    static std::shared_ptr<const PtcEvidence> learn(PtcCalSource,
        const ProcessingScanNativeProjection &,PtcSolverRequest);
    const auto &source() const noexcept{return source_;}
    const auto &groups() const noexcept{return groups_;}
    const auto &processing_binding() const noexcept{return processing_;}
    // Versioned realization of the existing five PTC named-use profiles.
    static constexpr std::string_view use_policy="ptc-cal-observed-entry-use-2026-09-18-v2";
private:
    explicit PtcEvidence(PtcCalSource source):source_(std::move(source)){}
    PtcCalSource source_;
    std::shared_ptr<const RtcExistingScanBinding> processing_;
    std::vector<PtcGroupEvidence> groups_;
};
class PtcPlan {
public:
    static std::shared_ptr<const PtcPlan> consider(std::shared_ptr<const PtcEvidence>,
        std::shared_ptr<const ValSnapshot>,std::uint64_t instance);
    const auto &evidence_handle()const noexcept{return evidence_;}
    const auto &snapshot_handle()const noexcept{return evidence_->source().val_snapshot_handle();}
    auto instance()const noexcept{return instance_;}
private:
    std::shared_ptr<const PtcEvidence> evidence_;
    std::uint64_t instance_=0;
};
class PtcAppliedSignal {
public:
    static std::shared_ptr<const PtcAppliedSignal> apply(std::shared_ptr<const PtcPlan>,
        const PtcCalSource &,std::shared_ptr<const ValSnapshot>);
    const auto &plan_handle()const noexcept{return plan_;}
    const auto &groups()const noexcept{return groups_;}
    std::size_t available_count()const noexcept{return available_;}
    // A supplied response is already on this exact CAL grid. This local J is
    // supported; no complete RTC->CAL astronomical response is manufactured.
    PtcApplied response(std::size_t group,const PtcCalSource &exact,const PtcMatrix &) const;
private:
    std::shared_ptr<const PtcPlan> plan_;
    std::vector<PtcApplied> groups_;
    std::size_t available_=0;
};
class ValPtcOutputFacts {
public:
    static std::shared_ptr<const ValPtcOutputFacts> preserve(std::shared_ptr<const PtcAppliedSignal>);
    const auto &signal_handle()const noexcept{return signal_;}
    std::uint8_t at(const std::shared_ptr<const PtcAppliedSignal> &exact,std::size_t group,std::size_t time,std::size_t detector) const;
private:
    explicit ValPtcOutputFacts(std::shared_ptr<const PtcAppliedSignal> signal):signal_(std::move(signal)){}
    std::shared_ptr<const PtcAppliedSignal> signal_;
};
} // namespace citlali::pipeline
