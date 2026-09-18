#pragma once

#include <citlali/core/pipeline/timestream_rtc_output_grid.h>

namespace citlali::pipeline {

// An immutable RTC producer fact block in VAL. Every scheduled target is
// covered, including unavailable values. The exact producer record is retained
// once; queries reconstruct its facts without copying a finding per sample.
// This is not a named-use evaluation and assigns no scientific eligibility.
class ValRtcOutputFacts {
public:
    static std::shared_ptr<const ValRtcOutputFacts> preserve(
        std::shared_ptr<const RtcOutputGrid> grid) {
        if (!grid) throw std::invalid_argument("VAL output facts require an RTC grid");
        return std::shared_ptr<const ValRtcOutputFacts>(new ValRtcOutputFacts{std::move(grid)});
    }
    const auto &grid_handle() const noexcept { return grid_; }
    const auto &input_snapshot_handle() const noexcept { return grid_->input_val_snapshot_handle(); }
    ValProducerProductIdentity producer() const {
        return {ValProducer::rtc, grid_->applied_handle()->plan_handle()->attempt()};
    }
    struct Fact {
        NativeReadoutCoordinate coordinate;
        bool numerical_available;
        RtcOutputGrid::Occurrence occurrence;
    };
    Fact at(const ValRtcOutputTarget &target) const {
        if (target.grid_handle().get() != grid_.get() ||
            !input_snapshot_handle()->contains(target))
            throw std::invalid_argument("VAL output query differs from its exact producer grid");
        auto occurrence = grid_->occurrence(target.detector_grid_index(), target.slot());
        return {target.coordinate(), target.coordinate() == NativeReadoutCoordinate::x
            ? occurrence.x_available : occurrence.r_available, std::move(occurrence)};
    }
    // Bind once at the consumer boundary. Indices are meaningful inside this
    // exact retained product; no identity object or VAL lookup per sample.
    class DetectorFacts {
    public:
        auto at(std::size_t slot) const { return grid_->state(detector_,slot); }
        const auto &times() const { return grid_->times(detector_); }
    private:
        friend class ValRtcOutputFacts;
        DetectorFacts(std::shared_ptr<const RtcOutputGrid> g,std::size_t d):grid_{std::move(g)},detector_{d} {}
        std::shared_ptr<const RtcOutputGrid> grid_;
        std::size_t detector_;
    };
    DetectorFacts bind_detector(const std::shared_ptr<const RtcOutputGrid> &exact,std::size_t d) const {
        if(exact.get()!=grid_.get())throw std::invalid_argument("VAL detector facts require the exact RTC grid");
        (void)grid_->detectors().at(d);
        return DetectorFacts{grid_,d};
    }
    static constexpr bool evaluates_scientific_use = false;
    std::size_t owned_bytes() const noexcept { return sizeof(*this); }
private:
    explicit ValRtcOutputFacts(std::shared_ptr<const RtcOutputGrid> grid) : grid_{std::move(grid)} {}
    std::shared_ptr<const RtcOutputGrid> grid_;
};

inline std::shared_ptr<const ValSnapshot> ValSnapshot::commit_rtc_output(
    std::shared_ptr<const ValSnapshot> base, std::shared_ptr<const ValRtcOutputFacts> facts) {
    if (!base || !facts || facts->input_snapshot_handle().get() != base.get())
        throw std::invalid_argument("VAL output commit requires the exact frozen input generation");
    if (base->generation_.value == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("VAL output generation would overflow");
    return std::shared_ptr<const ValSnapshot>(new ValSnapshot{std::move(base), {}, std::move(facts)});
}

} // namespace citlali::pipeline
