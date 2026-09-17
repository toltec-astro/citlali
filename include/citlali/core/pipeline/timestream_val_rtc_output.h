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
        const auto address = target.address();
        const auto &columns = grid_->detectors();
        const auto it = std::find_if(columns.begin(), columns.end(), [&](const auto &g) {
            return g.network == address.sample_identity().network_id() &&
                   g.detector == *address.detector_index();
        });
        if (it == columns.end()) throw std::invalid_argument("VAL output detector absent");
        auto occurrence = grid_->occurrence(static_cast<std::size_t>(it - columns.begin()), target.slot());
        return {target.coordinate(), target.coordinate() == NativeReadoutCoordinate::x
            ? occurrence.x_available : occurrence.r_available, std::move(occurrence)};
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
