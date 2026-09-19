#pragma once

#include <algorithm>
#include <citlali/core/pipeline/timestream_rtc_jump_exclusion.h>

namespace citlali::pipeline {

struct ProcessingScanNativeRecord {
    std::uint64_t scan = 0;
    RtcEventRange science_slots, context_slots;
    std::vector<RtcEventRange> science_native, context_native;
    std::size_t unmapped_science_slots = 0;
};

// One-way projection of an EXISTING processing generation. The common grid is
// only a relational locator; no native values, times, or physical runs change.
struct ProcessingScanNativeProjection {
    std::shared_ptr<const RtcExistingScanBinding> binding;
    std::vector<ProcessingScanNativeRecord> scans;
    std::vector<RtcEventRange> native_outside_processing;
    double maximum_association_residual_seconds = 0;
};

inline ProcessingScanNativeProjection project_processing_scans_to_native(
    std::shared_ptr<const NativePairedReadoutObservation> parent,
    TimestreamNetworkId network, const Eigen::VectorXd &common_times,
    const std::vector<NativeSlotAssociation> &associations,
    const Eigen::Matrix<Eigen::Index,Eigen::Dynamic,Eigen::Dynamic> &inclusive_indices, double association_tolerance,
    std::string generation, std::string relation_authority) {
    if (!parent || common_times.size() != static_cast<Eigen::Index>(associations.size()) ||
        common_times.size() == 0 || inclusive_indices.rows() != 4 || inclusive_indices.cols() == 0 ||
        !std::isfinite(association_tolerance) || association_tolerance <= 0)
        throw std::invalid_argument("processing scan projection requires exact existing slots and native relation");
    const auto &axis = parent->network(network).occurrence_axis();
    // The exact immutable native parent determines this partition once. Calling
    // run_for(axis,row) in the slot loop rebuilt it by scanning the observation
    // for every science/context occurrence (quadratic in observation length).
    const auto physical_runs = axis.contiguous_runs();
    ProcessingScanNativeProjection out;
    std::vector<bool> selected(axis.occurrence_count(),false);
    std::set<TimestreamNativeRow> seen;
    TimestreamNativeRow previous=-1;
    for (std::size_t s=0; s<associations.size(); ++s) {
        if (!std::isfinite(common_times[s]) || (s && common_times[s] <= common_times[s-1]))
            throw std::invalid_argument("processing reference time is not strictly increasing");
        if (!associations[s].mapped()) continue;
        const auto row=associations[s].native_row;
        if (row<axis.first_native_row() || row>=axis.past_last_native_row() || !seen.insert(row).second || row<=previous)
            throw std::invalid_argument("processing slot relation is outside or repeats native parent");
        previous=row;
        const double residual=std::abs(axis.native_identity(row).reconstructed_time_unix_sec()-common_times[s]);
        if (residual>association_tolerance)
            throw std::invalid_argument("processing slot relation violates its existing timing tolerance");
        out.maximum_association_residual_seconds=std::max(out.maximum_association_residual_seconds,residual);
    }
    auto ranges = [&](RtcEventRange slots, bool science, std::size_t &missing) {
        std::vector<RtcEventRange> result;
        for (auto slot=slots.first;slot<slots.past_last;++slot) {
            const auto &a=associations[slot];
            if (!a.mapped()) {++missing;continue;}
            const auto row=a.native_row;
            if (science) selected[row-axis.first_native_row()]=true;
            // Adjacent row numbers across a packet gap are not one interval.
            const auto boundary=std::lower_bound(physical_runs.begin(),physical_runs.end(),row,
                [](const NativeContiguousRun &run,TimestreamNativeRow value) {
                    return run.first_native_row<value;
                });
            const bool physical_start=boundary!=physical_runs.end() && boundary->first_native_row==row;
            if (!result.empty() && result.back().past_last==row && !physical_start)
                result.back().past_last=row+1;
            else result.push_back({row,row+1});
        }
        return result;
    };
    std::vector<RtcExistingScanNativeSupport> supports;
    for (Eigen::Index s=0;s<inclusive_indices.cols();++s) {
        ProcessingScanNativeRecord r;r.scan=s;
        r.science_slots={inclusive_indices(0,s),inclusive_indices(1,s)+1};
        r.context_slots={inclusive_indices(2,s),inclusive_indices(3,s)+1};
        if (!r.science_slots.present() || !r.context_slots.present() ||
            r.context_slots.first>r.science_slots.first || r.context_slots.past_last<r.science_slots.past_last ||
            r.context_slots.past_last>common_times.size())
            throw std::invalid_argument("processing scan inner/outer interval is malformed");
        r.science_native=ranges(r.science_slots,true,r.unmapped_science_slots);
        std::size_t ignored=0;r.context_native=ranges(r.context_slots,false,ignored);
        for (auto native:r.science_native) supports.push_back({r.scan,{network,native.first,native.past_last}});
        out.scans.push_back(std::move(r));
    }
    for (const auto &run:physical_runs) {
        auto begin=run.first_native_row;
        for (auto row=begin;row<run.past_last_native_row;++row) if (selected[row-axis.first_native_row()]) {
            if (begin<row) out.native_outside_processing.push_back({begin,row});
            begin=row+1;
        }
        if (begin<run.past_last_native_row) out.native_outside_processing.push_back({begin,run.past_last_native_row});
    }
    out.binding=RtcExistingScanBinding::admit(parent,std::move(generation),std::move(relation_authority),
        "exact-native-slot-association;inclusive-abs-residual<=dt/2;absolute-epoch-and-readout-integration-uncertainty-not-estimated",
        RtcExistingScanSupportState::conservative_native_support_bound,std::move(supports));
    return out;
}

} // namespace citlali::pipeline
