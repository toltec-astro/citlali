#pragma once

#include <citlali/core/pipeline/paired_readout.h>

#include <concepts>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace citlali::pipeline {

struct PairedReadoutNetworkIngress {
    std::shared_ptr<const PairedReadoutOccurrenceAxis> occurrence_axis;
    std::vector<PairedReadoutDetectorIdentity> detectors;
    std::shared_ptr<const NativeReadoutMappingIdentity> mapping_identity;
    std::vector<ReadoutMemberState> x_states;
    std::vector<ReadoutMemberState> r_states;
};

// Taking one rvalue solver result makes the x/r pairing atomic at the adapter
// boundary.  Exact matrix-type checks prevent a hidden layout conversion.
template <class SolverResult>
requires(!std::is_lvalue_reference_v<SolverResult>)
PairedReadoutNetwork take_paired_kids_solver_result(
    PairedReadoutNetworkIngress ingress, SolverResult &&solver_result) {
    using XMatrix = std::remove_cvref_t<
        decltype(solver_result.data_out.xs.data)>;
    using RMatrix = std::remove_cvref_t<
        decltype(solver_result.data_out.rs.data)>;
    static_assert(std::same_as<XMatrix, PairedReadoutMatrix>);
    static_assert(std::same_as<RMatrix, PairedReadoutMatrix>);

    auto x_values = std::move(solver_result.data_out.xs.data);
    auto r_values = std::move(solver_result.data_out.rs.data);
    return PairedReadoutNetwork::admit(
        std::move(ingress.occurrence_axis),
        std::move(ingress.detectors),
        std::move(ingress.mapping_identity), std::move(x_values),
        std::move(r_values), std::move(ingress.x_states),
        std::move(ingress.r_states));
}

}  // namespace citlali::pipeline
