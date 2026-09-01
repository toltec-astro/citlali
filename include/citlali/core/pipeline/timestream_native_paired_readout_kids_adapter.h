#pragma once

#include <citlali/core/pipeline/timestream_native_paired_readout.h>

#include <concepts>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace citlali::pipeline {

struct NativePairedReadoutNetworkIngress {
    std::shared_ptr<const NativePairedReadoutOccurrenceAxis> occurrence_axis;
    std::vector<NativeReadoutDetectorBinding> detectors;
    std::shared_ptr<const NativeReadoutMappingAuthority> mapping_authority;
    std::vector<NativeReadoutCoordinateState> x_states;
    std::vector<NativeReadoutCoordinateState> r_states;
};

// One mutable rvalue result is the atomic x/r source. Exact matrix-type checks
// make both ownership transfers fail at compile time if the KIDs boundary
// would require a hidden conversion, layout change, or independent source
// walk.
template <class SolverResult>
requires(!std::is_lvalue_reference_v<SolverResult> &&
         !std::is_const_v<std::remove_reference_t<SolverResult>>)
NativePairedReadoutNetwork take_native_paired_kids_solver_result(
    NativePairedReadoutNetworkIngress ingress,
    SolverResult &&solver_result) {
    using XMatrix = std::remove_cvref_t<
        decltype(solver_result.data_out.xs.data)>;
    using RMatrix = std::remove_cvref_t<
        decltype(solver_result.data_out.rs.data)>;
    static_assert(std::same_as<XMatrix, NativePairedReadoutMatrix>);
    static_assert(std::same_as<RMatrix, NativePairedReadoutMatrix>);

    auto x_values = std::move(solver_result.data_out.xs.data);
    auto r_values = std::move(solver_result.data_out.rs.data);
    return NativePairedReadoutNetwork::admit(
        std::move(ingress.occurrence_axis),
        std::move(ingress.detectors),
        std::move(ingress.mapping_authority), std::move(x_values),
        std::move(r_values), std::move(ingress.x_states),
        std::move(ingress.r_states));
}

}  // namespace citlali::pipeline
