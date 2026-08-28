#include <citlali/core/pipeline/paired_readout.h>
#include <citlali/core/pipeline/paired_readout_kids_adapter.h>

#include <kids/timestream/solver.h>

#include <concepts>
#include <type_traits>
#include <utility>

static_assert(std::same_as<
              std::remove_cvref_t<decltype(
                  std::declval<kids::TimeStreamSolverResult>()
                      .data_out.xs.data)>,
              citlali::pipeline::PairedReadoutMatrix>);
static_assert(std::same_as<
              std::remove_cvref_t<decltype(
                  std::declval<kids::TimeStreamSolverResult>()
                      .data_out.rs.data)>,
              citlali::pipeline::PairedReadoutMatrix>);

namespace {
[[maybe_unused]] auto paired_readout_header_is_self_contained =
    sizeof(citlali::pipeline::PairedReadout);
}
