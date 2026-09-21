#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace citlali::mapmaking {

// MAP-local numerical boundary. This does not perform PTC/VAL admission,
// declare a celestial frame, or turn a numerical result into a science bundle.
// The caller supplies structurally admitted occurrences, already expressed by
// AST in the exact target grid. FITS pixel centers are 1-based: cell (0,0)
// occupies [0.5,1.5) on each continuous axis. Storage is row-major (y,x).
struct ForwardMapGrid {
    std::size_t rows, columns;
    std::string identity;
};
std::optional<std::size_t> containing_pixel(const ForwardMapGrid &, double x, double y);

// Compact locator into one immutable, exact PTC publication, not a persistent
// occurrence identity on its own. The domain identity binds its ordered rows.
struct ForwardMapMember {
    std::size_t input_row, pixel;
    double coefficient;
};
struct ForwardMapSupport {
    double coverage_cut;
    // A nonempty identity must reference an independently authorized request
    // for this exact cut/scope; the kernel does not issue that authorization.
    std::string above_one_authority;
};
struct ForwardMapValues {
    std::vector<double> numerator; // full grid, pre-normalized, never sky
    std::vector<double> signal; // only operator.output_rows(), includes real zero
};

// Immutable resolved numerical operator. Sorted by exact input-domain row so
// caller ordering and processing blocks do not alter accumulation order.
// Positive coefficients are generic arithmetic here; selecting a scientific
// coefficient family remains exclusively the PTC producer's responsibility.
class ForwardMapOperator {
public:
    static ForwardMapOperator resolve(ForwardMapGrid, std::string input_domain,
        std::size_t input_rows, std::vector<ForwardMapMember>, ForwardMapSupport);
    ForwardMapValues apply(std::string_view exact_domain, std::span<const double>) const;
    const auto &grid() const noexcept { return grid_; }
    const auto &input_domain() const noexcept { return domain_; }
    auto input_rows() const noexcept { return input_rows_; }
    const auto &members() const noexcept { return members_; }
    const auto &normalization() const noexcept { return normalization_; }
    const auto &contributing_hits() const noexcept { return hits_; }
    const auto &normalization_support() const noexcept { return norm_support_; }
    const auto &science_support() const noexcept { return science_support_; }
    const auto &output_rows() const noexcept { return output_rows_; }
    auto reference_normalization() const noexcept { return reference_; }
    auto reference_index() const noexcept { return reference_index_; }
    auto positive_pixel_count() const noexcept { return positive_count_; }
    const auto &support_request() const noexcept { return support_; }
private:
    ForwardMapOperator() = default;
    ForwardMapGrid grid_;
    std::string domain_;
    std::size_t input_rows_ = 0, positive_count_ = 0;
    std::optional<std::size_t> reference_index_;
    std::vector<ForwardMapMember> members_;
    ForwardMapSupport support_;
    double reference_ = 0;
    std::vector<double> normalization_;
    std::vector<std::uint64_t> hits_;
    std::vector<std::uint8_t> norm_support_, science_support_;
    std::vector<std::size_t> output_rows_;
};

// Original-footprint accounting is separate from the linear MAP operator.
// Stable original rows and original_domain bind ALIGN occurrence identity;
// each original uses its own AST pixel. Repeated influence edges only union
// membership; no descendant pixel or retained-count time surrogate is used.
struct ForwardMapOriginal {
    // nullopt means a known invalid/outside coordinate, not missing authority.
    // Missing original-coordinate authority makes the whole exposure role
    // unavailable at the owning boundary, before calling this primitive.
    std::optional<std::size_t> pixel;
    double valid_original_seconds;
};
struct ForwardMapAncestor {
    std::size_t original_row;
    bool route_candidate, estimator_member;
};
struct ForwardMapExposure {
    std::vector<double> upstream_eligible, retained;
};
ForwardMapExposure accumulate_original_footprints(const ForwardMapGrid &,
    std::string_view original_domain, std::span<const ForwardMapOriginal>,
    std::span<const ForwardMapAncestor>);

} // namespace citlali::mapmaking
