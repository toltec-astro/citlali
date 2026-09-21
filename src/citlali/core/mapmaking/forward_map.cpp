#include <citlali/core/mapmaking/forward_map.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace citlali::mapmaking {
namespace {
std::size_t pixels(const ForwardMapGrid &g) {
    // Half-integer boundaries must remain exactly representable in binary64.
    constexpr std::size_t exact_limit = std::size_t{1} << 52;
    if (g.identity.empty() || !g.rows || !g.columns || g.rows >= exact_limit ||
        g.columns >= exact_limit || g.rows > std::numeric_limits<std::size_t>::max()/g.columns)
        throw std::invalid_argument("MAP grid identity/shape is invalid or unrepresentable");
    return g.rows*g.columns;
}
void finite_sum(double &sum, double value) {
    const double next = sum + value;
    if (!std::isfinite(next)) throw std::overflow_error("MAP aggregate is unrepresentable");
    sum = next;
}
}

std::optional<std::size_t> containing_pixel(const ForwardMapGrid &g, double x, double y) {
    pixels(g);
    if (!std::isfinite(x) || !std::isfinite(y) || x < 0.5 || y < 0.5 ||
        x >= static_cast<double>(g.columns)+0.5 || y >= static_cast<double>(g.rows)+0.5)
        return std::nullopt;
    return static_cast<std::size_t>(std::floor(y-0.5))*g.columns +
           static_cast<std::size_t>(std::floor(x-0.5));
}

ForwardMapOperator ForwardMapOperator::resolve(ForwardMapGrid grid, std::string domain,
    std::size_t rows, std::vector<ForwardMapMember> members, ForwardMapSupport support) {
    const auto n = pixels(grid);
    if (domain.empty() || !std::isfinite(support.coverage_cut) || support.coverage_cut < 0 ||
        (support.coverage_cut > 1 && support.above_one_authority.empty()))
        throw std::invalid_argument("MAP input domain or coverage-cut authority is absent/invalid");
    std::sort(members.begin(), members.end(), [](const auto &a,const auto &b){return a.input_row<b.input_row;});
    for (std::size_t i=0; i<members.size(); ++i) {
        const auto &m=members[i];
        if (m.input_row>=rows || m.pixel>=n || !std::isfinite(m.coefficient) || m.coefficient<=0 ||
            (i && members[i-1].input_row==m.input_row))
            throw std::invalid_argument("MAP member is duplicate, out of domain, or not finite positive");
    }
    ForwardMapOperator out;
    out.grid_=std::move(grid); out.domain_=std::move(domain); out.input_rows_=rows;
    out.members_=std::move(members); out.support_=std::move(support);
    out.normalization_.assign(n,0); out.hits_.assign(n,0);
    out.norm_support_.assign(n,0); out.science_support_.assign(n,0);
    for (const auto &m:out.members_) {
        finite_sum(out.normalization_[m.pixel],m.coefficient);
        if (out.hits_[m.pixel]==std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("MAP incidence count overflow");
        ++out.hits_[m.pixel];
    }
    std::vector<double> positive;
    for (auto q:out.normalization_) if(q>0) positive.push_back(q);
    std::sort(positive.begin(),positive.end()); out.positive_count_=positive.size();
    if (!positive.empty()) {
        // Integer arithmetic for floor((floor(3N/4)+N)/2), without 3N overflow.
        const auto count=positive.size();
        const auto three_quarters=(count/4)*3+((count%4)*3)/4;
        const auto k=three_quarters/2+count/2+(three_quarters%2+count%2)/2;
        out.reference_index_=k; out.reference_=positive[k];
    }
    const double science_threshold=out.reference_*out.support_.coverage_cut;
    const double norm_threshold=science_threshold/10;
    if (!std::isfinite(science_threshold) ||
        (out.reference_>0 && out.support_.coverage_cut>0 &&
         (science_threshold==0 || norm_threshold==0)))
        throw std::overflow_error("MAP support threshold is unrepresentable");
    for (std::size_t p=0;p<n;++p) {
        const auto q=out.normalization_[p];
        out.norm_support_[p]=q>0 && q>=norm_threshold;
        out.science_support_[p]=q>0 && q>=science_threshold;
        if (out.science_support_[p]) out.output_rows_.push_back(p);
    }
    return out;
}

ForwardMapValues ForwardMapOperator::apply(std::string_view domain, std::span<const double> values) const {
    if (domain!=domain_ || values.size()!=input_rows_)
        throw std::invalid_argument("MAP application requires the exact ordered input domain");
    // Validate every member before any arithmetic; excluded values are never read.
    for (const auto &m:members_) if (!std::isfinite(values[m.input_row]))
        throw std::invalid_argument("MAP admitted payload is nonfinite");
    ForwardMapValues out; out.numerator.assign(normalization_.size(),0);
    for (const auto &m:members_) {
        const double term=m.coefficient*values[m.input_row];
        if(!std::isfinite(term) || (values[m.input_row]!=0 && term==0))
            throw std::overflow_error("MAP weighted signal is unrepresentable");
        finite_sum(out.numerator[m.pixel],term);
    }
    out.signal.reserve(output_rows_.size());
    for (auto p:output_rows_) {
        const double value=out.numerator[p]/normalization_[p];
        if(!std::isfinite(value) || (out.numerator[p]!=0 && value==0))
            throw std::overflow_error("MAP normalized signal is unrepresentable");
        out.signal.push_back(value);
    }
    return out;
}

ForwardMapExposure accumulate_original_footprints(const ForwardMapGrid &grid,
    std::string_view domain, std::span<const ForwardMapOriginal> originals,
    std::span<const ForwardMapAncestor> ancestors) {
    const auto n=pixels(grid);
    if(domain.empty()) throw std::invalid_argument("MAP original domain identity absent");
    for(const auto &o:originals)
        if(!std::isfinite(o.valid_original_seconds) || o.valid_original_seconds<0 ||
           (o.pixel && *o.pixel>=n)) throw std::invalid_argument("MAP original footprint invalid");
    std::vector<std::uint8_t> up(originals.size(),0),ret(originals.size(),0);
    for(const auto &a:ancestors) {
        if(a.original_row>=originals.size() || (a.estimator_member && !a.route_candidate))
            throw std::invalid_argument("MAP original ancestry or membership invalid");
        up[a.original_row]|=a.route_candidate; ret[a.original_row]|=a.estimator_member;
    }
    ForwardMapExposure out; out.upstream_eligible.assign(n,0);out.retained.assign(n,0);
    for(std::size_t u=0;u<originals.size();++u) {
        const auto &o=originals[u]; if(!o.pixel)continue;
        if(up[u])finite_sum(out.upstream_eligible[*o.pixel],o.valid_original_seconds);
        if(ret[u])finite_sum(out.retained[*o.pixel],o.valid_original_seconds);
    }
    return out;
}
} // namespace citlali::mapmaking
