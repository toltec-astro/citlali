#pragma once

#include <citlali/core/pipeline/telescope_header_snapshot.h>

#include <Eigen/Core>

#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace citlali::engine_detail {

struct ResolvedEquatorialAliases {
    Eigen::VectorXd right_ascension;
    Eigen::VectorXd declination;
    std::string source_schema;
};

inline bool exact_vector_identity(const Eigen::VectorXd &lhs,
                                  const Eigen::VectorXd &rhs) {
    return lhs.size() == rhs.size() &&
           std::memcmp(lhs.data(), rhs.data(),
                       static_cast<std::size_t>(lhs.size()) *
                           sizeof(double)) == 0;
}

inline ResolvedEquatorialAliases resolve_equatorial_aliases(
    const std::map<std::string, Eigen::VectorXd> &raw) {
    constexpr const char *canonical_ra =
        "Data.TelescopeBackend.SourceRaAct";
    constexpr const char *canonical_dec =
        "Data.TelescopeBackend.SourceDecAct";
    constexpr const char *legacy_ra =
        "Data.TelescopeBackend.TelRaAct";
    constexpr const char *legacy_dec =
        "Data.TelescopeBackend.TelDecAct";

    const auto canonical_ra_it = raw.find(canonical_ra);
    const auto canonical_dec_it = raw.find(canonical_dec);
    const auto legacy_ra_it = raw.find(legacy_ra);
    const auto legacy_dec_it = raw.find(legacy_dec);
    const bool canonical_ra_present = canonical_ra_it != raw.end();
    const bool canonical_dec_present = canonical_dec_it != raw.end();
    const bool legacy_ra_present = legacy_ra_it != raw.end();
    const bool legacy_dec_present = legacy_dec_it != raw.end();
    if (canonical_ra_present != canonical_dec_present) {
        throw std::runtime_error(
            "canonical SourceRaAct/SourceDecAct telescope schema is partial");
    }
    if (legacy_ra_present != legacy_dec_present) {
        throw std::runtime_error(
            "legacy TelRaAct/TelDecAct telescope schema is partial");
    }
    if (!canonical_ra_present && !legacy_ra_present) {
        throw std::runtime_error(
            "required telescope RA/Dec schema pair is absent");
    }
    if (canonical_ra_present && legacy_ra_present &&
        (!exact_vector_identity(canonical_ra_it->second,
                                legacy_ra_it->second) ||
         !exact_vector_identity(canonical_dec_it->second,
                                legacy_dec_it->second))) {
        throw std::runtime_error(
            "canonical and legacy telescope RA/Dec schema pairs conflict");
    }
    if (canonical_ra_present) {
        return {canonical_ra_it->second, canonical_dec_it->second,
                "SourceRaAct_SourceDecAct"};
    }
    return {legacy_ra_it->second, legacy_dec_it->second,
            "TelRaAct_TelDecAct"};
}

inline pipeline::sci_align::TelescopeHeaderSnapshot
simulation_j2000_compatibility_header_snapshot() {
    using pipeline::sci_align::TelescopeHeaderDimensionSnapshot;
    using pipeline::sci_align::TelescopeHeaderNumericType;
    using pipeline::sci_align::TelescopeHeaderSnapshot;
    TelescopeHeaderSnapshot result;
    result.type = TelescopeHeaderNumericType::float64;
    // The governing application emitted every legacy telescope header on
    // this one-element dimension, including its synthesized simulation epoch.
    result.dimensions = {
        TelescopeHeaderDimensionSnapshot{"tel_header_n_pts", 1}};
    result.values = std::vector<double>{2000.0};
    return result;
}

}  // namespace citlali::engine_detail
