#pragma once

// Explicit historical experiment profiles. These are frozen compatibility
// boundaries, not general POINT/SCIENCE admission policies.
#include <citlali/core/pipeline/ast_scan_motion.h>
#include <yaml-cpp/yaml.h>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace citlali::cli::experiment_profiles {
namespace pipeline = citlali::pipeline;
struct IdentityRouteExperimentProfile {
    static constexpr std::string_view telescope_filename =
        "tel_toltec_2026-02-19_152390_00_0002.nc";
    static constexpr std::string_view telescope_sha256 =
        "2845455a620635955c00a4731e0d9720cfa456fece79d1729cf755a366a1ad6b";
    static constexpr std::uintmax_t telescope_byte_count = 24157872;
    static constexpr std::size_t telescope_record_count = 62109;
    static constexpr pipeline::AstScanMotionIdentityBinding ast_identity_binding{
        1523900001, 1523900002, 1523900003, 1523900004};
    static constexpr std::string_view mapping_domain_prefix =
        "observation=152390:network=";
};

// Keep short-circuit parsing identical to the historical saved MAP guard.
// A foreign observation does not parse its network as an admitted product.
inline bool masked_covariance_count_152430(const YAML::Node &input) {
    return input["observation"].as<int>() == 152430 &&
           (input["network"].as<int>() == 11 || input["network"].as<int>() == 12);
}
} // namespace citlali::cli::experiment_profiles
