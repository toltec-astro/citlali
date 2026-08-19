#pragma once

#include <citlali/core/pipeline/canonical_artifact_publication.h>

#include <iosfwd>
#include <optional>
#include <string>
#include <string_view>

namespace citlali::cli::canonical_apt_contract_protocol_v2 {

namespace publication =
    citlali::pipeline::canonical_artifact_publication;

inline constexpr std::string_view protocol_v2 =
    "citlali-canonical-apt-protocol-v2";
inline constexpr std::string_view cli_option_v2 =
    "--canonical-apt-contract-v2";
inline constexpr std::string_view describe_baseline_operation_v2 =
    "describe-baseline-v2";
inline constexpr std::string_view validate_bundle_operation_v2 =
    "validate-bundle-v2";
inline constexpr std::string_view canonicalize_target_operation_v2 =
    "canonicalize-target-v2";
inline constexpr std::string_view issue_observation_apt_operation_v2 =
    "issue-observation-apt-v2";
inline constexpr std::string_view migrate_v1_to_v2_operation =
    "migrate-v1-to-v2";

inline constexpr int success_exit_code = 0;
inline constexpr int contract_rejection_exit_code = 1;
inline constexpr int protocol_error_exit_code = 2;

struct ProtocolDependencies {
    publication::IssuanceFactory issuance_factory;
    publication::BundlePublicationHooks publication_hooks;
};

struct ProtocolResult {
    int exit_code = protocol_error_exit_code;
    // Exactly one strict-JSON response object without its terminating LF.
    std::string response_json;
};

// Process one complete request. JSON is only the machine protocol; canonical
// APT products remain the verified ECSV component bundle rooted at
// manifest.ecsv and completed by manifest.ecsv.sha256.
ProtocolResult process_request_line(
    std::string_view request_json,
    const ProtocolDependencies &dependencies);

ProtocolDependencies production_dependencies();

// Return nullopt when the option is absent. When present, consume exactly one
// LF-terminated request line, write one response line, and reject all other
// argv/input before ordinary reduction setup.
std::optional<int> dispatch_if_requested(
    int argc, char *argv[], std::istream &input, std::ostream &output,
    const ProtocolDependencies &dependencies);

inline std::optional<int> dispatch_if_requested(
    int argc, char *argv[], std::istream &input, std::ostream &output) {
    return dispatch_if_requested(argc, argv, input, output,
                                 production_dependencies());
}

}  // namespace citlali::cli::canonical_apt_contract_protocol_v2
