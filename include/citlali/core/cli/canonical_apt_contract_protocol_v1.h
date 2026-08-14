#pragma once

#include <citlali/core/pipeline/canonical_artifact_publication.h>

#include <iosfwd>
#include <optional>
#include <string>
#include <string_view>

namespace citlali::cli::canonical_apt_contract_protocol_v1 {

namespace publication =
    citlali::pipeline::canonical_artifact_publication;

inline constexpr std::string_view protocol_v1 =
    "citlali-canonical-apt-protocol-v1";
inline constexpr std::string_view cli_option_v1 =
    "--canonical-apt-contract-v1";
inline constexpr std::string_view describe_baseline_operation_v1 =
    "describe-baseline-v1";
inline constexpr std::string_view issue_observation_apt_operation_v1 =
    "issue-observation-apt-v1";
inline constexpr std::string_view validate_observation_apt_operation_v1 =
    "validate-observation-apt-v1";

inline constexpr int success_exit_code = 0;
inline constexpr int contract_rejection_exit_code = 1;
inline constexpr int protocol_error_exit_code = 2;

struct ProtocolDependencies {
    publication::IssuanceFactory issuance_factory;
    publication::PublicationHooks publication_hooks;
};

struct ProtocolResult {
    int exit_code = protocol_error_exit_code;
    // Exactly one strict-JSON response object without its terminating LF.
    std::string response_json;
};

// Process one complete JSON request. This API performs no reduction/config
// dispatch and never treats JSON as a persisted APT representation.
ProtocolResult process_request_line(
    std::string_view request_json,
    const ProtocolDependencies &dependencies);

// The production dependency set issues opaque occurrence/event references
// from local OS entropy. Tests inject their own factory and publication hooks.
ProtocolDependencies production_dependencies();

// Return nullopt when the versioned protocol option is absent. When present,
// consume exactly one LF-terminated request line, write exactly one response
// line, and return the protocol exit code. Extra arguments or input fail
// closed before normal reduction parsing or logging is initialized.
std::optional<int> dispatch_if_requested(
    int argc, char *argv[], std::istream &input, std::ostream &output,
    const ProtocolDependencies &dependencies);

inline std::optional<int> dispatch_if_requested(
    int argc, char *argv[], std::istream &input, std::ostream &output) {
    return dispatch_if_requested(argc, argv, input, output,
                                 production_dependencies());
}

}  // namespace citlali::cli::canonical_apt_contract_protocol_v1
