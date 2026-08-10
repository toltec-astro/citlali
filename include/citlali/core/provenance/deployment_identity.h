#pragma once

#include <citlali_config/config.h>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace citlali::provenance {

struct DeploymentIdentity {
    std::string profile;
    std::string lock_sha256;
    std::filesystem::path environment;

    [[nodiscard]] bool managed() const noexcept { return !profile.empty(); }
};

enum class DeploymentBinding { unmanaged, dag_match };

inline auto environment_value(const char *name) -> std::optional<std::string> {
    const auto *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return std::nullopt;
    }
    return std::string{value};
}

inline bool valid_sha256(std::string_view value) {
    return value.size() == 64 &&
           std::ranges::all_of(value, [](unsigned char character) {
               return std::isxdigit(character) != 0;
           });
}

inline DeploymentIdentity
deployment_identity_from_values(const std::optional<std::string> &profile,
                                const std::optional<std::string> &lock_sha256,
                                const std::optional<std::string> &environment) {
    const auto present = static_cast<int>(profile.has_value()) +
                         static_cast<int>(lock_sha256.has_value()) +
                         static_cast<int>(environment.has_value());
    if (present == 0) {
        return {};
    }
    if (present != 3) {
        throw std::runtime_error(
            "incomplete managed deployment identity: TOLTECA_SPACK_PROFILE, "
            "TOLTECA_SPACK_LOCK_SHA256, and TOLTECA_CPP_ENV must be set "
            "together");
    }
    if (!valid_sha256(*lock_sha256)) {
        throw std::runtime_error(
            "invalid TOLTECA_SPACK_LOCK_SHA256: expected 64 hexadecimal "
            "characters");
    }
    return DeploymentIdentity{*profile, *lock_sha256, *environment};
}

inline DeploymentIdentity runtime_deployment_identity() {
    return deployment_identity_from_values(
        environment_value("TOLTECA_SPACK_PROFILE"),
        environment_value("TOLTECA_SPACK_LOCK_SHA256"),
        environment_value("TOLTECA_CPP_ENV"));
}

inline std::string compiled_spack_dag_hash() {
#ifdef CITLALI_SPACK_DAG_HASH
    return CITLALI_SPACK_DAG_HASH;
#else
    return {};
#endif
}

inline std::string
deployment_profile_label(const DeploymentIdentity &identity) {
    return identity.managed() ? identity.profile : "unmanaged";
}

inline std::string deployment_lock_label(const DeploymentIdentity &identity) {
    return identity.managed() ? identity.lock_sha256 : "unavailable";
}

inline std::string deployment_binding_label(DeploymentBinding binding) {
    return binding == DeploymentBinding::dag_match ? "dag-match" : "unmanaged";
}

inline std::string
lock_root_dag_hash(const std::filesystem::path &environment) {
    const auto lock_path = environment / "spack.lock";
    YAML::Node lock;
    try {
        lock = YAML::LoadFile(lock_path.string());
    } catch (const std::exception &error) {
        throw std::runtime_error("cannot read managed deployment lock " +
                                 lock_path.string() + ": " + error.what());
    }
    const auto roots = lock["roots"];
    if (!roots || !roots.IsSequence() || roots.size() != 1) {
        throw std::runtime_error(
            "managed deployment lock must contain exactly one root: " +
            lock_path.string());
    }
    const auto hash = roots[0]["hash"];
    if (!hash || !hash.IsScalar()) {
        throw std::runtime_error(
            "managed deployment lock root is missing its DAG hash: " +
            lock_path.string());
    }
    return hash.as<std::string>();
}

inline DeploymentBinding
require_deployment_matches_build(const DeploymentIdentity &identity,
                                 std::string_view build_dag_hash) {
    if (!identity.managed()) {
        return DeploymentBinding::unmanaged;
    }
    if (build_dag_hash.empty()) {
        throw std::runtime_error(
            "managed deployment requires a Citlali executable with compiled "
            "Spack DAG identity");
    }
    const auto lock_hash = lock_root_dag_hash(identity.environment);
    if (lock_hash != build_dag_hash) {
        throw std::runtime_error(
            "managed deployment DAG mismatch: executable=" +
            std::string{build_dag_hash} + " lock=" + lock_hash);
    }
    return DeploymentBinding::dag_match;
}

inline DeploymentBinding require_runtime_deployment_matches_build() {
    return require_deployment_matches_build(runtime_deployment_identity(),
                                            compiled_spack_dag_hash());
}

} // namespace citlali::provenance
