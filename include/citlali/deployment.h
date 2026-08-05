#pragma once

#include <cstdlib>
#include <string>

namespace citlali::deployment {

inline auto environment_value(const char *name) -> std::string {
    const auto *value = std::getenv(name);
    return value == nullptr ? std::string{} : std::string{value};
}

inline auto spack_profile() -> std::string {
    return environment_value("TOLTECA_SPACK_PROFILE");
}

inline auto spack_lock_sha256() -> std::string {
    return environment_value("TOLTECA_SPACK_LOCK_SHA256");
}

}  // namespace citlali::deployment
