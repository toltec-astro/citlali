#pragma once

#include <string>
#include <vector>

namespace citlali::pipeline {

struct ConfigDiagnosticsState {
    using key_vec_t = std::vector<std::vector<std::string>>;

    key_vec_t missing_keys;
    key_vec_t invalid_keys;

    bool has_errors() const noexcept {
        return !missing_keys.empty() || !invalid_keys.empty();
    }

    key_vec_t &missing_key_paths() noexcept {
        return missing_keys;
    }

    const key_vec_t &missing_key_paths() const noexcept {
        return missing_keys;
    }

    key_vec_t &invalid_key_paths() noexcept {
        return invalid_keys;
    }

    const key_vec_t &invalid_key_paths() const noexcept {
        return invalid_keys;
    }
};

}  // namespace citlali::pipeline
