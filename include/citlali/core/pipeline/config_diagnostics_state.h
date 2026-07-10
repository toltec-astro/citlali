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
};

}  // namespace citlali::pipeline
