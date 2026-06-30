#pragma once

#include <stdexcept>
#include <string>
#include <utility>

namespace citlali::error {

enum class Code {
    invalid_config,
    io,
    runtime,
    internal
};

class Error : public std::runtime_error {
public:
    Error(Code code, std::string message)
        : std::runtime_error(message), code_(code) {}

    [[nodiscard]] Code code() const noexcept {
        return code_;
    }

private:
    Code code_;
};

inline Error invalid_config(std::string message) {
    return Error{Code::invalid_config, std::move(message)};
}

inline Error io(std::string message) {
    return Error{Code::io, std::move(message)};
}

inline Error runtime(std::string message) {
    return Error{Code::runtime, std::move(message)};
}

inline Error internal(std::string message) {
    return Error{Code::internal, std::move(message)};
}

}  // namespace citlali::error
