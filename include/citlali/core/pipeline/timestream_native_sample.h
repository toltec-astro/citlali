#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace citlali::pipeline {

using TimestreamNetworkId = std::int32_t;
using TimestreamNativeRow = std::int64_t;
using TimestreamNativeRevision = std::uint64_t;

struct NativeSampleKey {
    TimestreamNetworkId network_id = -1;
    TimestreamNativeRow native_row = -1;

    friend bool operator==(const NativeSampleKey &lhs,
                           const NativeSampleKey &rhs) noexcept {
        return lhs.network_id == rhs.network_id &&
               lhs.native_row == rhs.native_row;
    }

    friend bool operator<(const NativeSampleKey &lhs,
                          const NativeSampleKey &rhs) noexcept {
        if (lhs.network_id != rhs.network_id) {
            return lhs.network_id < rhs.network_id;
        }
        return lhs.native_row < rhs.native_row;
    }
};

// This identity names a delivered row and its reconstructed timestamp.  It
// deliberately makes no claim about which physical detector integration
// event that delivered timestamp represents.
class NativeSampleIdentity {
public:
    NativeSampleIdentity(TimestreamNetworkId network_id,
                         TimestreamNativeRow native_row,
                         double reconstructed_time_unix_sec)
        : key_{network_id, native_row},
          reconstructed_time_unix_sec_{reconstructed_time_unix_sec} {
        if (network_id < 0) {
            throw std::invalid_argument(
                "native sample network identity must be nonnegative");
        }
        if (native_row < 0) {
            throw std::invalid_argument(
                "native sample row identity must be nonnegative");
        }
        if (!std::isfinite(reconstructed_time_unix_sec)) {
            throw std::invalid_argument(
                "native reconstructed timestamp must be finite");
        }
    }

    const NativeSampleKey &key() const noexcept { return key_; }
    TimestreamNetworkId network_id() const noexcept {
        return key_.network_id;
    }
    TimestreamNativeRow native_row() const noexcept {
        return key_.native_row;
    }
    double reconstructed_time_unix_sec() const noexcept {
        return reconstructed_time_unix_sec_;
    }

    friend bool operator==(const NativeSampleIdentity &lhs,
                           const NativeSampleIdentity &rhs) noexcept {
        return lhs.key_ == rhs.key_ &&
               lhs.reconstructed_time_unix_sec_ ==
                   rhs.reconstructed_time_unix_sec_;
    }

private:
    NativeSampleKey key_;
    double reconstructed_time_unix_sec_;
};

struct NativeOperationIdentity {
    std::uint64_t sequence = 0;
    std::int64_t scan_index = -1;

    NativeOperationIdentity(std::uint64_t sequence_,
                            std::int64_t scan_index_)
        : sequence{sequence_}, scan_index{scan_index_} {
        if (scan_index < 0) {
            throw std::invalid_argument(
                "native operation scan index must be nonnegative");
        }
    }

    friend bool operator==(const NativeOperationIdentity &lhs,
                           const NativeOperationIdentity &rhs) noexcept {
        return lhs.sequence == rhs.sequence &&
               lhs.scan_index == rhs.scan_index;
    }
};

}  // namespace citlali::pipeline
