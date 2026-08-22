#pragma once

#include <citlali/core/pipeline/timestream_native_pointing.h>

#include <memory>
#include <stdexcept>
#include <utility>

namespace citlali::pipeline {

// Immutable observation-owned ALIGN/pointing carrier pair.  This is not the
// native-ready consumer binding: Stage 7 must also bind the exact compact-v2
// relation before any numerical route can activate.
class NativeObservationCarriers {
public:
    NativeObservationCarriers(
        NativeObservationScope expected_scope,
        std::shared_ptr<const NativeAlignmentPlan> alignment,
        std::shared_ptr<const NativePointingPlan> pointing)
        : expected_scope_{expected_scope},
          alignment_{std::move(alignment)},
          pointing_{std::move(pointing)} {
        if (!alignment_ || !pointing_) {
            throw std::invalid_argument(
                "native observation carrier pair is incomplete");
        }
        if (!(alignment_->scope() == expected_scope_) ||
            !(pointing_->scope() == expected_scope_)) {
            throw std::invalid_argument(
                "native observation carrier scope is stale or foreign");
        }
        if (!pointing_->bound_to(alignment_)) {
            throw std::invalid_argument(
                "native pointing candidate is stale for its alignment handle");
        }
    }

    const NativeObservationScope &scope() const noexcept {
        return expected_scope_;
    }
    const std::shared_ptr<const NativeAlignmentPlan> &alignment_handle()
        const noexcept {
        return alignment_;
    }
    const std::shared_ptr<const NativePointingPlan> &pointing_handle() const
        noexcept {
        return pointing_;
    }

private:
    NativeObservationScope expected_scope_;
    std::shared_ptr<const NativeAlignmentPlan> alignment_;
    std::shared_ptr<const NativePointingPlan> pointing_;
};

// The owner validates a complete candidate first and swaps one immutable
// handle only after admission.  Failed publication leaves the prior pair
// pointer-identical.
class NativeObservationCarrierSlot {
public:
    explicit NativeObservationCarrierSlot(
        NativeObservationScope expected_scope)
        : expected_scope_{expected_scope} {}

    void publish(
        std::shared_ptr<const NativeAlignmentPlan> alignment,
        std::shared_ptr<const NativePointingPlan> pointing) {
        auto candidate = std::make_shared<const NativeObservationCarriers>(
            expected_scope_, std::move(alignment), std::move(pointing));
        current_.swap(candidate);
    }

    void reset() noexcept { current_.reset(); }
    bool has_value() const noexcept { return current_ != nullptr; }
    const std::shared_ptr<const NativeObservationCarriers> &handle() const
        noexcept {
        return current_;
    }
    const NativeObservationCarriers &require() const {
        if (!current_) {
            throw std::logic_error(
                "native observation carriers are not published");
        }
        return *current_;
    }

private:
    NativeObservationScope expected_scope_;
    std::shared_ptr<const NativeObservationCarriers> current_;
};

}  // namespace citlali::pipeline
