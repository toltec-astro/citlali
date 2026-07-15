#pragma once

#include <citlali/core/pipeline/stage_profile.h>
#include <citlali/core/session/reduction_result.h>

#include <cstddef>
#include <exception>
#include <functional>
#include <string>
#include <utility>

namespace citlali::session {

enum class ReductionSessionState {
    ready,
    running,
    succeeded,
    failed
};

class ReductionSession {
public:
    ReductionSession() = default;
    ReductionSession(const ReductionSession &) = delete;
    ReductionSession &operator=(const ReductionSession &) = delete;
    ReductionSession(ReductionSession &&) = delete;
    ReductionSession &operator=(ReductionSession &&) = delete;

    ReductionSessionState state() const noexcept {
        return state_;
    }

    std::size_t runs_started() const noexcept {
        return runs_started_;
    }

    const citlali::pipeline::StageProfileCollector &stage_profile() const
        noexcept {
        return stage_profile_;
    }

    template <class Operation>
    ReductionResult run(Operation &&operation) {
        if (state_ == ReductionSessionState::running) {
            return failed_reduction_result(
                ReductionStatus::invalid_session_state,
                "session.already_running",
                "the reduction session is already running");
        }

        state_ = ReductionSessionState::running;
        ++runs_started_;
        stage_profile_.reset();

        ReductionResult result;
        try {
            result = std::invoke(std::forward<Operation>(operation),
                                 stage_profile_);
        } catch (const citlali::error::Error &error) {
            result = failed_reduction_result(error);
        } catch (const std::exception &error) {
            result = failed_reduction_result(
                ReductionStatus::unhandled_exception,
                "session.unhandled_exception", error.what());
        } catch (...) {
            result = failed_reduction_result(
                ReductionStatus::unhandled_exception,
                "session.unhandled_exception",
                "unknown exception escaped reduction execution");
        }

        state_ = result.succeeded()
            ? ReductionSessionState::succeeded
            : ReductionSessionState::failed;
        return result;
    }

private:
    ReductionSessionState state_ = ReductionSessionState::ready;
    std::size_t runs_started_ = 0;
    citlali::pipeline::StageProfileCollector stage_profile_;
};

}  // namespace citlali::session
