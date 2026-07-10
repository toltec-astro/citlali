#pragma once

#include <citlali/core/pipeline/ordered_writer.h>
#include <citlali/core/pipeline/output_policy.h>

#include <exception>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

namespace citlali::pipeline {

struct TimestreamOutputFlags {
    bool write_rtc = false;
    bool write_ptc = false;
    bool write_rtcdiag = false;
    bool write_ptcdiag = false;
};

struct TimestreamOutputExpectations {
    Eigen::Index rtc = 0;
    Eigen::Index ptc = 0;
    Eigen::Index rtcdiag = 0;
    Eigen::Index ptcdiag = 0;
};

struct TimestreamOutputWriters {
    std::shared_ptr<OrderedWriter> rtc;
    std::shared_ptr<OrderedWriter> ptc;
    std::shared_ptr<OrderedWriter> rtcdiag;
    std::shared_ptr<OrderedWriter> ptcdiag;
    std::shared_ptr<OutputFailureState> failure_state;

    void cancel_all(std::exception_ptr error) const noexcept {
        if (failure_state != nullptr) {
            failure_state->record(error);
        }
        for (const auto &writer : {rtc, ptc, rtcdiag, ptcdiag}) {
            if (writer != nullptr) {
                writer->cancel(error);
            }
        }
    }

    template <class Write>
    bool write_when_ready(
        const std::shared_ptr<OrderedWriter> &writer,
        Eigen::Index index,
        Write &&write) const {
        if (writer == nullptr) {
            cancel_all(std::make_exception_ptr(
                std::logic_error("required output writer is not configured")));
            return false;
        }
        try {
            writer->write_when_ready(index, std::forward<Write>(write));
        } catch (...) {
            cancel_all(std::current_exception());
            return false;
        }
        return true;
    }

    bool failed() const noexcept {
        return failure_state != nullptr && failure_state->failed();
    }

    void rethrow_if_failed() const {
        if (failure_state != nullptr) {
            failure_state->rethrow_if_failed();
        }
    }

    void verify_complete(const TimestreamOutputExpectations &expected) const {
        const auto verify = [](const char *name,
                               const std::shared_ptr<OrderedWriter> &writer,
                               Eigen::Index count) {
            const Eigen::Index completed =
                writer == nullptr ? 0 : writer->completed_count();
            if (completed != count) {
                std::ostringstream message;
                message << "incomplete required " << name << " output: expected "
                        << count << " writes, completed " << completed;
                throw std::runtime_error(message.str());
            }
        };
        verify("RTC TOD", rtc, expected.rtc);
        verify("PTC TOD", ptc, expected.ptc);
        verify("RTC diagnostics", rtcdiag, expected.rtcdiag);
        verify("PTC diagnostics", ptcdiag, expected.ptcdiag);
    }
};

inline std::shared_ptr<OrderedWriter> make_ordered_writer_if(bool enabled) {
    return enabled ? std::make_shared<OrderedWriter>() : nullptr;
}

template <class Engine>
TimestreamOutputFlags standard_timestream_output_flags(const Engine &engine) {
    TimestreamOutputFlags flags;
    flags.write_rtc = raw_tod_output_files_available(engine);
    flags.write_ptc = processed_tod_output_files_available(engine);
    flags.write_rtcdiag = !engine.output_paths.rtcdiag_filename.empty();
    flags.write_ptcdiag = !engine.output_paths.ptcdiag_filename.empty();
    return flags;
}

template <class Engine>
TimestreamOutputFlags beammap_timestream_output_flags(
    const Engine &engine, bool write_outputs) {
    TimestreamOutputFlags flags;
    flags.write_rtc = write_outputs && raw_tod_output_files_available(engine);
    flags.write_rtcdiag = write_outputs && !engine.output_paths.rtcdiag_filename.empty();
    return flags;
}

template <class Engine>
TimestreamOutputExpectations standard_timestream_output_expectations(
    const Engine &engine, const TimestreamOutputFlags &flags) {
    const Eigen::Index scan_count = engine.telescope.scan_indices.cols();
    return {
        flags.write_rtc ? engine.tod_outputs.n_rtc_output_scans : 0,
        flags.write_ptc ? engine.tod_outputs.n_ptc_output_scans : 0,
        flags.write_rtcdiag ? scan_count : 0,
        flags.write_ptcdiag ? scan_count : 0,
    };
}

template <class Engine>
TimestreamOutputExpectations beammap_timestream_output_expectations(
    const Engine &engine, const TimestreamOutputFlags &flags) {
    const Eigen::Index scan_count = engine.telescope.scan_indices.cols();
    return {
        flags.write_rtc ? engine.tod_outputs.n_rtc_output_scans : 0,
        0,
        flags.write_rtcdiag ? scan_count : 0,
        0,
    };
}

inline TimestreamOutputWriters make_timestream_output_writers(
    const TimestreamOutputFlags &flags,
    std::shared_ptr<OutputFailureState> failure_state =
        std::make_shared<OutputFailureState>()) {
    return {
        make_ordered_writer_if(flags.write_rtc),
        make_ordered_writer_if(flags.write_ptc),
        make_ordered_writer_if(flags.write_rtcdiag),
        make_ordered_writer_if(flags.write_ptcdiag),
        std::move(failure_state),
    };
}

}  // namespace citlali::pipeline
