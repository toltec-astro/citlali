#pragma once

#include <citlali/core/pipeline/ordered_writer.h>
#include <citlali/core/pipeline/output_policy.h>

#include <exception>
#include <memory>
#include <utility>

namespace citlali::pipeline {

struct TimestreamOutputFlags {
    bool write_rtc = false;
    bool write_ptc = false;
    bool write_rtcdiag = false;
    bool write_ptcdiag = false;
};

struct TimestreamOutputWriters {
    std::shared_ptr<OrderedWriter> rtc;
    std::shared_ptr<OrderedWriter> ptc;
    std::shared_ptr<OrderedWriter> rtcdiag;
    std::shared_ptr<OrderedWriter> ptcdiag;

    void cancel_all(std::exception_ptr error) const noexcept {
        for (const auto &writer : {rtc, ptc, rtcdiag, ptcdiag}) {
            if (writer != nullptr) {
                writer->cancel(error);
            }
        }
    }

    template <class Write>
    void write_when_ready(
        const std::shared_ptr<OrderedWriter> &writer,
        Eigen::Index index,
        Write &&write) const {
        try {
            writer->write_when_ready(index, std::forward<Write>(write));
        } catch (...) {
            cancel_all(std::current_exception());
            throw;
        }
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

inline TimestreamOutputWriters make_timestream_output_writers(
    const TimestreamOutputFlags &flags) {
    return {
        make_ordered_writer_if(flags.write_rtc),
        make_ordered_writer_if(flags.write_ptc),
        make_ordered_writer_if(flags.write_rtcdiag),
        make_ordered_writer_if(flags.write_ptcdiag),
    };
}

}  // namespace citlali::pipeline
