#pragma once

// Engine member-function implementations split from engine.h.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/output_policy.h>

void Engine::setup_tod_output_chunk_selection() {
    const Eigen::Index n_scans = telescope.scan_indices.cols();

    auto build_uniform_plus_source_crossing_chunks =
        [&](const std::string &stream_name, int n_uniform, int n_source_dense) {
            if (n_scans <= 0) {
                return std::vector<Eigen::Index>{};
            }

            n_uniform = std::max(0, n_uniform);
            n_source_dense = std::max(0, n_source_dense);

            citlali::pipeline::TodSourceCrossingScan source_crossing{
                (n_scans - 1) / 2,
                std::numeric_limits<double>::infinity()};
            try {
                source_crossing =
                    citlali::pipeline::find_source_crossing_scan(
                        telescope, pointing_offsets_arcsec,
                        typed_config.mapmaking.grouping);
            }
            catch (const std::exception &e) {
                logger->warn(
                    "{} TOD uniform_plus_source_crossing selection could not calculate source-crossing scan ({}); using scan {}",
                    stream_name, e.what(), source_crossing.scan_index + 1);
            }

            const auto selected_1based =
                citlali::pipeline::uniform_plus_source_tod_output_chunks(
                    n_scans, n_uniform, n_source_dense,
                    source_crossing.scan_index);

            logger->info(
                "{} TOD output selection mode uniform_plus_source_crossing: n_uniform={} n_source_dense={} source_scan={} source_min_distance_arcsec={:.3f} selected={}",
                stream_name,
                n_uniform,
                n_source_dense,
                source_crossing.scan_index + 1,
                std::isfinite(source_crossing.min_distance2)
                    ? std::sqrt(source_crossing.min_distance2) * RAD_TO_ASEC
                    : std::numeric_limits<double>::quiet_NaN(),
                citlali::pipeline::tod_output_chunks_to_string(
                    selected_1based));
            return selected_1based;
        };

    auto setup_one = [&](const std::string &stream_name, bool output_enabled,
                         const citlali::config::TodStreamOutputConfig &config,
                         Eigen::VectorXI &scan_to_output,
                         Eigen::Index &n_output_scans) {
        std::vector<Eigen::Index> uniform_source_chunks;
        if (citlali::config::is_uniform_source_tod_output_selection_mode(
                config.selection_mode)) {
            uniform_source_chunks = build_uniform_plus_source_crossing_chunks(
                stream_name, config.selection_n_uniform,
                config.selection_n_source_dense);
        }
        citlali::pipeline::configure_tod_output_stream_selection(
            stream_name, output_enabled, config, n_scans,
            uniform_source_chunks, scan_to_output, n_output_scans, logger);
    };

    if (!citlali::pipeline::tod_output_enabled(*this)) {
        citlali::pipeline::reset(tod_outputs);
    }
    else {
        const auto &output_config = typed_config.timestream.output;
        const auto &rtc_output_config = output_config.raw_time_chunk;
        const auto &ptc_output_config = output_config.processed_time_chunk;
        setup_one("RTC", citlali::pipeline::raw_tod_output_enabled(*this),
                  rtc_output_config, tod_outputs.rtc_scan_to_output_scan,
                  tod_outputs.n_rtc_output_scans);
        setup_one("PTC",
                  citlali::pipeline::processed_tod_output_enabled(*this),
                  ptc_output_config, tod_outputs.ptc_scan_to_output_scan,
                  tod_outputs.n_ptc_output_scans);
    }
}

bool Engine::should_write_tod_chunk(Eigen::Index scan_index) const {
    return tod_output_scan_row(scan_index) >= 0;
}

Eigen::Index Engine::tod_output_scan_row(Eigen::Index scan_index) const {
    if (citlali::pipeline::raw_tod_output_enabled(*this)) {
        return tod_output_scan_row(
            scan_index, citlali::config::TodOutputStream::rtc);
    }
    if (citlali::pipeline::processed_tod_output_enabled(*this)) {
        return tod_output_scan_row(
            scan_index, citlali::config::TodOutputStream::ptc);
    }
    return -1;
}

Eigen::Index Engine::tod_output_scan_row(
    Eigen::Index scan_index, citlali::config::TodOutputStream stream) const {
    const Eigen::VectorXI *scan_to_output = nullptr;
    if (citlali::config::is_rtc_tod_output_stream(stream)) {
        scan_to_output = &tod_outputs.rtc_scan_to_output_scan;
    }
    else if (citlali::config::is_ptc_tod_output_stream(stream)) {
        scan_to_output = &tod_outputs.ptc_scan_to_output_scan;
    }
    else {
        logger->error(
            "invalid TOD stream '{}' for output row lookup",
            citlali::config::to_string(stream));
        return -1;
    }

    if (scan_index < 0 || scan_index >= scan_to_output->size()) {
        return -1;
    }
    return (*scan_to_output)(scan_index);
}
