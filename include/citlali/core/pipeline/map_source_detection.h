#pragma once

// Included by map_source_finding.h inside namespace citlali::pipeline.

template <class SourceCounts, class SourceLocations>
void append_missing_source_location(SourceCounts &source_counts,
                                    SourceLocations &row_source_locations,
                                    SourceLocations &col_source_locations) {
    source_counts.push_back(0);
    row_source_locations.push_back(Eigen::VectorXi::Ones(1));
    col_source_locations.push_back(Eigen::VectorXi::Ones(1));

    row_source_locations.back() *= missing_source_location();
    col_source_locations.back() *= missing_source_location();
}

inline bool has_sources(Eigen::Index n_sources) {
    return n_sources > 0;
}

template <class SourceCounts, class Logger>
void log_source_detection_result(bool sources_found,
                                 const SourceCounts &source_counts,
                                 const Logger &logger) {
    if (sources_found) {
        logger->info("{} source(s) found", source_counts.back());
    }
    else {
        logger->info("no sources found");
    }
}

template <class SourceCounts>
Eigen::Index count_map_sources(const SourceCounts &source_counts) {
    Eigen::Index n_sources = 0;
    for (const auto &sources : source_counts) {
        n_sources += sources;
    }
    return n_sources;
}

template <class MapBuffer>
void clear_source_detection_vectors(MapBuffer &map_buffer) {
    map_buffer.n_sources.clear();
    map_buffer.row_source_locs.clear();
    map_buffer.col_source_locs.clear();
}

template <class MapBuffer>
void initialize_source_fit_tables(MapBuffer &map_buffer,
                                  Eigen::Index n_params) {
    const Eigen::Index n_sources =
        count_map_sources(map_buffer.n_sources);

    map_buffer.source_params.setZero(n_sources, n_params);
    map_buffer.source_perror.setZero(n_sources, n_params);
}

template <class MapBuffer, class MapCount, class Logger>
void detect_map_sources(MapBuffer &map_buffer, MapCount n_maps,
                        const Logger &logger) {
    clear_source_detection_vectors(map_buffer);

    for (Eigen::Index i = 0; i < n_maps; ++i) {
        append_missing_source_location(
            map_buffer.n_sources, map_buffer.row_source_locs,
            map_buffer.col_source_locs);

        const auto sources_found = map_buffer.find_sources(i);
        log_source_detection_result(
            sources_found, map_buffer.n_sources, logger);
    }
}

