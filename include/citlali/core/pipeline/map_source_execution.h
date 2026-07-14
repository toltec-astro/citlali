#pragma once

// Included by map_source_finding.h inside namespace citlali::pipeline.

template <class SourceCount>
std::vector<int> source_index_vector(SourceCount n_sources) {
    std::vector<int> source_indices(static_cast<std::size_t>(n_sources));
    std::iota(source_indices.begin(), source_indices.end(), 0);
    return source_indices;
}

template <class ParallelPolicy, class SourceCount, class FitSource>
std::size_t fit_source_candidates(const ParallelPolicy &parallel_policy,
                                  SourceCount n_map_sources,
                                  const FitSource &fit_source) {
    const auto source_in_vec = source_index_vector(n_map_sources);
    std::vector<int> source_out_vec(source_in_vec.size());

    grppi::map(tula::grppi_utils::dyn_ex(parallel_policy),
               source_in_vec, source_out_vec, [&](auto source_index) {
        return fit_source(source_index) ? 1 : 0;
    });
    return static_cast<std::size_t>(
        std::accumulate(source_out_vec.begin(), source_out_vec.end(), 0));
}

template <auto MapType, class Engine, class MapBuffer, class Logger>
SourceFitCardinality find_map_sources_with_log(
    Engine &engine, MapBuffer &map_buffer, const Logger &logger,
    const char *log_message) {
    logger->info("{}", log_message);
    return engine.template find_sources<MapType>(map_buffer);
}

template <auto MapType, class Engine, class MapBuffer, class Logger>
std::optional<SourceFitCardinality> find_map_sources_if_needed(
    Engine &engine, MapBuffer &map_buffer, const Logger &logger,
    bool should_find, const char *log_message) {
    if (should_find) {
        return find_map_sources_with_log<MapType>(
            engine, map_buffer, logger, log_message);
    }
    return std::nullopt;
}
