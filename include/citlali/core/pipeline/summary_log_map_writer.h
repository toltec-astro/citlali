#pragma once

// Included by summary_log.h inside namespace citlali::pipeline.

template <class MapBuffer>
void write_map_product_presence_summary(std::ostream &stream,
                                        const MapBuffer &mb) {
    stream << "-Kernel maps generated: " << !mb.kernel.empty() << "\n";
    stream << "-Coverage maps generated: " << !mb.coverage.empty() << "\n";
    stream << "-Noise maps generated: " << !mb.noise.empty() << "\n";
    stream << "-Number of noise maps: " << mb.noise.size() << "\n";
}

template <class NonfiniteCounts>
void write_map_nonfinite_summary(std::ostream &stream,
                                 const NonfiniteCounts &counts) {
    for (auto const& [key, val] : counts.n_nans) {
         stream << "-Number of " + key + " NaNs: " << val << "\n";
    }

    for (auto const& [key, val] : counts.n_infs) {
        stream << "-Number of " + key + " Infs: " << val << "\n";
    }
}

template <class MapBuffer, class NonfiniteCounts>
void write_map_summary_log(std::ostream &stream,
                           const std::string &citlali_version,
                           const std::string &kids_version,
                           const std::string &write_time,
                           std::string_view reduction_type,
                           std::string_view tod_type,
                           std::string_view map_grouping,
                           long long n_maps, const MapBuffer &mb,
                           const NonfiniteCounts &nonfinite_counts) {
    stream << "Summary file for maps\n";
    write_pipeline_version_summary(stream, citlali_version, kids_version);
    write_file_time_summary(stream, write_time);
    write_map_identity_summary(
        stream, std::string(reduction_type), std::string(tod_type),
        std::string(map_grouping), mb.n_rows, mb.n_cols, n_maps,
        mb.sig_unit);
    write_map_product_presence_summary(stream, mb);
    write_map_nonfinite_summary(stream, nonfinite_counts);
}

