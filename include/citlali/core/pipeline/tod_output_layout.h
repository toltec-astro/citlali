#pragma once

// TOD output NetCDF metadata implementation detail.
// Include this only from output_netcdf_metadata.h inside citlali::pipeline.

inline double tod_output_fill_double() {
    return std::numeric_limits<double>::quiet_NaN();
}

constexpr int tod_output_fill_int() {
    return -2147483647;
}

inline std::string tod_output_directory(const std::string &obsnum_dir_name,
                                        const std::string &subdir_name) {
    std::string dir_name = obsnum_dir_name + "raw/";
    if (subdir_name != "null") {
        dir_name += subdir_name + "/";
    }
    return dir_name;
}

inline const char *tod_stream_output_key(bool is_rtc_stream) {
    return is_rtc_stream ? "rtc" : "ptc";
}

template <class TodFilenameMap>
std::string register_tod_output_file(TodFilenameMap &tod_filename,
                                     const std::string &key,
                                     const std::string &filename_base) {
    tod_filename[key] = filename_base + ".nc";
    return key;
}

template <auto DataType, auto ProductType, auto FilterType, class ToltecIo>
std::string tod_stream_output_filename(
    ToltecIo &toltec_io, const std::string &dir_name,
    const std::string &reduction_type, const std::string &obsnum,
    bool simulated_observation) {
    return toltec_io.template create_filename<DataType, ProductType,
                                              FilterType>(
        dir_name, reduction_type, "", obsnum, simulated_observation);
}

template <auto DataType, auto ProductType, auto FilterType, class ToltecIo,
          class TodFilenameMap>
std::string register_tod_stream_output_file(
    ToltecIo &toltec_io, TodFilenameMap &tod_filename,
    const std::string &dir_name, const std::string &reduction_type,
    const std::string &obsnum, bool simulated_observation,
    bool is_rtc_stream) {
    const auto filename =
        tod_stream_output_filename<DataType, ProductType, FilterType>(
            toltec_io, dir_name, reduction_type, obsnum,
            simulated_observation);
    return register_tod_output_file(
        tod_filename, tod_stream_output_key(is_rtc_stream), filename);
}

struct TodFileDims {
    netCDF::NcDim n_pts;
    netCDF::NcDim n_raw_scan_indices;
    netCDF::NcDim n_scan_indices;
    netCDF::NcDim n_scans;
    netCDF::NcDim n_dets;
    std::vector<netCDF::NcDim> signal;
    std::vector<netCDF::NcDim> raw_scans;
    std::vector<netCDF::NcDim> scans;
};

struct TodStreamLayout {
    Eigen::Index n_output_scans;
    bool mini_output;
    bool outer_output;
};

struct TodFileCounts {
    std::size_t n_output_scans;
    std::size_t n_raw_scan_indices;
    std::size_t n_dets;
};

struct TodChunking {
    netCDF::NcVar::ChunkMode mode;
    std::vector<std::size_t> sizes;
};

struct TodPreparedLayout {
    TodStreamLayout stream;
    TodFileCounts counts;
    TodFileDims dims;
    TodChunking chunking;
};

inline TodFileCounts tod_file_counts(Eigen::Index n_output_scans,
                                     Eigen::Index n_raw_scan_indices,
                                     Eigen::Index n_dets) {
    return {
        static_cast<std::size_t>(n_output_scans),
        static_cast<std::size_t>(n_raw_scan_indices),
        static_cast<std::size_t>(n_dets),
    };
}

template <class RtcProc, class PtcProc>
TodStreamLayout tod_stream_layout(bool is_rtc_stream,
                                  Eigen::Index n_rtc_output_scans,
                                  Eigen::Index n_ptc_output_scans,
                                  const RtcProc &rtcproc,
                                  const PtcProc &ptcproc) {
    return {
        is_rtc_stream ? n_rtc_output_scans : n_ptc_output_scans,
        is_rtc_stream ? rtcproc.tod_output_mini : ptcproc.tod_output_mini,
        is_rtc_stream ? rtcproc.tod_output_outer : ptcproc.tod_output_outer,
    };
}

inline TodFileDims add_tod_file_dims(netCDF::NcFile &fo,
                                     std::size_t n_output_scans,
                                     std::size_t n_raw_scan_indices,
                                     std::size_t n_dets) {
    TodFileDims dims;
    dims.n_pts = fo.addDim("n_pts");
    dims.n_raw_scan_indices =
        fo.addDim("n_raw_scan_indices", n_raw_scan_indices);
    dims.n_scan_indices = fo.addDim("n_scan_indices", 2);
    dims.n_scans = fo.addDim("n_scans", n_output_scans);
    dims.n_dets = fo.addDim("n_dets", n_dets);
    dims.signal = {dims.n_pts, dims.n_dets};
    dims.raw_scans = {dims.n_scans, dims.n_raw_scan_indices};
    dims.scans = {dims.n_scans, dims.n_scan_indices};
    return dims;
}

template <class ScanIndices>
std::vector<std::size_t> tod_data_chunk_sizes(const ScanIndices &scan_indices,
                                              std::size_t n_dets) {
    const auto mean_scan_size =
        ((scan_indices.row(3) - scan_indices.row(2)).array() + 1).mean();
    return {static_cast<std::size_t>(mean_scan_size), n_dets};
}

template <class ScanIndices>
TodChunking tod_data_chunking(const ScanIndices &scan_indices,
                              std::size_t n_dets) {
    return {
        netCDF::NcVar::nc_CHUNKED,
        tod_data_chunk_sizes(scan_indices, n_dets),
    };
}

inline void add_tod_output_type_label(netCDF::NcFile &fo,
                                      const std::string &label) {
    netCDF::NcDim dim = fo.addDim("n_tod_output_type", 1);
    netCDF::NcVar var = fo.addVar("tod_output_type", netCDF::ncString, dim);
    const std::vector<std::size_t> index = {0};
    std::string value = label;
    var.putVar(index, value);
}

inline void add_tod_stream_output_type_label(netCDF::NcFile &fo,
                                             bool is_rtc_stream) {
    add_tod_output_type_label(fo, tod_stream_output_key(is_rtc_stream));
}
