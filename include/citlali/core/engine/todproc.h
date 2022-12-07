#pragma once

#include <unsupported/Eigen/CXX11/Tensor>

#include <tula/eigen.h>

#include <citlali/core/utils/pointing.h>
#include <tula/algorithm/mlinterp/mlinterp.hpp>

#include <citlali/core/utils/fits_io.h>
#include <citlali/core/utils/netcdf_io.h>
#include <citlali/core/utils/toltec_io.h>

namespace fs = std::filesystem;

struct DummyEngine {
    template <typename OStream>
    friend OStream &operator<<(OStream &os, const DummyEngine &e) {
        return os << fmt::format("DummyEngine()");
    }
};

/**
 * @brief The time ordered data processing struct
 * This wraps around the lali config
 */

template <class EngineType>
struct TimeOrderedDataProc : ConfigMapper<TimeOrderedDataProc<EngineType>> {
    using Base = ConfigMapper<TimeOrderedDataProc<EngineType>>;
    using config_t = typename Base::config_t;
    using Engine = EngineType;
    using scanindicies_t = Eigen::MatrixXI;
    using map_extent_t = std::vector<double>;
    using map_coord_t = std::vector<Eigen::VectorXd>;
    using map_count_t = std::size_t;
    using array_indices_t = std::vector<std::tuple<Eigen::Index, Eigen::Index>>;
    using det_indices_t = std::vector<std::tuple<Eigen::Index, Eigen::Index>>;

    TimeOrderedDataProc(config_t config) : Base{std::move(config)} {}

    static auto check_config(const config_t &config)
        -> std::optional<std::string> {
        std::vector<std::string> missing_keys;
        SPDLOG_TRACE("check TOD proc config\n{}", config);
        if (!config.has("runtime")) {
            missing_keys.push_back("runtime");
        }
        if (!config.has("timestream")) {
            missing_keys.push_back("timestream");
        }
        if (!config.has("mapmaking")) {
            missing_keys.push_back("mapmaking");
        }
        if (!config.has("beammap")) {
            missing_keys.push_back("beammap");
        }
        if (!config.has("coadd")) {
            missing_keys.push_back("coadd");
        }
        if (missing_keys.empty()) {
            return std::nullopt;
        }
        return fmt::format("invalid or missing keys={}", missing_keys);
    }

    void create_output_dir();
    void align_timestreams(const RawObs &rawobs);
    void calc_map_num();
    void allocate_cmb(std::vector<map_extent_t> &map_extents, std::vector<map_coord_t> &map_coords);

    template<class map_buffer_t>
    void allocate_nmb(map_buffer_t &mb);

    void allocate_omb(map_extent_t &map_extent, map_coord_t &map_coord);
    void calc_map_size(std::vector<map_extent_t> &map_extents, std::vector<map_coord_t> &map_coords);
    void coadd();
    void make_index_file(std::string filepath);
    void setup_filenames();


    // TODO fix the const correctness
    Engine &engine() { return m_engine; }

    const Engine &engine() const { return m_engine; }

    template <typename OStream>
    friend OStream &operator<<(OStream &os,
                               const TimeOrderedDataProc &todproc) {
        return os << fmt::format("TimeOrderedDataProc(engine={})",
                                 todproc.engine());
    }

private:
    Engine m_engine;
};

// create output directories
template <class EngineType>
void TimeOrderedDataProc<EngineType>::create_output_dir() {

    // redu subdir
    engine().redu_dir_name = "";

    // create reduction subdir
    if (engine().use_subdir) {
        int redu_dir_num = 0;

        std::stringstream ss_redu_dir_num;
        ss_redu_dir_num << std::setfill('0') << std::setw(2) << redu_dir_num;

        std::string redu_dir_name = "redu" + ss_redu_dir_num.str();

        while (fs::exists(fs::status(engine().output_dir + "/" + redu_dir_name))) {
            redu_dir_num++;
            std::stringstream ss_redu_dir_num_i;
            ss_redu_dir_num_i << std::setfill('0') << std::setw(2) << redu_dir_num;
            redu_dir_name = "redu" + ss_redu_dir_num_i.str();
        }

        engine().redu_dir_name = engine().output_dir + "/" + redu_dir_name;

        fs::create_directories(engine().redu_dir_name);
    }

    // coaddition directories
    if (engine().run_coadd) {
        engine().coadd_dir_name = engine().redu_dir_name + "/coadded";
        if (!fs::exists(fs::status(engine().coadd_dir_name + "/raw"))) {
            fs::create_directories(engine().coadd_dir_name + "/raw");
        }
        else {
            SPDLOG_WARN("directory {} already exists", engine().coadd_dir_name + "/raw");
        }

        // filtered directories
        if (engine().run_map_filter) {
            if (!fs::exists(fs::status(engine().coadd_dir_name + "/filtered"))) {
                fs::create_directories(engine().coadd_dir_name + "/filtered");
            }
            else {
                SPDLOG_WARN("directory {} already exists", engine().coadd_dir_name + "/filtered");
            }
        }
    }
}

// align tod with telescope
template <class EngineType>
void TimeOrderedDataProc<EngineType>::align_timestreams(const RawObs &rawobs) {
    using namespace netCDF;
    using namespace netCDF::exceptions;

    // clear start and end indices for each observation
    engine().start_indices.clear();
    engine().end_indices.clear();

    // vector of network times
    std::vector<Eigen::VectorXd> nw_ts;
    // start and end times
    std::vector<double> nw_t0, nw_tn;

    // maximum start time
    double max_t0 = -99;

    // minimum end time
    double min_tn = std::numeric_limits<double>::max();
    // indices of max start time and min end time
    Eigen::Index max_t0_i, min_tn_i;

    // set network
    Eigen::Index nw = 0;

    // loop through input files
    for (const RawObs::DataItem &data_item : rawobs.kidsdata()) {
        try {
            // load data file
            NcFile fo(data_item.filepath(), NcFile::read);
            auto vars = fo.getVars();

            // get roach index for offsets
            int roach_index;
            vars.find("Header.Toltec.RoachIndex")->second.getVar(&roach_index);

            // get dimensions for time matrix
            Eigen::Index n_pts = vars.find("Data.Toltec.Ts")->second.getDim(0).getSize();
            Eigen::Index n_time = vars.find("Data.Toltec.Ts")->second.getDim(1).getSize();

            // get time matrix
            Eigen::MatrixXi ts(n_time,n_pts);
            vars.find("Data.Toltec.Ts")->second.getVar(ts.data());

            // transpose due to row-major order
            ts.transposeInPlace();

            // get fpga frequency
            double fpga_freq;
            vars.find("Header.Toltec.FpgaFreq")->second.getVar(&fpga_freq);

            // seconds
            auto sec0 = ts.cast <double> ().col(0);
            // nanoseconds
            auto nsec0 = ts.cast <double> ().col(5);
            // pps count
            auto pps = ts.cast <double> ().col(1);
            // milliseconds
            auto msec = ts.cast <double> ().col(2)/fpga_freq;
            // count
            auto count = ts.cast <double> ().col(3);
            // pps milliseconds
            auto pps_msec = ts.cast <double> ().col(4)/fpga_freq;
            // get start time
            auto t0 = sec0 + nsec0*1e-9;

            // shift start time
            int start_t = int(t0[0] - 0.5);
            //int start_t = int(t0[0]);

            // convert start time to double
            double start_t_dbl = start_t;

            Eigen::VectorXd dt = msec - pps_msec;

            // remove overflow due to int32
            dt = (dt.array() < 0).select(msec.array() - pps_msec.array() + (pow(2.0,32)-1)/fpga_freq,msec - pps_msec);

            // get network time and add offsets
            nw_ts.push_back(start_t_dbl + pps.array() + dt.array() +
                            engine().interface_sync_offset["toltec"+std::to_string(roach_index)]);

            // push back start time
            nw_t0.push_back(nw_ts.back()[0]);

            // push back end time
            nw_tn.push_back(nw_ts.back()[n_pts - 1]);

            // get global max start time and index
            if (nw_t0.back() > max_t0) {
                max_t0 = nw_t0.back();
                max_t0_i = nw;
            }

            // get global min end time and index
            if (nw_tn.back() < min_tn) {
                min_tn = nw_tn.back();
                min_tn_i = nw;
            }

            nw++;

            fo.close();

        } catch (NcException &e) {
            SPDLOG_ERROR("{}", e.what());
            throw DataIOError{fmt::format(
                "failed to load data from netCDF file {}", data_item.filepath())};
        }
    }

    // size of smallest data time vector
    Eigen::Index min_size = nw_ts[0].size();

    // loop through time vectors and get the smallest
    for (Eigen::Index i=0; i<nw_ts.size(); i++) {

        // find start index that is larger than max start
        Eigen::Index si, ei;
        auto s = (abs(nw_ts[i].array() - max_t0)).minCoeff(&si);

        while (nw_ts[i][si] < max_t0) {
            si++;
        }

        engine().start_indices.push_back(si);

        // find end index that is smaller than min end
        auto e = (abs(nw_ts[i].array() - min_tn)).minCoeff(&ei);
        while (nw_ts[i][ei] > min_tn) {
            ei--;
        }
        engine().end_indices.push_back(ei);
    }

    for (Eigen::Index i=0; i<nw_ts.size(); i++) {
        auto si = engine().start_indices[i];
        auto ei = engine().end_indices[i];

        if ((ei - si + 1) < min_size) {
            min_size = ei - si + 1;
        }
    }

    // size of telescope data
    Eigen::Matrix<Eigen::Index,1,1> nd;
    nd << engine().telescope.tel_data["TelTime"].size();

    // shortest data time vector
    Eigen::VectorXd xi = nw_ts[max_t0_i].head(min_size);

    // interpolate telescope data
    for (const auto &tel_it : engine().telescope.tel_data) {
        if (tel_it.first !="TelTime") {
            // telescope vector to interpolate
            Eigen::VectorXd yd = engine().telescope.tel_data[tel_it.first];
            Eigen::VectorXd yi(min_size);

            mlinterp::interp(nd.data(), min_size, // nd, ni
                             yd.data(), yi.data(), // yd, yi
                             engine().telescope.tel_data["TelTime"].data(), xi.data()); // xd, xi

            engine().telescope.tel_data[tel_it.first] = std::move(yi);
        }
    }

    // replace telescope time vector
    engine().telescope.tel_data["TelTime"] = xi;
}

// get map number
template <class EngineType>
void TimeOrderedDataProc<EngineType>::calc_map_num() {
    // for science and pointing maps
    if ((engine().redu_type == "science") || (engine().redu_type == "pointing")) {
        engine().n_maps = engine().calib.n_arrays;
    }

    // for beammaps
    else if ((engine().redu_type == "beammap")) {
        engine().n_maps = engine().calib.n_dets;
    }

    // overwrite map number for networks
    if (engine().map_grouping == "nw") {
        engine().n_maps = engine().calib.n_nws;
    }

    // overwrite map number for arrays
    else if (engine().map_grouping == "array") {
        engine().n_maps = engine().calib.n_arrays;
    }

    // overwrite map number for detectors
    else if (engine().map_grouping == "detector") {
        engine().n_maps = engine().calib.n_dets;
    }

    if (engine().rtcproc.run_polarization) {
        // multiply by number of polarizations
        engine().n_maps = engine().n_maps*3;
    }

    engine().maps_to_arrays.resize(engine().n_maps);

    if (((engine().redu_type == "science") || (engine().redu_type == "pointing") || (engine().map_grouping == "array"))) {
        Eigen::Index i=0;
        for (const auto &stokes_param: engine().rtcproc.polarization.stokes_params) {
            engine().maps_to_arrays.segment(i,engine().calib.arrays.size()) = engine().calib.arrays;
            i = i + engine().calib.arrays.size();
        }
    }

    if (((engine().redu_type == "beammap") || (engine().map_grouping == "detector"))) {
        Eigen::Index i=0;
        for (const auto &stokes_param: engine().rtcproc.polarization.stokes_params) {
            Eigen::Index n_dets;
            Eigen::VectorXI array_indices;

            if (stokes_param == "I") {
                n_dets = engine().calib.n_dets;
                array_indices = engine().calib.apt["array"].template cast<Eigen::Index> ();
            }

            else if ((stokes_param == "Q") || (stokes_param == "U")) {
                n_dets = (engine().calib.apt["fg"].array() == 0).count() + (engine().calib.apt["fg"].array() == 1).count();
                Eigen::Index j=0;
                // loop through all detectors
                for (Eigen::Index i=0; i<engine().calib.n_dets-1; i=i+2) {
                    array_indices(j) = engine().calib.apt["array"](i);
                    j++;
                }
            }

            engine().maps_to_arrays.segment(i,n_dets) = array_indices;
            i = i + n_dets;
        }
    }

    if (engine().map_grouping == "nw") {
        Eigen::VectorXI array_indices(engine().calib.nws.size());

        for (Eigen::Index i=0; i<engine().calib.nws.size(); i++) {
            for (Eigen::Index j=0; j<engine().calib.n_dets; j++) {
                if (engine().calib.apt["nw"](j) == engine().calib.nws(i)) {
                    array_indices(i) = engine().calib.apt["array"](j);
                }
            }
        }
        Eigen::Index i=0;
        for (const auto &stokes_param: engine().rtcproc.polarization.stokes_params) {
            engine().maps_to_arrays.segment(i,engine().calib.nws.size()) = engine().calib.nws;
            i = i + engine().calib.nws.size();
        }
    }

    SPDLOG_INFO("n_maps {}", engine().n_maps);
    SPDLOG_INFO("engine().maps_to_arrays {}", engine().maps_to_arrays);
}

// determine the map dimensions and allocate the coadded map buffer
template <class EngineType>
void TimeOrderedDataProc<EngineType>::allocate_cmb(std::vector<map_extent_t> &map_extents, std::vector<map_coord_t> &map_coords) {
    // min/max rows and cols
    double min_row, max_row, min_col, max_col;

    min_row = map_coords.at(0).front()(0);
    max_row = map_coords.at(0).front()( map_coords.at(0).front().size() - 1);

    min_col = map_coords.at(0).back()(0);
    max_col = map_coords.at(0).back()( map_coords.at(0).back().size() - 1);

    // loop through physical coordinates and get min/max
    for (Eigen::Index i=0; i<map_coords.size(); i++) {
        auto rows_tan_vec = map_coords.at(i).front();
        auto cols_tan_vec = map_coords.at(i).back();

        auto n_pts_rows = rows_tan_vec.size();
        auto n_pts_cols = cols_tan_vec.size();

        // check global minimum row
        if (rows_tan_vec(0) < min_row) {
            min_row = rows_tan_vec(0);
        }

        // check global maximum row
        if (rows_tan_vec(n_pts_rows-1) > max_row) {
            max_row = rows_tan_vec(n_pts_rows-1);
        }

        // check global minimum col
        if (cols_tan_vec(0) < min_col) {
            min_col = cols_tan_vec(0);
        }

        // check global maximum row
        if (cols_tan_vec(n_pts_cols-1) > max_col) {
            max_col = cols_tan_vec(n_pts_cols-1);
        }
    }

    // calculate dimensions
    auto calc_map_dims = [&](auto min_dim, auto max_dim) {
        auto min_pix = ceil(abs(min_dim/engine().cmb.pixel_size_rad));
        auto max_pix = ceil(abs(max_dim/engine().cmb.pixel_size_rad));

        max_pix = std::max(min_pix, max_pix);
        auto ndim = 2*max_pix + 4;

        Eigen::VectorXd dim_vec = (Eigen::VectorXd::LinSpaced(ndim,0,ndim-1).array() -
                                   (ndim)/2.)*engine().cmb.pixel_size_rad;

        return std::tuple{ndim, std::move(dim_vec)};
    };

    // get number of rows and n_cols
    auto [n_rows, rows_tan_vec] = calc_map_dims(min_row, max_row);
    auto [n_cols, cols_tan_vec] = calc_map_dims(min_col, max_col);

    SPDLOG_INFO("map buffer n_rows {} n_cols {}", n_rows, n_cols);

    // clear map vectors
    engine().cmb.signal.clear();
    engine().cmb.weight.clear();
    engine().cmb.kernel.clear();
    engine().cmb.coverage.clear();

    // set number of rows and cols
    engine().cmb.n_rows = n_rows;
    engine().cmb.n_cols = n_cols;

    engine().cmb.wcs.naxis[1] = n_rows;
    engine().cmb.wcs.naxis[0] = n_cols;

    // pixel corresponding to reference value
    double ref_pix_cols = n_cols/2;
    double ref_pix_rows = n_rows/2;

    if ((int)ref_pix_cols == ref_pix_cols) {
        ref_pix_cols += 0.5;
    }

    if ((int)ref_pix_rows == ref_pix_rows) {
        ref_pix_rows += 0.5;
    }

    engine().cmb.wcs.crpix[0] = ref_pix_cols;
    engine().cmb.wcs.crpix[1] = ref_pix_rows;

    // loop through maps and allocate space
    for (Eigen::Index i=0; i<engine().n_maps; i++) {
        engine().cmb.signal.push_back(Eigen::MatrixXd::Zero(engine().cmb.n_rows, engine().cmb.n_cols));
        engine().cmb.weight.push_back(Eigen::MatrixXd::Zero(engine().cmb.n_rows, engine().cmb.n_cols));
        if (engine().rtcproc.run_kernel) {
            engine().cmb.kernel.push_back(Eigen::MatrixXd::Zero(engine().cmb.n_rows, engine().cmb.n_cols));
        }

        if (engine().redu_type!="beammap") {
            engine().cmb.coverage.push_back(Eigen::MatrixXd::Zero(engine().cmb.n_rows, engine().cmb.n_cols));
        }
    }

    // set row and col vectors
    engine().cmb.rows_tan_vec = rows_tan_vec;
    engine().cmb.cols_tan_vec = cols_tan_vec;
}

template <class EngineType>
template <class map_buffer_t>
void TimeOrderedDataProc<EngineType>::allocate_nmb(map_buffer_t &mb) {
    // resize noise maps (n_maps, [n_rows, n_cols, n_noise])
    for (Eigen::Index i=0; i<engine().n_maps; i++) {
        mb.noise.push_back(Eigen::Tensor<double,3>(mb.n_rows, mb.n_cols, mb.n_noise));
        mb.noise.at(i).setZero();
    }
}

// allocate observation map buffer
template <class EngineType>
void TimeOrderedDataProc<EngineType>::allocate_omb(map_extent_t &map_extent, map_coord_t &map_coord) {
    // clear map vectors for each obs
    engine().omb.signal.clear();
    engine().omb.weight.clear();
    engine().omb.kernel.clear();
    engine().omb.coverage.clear();

    // set omb dim variables
    engine().omb.n_rows = map_extent[0];
    engine().omb.n_cols = map_extent[1];

    // set omb wcs
    engine().omb.wcs.naxis[1] = map_extent[0];
    engine().omb.wcs.naxis[0] = map_extent[1];

    // pixel corresponding to reference value
    double ref_pix_cols = engine().omb.n_cols/2;
    double ref_pix_rows = engine().omb.n_rows/2;

    if ((int)ref_pix_cols == ref_pix_cols) {
        ref_pix_cols += 0.5;
    }

    if ((int)ref_pix_rows == ref_pix_rows) {
        ref_pix_rows += 0.5;
    }

    // set crpix
    engine().omb.wcs.crpix[0] = ref_pix_cols;
    engine().omb.wcs.crpix[1] = ref_pix_rows;

    // loop through maps and allocate
    for (Eigen::Index i=0; i<engine().n_maps; i++) {
        engine().omb.signal.push_back(Eigen::MatrixXd::Zero(engine().omb.n_rows, engine().omb.n_cols));
        engine().omb.weight.push_back(Eigen::MatrixXd::Zero(engine().omb.n_rows, engine().omb.n_cols));

        if (engine().rtcproc.run_kernel) {
            // allocate kernel
            engine().omb.kernel.push_back(Eigen::MatrixXd::Zero(engine().omb.n_rows, engine().omb.n_cols));
        }

        engine().omb.coverage.push_back(Eigen::MatrixXd::Zero(engine().omb.n_rows, engine().omb.n_cols));
    }

    engine().omb.rows_tan_vec = map_coord[0];
    engine().omb.cols_tan_vec = map_coord[1];
}

// calculate map dimensions
template <class EngineType>
void TimeOrderedDataProc<EngineType>::calc_map_size(std::vector<map_extent_t> &map_extents, std::vector<map_coord_t> &map_coords) {

    if ((engine().omb.wcs.naxis[0] <= 0) || (engine().omb.wcs.naxis[1] <= 0)) {

        Eigen::MatrixXd det_lat_limits, det_lon_limits, map_limits;
        det_lat_limits.setZero(engine().calib.n_dets,2);
        det_lon_limits.setZero(engine().calib.n_dets,2);
        map_limits.setZero(2,2);

        // loop through scans
        for (Eigen::Index i=0; i<engine().telescope.scan_indices.cols(); i++) {
            auto si = engine().telescope.scan_indices(2, i);
            auto sl = engine().telescope.scan_indices(3, i) - engine().telescope.scan_indices(2, i) + 1;

            std::map<std::string, Eigen::VectorXd> tel_data;
            for (auto const& x: engine().telescope.tel_data) {
                tel_data[x.first] = engine().telescope.tel_data[x.first].segment(si, sl);
            }

            // loop through detectors
            for (Eigen::Index j=0; j<engine().calib.n_dets; j++) {

                double az_off = 0;
                double el_off = 0;

                if (engine().redu_type!="beammap") {
                    az_off = engine().calib.apt["x_t"](j);
                    el_off = engine().calib.apt["y_t"](j);
                }

                auto [lat, lon] = engine_utils::calc_det_pointing(tel_data, az_off, el_off,
                                                                  engine().telescope.pixel_axes,
                                                                  engine().pointing_offsets_arcsec);
                // check for min and max
                if (lat.minCoeff() < det_lat_limits(j,0)) {
                    det_lat_limits(j,0) = lat.minCoeff();
                }
                if (lat.maxCoeff() > det_lat_limits(j,1)) {
                    det_lat_limits(j,1) = lat.maxCoeff();
                }
                if (lon.minCoeff() < det_lon_limits(j,0)) {
                    det_lon_limits(j,0) = lon.minCoeff();
                }
                if (lon.maxCoeff() > det_lon_limits(j,1)) {
                    det_lon_limits(j,1) = lon.maxCoeff();
                }
            }
        }

        // get the global min and max
        map_limits(0,0)  = det_lat_limits.col(0).minCoeff();
        map_limits(1,0)  = det_lat_limits.col(1).maxCoeff();
        map_limits(0,1)  = det_lon_limits.col(0).minCoeff();
        map_limits(1,1)  = det_lon_limits.col(1).maxCoeff();

        // calculate dimensions
        auto calc_map_dims = [&](auto min_dim, auto max_dim) {
            auto min_pix = ceil(abs(min_dim/engine().omb.pixel_size_rad));
            auto max_pix = ceil(abs(max_dim/engine().omb.pixel_size_rad));

            max_pix = std::max(min_pix, max_pix);
            auto ndim = 2*max_pix + 4;

            Eigen::VectorXd dim_vec = (Eigen::VectorXd::LinSpaced(ndim,0,ndim-1).array() -
                                       (ndim)/2.)*engine().omb.pixel_size_rad;

            return std::tuple{ndim, std::move(dim_vec)};
        };

        // get n_rows and n_cols
        auto [n_rows, rows_tan_vec] = calc_map_dims(map_limits(0,0), map_limits(1,0));
        auto [n_cols, cols_tan_vec] = calc_map_dims(map_limits(0,1), map_limits(1,1));

        SPDLOG_INFO("n_rows {}, n_cols {}", n_rows, n_cols);

        map_extent_t map_extent = {n_rows, n_cols};
        map_coord_t map_coord = {rows_tan_vec, cols_tan_vec};

        map_extents.push_back(map_extent);
        map_coords.push_back(map_coord);
    }

    else {
        auto n_rows = engine().omb.wcs.naxis[1];
        auto n_cols = engine().omb.wcs.naxis[0];

        Eigen::VectorXd rows_tan_vec = (Eigen::VectorXd::LinSpaced(n_rows,0,n_rows-1).array() -
                                        (n_rows)/2.)*engine().omb.pixel_size_rad;
        Eigen::VectorXd cols_tan_vec = (Eigen::VectorXd::LinSpaced(n_cols,0,n_cols-1).array() -
                                        (n_cols)/2.)*engine().omb.pixel_size_rad;
        map_extent_t map_extent = {n_rows, n_cols};
        map_coord_t map_coord = {rows_tan_vec, cols_tan_vec};

        map_extents.push_back(map_extent);
        map_coords.push_back(map_coord);
    }
}

// coadd maps
template <class EngineType>
void TimeOrderedDataProc<EngineType>::coadd() {
    // offset between cmb and omb physical coordinates
    int delta_row = (engine().omb.rows_tan_vec(0) - engine().cmb.rows_tan_vec(0))/engine().cmb.pixel_size_rad;
    int delta_col = (engine().omb.cols_tan_vec(0) - engine().cmb.cols_tan_vec(0))/engine().cmb.pixel_size_rad;

    // loop through the maps
    for (Eigen::Index i=0; i<engine().n_maps; i++) {
        // weight += weight
        engine().cmb.weight.at(i).block(delta_row, delta_col, engine().omb.n_rows, engine().omb.n_cols) =
            engine().cmb.weight.at(i).block(delta_row, delta_col, engine().omb.n_rows, engine().omb.n_cols).array() +
            engine().omb.weight.at(i).array();

        // signal += signal*weight
        engine().cmb.signal.at(i).block(delta_row, delta_col,engine().omb.n_rows, engine().omb.n_cols) =
            engine().cmb.signal.at(i).block(delta_row, delta_col, engine().omb.n_rows, engine().omb.n_cols).array() +
            (engine().omb.weight.at(i).array()*engine().omb.signal.at(i).array()).array();

        if (engine().rtcproc.run_kernel) {
            // kernel += kernel*weight
            engine().cmb.kernel.at(i).block(delta_row, delta_col, engine().omb.n_rows, engine().omb.n_cols) =
                engine().cmb.kernel.at(i).block(delta_row, delta_col, engine().omb.n_rows, engine().omb.n_cols).array() +
                (engine().omb.weight.at(i).array()*engine().omb.kernel.at(i).array()).array();
        }

        // coverage +=coverage
        engine().cmb.coverage.at(i).block(delta_row, delta_col, engine().omb.n_rows, engine().omb.n_cols) =
            engine().cmb.coverage.at(i).block(delta_row, delta_col, engine().omb.n_rows, engine().omb.n_cols).array() +
            engine().omb.coverage.at(i).array();
    }
}

template <class EngineType>
void TimeOrderedDataProc<EngineType>::make_index_file(std::string filepath) {
    std::set<fs::path> sorted_by_name;

    for (auto &entry : fs::directory_iterator(filepath))
        sorted_by_name.insert(entry);

    YAML::Node node;
    node["description"].push_back("citlali data products");
    node["date"].push_back(engine_utils::current_date_time());
    node["version"].push_back(CITLALI_GIT_VERSION);

    for (const auto & entry : sorted_by_name) {
        std::string path_string{entry.generic_string()};
        if (fs::is_directory(entry)) {
            make_index_file(path_string);
        }
        node["files"].push_back(path_string.substr(path_string.find_last_of("/") + 1));
    }
    std::ofstream fout(filepath + "/index.yaml");
    fout << node;
}

template <class EngineType>
void TimeOrderedDataProc<EngineType>::setup_filenames() {
    // if coaddition is requested
    if (engine().run_coadd) {
        for (Eigen::Index i=0; i<engine().calib.n_arrays; i++) {
            auto array = engine().calib.arrays[i];
            std::string array_name = engine().toltec_io.array_name_map[array];
            auto filename = engine().toltec_io.template create_filename<engine_utils::toltecIO::toltec,
                                                                        engine_utils::toltecIO::map,
                                                                        engine_utils::toltecIO::raw>(engine().coadd_dir_name + "/raw/",
                                                                                                     "", array_name, "",
                                                                                                     engine().telescope.sim_obs);
            fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*> fits_io(filename);
            engine().coadd_fits_io_vec.push_back(std::move(fits_io));

            // if noise maps requested
            if (engine().run_noise) {
                auto filename = engine().toltec_io.template create_filename<engine_utils::toltecIO::toltec,
                                                                            engine_utils::toltecIO::noise,
                                                                            engine_utils::toltecIO::raw>(engine().coadd_dir_name + "/raw/",
                                                                                                         "", array_name,
                                                                                                         "", engine().telescope.sim_obs);
                fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*> fits_io(filename);
                engine().coadd_noise_fits_io_vec.push_back(std::move(fits_io));
            }
        }

        // if map filtering are requested
        if (engine().run_map_filter) {
            for (Eigen::Index i=0; i<engine().calib.n_arrays; i++) {
                auto array = engine().calib.arrays[i];
                std::string array_name = engine().toltec_io.array_name_map[array];
                auto filename = engine().toltec_io.template create_filename<engine_utils::toltecIO::toltec,
                                                                            engine_utils::toltecIO::map,
                                                                            engine_utils::toltecIO::filtered>(engine().coadd_dir_name + "/filtered/",
                                                                                                         "", array_name,
                                                                                                         "", engine().telescope.sim_obs);
                fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*> fits_io(filename);
                engine().filtered_coadd_fits_io_vec.push_back(std::move(fits_io));

                // if noise maps requested
                if (engine().run_noise) {
                    auto filename = engine().toltec_io.template create_filename<engine_utils::toltecIO::toltec,
                                                                                engine_utils::toltecIO::noise,
                                                                                engine_utils::toltecIO::filtered>(engine().coadd_dir_name + "/filtered/",
                                                                                                               "", array_name,
                                                                                                               "", engine().telescope.sim_obs);
                    fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*> fits_io(filename);
                    engine().filtered_coadd_noise_fits_io_vec.push_back(std::move(fits_io));
                }
            }
        }
    }
}
