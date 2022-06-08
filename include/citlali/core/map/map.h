#pragma once

#include <string>
#include <vector>
#include <map>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <boost/random.hpp>
#include <boost/random/random_device.hpp>

#include <tula/grppi.h>

#include <citlali/core/utils/pointing.h>
#include <citlali/core/map/psd.h>
#include <citlali/core/map/histogram.h>
#include <citlali/core/map/coadd.h>
#include <citlali/core/map/noise.h>

namespace mapmaking {


struct WCS {
    // pixel size
    double pixel_size;

    // map size in pixels
    double x_size_pix, y_size_pix;

    // reference pixel
    double crpix1, crpix2;

    // map unit
    std::string cunit;
};

using map_dims_t = std::tuple<int, int, Eigen::VectorXd, Eigen::VectorXd>;
using map_extent_t = std::vector<double>;
using map_coord_t = std::vector<Eigen::VectorXd>;
using map_count_t = std::size_t;

class MapBuffer {
public:
    Eigen::Index nrows, ncols;
    map_count_t map_count;

    // for map fit parameters
    Eigen::MatrixXd pfit;
    // for map fit errors
    Eigen::MatrixXd perror;

    double pixel_size;

    // physical coordinates for rows and cols (radians)
    Eigen::VectorXd rcphys, ccphys;

    // elevations when detector is on source
    Eigen::VectorXd min_el;

    // src-det el when detector is on source
    Eigen::VectorXd el_dist;

    // vector of psd classes
    std::vector<PSD> psd;

    // vector of histogram psds
    std::vector<Histogram> histogram;

    // map types
    std::vector<Eigen::MatrixXd> signal, weight, kernel, coverage;

    void setup_maps(map_extent_t map_extent, map_coord_t map_coord, map_count_t _mc,
                    const bool run_kernel, std::string map_grouping) {
        map_count = _mc;

        nrows = map_extent.at(0);
        ncols = map_extent.at(1);
        rcphys = map_coord.at(0);
        ccphys = map_coord.at(1);

        // empty map vectors for each observation
        signal.clear();
        weight.clear();
        kernel.clear();
        coverage.clear();
        psd.clear();

        // resize the maps (nobs, [nrows, ncols])
        for (Eigen::Index i=0; i<map_count; i++) {
            signal.push_back(Eigen::MatrixXd::Zero(nrows, ncols));
            weight.push_back(Eigen::MatrixXd::Zero(nrows, ncols));
            // only make kernel map if requested to save space
            if (run_kernel) {
                kernel.push_back(Eigen::MatrixXd::Zero(nrows, ncols));
            }
            // only make coverage maps if not in beammap mode
            if (map_grouping != "beammap") {
                coverage.push_back(Eigen::MatrixXd::Zero(nrows, ncols));
            }
        }
    }

    void normalize_maps(const bool run_kernel, std::string ex_name) {
        // normalize maps by weight map
        /*for (Eigen::Index mc = 0; mc < map_count; mc++) {
                signal.at(mc) = (weight.at(mc).array() == 0).select(0, signal.at(mc).array() / weight.at(mc).array());
            if (run_kernel) {
                kernel.at(mc) = (weight.at(mc).array() == 0).select(0, kernel.at(mc).array() / weight.at(mc).array());
            }
        }*/

        std::vector<int> det_in_vec, det_out_vec;

        det_in_vec.resize(map_count);
        std::iota(det_in_vec.begin(), det_in_vec.end(), 0);
        det_out_vec.resize(map_count);

        //for (Eigen::Index mc=0; mc<map_count; mc++) {
        grppi::map(tula::grppi_utils::dyn_ex(ex_name), det_in_vec, det_out_vec, [&](int mc) {
            for (Eigen::Index i=0; i<nrows; i++) {
                for (Eigen::Index j=0; j<ncols; j++) {
                    auto pixel_weight = weight.at(mc)(i,j);
                    if (pixel_weight != pixel_weight) {
                            SPDLOG_INFO("bad pixel weight {}", pixel_weight);
                    }
                    if (pixel_weight != 0. && pixel_weight == pixel_weight) {
                        signal.at(mc)(i,j) = (signal.at(mc)(i,j)) / pixel_weight;
                        if (run_kernel) {
                            kernel.at(mc)(i,j) = (kernel.at(mc)(i,j)) / pixel_weight;
                        }
                    }
                    else {
                        signal.at(mc)(i,j) = 0;
                        if (run_kernel) {
                            kernel.at(mc)(i,j) = 0;
                        }
                    }
                }
            }
            return 0;
        });
    }
};

class MapBase: public WCS {
public:
    MapBuffer mb;
    CoaddedMapBuffer cmb;
    NoiseMapBuffer nmb;

    enum MapType {
        obs = 0,
        coadd = 1,
    };

    // icrs or altaz
    std::string map_type;

    // grouping of detectors
    std::string map_grouping;

    // mapping method (naive, etc)
    std::string mapping_method;

    Eigen::VectorXd cflux;

    template <typename tel_meta_t, typename C, typename S>
    map_dims_t get_dims(tel_meta_t &, C &, S &, std::string, std::string,
                        const double, const double, std::map<std::string,double> &);
};

template <typename tel_meta_t, typename C, typename S>
map_dims_t MapBase::get_dims(tel_meta_t &tel_meta_data, C &calib_data, S &scan_indices,
                             std::string ex_name, std::string reduction_type,
                             const double _xs, const double _ys,
                             std::map<std::string,double> &pointing_offsets) {

    if ((x_size_pix==0) && (y_size_pix==0)) {
        SPDLOG_INFO("calculating map dimensions based on pointing max/min");
        // matrices to hold min and max lat/lon values for each detector
        Eigen::MatrixXd lat_limits, lon_limits;

        lat_limits.setZero(calib_data["y_t"].size(), 2);
        lon_limits.setZero(calib_data["x_t"].size(), 2);

        // global max and min lat/lon
        Eigen::MatrixXd map_dims = Eigen::MatrixXd::Zero(2,2);

        // placeholder vectors for grppi maps
        std::vector<int> scan_in_vec, scan_out_vec;
        std::vector<int> det_in_vec, det_out_vec;

        det_in_vec.resize(calib_data["x_t"].size());
        std::iota(det_in_vec.begin(), det_in_vec.end(), 0);
        det_out_vec.resize(calib_data["x_t"].size());


        // loop through scans
        for (Eigen::Index s = 0; s < scan_indices.cols(); s++) {
            auto si = scan_indices(2, s);
            auto scan_length = scan_indices(3, s) - scan_indices(2, s) + 1;

            // copy tel meta data for scan
            tel_meta_t tel_meta_data_scan;
            tel_meta_data_scan["TelElDes"] = tel_meta_data["TelElDes"].segment(si, scan_length);
            tel_meta_data_scan["ParAng"] = tel_meta_data["ParAng"].segment(si, scan_length);

            tel_meta_data_scan["TelLatPhys"] = tel_meta_data["TelLatPhys"].segment(si, scan_length);
            tel_meta_data_scan["TelLonPhys"] = tel_meta_data["TelLonPhys"].segment(si, scan_length);

            // loop through detectors
            grppi::map(tula::grppi_utils::dyn_ex(ex_name), det_in_vec, det_out_vec, [&](auto di) {

                double azoff, eloff;

                // get detector offsets from apt table
                if (reduction_type == "science" || reduction_type == "pointing") {
                    azoff = calib_data["x_t"](di);
                    eloff = calib_data["y_t"](di);
                }

                // if beammap, detector offsets are zero
                else if (reduction_type == "beammap") {
                    azoff = 0;
                    eloff = 0;
                }

                // get detector pointing
                auto [lat, lon] = engine_utils::get_det_pointing(tel_meta_data_scan,
                        azoff, eloff, map_type, pointing_offsets);

                // check for min and max
                if (lat.minCoeff() < lat_limits(di,0)) {
                    lat_limits(di,0) = lat.minCoeff();
                }
                else if (lat.maxCoeff() > lat_limits(di,1)) {
                    lat_limits(di,1) = lat.maxCoeff();
                }
                if (lon.minCoeff() < lon_limits(di,0)) {
                    lon_limits(di,0) = lon.minCoeff();
                }
                else if (lon.maxCoeff() > lon_limits(di,1)) {
                    lon_limits(di,1) = lon.maxCoeff();
                }
                return 0;
            });
        }

        // get the global min and max
        map_dims(0,0)  = lat_limits.col(0).minCoeff();
        map_dims(1,0)  = lat_limits.col(1).maxCoeff();
        map_dims(0,1)  = lon_limits.col(0).minCoeff();
        map_dims(1,1)  = lon_limits.col(1).maxCoeff();

        // calculate dimension and corresponding vector
        auto get_dim_size = [&](auto min_dim, auto max_dim) {
            auto min_pix = ceil(abs(min_dim/pixel_size));
            auto max_pix = ceil(abs(max_dim/pixel_size));

            max_pix = std::max(min_pix, max_pix);
            auto ndim = 2*max_pix + 4;

            Eigen::VectorXd dim_vec = (Eigen::VectorXd::LinSpaced(ndim,0,ndim-1).array() -
                    (ndim)/2.)*pixel_size;

            return std::tuple{ndim, std::move(dim_vec)};
        };

        // get nrows and ncols
        auto [nr, rcp] = get_dim_size(map_dims(0,0), map_dims(1,0));
        auto [nc, ccp] = get_dim_size(map_dims(0,1), map_dims(1,1));

        SPDLOG_INFO("nrows {} ncols {}", nr, nc);

        return map_dims_t {nr, nc, std::move(rcp), std::move(ccp)};
    }

    else {
        // set rows and cols to requested sizes
        Eigen::Index nr = y_size_pix;
        Eigen::Index nc = x_size_pix;

        Eigen::VectorXd rcp = (Eigen::VectorXd::LinSpaced(nr,0,nr-1).array() -
                (nr)/2.)*pixel_size;

        Eigen::VectorXd ccp = (Eigen::VectorXd::LinSpaced(nc,0,nc-1).array() -
                (nc)/2.)*pixel_size;

        SPDLOG_INFO("nrows {} ncols {}", nr, nc);

        return map_dims_t {nr, nc, std::move(rcp), std::move(ccp)};
    }
}

} // namespace mapmaking
