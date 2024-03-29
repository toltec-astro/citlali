#pragma once

#include <tula/logging.h>
#include <tula/nc.h>
#include <tula/algorithm/ei_stats.h>

#include <citlali/core/utils/utils.h>
#include <citlali/core/utils/pointing.h>

#include <citlali/core/timestream/timestream.h>
#include <citlali/core/timestream/ptc/clean.h>

#include <citlali/core/utils/toltec_io.h>

namespace timestream {

using timestream::TCData;

class PTCProc: public TCProc {
public:
    // controls for timestream reduction
    bool run_clean;
    // median weight factor
    double med_weight_factor;
    // weight type (full, approximate, const)
    std::string weighting_type;

    // ptc tod proc
    timestream::Cleaner cleaner;

    // get config file
    template <typename config_t>
    void get_config(config_t &, std::vector<std::vector<std::string>> &,
                    std::vector<std::vector<std::string>> &);

    // subtract detector means
    void subtract_mean(TCData<TCDataKind::PTC, Eigen::MatrixXd> &);

    // run main processing stage
    template <class calib_type, typename Derived>
    void run(TCData<TCDataKind::PTC, Eigen::MatrixXd> &,
             TCData<TCDataKind::PTC, Eigen::MatrixXd> &,
             calib_type &, Eigen::DenseBase<Derived> &,
             std::string, std::string);

    // calculate detector weights
    template <typename apt_type, class tel_type, typename Derived>
    void calc_weights(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, apt_type &, tel_type &,
                      Eigen::DenseBase<Derived> &);

    // reset outlier weights to the median
    template <typename calib_t, typename Derived>
    auto reset_weights(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, calib_t &,
                       Eigen::DenseBase<Derived> &det_indices);

    // append time chunk to tod netcdf file
    template <typename Derived, typename calib_t, typename pointing_offset_t>
    void append_to_netcdf(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, std::string, std::string, std::string &,
                          pointing_offset_t &, Eigen::DenseBase<Derived> &, calib_t &);
};

// get config file
template <typename config_t>
void PTCProc::get_config(config_t &config, std::vector<std::vector<std::string>> &missing_keys,
                         std::vector<std::vector<std::string>> &invalid_keys) {

    // weight type
    get_config_value(config, weighting_type, missing_keys, invalid_keys,
                     std::tuple{"timestream","processed_time_chunk","weighting","type"},{"full","approximate","const"});
    // median weight factor
    get_config_value(config, med_weight_factor, missing_keys, invalid_keys,
                     std::tuple{"timestream","processed_time_chunk","weighting","median_weight_factor"});
    // lower weight factor
    get_config_value(config, lower_weight_factor, missing_keys, invalid_keys,
                     std::tuple{"timestream","processed_time_chunk","flagging","lower_weight_factor"});
    // upper weight factor
    get_config_value(config, upper_weight_factor, missing_keys, invalid_keys,
                     std::tuple{"timestream","processed_time_chunk","flagging","upper_weight_factor"});

    // run fruit loops?
    get_config_value(config, run_fruit_loops, missing_keys, invalid_keys,
                     std::tuple{"timestream","fruit_loops","enabled"});

    if (run_fruit_loops) {
        // save all fruit loops iterations?
        get_config_value(config, save_all_iters, missing_keys, invalid_keys,
                         std::tuple{"timestream","fruit_loops","save_all_iters"});
        // fruit looops path
        get_config_value(config, fruit_loops_path, missing_keys, invalid_keys,
                         std::tuple{"timestream","fruit_loops","path"});
        // fruit looops type
        get_config_value(config, fruit_loops_type, missing_keys, invalid_keys,
                         std::tuple{"timestream","fruit_loops","type"});
        // let user specify "coadd" or "coadded"
        if (fruit_loops_type == "coadded") {
            fruit_loops_type = "coadd";
        }
        // fruit loops signal-to-noise
        get_config_value(config, fruit_loops_sig2noise, missing_keys, invalid_keys,
                         std::tuple{"timestream","fruit_loops","lower_sig2noise_limit"});
        // fruit loops flux density limit
        auto fruit_loops_flux_vec = config.template get_typed<std::vector<double>>(std::tuple{"timestream","fruit_loops","array_lower_flux_limit"});
        fruit_loops_flux = Eigen::Map<Eigen::VectorXd>(fruit_loops_flux_vec.data(), fruit_loops_flux_vec.size());

        // maximum fruit loops iterations
        get_config_value(config, fruit_loops_iters, missing_keys, invalid_keys,
                         std::tuple{"timestream","fruit_loops","max_iters"});
    }

    // run clean?
    get_config_value(config, run_clean, missing_keys, invalid_keys,
                     std::tuple{"timestream","processed_time_chunk","clean", "enabled"});

    if (run_clean) {
        // get cleaning grouping vector
        cleaner.grouping = config.template get_typed<std::vector<std::string>>(std::tuple{"timestream","processed_time_chunk","clean","grouping"});
        // get cleaning number of eigenvalues vector
        for (auto const& [arr_index, arr_name] : toltec_io.array_name_map) {
            auto n_eig_to_cut = config.template get_typed<std::vector<Eigen::Index>>(std::tuple{"timestream","processed_time_chunk","clean",
                                                                                                "n_eig_to_cut",arr_name});
            // add eigenvalues to cleaner class
            cleaner.n_eig_to_cut[arr_index] = (Eigen::Map<Eigen::VectorXI>(n_eig_to_cut.data(),n_eig_to_cut.size()));
        }

        // stddev limit
        get_config_value(config, cleaner.stddev_limit, missing_keys, invalid_keys,
                         std::tuple{"timestream","processed_time_chunk","clean","stddev_limit"});
        // mask radius in arcseconds
        get_config_value(config, mask_radius_arcsec, missing_keys, invalid_keys,
                         std::tuple{"timestream","processed_time_chunk","clean","mask_radius_arcsec"});
    }
}

void PTCProc::subtract_mean(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in) {
    // cast flags to double and flip 1's and 0's so we can multiply by the data
    auto f = (in.flags.data.derived().array().cast <double> ().array() - 1).abs();
    // mean of each detector
    Eigen::RowVectorXd col_mean = (in.scans.data.derived().array()*f).colwise().sum()/
                                   f.colwise().sum();

    // remove nans from completely flagged detectors
    Eigen::RowVectorXd dm = (col_mean).array().isNaN().select(0,col_mean);

    // subtract mean from data and copy into det matrix
    in.scans.data.noalias() = in.scans.data.derived().rowwise() - dm;

    // subtract kernel mean
    if (in.kernel.data.size()!=0) {
        Eigen::RowVectorXd col_mean = (in.kernel.data.derived().array()*f).colwise().sum()/
                                      f.colwise().sum();

        // remove nans from completely flagged detectors
        Eigen::RowVectorXd dm = (col_mean).array().isNaN().select(0,col_mean);

        // subtract mean from data and copy into det matrix
        in.kernel.data.noalias() = in.kernel.data.derived().rowwise() - dm;
    }
}

template <class calib_type, typename Derived>
void PTCProc::run(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in,
                  TCData<TCDataKind::PTC, Eigen::MatrixXd> &out, calib_type &calib,
                  Eigen::DenseBase<Derived> &det_indices, std::string pixel_axes,
                  std::string map_grouping) {

    // subtract mean from data and kernel
    subtract_mean(in);

    if (run_clean) {
        // number of samples
        Eigen::Index n_pts = in.scans.data.rows();
        // index for number of cleaning groups in vectors
        Eigen::Index indx = 0;

        // loop through config groupings
        for (const auto & group: cleaner.grouping) {

            logger->debug("cleaning with {} grouping", group);

            if (run_tod_output || write_evals) {
                // add current group to eval/evec vectors
                out.evals.data.emplace_back();
                out.evecs.data.emplace_back();
            }

            // map of tuples to hold detector limits
            std::map<Eigen::Index, std::tuple<Eigen::Index, Eigen::Index>> grp_limits;

            // use all detectors for cleaning
            if (group == "all") {
                grp_limits[0] = std::make_tuple(0,in.scans.data.cols());
            }
            else {
                // get group limits
                grp_limits = get_grouping(group, det_indices, calib, in.scans.data.cols());
            }
            // loop through cleaning groups
            for (auto const& [key, val] : grp_limits) {
                Eigen::Index arr_index;
                // use all detectors
                if (group=="all") {
                    arr_index = calib.arrays(0);
                }
                // use network grouping
                else if (group=="nw" || group=="network") {
                    arr_index = toltec_io.nw_to_array_map[key];
                }
                // use array grouping
                else if (group=="array") {
                    arr_index = key;
                }

                // start index and number of detectors
                auto [start_index, n_dets] = std::make_tuple(std::get<0>(val), std::get<1>(val) - std::get<0>(val));

                // matrix for flags so we don't overwrite the raw flags
                Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> masked_flags;

                // mask region if radius is >0
                if (mask_radius_arcsec > 0) {
                    // samples that were masked will be flagged
                    masked_flags = mask_region(in, calib, det_indices, pixel_axes, map_grouping, n_pts, n_dets, start_index);
                }
                // otherwise just use input flags
                else {
                    masked_flags = in.flags.data.block(0, start_index, n_pts, n_dets);
                }

                auto in_scans_block = in.scans.data.block(0, start_index, n_pts, n_dets);
                auto out_scans_block = out.scans.data.block(0, start_index, n_pts, n_dets);

                // get the block of out scans that corresponds to the current array
                auto apt_flags = calib.apt["flag"].segment(start_index, n_dets);

                // check if any good flags
                if ((apt_flags.array()==0).any()) {
                    // calculate eigenvalues and eigenvalues
                    auto [evals, evecs] = cleaner.calc_eig_values<timestream::Cleaner::SpectraBackend>(in_scans_block, masked_flags, apt_flags,
                                                                                                       cleaner.n_eig_to_cut[arr_index](indx));
                    logger->debug("evals {}", evals);
                    logger->debug("evecs {}", evecs);

                    if (run_tod_output || write_evals) {
                        // get first 64 eigenvalues and eigenvectors
                        Eigen::VectorXd ev = evals.head(cleaner.n_calc);
                        Eigen::MatrixXd evc = evecs.leftCols(cleaner.n_calc);

                        // copy evals and evecs to ptcdata
                        out.evals.data[indx].push_back(std::move(ev));
                        out.evecs.data[indx].push_back(std::move(evc));
                    }

                    // remove eigenvalues from the data and reconstruct the tod
                    cleaner.remove_eig_values<timestream::Cleaner::SpectraBackend>(in_scans_block, masked_flags, evals, evecs, out_scans_block,
                                                                                   cleaner.n_eig_to_cut[arr_index](indx));

                    if (in.kernel.data.size()!=0) {
                        // check if any good flags
                            logger->debug("cleaning kernel");
                            auto in_kernel_block = in.kernel.data.block(0, start_index, n_pts, n_dets);
                            auto out_kernel_block = in.kernel.data.block(0, start_index, n_pts, n_dets);

                            // remove eigenvalues from the kernel and reconstruct the tod
                            cleaner.remove_eig_values<timestream::Cleaner::SpectraBackend>(in_kernel_block, masked_flags, evals, evecs, out_kernel_block,
                                                                                           cleaner.n_eig_to_cut[arr_index](indx));
                    }
                }
                // otherwise just copy the data
                else {
                    logger->debug("no good detectors found. skipping clean.");
                    // copy scans
                    out.scans.data.block(0, start_index, n_pts, n_dets) = in.scans.data.block(0, start_index, n_pts, n_dets);
                    // copy kernel
                    if (in.kernel.data.size()!=0) {
                        out.kernel.data.block(0, start_index, n_pts, n_dets) = in.kernel.data.block(0, start_index, n_pts, n_dets);
                    }
                }
            }
            indx++;
            // set as cleaned
            out.status.cleaned = true;
        }
    }
}

template <typename apt_type, class tel_type, typename Derived>
void PTCProc::calc_weights(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, apt_type &apt, tel_type &telescope,
                           Eigen::DenseBase<Derived> &det_indices) {
    // number of detectors
    Eigen::Index n_dets = in.scans.data.cols();

    // resize weights to number of detectors
    in.weights.data = Eigen::VectorXd::Zero(n_dets);

    // approximate weighting
    if (weighting_type == "approximate") {
        logger->debug("calculating weights using detector sensitivities");
        // unit conversion x flux calibration factor x 1/exp(-tau)
        double conversion_factor;

        // loop through detectors and calculate weights
        for (Eigen::Index i=0; i<n_dets; ++i) {
            // current detector index
            Eigen::Index det_index = det_indices(i);
            // if flux calibrated, get flux conversion factor
            if (in.status.calibrated) {
                conversion_factor = in.fcf.data(i);
            }
            // otherwise fcf is unity
            else {
                conversion_factor = 1;
            }
            // make sure flux conversion is not zero (otherwise weight=0)
            if (conversion_factor*apt["sens"](det_index)!=0) {
                // calculate weights while applying flux calibration
                in.weights.data(i) = pow(sqrt(telescope.d_fsmp)*apt["sens"](det_index)*conversion_factor,-2.0);
            }
            else {
                in.weights.data(i) = 0;
            }
        }
    }
    // use full weighting
    else if (weighting_type == "full"){
        logger->debug("calculating weights using timestream variance");

        // loop through detectors
        for (Eigen::Index i=0; i<n_dets; ++i) {
            // only calculate weights if detector is unflagged
            if (apt["flag"](det_indices(i))==0) {
                // make Eigen::Maps for each detector's scan
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> scans(
                    in.scans.data.col(i).data(), in.scans.data.rows());
                Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> flags(
                    in.flags.data.col(i).data(), in.flags.data.rows());

                // unflagged detector stddev
                double det_std_dev = engine_utils::calc_std_dev(scans, flags);
                // if stddev is not zero
                if (det_std_dev !=0) {
                    // weight = 1/(stddev)^2
                    in.weights.data(i) = pow(det_std_dev,-2);
                }
                // otherwise weight = 0 (not included in maps)
                else {
                    in.weights.data(i) = 0;
                }
            }
            // otherwise weight = 0 (not included in maps)
            else {
                in.weights.data(i) = 0;
            }
        }
    }
    // constant weighting
    else if (weighting_type == "const") {
        for (Eigen::Index i=0; i<n_dets; ++i) {
            // only calculate weights if detector is unflagged
            if (apt["flag"](det_indices(i))==0) {
                in.weights.data(i) = 1;
            }
            // otherwise set to zero
            else {
                in.weights.data(i) = 0;
            }
        }
    }
}

template <typename calib_t, typename Derived>
auto PTCProc::reset_weights(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, calib_t &calib,
                            Eigen::DenseBase<Derived> &det_indices) {

    // only need to run if median weight factor >=1
    if (med_weight_factor >= 1) {
        // number of detectors
        Eigen::Index n_dets = in.scans.data.cols();

        // get group limits
        auto grp_limits = get_grouping("array", det_indices, calib, n_dets);

        // collect detectors that are un-flagged and have non-zero weights
        for (auto const& [key, val] : grp_limits) {
            // weights for current group
            auto grp_weights = in.weights.data(Eigen::seq(std::get<0>(grp_limits[key]),
                                                         std::get<1>(grp_limits[key])-1));
            // number of good detectors
            Eigen::Index n_good_dets = 0;
            // start index of current group
            Eigen::Index j = std::get<0>(grp_limits[key]);

            // loop through detectors in current group
            for (Eigen::Index m=0; m<grp_weights.size(); m++) {
                // if detector is good and weight is not zero
                if (calib.apt["flag"](det_indices(j))==0 && grp_weights(m)>0) {
                    n_good_dets++;
                }
                j++;
            }

            // to hold good detectors
            Eigen::VectorXd good_wt;

            // if good detectors were found
            if (n_good_dets>0) {
                good_wt.resize(n_good_dets);

                // remove flagged dets
                j = std::get<0>(grp_limits[key]);
                Eigen::Index k = 0;
                for (Eigen::Index m=0; m<grp_weights.size(); m++) {
                    if (calib.apt["flag"](det_indices(j))==0 && grp_weights(m)>0) {
                        good_wt(k) = grp_weights(m);
                        k++;
                    }
                    j++;
                }
            }
            // otherwise just use all detectors
            else {
                good_wt = grp_weights;
            }

            // get median weight
            auto med_wt = tula::alg::median(good_wt);
            // store median weights
            in.median_weights.data.push_back(med_wt);

            int outliers = 0;

            // start index of current group
            j = std::get<0>(grp_limits[key]);
            // loop through detectors in current group
            for (Eigen::Index m=0; m<grp_weights.size(); m++) {
                // if detector weight is med_weight_factor times larger than med_wt
                if (in.weights.data(j) > med_weight_factor*med_wt) {
                    // reset high weights to median
                    in.weights.data(j) = med_wt;
                    outliers++;
                }
                j++;
            }
            logger->info("array {} had {} outlier weights", key, outliers);
        }
    }
}

template <typename Derived, typename calib_t, typename pointing_offset_t>
void PTCProc::append_to_netcdf(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, std::string filepath, std::string map_grouping,
                              std::string &pixel_axes, pointing_offset_t &pointing_offsets_arcsec, Eigen::DenseBase<Derived> &det_indices,
                              calib_t &calib) {

    using netCDF::NcDim;
    using netCDF::NcFile;
    using netCDF::NcType;
    using netCDF::NcVar;
    using namespace netCDF::exceptions;

    try {
        // open netcdf file
        NcFile fo(filepath, netCDF::NcFile::write);

        // append common time chunk variables
        append_base_to_netcdf(fo, in, map_grouping, pixel_axes, pointing_offsets_arcsec, det_indices, calib);

        // get dimensions
        NcDim n_dets_dim = fo.getDim("n_dets");

        // number of detectors currently in file
        unsigned long n_dets_exists = n_dets_dim.getSize();

        // append weights
        std::vector<std::size_t> start_index_weights = {static_cast<unsigned long>(in.index.data), 0};
        std::vector<std::size_t> size_weights = {1, n_dets_exists};

        // get weight variable
        NcVar weights_v = fo.getVar("weights");

        // add weights to tod output
        weights_v.putVar(start_index_weights, size_weights, in.weights.data.data());

        // get number of eigenvalues to save
        NcDim n_eigs_dim = fo.getDim("n_eigs");
        netCDF::NcDim n_eig_grp_dim = fo.getDim("n_eig_grp");

        // if eigenvalue dimension is null, add it
        if (n_eig_grp_dim.isNull()) {
            n_eig_grp_dim = fo.addDim("n_eig_grp",in.evals.data[0].size());
        }

        // dimensions for eigenvalue data
        std::vector<netCDF::NcDim> eval_dims = {n_eig_grp_dim, n_eigs_dim};

        // loop through cleaner gropuing
        for (Eigen::Index i=0; i<in.evals.data.size(); ++i) {
            NcVar eval_v = fo.addVar("evals_" + cleaner.grouping[i] + "_" + std::to_string(i) +
                                         "_chunk_" + std::to_string(in.index.data), netCDF::ncDouble,eval_dims);
            std::vector<std::size_t> start_eig_index = {0, 0};
            std::vector<std::size_t> size = {1, TULA_SIZET(cleaner.n_calc)};

            // loop through eigenvalues in current group
            for (const auto &evals: in.evals.data[i]) {
                eval_v.putVar(start_eig_index,size,evals.data());
                start_eig_index[0] += 1;
            }
        }

        // number of dimensions for eigenvectors
        std::vector<netCDF::NcDim> eig_dims = {n_dets_dim, n_eigs_dim};

        // loop through cleaner gropuing
        for (Eigen::Index i=0; i<in.evecs.data.size(); ++i) {
            // start at first row and col
            std::vector<std::size_t> start_eig_index = {0, 0};

            NcVar evec_v = fo.addVar("evecs_" + cleaner.grouping[i] + "_" + std::to_string(i) + "_chunk_" +
                                         std::to_string(in.index.data),netCDF::ncDouble,eig_dims);

            // loop through eigenvectors in current group
            for (const auto &evecs: in.evecs.data[i]) {
                std::vector<std::size_t> size = {TULA_SIZET(evecs.rows()), TULA_SIZET(cleaner.n_calc)};

                // transpose eigenvectors
                Eigen::MatrixXd ev = evecs.transpose();
                evec_v.putVar(start_eig_index, size, ev.data());

                // increment start
                start_eig_index[0] += TULA_SIZET(evecs.rows());
            }
        }

        // sync file to make sure it gets updated
        fo.sync();
        // close file
        fo.close();
        logger->info("tod chunk written to {}", filepath);

    } catch (NcException &e) {
        logger->error("{}", e.what());
    }
}

} // namespace timestream
