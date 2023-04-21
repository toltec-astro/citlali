#pragma once

#include <tula/logging.h>
#include <tula/nc.h>
#include <tula/algorithm/ei_stats.h>

#include <citlali/core/utils/utils.h>
#include <citlali/core/utils/pointing.h>

#include <citlali/core/timestream/timestream.h>
#include <citlali/core/timestream/ptc/clean.h>

namespace timestream {

using timestream::TCData;

class PTCProc {
public:
    bool run_clean, run_calibrate, run_stokes_clean, reset_weighting;
    std::string weighting_type;

    // ptc tod proc
    timestream::Cleaner cleaner;

    double lower_std_dev, upper_std_dev;

    void subtract_mean(TCData<TCDataKind::PTC, Eigen::MatrixXd> &);

    template <class calib_type, typename Derived>
    void run(TCData<TCDataKind::PTC, Eigen::MatrixXd> &,
             TCData<TCDataKind::PTC, Eigen::MatrixXd> &, calib_type &,
             Eigen::DenseBase<Derived> &, std::string);

    template <typename apt_type, class tel_type, typename Derived>
    void calc_weights(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, apt_type &, tel_type &,
                      Eigen::DenseBase<Derived> &);

    template <typename calib_t>
    auto reset_weights(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, calib_t &);

    template <typename apt_t, typename Derived>
    void remove_flagged_dets(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, apt_t &, Eigen::DenseBase<Derived> &);

    template <typename calib_t, typename Derived>
    auto remove_bad_dets(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, calib_t &, Eigen::DenseBase<Derived> &,
                            Eigen::DenseBase<Derived> &, Eigen::DenseBase<Derived> &, std::string, std::string);

    template <typename calib_t, typename Derived>
    auto remove_bad_dets_iter(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, calib_t &, Eigen::DenseBase<Derived> &,
                            Eigen::DenseBase<Derived> &, Eigen::DenseBase<Derived> &, std::string, std::string);

    template <typename Derived, typename apt_t, typename pointing_offset_t>
    void append_to_netcdf(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, std::string, std::string, std::string &,
                          pointing_offset_t &, Eigen::DenseBase<Derived> &, apt_t &, std::string, bool, double);

    template <typename DerivedB, typename DerivedC, typename apt_t, typename pointing_offset_t>
    void add_gaussian(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, Eigen::DenseBase<DerivedB> &, std::string &,
                      std::string &, apt_t &, pointing_offset_t &, double,
                      Eigen::Index, Eigen::Index, Eigen::DenseBase<DerivedC> &, Eigen::DenseBase<DerivedC> &,
                      std::string);
};

void PTCProc::subtract_mean(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in) {
    // mean of each detector
    Eigen::RowVectorXd col_mean = (in.scans.data.derived().array()*in.flags.data.derived().array().cast <double> ()).colwise().sum()/
                                   in.flags.data.derived().array().cast <double> ().colwise().sum();

    // remove nans from completely flagged detectors
    Eigen::RowVectorXd dm = (col_mean).array().isNaN().select(0,col_mean);

    // subtract mean from data and copy into det matrix
    in.scans.data.noalias() = in.scans.data.derived().rowwise() - dm;

    // subtract kernel mean
    if (in.kernel.data.size()!=0) {
        Eigen::RowVectorXd col_mean = (in.kernel.data.derived().array()*in.flags.data.derived().array().cast <double> ()).colwise().sum()/
                                      in.flags.data.derived().array().cast <double> ().colwise().sum();

        // remove nans from completely flagged detectors
        Eigen::RowVectorXd dm = (col_mean).array().isNaN().select(0,col_mean);

        // subtract mean from data and copy into det matrix
        in.kernel.data.noalias() = in.kernel.data.derived().rowwise() - dm;
    }
}

template <class calib_type, typename Derived>
void PTCProc::run(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in,
                  TCData<TCDataKind::PTC, Eigen::MatrixXd> &out, calib_type &calib,
                  Eigen::DenseBase<Derived> &det_indices, std::string stokes_param) {

    if (run_clean) {
        // number of samples
        Eigen::Index n_pts = in.scans.data.rows();
        Eigen::Index i = 0;

        // loop through config groupings
        for (const auto & group: cleaner.grouping) {
            // map of tuples to hold detector limits
            std::map<Eigen::Index, std::tuple<Eigen::Index, Eigen::Index>> grouping_limits;

            if (stokes_param=="I") {
                // network cleaning
                if (group == "nw" || group == "network") {
                    grouping_limits = calib.nw_limits;
                }

                // array cleaning
                else if (group == "array") {
                    grouping_limits = calib.array_limits;
                }

                // use all detectors for cleaning
                else if (group == "all") {
                    grouping_limits[0] = std::make_tuple(0,in.scans.data.cols());
                }
            }

            else if (run_stokes_clean) {
                Eigen::Index grp_i = calib.apt[group](det_indices(0));
                grouping_limits[grp_i] = std::tuple<Eigen::Index, Eigen::Index>{0, 0};
                Eigen::Index j = 0;
                // loop through apt table arrays, get highest index for current array
                for (Eigen::Index i=0; i<in.scans.data.cols(); i++) {
                    auto det_index = det_indices(i);
                    if (calib.apt[group](det_index) == grp_i) {
                        std::get<1>(grouping_limits[grp_i]) = i+1;
                    }
                    else {
                        grp_i = calib.apt[group](det_index);
                        j += 1;
                        grouping_limits[grp_i] = std::tuple<Eigen::Index, Eigen::Index>{i, 0};
                    }
                }
            }

            SPDLOG_DEBUG("cleaning with {} grouping", group);

            if (stokes_param=="I" || run_stokes_clean) {
                for (auto const& [key, val] : grouping_limits) {
                    // starting index
                    auto start_index = std::get<0>(val);
                    // size of block for each grouping
                    auto n_dets = std::get<1>(val) - std::get<0>(val);

                    // get the reference block of in scans that corresponds to the current array
                    Eigen::Ref<Eigen::MatrixXd> in_scans_ref = in.scans.data.block(0, start_index, n_pts, n_dets);

                    Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>>
                        in_scans(in_scans_ref.data(), in_scans_ref.rows(), in_scans_ref.cols(),
                                 Eigen::OuterStride<>(in_scans_ref.outerStride()));

                    // get the block of in flags that corresponds to the current array
                    Eigen::Ref<Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic>> in_flags_ref =
                        in.flags.data.block(0, start_index, n_pts, n_dets);

                    Eigen::Map<Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic>, 0, Eigen::OuterStride<> >
                        in_flags(in_flags_ref.data(), in_flags_ref.rows(), in_flags_ref.cols(),
                                 Eigen::OuterStride<>(in_flags_ref.outerStride()));

                    // get the block of out scans that corresponds to the current array
                    Eigen::Ref<Eigen::MatrixXd> out_scans_ref = out.scans.data.block(0, start_index, n_pts, n_dets);

                    Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>>
                        out_scans(out_scans_ref.data(), out_scans_ref.rows(), out_scans_ref.cols(),
                                  Eigen::OuterStride<>(out_scans_ref.outerStride()));

                    // get the block of out scans that corresponds to the current array
                    auto apt_flags = calib.apt["flag"].segment(start_index, n_dets);

                    // check if any good flags
                    if ((apt_flags.array()==1).any()) {
                        auto [evals, evecs] = cleaner.calc_eig_values<timestream::Cleaner::SpectraBackend>(in_scans, in_flags, apt_flags,
                                                                                                           cleaner.n_eig_to_cut(i));
                        SPDLOG_DEBUG("evals {}", evals);
                        SPDLOG_DEBUG("evecs {}", evecs);

                        cleaner.remove_eig_values<timestream::Cleaner::SpectraBackend>(in_scans, in_flags, evals, evecs, out_scans,
                                                                                       cleaner.n_eig_to_cut(i));

                        if (in.kernel.data.size()!=0) {
                            // check if any good flags
                                SPDLOG_DEBUG("cleaning kernel");
                                // get the reference block of in scans that corresponds to the current array
                                Eigen::Ref<Eigen::MatrixXd> in_kernel_ref = in.kernel.data.block(0, start_index, n_pts, n_dets);

                                Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>>
                                    in_kernel(in_kernel_ref.data(), in_kernel_ref.rows(), in_kernel_ref.cols(),
                                              Eigen::OuterStride<>(in_kernel_ref.outerStride()));

                                // get the block of out scans that corresponds to the current array
                                Eigen::Ref<Eigen::MatrixXd> out_kernel_ref = out.kernel.data.block(0, start_index, n_pts, n_dets);

                                Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<> >
                                    out_kernel(out_kernel_ref.data(), out_kernel_ref.rows(), out_scans_ref.cols(),
                                               Eigen::OuterStride<>(out_kernel_ref.outerStride()));

                                cleaner.remove_eig_values<timestream::Cleaner::SpectraBackend>(in_kernel, in_flags, evals, evecs, out_kernel,
                                                                                               cleaner.n_eig_to_cut(i));
                        }
                    }
                    else {
                        out.scans.data.block(0, start_index, n_pts, n_dets) = in.scans.data.block(0, start_index, n_pts, n_dets);
                        if (in.kernel.data.size()!=0) {
                            out.kernel.data.block(0, start_index, n_pts, n_dets) = in.kernel.data.block(0, start_index, n_pts, n_dets);
                        }
                    }
                }
                i++;
            }
            out.cleaned = true;
        }
    }
}

template <typename apt_type, class tel_type, typename Derived>
void PTCProc::calc_weights(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, apt_type &apt, tel_type &telescope,
                           Eigen::DenseBase<Derived> &det_indices) {
    if (weighting_type == "approximate") {
        SPDLOG_DEBUG("calculating weights using detector sensitivities");
        // resize weights to number of detectors
        in.weights.data = Eigen::VectorXd::Zero(in.scans.data.cols());

        double conversion_factor;

        // loop through detectors and calculate weights
        for (Eigen::Index i=0; i<in.scans.data.cols(); i++) {
            Eigen::Index det_index = det_indices(i);
            if (run_calibrate) {
                conversion_factor = in.fcf.data(det_index);
            }
            else {
                conversion_factor = 1;
            }
            // make sure flux conversion is not zero (otherwise weight=0)
            if (conversion_factor!=0) {// && apt["flag"](det_index)!=0) {
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
        SPDLOG_DEBUG("calculating weights using timestream variance");
        Eigen::Index n_dets = in.scans.data.cols();
        in.weights.data = Eigen::VectorXd::Zero(n_dets);

        for (Eigen::Index i=0; i<n_dets; i++) {
            // only calculate weights if detector is unflagged
            if (apt["flag"](i)!=0) {
                // make Eigen::Maps for each detector's scan
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> scans(
                    in.scans.data.col(i).data(), in.scans.data.rows());
                Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> flags(
                    in.flags.data.col(i).data(), in.flags.data.rows());

                double det_std_dev = engine_utils::calc_std_dev(scans, flags);

                if (det_std_dev !=0) {
                    in.weights.data(i) = pow(det_std_dev,-2);
                }
                else {
                    in.weights.data(i) = 0;
                }
            }
            else {
                in.weights.data(i) = 0;
            }
        }
    }
}

template <typename calib_t>
auto PTCProc::reset_weights(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, calib_t &calib) {
    // number of detectors
    Eigen::Index n_dets = in.scans.data.cols();
    for (auto const& [key, val] : calib.nw_limits) {
        auto nw_weights = in.weights.data(Eigen::seq(std::get<0>(calib.nw_limits[key]),
                                                     std::get<1>(calib.nw_limits[key])-1));

        // number of good detectors
        Eigen::Index n_good_det = calib.apt["flag"](Eigen::seq(std::get<0>(calib.nw_limits[key]),
                                                               std::get<1>(calib.nw_limits[key])-1)).sum();

        // to hold good detectors
        Eigen::VectorXd good_wt;

        if (n_good_det>0) {
            good_wt.resize(n_good_det);

            // remove flagged dets
            Eigen::Index j = std::get<0>(calib.nw_limits[key]);
            Eigen::Index k = 0;
            for (Eigen::Index m=0; m<nw_weights.size(); m++) {
                if (calib.apt["flag"](j)) {
                    good_wt(k) = nw_weights(m);
                    k++;
                }
                j++;
            }
        }
        else {
            good_wt = nw_weights;
        }

        auto med_wt = tula::alg::median(good_wt);
        SPDLOG_INFO("nw{} med_wt {}",key, med_wt);

        int outliers = 0;

        Eigen::Index j = std::get<0>(calib.nw_limits[key]);
        for (Eigen::Index m=0; m<nw_weights.size(); m++) {
            if (in.weights.data(j) > med_wt) {
                in.weights.data(j) = med_wt;
                outliers++;
            }
            j++;
        }
        SPDLOG_INFO("nw{} had {} outlier weights",key, outliers);
    }
}

template <typename apt_t, typename Derived>
void PTCProc::remove_flagged_dets(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, apt_t &apt, Eigen::DenseBase<Derived> &det_indices) {

    // number of detectors
    Eigen::Index n_dets = in.scans.data.cols();

    // number of detectors flagged in apt
    Eigen::Index n_flagged = (apt["flag"].array()==0).count();
    SPDLOG_INFO("removing {} detectors flagged in APT table ({}%)",n_flagged,
                (static_cast<float>(n_flagged)/static_cast<float>(n_dets))*100);

    // loop through detectors and set flags to zero
    // for those flagged in apt table
    for (Eigen::Index i=0; i<n_dets; i++) {
        Eigen::Index det_index = det_indices(i);
        if (!apt["flag"](det_index)) {
            in.flags.data.col(i).setZero();
        }
    }
}

template <typename calib_t, typename Derived>
auto PTCProc::remove_bad_dets(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, calib_t &calib, Eigen::DenseBase<Derived> &det_indices,
                                 Eigen::DenseBase<Derived> &nw_indices, Eigen::DenseBase<Derived> &array_indices, std::string redu_type,
                                 std::string map_grouping) {

    // make a copy of the calib class for flagging
    calib_t calib_scan = calib;

    // number of detectors
    Eigen::Index n_dets = in.scans.data.cols();

    in.n_low_dets = 0;
    in.n_high_dets = 0;

    // only run if limits are not zero
    if (lower_std_dev !=0 || upper_std_dev !=0) {
        for (Eigen::Index i=0; i<calib.n_nws; i++) {
            // number of unflagged detectors
            Eigen::Index n_good_dets = 0;

            for (Eigen::Index j=0; j<n_dets; j++) {
                Eigen::Index det_index = det_indices(j);
                if (calib.apt["flag"](det_index) && calib.apt["nw"](det_index)==calib.nws(i)) {
                    n_good_dets++;
                }
            }

            Eigen::VectorXd det_std_dev(n_good_dets);
            Eigen::VectorXI dets(n_good_dets);
            Eigen::Index k = 0;

            // collect standard deviation from good detectors
            for (Eigen::Index j=0; j<n_dets; j++) {
                Eigen::Index det_index = det_indices(j);
                if (calib.apt["flag"](det_index) && calib.apt["nw"](det_index)==calib.nws(i)) {
                    // make Eigen::Maps for each detector's scan
                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> scans(
                        in.scans.data.col(j).data(), in.scans.data.rows());
                    Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> flags(
                        in.flags.data.col(j).data(), in.flags.data.rows());

                    // calc standard deviation
                    det_std_dev(k) = engine_utils::calc_std_dev(scans, flags);
                    //det_std_dev(k) = engine_utils::calc_rms(scans, flags);

                    // convert to 1/variance
                    if (det_std_dev(k) !=0) {
                        det_std_dev(k) = std::pow(det_std_dev(k),-2);
                    }
                    else {
                        det_std_dev(k) = 0;
                    }

                    dets(k) = j;
                    k++;
                }
            }

            // get mean standard deviation
            //double mean_std_dev = det_std_dev.mean();
            double mean_std_dev = tula::alg::median(det_std_dev);

            int n_low_dets = 0;
            int n_high_dets = 0;

            // loop through good detectors and flag those that have std devs beyond the limits
            for (Eigen::Index j=0; j<n_good_dets; j++) {
                Eigen::Index det_index = det_indices(dets(j));
                // flag those below limit
                if (calib.apt["flag"](det_index) && calib.apt["nw"](det_index)==calib.nws(i)) {
                    if ((det_std_dev(j) < (lower_std_dev*mean_std_dev))  && lower_std_dev!=0) {
                        if (map_grouping!="detector") {
                            in.flags.data.col(dets(j)).setZero();
                        }
                        else {
                            calib_scan.apt["flag"](det_index) = 0;
                        }
                        in.n_low_dets++;
                        n_low_dets++;
                    }

                    // flag those above limit
                    if ((det_std_dev(j) > (upper_std_dev*mean_std_dev))  && upper_std_dev!=0) {
                        if (map_grouping!="detector") {
                            in.flags.data.col(dets(j)).setZero();
                        }
                        else {
                            calib_scan.apt["flag"](det_index) = 0;
                        }
                        in.n_high_dets++;
                        n_high_dets++;
                    }
                }
            }

            SPDLOG_INFO("nw{}: {}/{} dets below limit. {}/{} dets above limit.", calib.nws(i), n_low_dets, n_good_dets,
                        n_high_dets, n_good_dets);
        }
    }

    // set up scan calib
    calib_scan.setup();

    return std::move(calib_scan);
}


template <typename calib_t, typename Derived>
auto PTCProc::remove_bad_dets_iter(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, calib_t &calib, Eigen::DenseBase<Derived> &det_indices,
                                 Eigen::DenseBase<Derived> &nw_indices, Eigen::DenseBase<Derived> &array_indices, std::string redu_type,
                                 std::string map_grouping) {

    // make a copy of the calib class for flagging
    calib_t calib_scan = calib;

    // number of detectors
    Eigen::Index n_dets = in.scans.data.cols();

    // number of dets lower than limit
    in.n_low_dets = 0;
    // number of dets higher than limit
    in.n_high_dets = 0;

    // only do if config options are not zero
    if (upper_std_dev!=0 || lower_std_dev!=0) {
        SPDLOG_INFO("removing outlier weights");
        for (Eigen::Index i=0; i<calib.n_nws; i++) {
            bool keep_going = true;
            Eigen::Index n_iter = 0;

            while (keep_going) {
                SPDLOG_INFO("iteration {}", n_iter);
                // number of unflagged detectors
                Eigen::Index n_good_dets = 0;

                // get number of unflagged detectors
                for (Eigen::Index j=0; j<n_dets; j++) {
                    Eigen::Index det_index = det_indices(j);
                    if (calib_scan.apt["flag"](det_index) && calib_scan.apt["nw"](det_index)==calib_scan.nws(i)) {
                        n_good_dets++;
                    }
                }

                Eigen::VectorXd det_std_dev(n_good_dets);
                Eigen::VectorXI dets(n_good_dets);
                Eigen::Index k = 0;

                // collect standard deviation from good detectors
                for (Eigen::Index j=0; j<n_dets; j++) {
                    Eigen::Index det_index = det_indices(j);
                    if (calib_scan.apt["flag"](det_index) && calib_scan.apt["nw"](det_index)==calib_scan.nws(i)) {

                        // make Eigen::Maps for each detector's scan
                        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> scans(
                            in.scans.data.col(j).data(), in.scans.data.rows());
                        Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> flags(
                            in.flags.data.col(j).data(), in.flags.data.rows());

                        det_std_dev(k) = engine_utils::calc_std_dev(scans, flags);

                        if (det_std_dev(k) !=0) {
                            det_std_dev(k) = std::pow(det_std_dev(k),-2);
                        }
                        else {
                            det_std_dev(k) = 0;
                        }
                        //det_std_dev(k) = engine_utils::calc_rms(scans, flags);

                        dets(k) = j;
                        k++;
                    }
                }

                // get mean standard deviation
                //double mean_std_dev = det_std_dev.mean();
                double mean_std_dev = tula::alg::median(det_std_dev);

                int n_low_dets = 0;
                int n_high_dets = 0;

                // loop through good detectors and flag those that have std devs beyond the limits
                for (Eigen::Index j=0; j<n_good_dets; j++) {
                    Eigen::Index det_index = det_indices(dets(j));
                    // flag those below limit
                    if (calib_scan.apt["flag"](det_index) && calib_scan.apt["nw"](det_index)==calib_scan.nws(i)) {
                        if ((det_std_dev(j) < (lower_std_dev*mean_std_dev)) && lower_std_dev!=0) {
                            if (map_grouping!="detector") {
                                in.flags.data.col(dets(j)).setZero();
                                calib_scan.apt["flag"](det_index) = 0;
                            }
                            else {
                                calib_scan.apt["flag"](det_index) = 0;
                            }
                            in.n_low_dets++;
                            n_low_dets++;
                        }

                        // flag those above limit
                        if ((det_std_dev(j) > (upper_std_dev*mean_std_dev)) && upper_std_dev!=0) {
                            if (map_grouping!="detector") {
                                in.flags.data.col(dets(j)).setZero();
                                calib_scan.apt["flag"](det_index) = 0;
                            }
                            else {
                                calib_scan.apt["flag"](det_index) = 0;
                            }
                            in.n_high_dets++;
                            n_high_dets++;
                        }
                    }
                }

                if ((n_low_dets==0 && n_high_dets==0) || n_iter > 10) {
                    keep_going = false;
                }

                n_iter++;

                SPDLOG_INFO("nw{}: {}/{} dets below limit. {}/{} dets above limit.", calib.nws(i), n_low_dets, n_good_dets,
                            n_high_dets, n_good_dets);
            }

        }
    }

    // set up calib scan
    calib_scan.setup();

    return std::move(calib_scan);
}

template <typename Derived, typename apt_t, typename pointing_offset_t>
void PTCProc::append_to_netcdf(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, std::string filepath, std::string redu_type,
                               std::string &pixel_axes, pointing_offset_t &pointing_offsets_arcsec, Eigen::DenseBase<Derived> &det_indices,
                               apt_t &apt, std::string tod_output_type,bool verbose_mode, double fsmp) {

    // tangent plane pointing for each detector
    Eigen::MatrixXd lats(in.scans.data.rows(), in.scans.data.cols()), lons(in.scans.data.rows(), in.scans.data.cols());

    for (Eigen::Index i=0; i<in.scans.data.cols(); i++) {
        double az_off = 0;
        double el_off = 0;

        if (redu_type!="beammap") {
            auto det_index = det_indices(i);
            az_off = apt["x_t"](det_index);
            el_off = apt["y_t"](det_index);
        }

        // get tangent pointing
        auto [lat, lon] = engine_utils::calc_det_pointing(in.tel_data.data, az_off, el_off,
                                                          pixel_axes, pointing_offsets_arcsec);
        lats.col(i) = std::move(lat);
        lons.col(i) = std::move(lon);
    }

    using netCDF::NcDim;
    using netCDF::NcFile;
    using netCDF::NcType;
    using netCDF::NcVar;
    using namespace netCDF::exceptions;

    try {
        netCDF::NcFile fo(filepath, netCDF::NcFile::write);
        auto vars = fo.getVars();

        double cra, cdec;
        vars.find("SourceRa")->second.getVar(&cra);
        vars.find("SourceDec")->second.getVar(&cdec);

        NcDim n_pts_dim = fo.getDim("n_pts");
        NcDim n_dets_dim = fo.getDim("n_dets");

        unsigned long n_pts_exists = n_pts_dim.getSize();
        unsigned long n_dets_exists = n_dets_dim.getSize();

        // start indices for data
        std::vector<std::size_t> start_index = {n_pts_exists, 0};
        // size for data
        std::vector<std::size_t> size = {1, n_dets_exists};

        // start index for telescope data
        std::vector<std::size_t> start_index_tel = {n_pts_exists};
        // size for telescope data
        std::vector<std::size_t> size_tel = {1};

        // start index for apt table
        std::vector<std::size_t> start_index_apt = {0};
        // size for apt
        std::vector<std::size_t> size_apt = {1};

        // get timestream variables
        NcVar signal_v = fo.getVar("signal");
        NcVar flags_v = fo.getVar("flags");
        NcVar kernel_v = fo.getVar("kernel");

        // detector tangent plane pointing
        NcVar det_lat_v = fo.getVar("det_lat");
        NcVar det_lon_v = fo.getVar("det_lon");

        // detector absolute pointing
        NcVar det_ra_v = fo.getVar("det_ra");
        NcVar det_dec_v = fo.getVar("det_dec");

        // append weights if output type is ptc
        if (tod_output_type == "ptc") {
            std::vector<std::size_t> start_index_weights = {static_cast<unsigned long>(in.index.data), 0};
            std::vector<std::size_t> size_weights = {1, n_dets_exists};

            NcVar weights_v = fo.getVar("weights");

            weights_v.putVar(start_index_weights, size_weights, in.weights.data.data());
        }

        // append data
        for (std::size_t i = 0; i < TULA_SIZET(in.scans.data.rows()); ++i) {
            start_index[0] = n_pts_exists + i;
            start_index_tel[0] = n_pts_exists + i;

            // append scans
            Eigen::VectorXd scans = in.scans.data.row(i);
            signal_v.putVar(start_index, size, scans.data());

            // append flags
            Eigen::VectorXi flags_int = in.flags.data.row(i).cast<int> ();
            flags_v.putVar(start_index, size, flags_int.data());

            // append kernel
            if (!kernel_v.isNull()) {
                Eigen::VectorXd kernel = in.kernel.data.row(i);
                kernel_v.putVar(start_index, size, kernel.data());
            }

            // append detector latitudes
            Eigen::VectorXd lats_row = lats.row(i);
            det_lat_v.putVar(start_index, size, lats_row.data());

            // append detector longitudes
            Eigen::VectorXd lons_row = lons.row(i);
            det_lon_v.putVar(start_index, size, lons_row.data());

            if (pixel_axes == "icrs") {
                // get absolute pointing
                auto [decs, ras] = engine_utils::tangent_to_abs(lats_row, lons_row, cra, cdec);

                // append detector ra
                det_ra_v.putVar(start_index, size, ras.data());

                // append detector dec
                det_dec_v.putVar(start_index, size, decs.data());
            }

            // append telescope
            for (auto const& x: in.tel_data.data) {
                NcVar tel_data_v = fo.getVar(x.first);
                tel_data_v.putVar(start_index_tel, size_tel, x.second.row(i).data());
            }

            // append pointing offsets
            for (auto const& x: in.pointing_offsets_arcsec.data) {
                NcVar offset_v = fo.getVar("pointing_offset_"+x.first);
                offset_v.putVar(start_index_tel, size_tel, x.second.row(i).data());
            }
        }

        // overwrite apt table
        for (auto const& x: apt) {
            netCDF::NcVar apt_v = fo.getVar("apt_" + x.first);
            for (std::size_t i=0; i< TULA_SIZET(n_dets_exists); ++i) {
                start_index_apt[0] = i;
                apt_v.putVar(start_index_apt, size_apt, &apt[x.first](det_indices(i)));
            }
        }

        // vector to hold current scan indices
        Eigen::VectorXd scan_indices(2);

        // if not on first scan, grab last scan and add size of current scan
        if (in.index.data > 0) {
            // start indices for data
            std::vector<std::size_t> scan_indices_start_index = {TULA_SIZET(in.index.data-1), 0};
            // size for data
            std::vector<std::size_t> scan_indices_size = {1, 2};
            vars.find("scan_indices")->second.getVar(scan_indices_start_index,scan_indices_size,scan_indices.data());

            scan_indices = scan_indices.array() + in.scans.data.rows();
        }
        // otherwise, use size of this scan
        else {
            scan_indices(0) = 0;
            scan_indices(1) = in.scans.data.rows() - 1;
        }

        // add current scan indices row
        std::vector<std::size_t> scan_indices_start_index = {TULA_SIZET(in.index.data), 0};
        std::vector<std::size_t> scan_indices_size = {1, 2};
        NcVar scan_indices_v = fo.getVar("scan_indices");
        scan_indices_v.putVar(scan_indices_start_index, scan_indices_size,scan_indices.data());

        fo.sync();
        fo.close();

    } catch (NcException &e) {
        SPDLOG_ERROR("{}", e.what());
    }
}

template <typename DerivedB, typename DerivedC, typename apt_t, typename pointing_offset_t>
void PTCProc::add_gaussian(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, Eigen::DenseBase<DerivedB> &params, std::string &pixel_axes,
                  std::string &map_grouping, apt_t &apt, pointing_offset_t &pointing_offsets_arcsec,
                  double pixel_size_rad, Eigen::Index n_rows, Eigen::Index n_cols,
                  Eigen::DenseBase<DerivedC> &map_indices, Eigen::DenseBase<DerivedC> &det_indices, std::string type) {

    Eigen::Index n_dets = in.scans.data.cols();
    Eigen::Index n_pts = in.scans.data.rows();

    // loop through detectors
    for (Eigen::Index i=0; i<n_dets; i++) {
        double azoff, eloff;

        auto det_index = det_indices(i);
        auto map_index = map_indices(i);

        double az_off = 0;
        double el_off = 0;

        if (map_grouping!="detector") {
            az_off = apt["x_t"](det_index);
            el_off = apt["y_t"](det_index);
        }

        auto [lat, lon] = engine_utils::calc_det_pointing(in.tel_data.data, az_off, el_off,
                                                          pixel_axes, pointing_offsets_arcsec);

        // get parameters for current detector
        double amp = params(map_index,0);
        double off_lat = params(map_index,2);
        double off_lon = params(map_index,1);
        double sigma_lat = params(map_index,4);
        double sigma_lon = params(map_index,3);
        double rot_ang = params(map_index,5);

        double sigma = std::max(sigma_lat, sigma_lon);

        if (type=="subtract") {
            amp = -amp;
        }

        // rescale offsets and stddev to on-sky units
        off_lat = pixel_size_rad*(off_lat - (n_rows)/2);
        off_lon = pixel_size_rad*(off_lon - (n_cols)/2);

        sigma_lon = pixel_size_rad*sigma;
        sigma_lat = pixel_size_rad*sigma;
        sigma = pixel_size_rad*sigma;

        auto cost2 = cos(rot_ang) * cos(rot_ang);
        auto sint2 = sin(rot_ang) * sin(rot_ang);
        auto sin2t = sin(2. * rot_ang);
        auto xstd2 = sigma * sigma;
        auto ystd2 = sigma * sigma;
        auto a = - 0.5 * ((cost2 / xstd2) + (sint2 / ystd2));
        auto b = - 0.5 * ((sin2t / xstd2) - (sin2t / ystd2));
        auto c = - 0.5 * ((sint2 / xstd2) + (cost2 / ystd2));

        // calculate distance to source to truncate it
        auto distance = ((lat.array() - off_lat).pow(2) + (lon.array() - off_lon).pow(2)).sqrt();

        Eigen::VectorXd gauss(n_pts);
        // make gaussian
        for (Eigen::Index j=0; j<n_pts; j++) {
            //if (distance(j) < 10*(sigma_lat+sigma_lon)/2) {
                gauss(j) = amp*exp(pow(lon(j) - off_lon, 2) * a +
                                     (lon(j) - off_lon) * (lat(j) - off_lat) * b +
                                     pow(lat(j) - off_lat, 2) * c);
            //}
            //else {
            //    gauss(j) = 0;
            //}
        }

        /*for (Eigen::Index j=0; j<n_pts; j++) {
            if (distance(j) <= 3.*(sigma_lat+sigma_lon)/2) {
                gauss(j,i) = exp(-0.5*pow(distance(j)/(sigma_lat+sigma_lon)/2,2));
            }
            else {
                gauss(j,i) = 0;
            }
        }*/

        if (!gauss.array().isNaN().any()) {
            // add gaussian to detector scan
            in.scans.data.col(i) = in.scans.data.col(i).array() + gauss.array();
        }
    }
}

} // namespace timestream
