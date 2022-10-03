#pragma once

#include <tula/logging.h>
#include <tula/nc.h>

namespace timestream {

class PTCProc {
public:
    bool run_clean;
    std::string weighting_type;

    // ptc tod proc
    timestream::Cleaner cleaner;

    double lower_std_dev, upper_std_dev;

    template <class calib_type>
    void run(TCData<TCDataKind::PTC, Eigen::MatrixXd> &,
                   TCData<TCDataKind::PTC, Eigen::MatrixXd> &, calib_type &);

    template <class calib_type, class tel_type>
    void calc_weights(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, calib_type &, tel_type &);

    template <typename apt_t, typename Derived>
    void remove_flagged_dets(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, apt_t &, Eigen::DenseBase<Derived> &);

    template <typename apt_t, typename Derived>
    void remove_bad_dets(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, apt_t &, Eigen::DenseBase<Derived> &);

    template <typename Derived, typename apt_t>
    void append_to_netcdf(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, std::string,
                          Eigen::DenseBase<Derived> &, apt_t &);
};

template <class calib_type>
void PTCProc::run(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in,
                  TCData<TCDataKind::PTC, Eigen::MatrixXd> &out, calib_type &calib) {

    Eigen::Index n_pts = in.scans.data.rows();

    std::map<Eigen::Index, std::tuple<Eigen::Index, Eigen::Index>> grouping_limits;

    if (cleaner.grouping == "nw") {
        grouping_limits = calib.nw_limits;
    }

    else if (cleaner.grouping == "array") {
        grouping_limits = calib.array_limits;
    }

    for (auto const& [key, val] : grouping_limits) {
        // starting index
        auto start_index = std::get<0>(val);
        // size of block for each grouping
        auto n_dets = std::get<1>(val) - std::get<0>(val);

        SPDLOG_INFO("start_index {} n_dets {}", start_index, n_dets);

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

        Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<> >
            out_scans(out_scans_ref.data(), out_scans_ref.rows(), out_scans_ref.cols(),
                      Eigen::OuterStride<>(out_scans_ref.outerStride()));

        auto [evals, evecs] = cleaner.calc_eig_values<SpectraBackend>(in_scans, in_flags);
        cleaner.remove_eig_values(in_scans, in_flags, evals, evecs, out_scans);
    }

}

template <class calib_type, class tel_type>
void PTCProc::calc_weights(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, calib_type &calib, tel_type &telescope) {
    if (weighting_type == "approximate") {
        in.weights.data = pow(sqrt(telescope.d_fsmp)*calib.apt["sens"].array(),-2.0);
    }

    else {
        Eigen::Index n_dets = in.scans.data.cols();
        in.weights.data = Eigen::VectorXd::Zero(n_dets);

        for (Eigen::Index i=0; i<n_dets; i++) {
            // make Eigen::Maps for each detector's scan
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> scans(
                in.scans.data.col(i).data(), in.scans.data.rows());
            Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> flags(
                in.flags.data.col(i).data(), in.flags.data.rows());

            double det_std_dev = engine_utils::calc_std_dev(scans, flags);

            in.weights.data(i) = pow(det_std_dev,-2);
        }
    }
}

template <typename apt_t, typename Derived>
void PTCProc::remove_flagged_dets(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, apt_t &apt, Eigen::DenseBase<Derived> &det_indices) {

    Eigen::Index n_dets = in.scans.data.cols();

    for (Eigen::Index i=0; i<n_dets; i++) {
        Eigen::Index det_index = det_indices(i);
        if (!apt["flag"](det_index)) {
            in.flags.data.col(i).setZero();
        }
    }
}

template <typename apt_t, typename Derived>
void PTCProc::remove_bad_dets(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, apt_t &apt, Eigen::DenseBase<Derived> &det_indices) {
    Eigen::Index n_dets = in.scans.data.cols();

    // number of unflagged detectors
    Eigen::Index n_good_dets = 0;

    for (Eigen::Index i=0; i<n_dets; i++) {
        Eigen::Index det_index = det_indices(i);
        if (apt["flag"](det_index)) {
            n_good_dets++;
        }
    }

    Eigen::VectorXd good_weights(n_good_dets), det_std_dev(n_good_dets);
    Eigen::VectorXI dets(n_good_dets);
    Eigen::Index j = 0;

    // collect standard deviation from good detectors
    for (Eigen::Index i=0; i<n_dets; i++) {
        Eigen::Index det_index = det_indices(i);
        if (apt["flag"](det_index)) {
            good_weights(j) = in.weights.data(i);

            // make Eigen::Maps for each detector's scan
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> scans(
                in.scans.data.col(i).data(), in.scans.data.rows());
            Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> flags(
                in.flags.data.col(i).data(), in.flags.data.rows());

            det_std_dev(j) = engine_utils::calc_std_dev(scans, flags);
            dets(j) = i;
            j++;
        }
    }

    // get mean standard deviation
    double mean_std_dev = det_std_dev.mean();

    // loop through good detectors and flag those that have std devs beyond the limits.
    for (Eigen::Index i=0; i<n_good_dets; i++) {
        Eigen::Index det_index = det_indices(dets(i));
        if (apt["flag"](det_index)) {
            if ((det_std_dev(i) < (lower_std_dev*mean_std_dev)) && lower_std_dev!=0) {
                in.flags.data.col(dets(i)).setZero();
            }

            if ((det_std_dev(i) > (upper_std_dev*mean_std_dev)) && upper_std_dev!=0) {
                in.flags.data.col(dets(i)).setZero();
            }
        }
    }
}

template <typename Derived, typename apt_t>
void PTCProc::append_to_netcdf(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, std::string filepath,
                               Eigen::DenseBase<Derived> &det_indices, apt_t &apt) {
    using netCDF::NcDim;
    using netCDF::NcFile;
    using netCDF::NcType;
    using netCDF::NcVar;
    using namespace netCDF::exceptions;

    //try {
        netCDF::NcFile fo(filepath, netCDF::NcFile::write);
        auto vars = fo.getVars();

        NcDim n_pts_dim = fo.getDim("n_pts");
        NcDim n_dets_dim = fo.getDim("n_dets");
        NcDim n_scans_dim = fo.getDim("n_scans");

        unsigned long n_pts_exists = n_pts_dim.getSize();
        unsigned long n_dets_exists = n_dets_dim.getSize();
        unsigned long n_scans_exists = n_scans_dim.getSize();

        std::vector<std::size_t> start_index = {n_pts_exists, 0};
        std::vector<std::size_t> size = {1, n_dets_exists};

        std::vector<std::size_t> start_index_apt = {0};
        std::vector<std::size_t> size_apt = {1};

        NcVar scans_var = fo.getVar("scans");
        NcVar flags_var = fo.getVar("flags");
        NcVar kernel_var = fo.getVar("kernel");

        NcDim n_weights_dim = fo.getDim("n_weights");

        if (!n_weights_dim.isNull()) {

            std::vector<std::size_t> start_index_weights = {n_scans_exists,0};
            std::vector<std::size_t> size_weights = {1, n_dets_exists};

            NcVar weights_var = fo.getVar("weights");

            weights_var.putVar(start_index_weights, size_weights, in.weights.data.data());
        }

        // append data
        for (std::size_t i = 0; i < TULA_SIZET(in.scans.data.rows()); ++i) {
            start_index[0] = n_pts_exists + i;
            scans_var.putVar(start_index, size, in.scans.data.row(i).data());

            Eigen::VectorXi flags_double = in.flags.data.row(i).cast<int> ();

            flags_var.putVar(start_index, size, flags_double.data());
            kernel_var.putVar(start_index, size, in.kernel.data.row(i).data());

            // append telescope
            for (auto const& x: in.tel_data.data) {
                NcVar tel_data_var = fo.getVar(x.first);
                tel_data_var.putVar(start_index, size, x.second.data());
            }
        }

        // overwrite apt table
        for (auto const& x: apt) {
            netCDF::NcVar apt_var = fo.getVar(x.first);
            for (std::size_t i=0; i< TULA_SIZET(n_dets_exists); ++i) {
                start_index_apt[0] = i;
                apt_var.putVar(start_index_apt, size_apt, &apt[x.first](det_indices(i)));
            }
        }

        fo.sync();
        fo.close();

    //} catch (NcException &e) {
      //  SPDLOG_ERROR("{}", e.what());
    //}
}

} // namespace timestream
