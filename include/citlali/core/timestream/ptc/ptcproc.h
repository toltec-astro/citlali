#pragma once

#include <tula/logging.h>
#include <tula/nc.h>
#include <tula/algorithm/ei_stats.h>

#include <citlali/core/utils/utils.h>
#include <citlali/core/utils/pointing.h>


namespace timestream {

class PTCProc {
public:
    bool run_clean, run_calibrate;
    std::string weighting_type;

    // ptc tod proc
    timestream::Cleaner cleaner;

    double lower_std_dev, upper_std_dev;

    template <class calib_type>
    void run(TCData<TCDataKind::PTC, Eigen::MatrixXd> &,
                   TCData<TCDataKind::PTC, Eigen::MatrixXd> &, calib_type &);

    template <typename apt_type, class tel_type>
    void calc_weights(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, apt_type &, tel_type &);

    template <typename apt_t, typename Derived>
    void remove_flagged_dets(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, apt_t &, Eigen::DenseBase<Derived> &);

    template <typename calib_t, typename Derived>
    auto remove_bad_dets_nw(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, calib_t &, Eigen::DenseBase<Derived> &,
                            Eigen::DenseBase<Derived> &, Eigen::DenseBase<Derived> &, std::string, std::string);

    template <typename Derived, typename apt_t, typename pointing_offset_t>
    void append_to_netcdf(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, std::string, std::string, std::string &,
                          pointing_offset_t &, Eigen::DenseBase<Derived> &, apt_t &, std::string, bool, double);

    template <typename DerivedB, typename DerivedC, typename apt_t, typename pointing_offset_t>
    void add_gaussian(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, Eigen::DenseBase<DerivedB> &, std::string &,
                      std::string &, apt_t &, pointing_offset_t &, double,
                      Eigen::Index, Eigen::Index, Eigen::DenseBase<DerivedC> &, Eigen::DenseBase<DerivedC> &);
};

template <class calib_type>
void PTCProc::run(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in,
                  TCData<TCDataKind::PTC, Eigen::MatrixXd> &out, calib_type &calib) {

    if (run_clean) {
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

            /*Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>>
                apt_flags(apt_flags_ref.data(), apt_flags_ref.rows(), apt_flags_ref.cols(),
                          Eigen::OuterStride<>(apt_flags_ref.outerStride()));*/

            auto [evals, evecs] = cleaner.calc_eig_values<SpectraBackend>(in_scans, in_flags, apt_flags);
            cleaner.remove_eig_values(in_scans, in_flags, evals, evecs, out_scans);

            in.evals.data = evals.head(cleaner.n_eig_to_cut);

            if (in.kernel.data.size()!=0) {
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

                cleaner.remove_eig_values(in_kernel, in_flags, evals, evecs, out_kernel);
            }
        }
        out.cleaned = true;
    }
}

template <typename apt_type, class tel_type>
void PTCProc::calc_weights(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, apt_type &apt, tel_type &telescope) {
    if (weighting_type == "approximate") {
        SPDLOG_DEBUG("calculating weights using approximate method");
        // resize weights to number of detectors
        in.weights.data = Eigen::VectorXd::Zero(in.scans.data.cols());

        double conversion_factor;

        // loop through detectors and calculate weights
        for (Eigen::Index i=0; i<in.scans.data.cols(); i++) {
            if (run_calibrate) {
                conversion_factor = apt["flxscale"](i);
            }

            else {
                conversion_factor = 1;
            }
            // make sure flux conversion is not zero (otherwise weight=0)
            if (conversion_factor!=0) {
                // calculate weights while applying flux calibration
                in.weights.data(i) = pow(sqrt(telescope.d_fsmp)*apt["sens"](i)*conversion_factor,-2.0);
            }
            else {
                in.weights.data(i) = 0;
            }
        }
    }

    else if (weighting_type == "full"){
        SPDLOG_DEBUG("calculating weights using full method");
        Eigen::Index n_dets = in.scans.data.cols();
        in.weights.data = Eigen::VectorXd::Zero(n_dets);

        for (Eigen::Index i=0; i<n_dets; i++) {
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
    }
}

template <typename apt_t, typename Derived>
void PTCProc::remove_flagged_dets(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, apt_t &apt, Eigen::DenseBase<Derived> &det_indices) {

    Eigen::Index n_dets = in.scans.data.cols();
    SPDLOG_INFO("removing {} detectors flagged in APT table",(apt["flag"].array()==0).count());

    for (Eigen::Index i=0; i<n_dets; i++) {
        Eigen::Index det_index = det_indices(i);
        if (!apt["flag"](det_index)) {
            in.flags.data.col(i).setZero();
        }
    }
}

template <typename calib_t, typename Derived>
auto PTCProc::remove_bad_dets_nw(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, calib_t &calib, Eigen::DenseBase<Derived> &det_indices,
                                 Eigen::DenseBase<Derived> &nw_indices, Eigen::DenseBase<Derived> &array_indices, std::string redu_type,
                                 std::string map_grouping) {

    calib_t calib_scan = calib;

    Eigen::Index n_dets = in.scans.data.cols();

    in.n_low_dets = 0;
    in.n_high_dets = 0;

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

                det_std_dev(k) = engine_utils::calc_std_dev(scans, flags);
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
                if ((det_std_dev(j) < (lower_std_dev*mean_std_dev)) && lower_std_dev!=0) {
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
                if ((det_std_dev(j) > (upper_std_dev*mean_std_dev)) && upper_std_dev!=0) {
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

    //TCData<TCDataKind::PTC, Eigen::MatrixXd> out = in;

    /*Eigen::Index n_good_dets = 0;
    for (Eigen::Index i=0; i<n_dets; i++) {
        if ((in.flags.data.col(i).array()!=0).all()) {
            n_good_dets++;
        }
    }

    out.scans.data.resize(in.scans.data.rows(), n_good_dets);
    out.flags.data.resize(in.flags.data.rows(), n_good_dets);

    if (in.kernel.data.size()!=0) {
        out.kernel.data.resize(in.kernel.data.rows(), n_good_dets);
    }*/

    //Eigen::VectorXI array_indices_temp(n_good_dets), nw_indices_temp(n_good_dets), det_indices_temp(n_good_dets);

    /*Eigen::Index j = 0;
    for (Eigen::Index i=0; i<n_dets; i++) {
        if ((in.flags.data.col(i).array()!=0).all()) {
            out.scans.data.col(j) = in.scans.data.col(i);
            out.flags.data.col(j) = in.flags.data.col(i);

            if (in.kernel.data.size()!=0) {
                out.kernel.data.col(j) = in.kernel.data.col(i);
            }

            det_indices_temp(j) = det_indices(i);
            nw_indices_temp(j) = nw_indices(i);
            array_indices_temp(j) = array_indices(i);
            j++;
        }
    }*/

    /*det_indices = det_indices_temp;
    nw_indices = nw_indices_temp;
    array_indices = array_indices_temp;

    in = out;*/

    /*for (auto const& [key, val]: calib.apt) {
        calib_temp.apt[key].setZero(n_good_dets);
        Eigen::Index i = 0;
        for (Eigen::Index j=0; j<calib.apt["nw"].size(); j++) {
            if ((in.flags.data.col(j).array()!=0).all()) {
                calib_temp.apt[key](i) = calib.apt[key](j);
                i++;
            }
        }
    }

    calib_temp.setup();*/

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
        NcVar scans_var = fo.getVar("signal");
        NcVar flags_var = fo.getVar("flags");
        NcVar kernel_var = fo.getVar("kernel");

        NcVar det_lat_var = fo.getVar("det_lat");
        NcVar det_lon_var = fo.getVar("det_lon");

        // append weights if output type is ptc
        if (tod_output_type == "ptc") {
            std::vector<std::size_t> start_index_weights = {static_cast<unsigned long>(in.index.data), 0};
            std::vector<std::size_t> size_weights = {1, n_dets_exists};

            NcVar weights_var = fo.getVar("weights");

            weights_var.putVar(start_index_weights, size_weights, in.weights.data.data());
        }

        // append data
        for (std::size_t i = 0; i < TULA_SIZET(in.scans.data.rows()); ++i) {
            start_index[0] = n_pts_exists + i;
            start_index_tel[0] = n_pts_exists + i;

            // append scans
            Eigen::VectorXd scans = in.scans.data.row(i);
            scans_var.putVar(start_index, size, scans.data());

            // append flags
            Eigen::VectorXi flags_int = in.flags.data.row(i).cast<int> ();
            flags_var.putVar(start_index, size, flags_int.data());

            // append kernel
            if (!kernel_var.isNull()) {
                Eigen::VectorXd kernel = in.kernel.data.row(i);
                kernel_var.putVar(start_index, size, in.kernel.data.row(i).data());
            }

            // append detector lats
            Eigen::VectorXd lats_row = lats.row(i);
            det_lat_var.putVar(start_index, size, lats_row.data());

            // append detector lons
            Eigen::VectorXd lons_row = lons.row(i);
            det_lon_var.putVar(start_index, size, lons_row.data());

            // append telescope
            for (auto const& x: in.tel_data.data) {
                NcVar tel_data_var = fo.getVar(x.first);
                tel_data_var.putVar(start_index_tel, size_tel, x.second.row(i).data());
            }
        }

        // overwrite apt table
        for (auto const& x: apt) {
            netCDF::NcVar apt_var = fo.getVar("apt_" + x.first);
            for (std::size_t i=0; i< TULA_SIZET(n_dets_exists); ++i) {
                start_index_apt[0] = i;
                apt_var.putVar(start_index_apt, size_apt, &apt[x.first](det_indices(i)));
            }
        }

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
                  Eigen::DenseBase<DerivedC> &map_indices, Eigen::DenseBase<DerivedC> &det_indices) {

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
        auto amp = params(map_index,0);
        auto off_lat = params(map_index,2);
        auto off_lon = params(map_index,1);
        auto sigma_lat = params(map_index,4);
        auto sigma_lon = params(map_index,3);
        auto rot_ang = params(map_index,5);

        // rescale offsets and stddev to on-sky units
        off_lat = pixel_size_rad*(off_lat - (n_rows)/2);
        off_lon = pixel_size_rad*(off_lon - (n_cols)/2);

        sigma_lon = pixel_size_rad*sigma_lon;
        sigma_lat = pixel_size_rad*sigma_lat;

        auto cost2 = cos(rot_ang) * cos(rot_ang);
        auto sint2 = sin(rot_ang) * sin(rot_ang);
        auto sin2t = sin(2. * rot_ang);
        auto xstd2 = sigma_lon * sigma_lon;
        auto ystd2 = sigma_lat * sigma_lat;
        auto a = - 0.5 * ((cost2 / xstd2) + (sint2 / ystd2));
        auto b = - 0.5 * ((sin2t / xstd2) - (sin2t / ystd2));
        auto c = - 0.5 * ((sint2 / xstd2) + (cost2 / ystd2));

        Eigen::VectorXd gauss(n_pts);

        // make gaussian
        for (Eigen::Index j=0; j<n_pts; j++) {
            gauss(j) = amp*exp(pow(lat(j) - off_lat, 2) * a +
                                 (lon(j) - off_lon) * (lat(j) - off_lat) * b +
                                 pow(lon(j) - off_lon, 2) * c);
        }

        if (!gauss.array().isNaN().any()) {
            // add gaussian to detector scan
            in.scans.data.col(i) = in.scans.data.col(i).array() + gauss.array();
        }
    }
}

} // namespace timestream
