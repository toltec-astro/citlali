#pragma once

#include <tula/logging.h>
#include <tula/nc.h>
#include <tula/algorithm/ei_stats.h>

#include <citlali/core/utils/utils.h>
#include <citlali/core/utils/pointing.h>


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

    template <typename apt_type, class tel_type>
    void calc_weights(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, apt_type &, tel_type &);

    template <typename apt_t, typename Derived>
    void remove_flagged_dets(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, apt_t &, Eigen::DenseBase<Derived> &);

    template <typename calib_t, typename Derived>
    auto remove_bad_dets_nw(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, calib_t &, Eigen::DenseBase<Derived> &,
                            Eigen::DenseBase<Derived> &, Eigen::DenseBase<Derived> &);

    template <typename apt_t, typename Derived>
    void remove_bad_dets(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, apt_t &, Eigen::DenseBase<Derived> &);

    template <typename Derived, typename apt_t, typename pointing_offset_t>
    void append_to_netcdf(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, std::string, std::string, std::string &,
                          pointing_offset_t &, Eigen::DenseBase<Derived> &, apt_t &, std::string, bool, double);
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
            SPDLOG_INFO("array_limits {}", calib.array_limits);
        }

        for (auto const& [key, val] : grouping_limits) {
            // starting index
            auto start_index = std::get<0>(val);
            // size of block for each grouping
            auto n_dets = std::get<1>(val) - std::get<0>(val);

            SPDLOG_INFO("start_index {} ndets {}", start_index, n_dets);

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

            SPDLOG_INFO("in_scans {}", in_scans);
            SPDLOG_INFO("in_flags {}", in_flags);

            auto [evals, evecs] = cleaner.calc_eig_values<SpectraBackend>(in_scans, in_flags);
            cleaner.remove_eig_values(in_scans, in_flags, evals, evecs, out_scans);

            if (in.kernel.data.size()!=0) {
                SPDLOG_INFO("cleaning kernel");
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
    }
}

template <typename apt_type, class tel_type>
void PTCProc::calc_weights(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, apt_type &apt, tel_type &telescope) {
    if (weighting_type == "approximate") {
        in.weights.data = pow(sqrt(telescope.d_fsmp)*apt["sens"].array(),-2.0);
    }

    else if (weighting_type == "full"){
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
    SPDLOG_INFO("removing {} flagged detectors",(apt["flag"].array()==0).count());

    for (Eigen::Index i=0; i<n_dets; i++) {
        Eigen::Index det_index = det_indices(i);
        if (!apt["flag"](det_index)) {
            in.flags.data.col(i).setZero();
        }
    }
}

template <typename calib_t, typename Derived>
auto PTCProc::remove_bad_dets_nw(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, calib_t &calib, Eigen::DenseBase<Derived> &det_indices,
                                 Eigen::DenseBase<Derived> &nw_indices, Eigen::DenseBase<Derived> &array_indices) {

    Eigen::Index n_dets = in.scans.data.cols();

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
            if (calib.apt["flag"](det_index) && calib.apt["nw"](det_index)==calib.nws(i)) {
                if ((det_std_dev(j) < (lower_std_dev*mean_std_dev)) && lower_std_dev!=0) {
                    in.flags.data.col(dets(j)).setZero();
                    n_low_dets++;
                }

                if ((det_std_dev(j) > (upper_std_dev*mean_std_dev)) && upper_std_dev!=0) {
                    in.flags.data.col(dets(j)).setZero();
                    n_high_dets++;
                }
            }
        }
        SPDLOG_INFO("{}/{} dets below limit. {}/{} dets above limit.", n_low_dets, n_good_dets, n_high_dets, n_good_dets);
    }

    TCData<TCDataKind::PTC, Eigen::MatrixXd> out = in;

    Eigen::Index n_good_dets = 0;
    for (Eigen::Index i=0; i<n_dets; i++) {
        if ((in.flags.data.col(i).array()!=0).all()) {
            n_good_dets++;
        }
    }

    out.scans.data.resize(in.scans.data.rows(), n_good_dets);
    out.flags.data.resize(in.flags.data.rows(), n_good_dets);

    if (in.kernel.data.size()!=0) {
        out.kernel.data.resize(in.kernel.data.rows(), n_good_dets);
    }

    calib_t calib_temp;

    Eigen::VectorXI array_indices_temp(n_good_dets), nw_indices_temp(n_good_dets), det_indices_temp(n_good_dets);

    Eigen::Index j = 0;
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
    }

    /*det_indices = det_indices_temp;
    nw_indices = nw_indices_temp;
    array_indices = array_indices_temp;

    in = out;*/

    for (auto const& [key, val]: calib.apt) {
        calib_temp.apt[key].setZero(n_good_dets);
        Eigen::Index i = 0;
        for (Eigen::Index j=0; j<calib.apt["nw"].size(); j++) {
            if ((in.flags.data.col(j).array()!=0).all()) {
                calib_temp.apt[key](i) = calib.apt[key](j);
                i++;
            }
        }
    }

    calib_temp.setup();

    return std::move(calib_temp);
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

    Eigen::VectorXd det_std_dev(n_good_dets);
    Eigen::VectorXI dets(n_good_dets);
    Eigen::Index j = 0;

    // collect standard deviation from good detectors
    for (Eigen::Index i=0; i<n_dets; i++) {
        Eigen::Index det_index = det_indices(i);
        if (apt["flag"](det_index)) {
            //good_weights(j) = in.weights.data(i);

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
    //double mean_std_dev = tula::alg::median(det_std_dev);

    int n_low_dets = 0;
    int n_high_dets = 0;

    // loop through good detectors and flag those that have std devs beyond the limits
    for (Eigen::Index i=0; i<n_good_dets; i++) {
        Eigen::Index det_index = det_indices(dets(i));
        if (apt["flag"](det_index)) {
            if ((det_std_dev(i) < (lower_std_dev*mean_std_dev)) && lower_std_dev!=0) {
                in.flags.data.col(dets(i)).setZero();
                n_low_dets++;
            }

            if ((det_std_dev(i) > (upper_std_dev*mean_std_dev)) && upper_std_dev!=0) {
                in.flags.data.col(dets(i)).setZero();
                n_high_dets++;
            }
        }
    }

    SPDLOG_INFO("{}/{} dets below limit. {}/{} dets above limit.", n_low_dets, n_good_dets, n_high_dets, n_good_dets);
}

template <typename Derived, typename apt_t, typename pointing_offset_t>
void PTCProc::append_to_netcdf(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, std::string filepath, std::string redu_type,
                               std::string &pixel_axes, pointing_offset_t &pointing_offsets_arcsec, Eigen::DenseBase<Derived> &det_indices,
                               apt_t &apt, std::string tod_output_type,bool verbose_mode, double fsmp) {

    Eigen::MatrixXd lats(in.scans.data.rows(), in.scans.data.cols()), lons(in.scans.data.rows(), in.scans.data.cols());
    for (Eigen::Index i=0; i<in.scans.data.cols(); i++) {
        // skip completely flagged detectors
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

        // if in verbose mode
        if (verbose_mode) {
            // number of samples
            unsigned long n_pts = in.scans.data.rows();

            // make sure its even
            if (n_pts % 2 == 1) {
                n_pts--;
            }

            // containers for frequency domain
            Eigen::Index n_freqs = n_pts / 2 + 1; // number of one sided freq bins
            double d_freq = fsmp / n_pts;

            // number of bins dimension
            NcDim n_hist_dim = fo.getDim("n_hist_bins");
            // get number of bins
            auto n_hist_bins = n_hist_dim.getSize();

            std::vector<netCDF::NcDim> hist_dims = {n_dets_dim, n_hist_dim};

            // histogram variable
            netCDF::NcVar hist_var = fo.addVar("scan_hist_"+std::to_string(in.index.data),netCDF::ncDouble, hist_dims);
            // histogram bins variable
            NcVar hist_bin_var = fo.addVar("scan_hist_bins_"+std::to_string(in.index.data),netCDF::ncDouble, hist_dims);

            // add psd variable
            NcDim n_psd_dim = fo.addDim("scan_n_psd_"+std::to_string(in.index.data), n_freqs);
            std::vector<netCDF::NcDim> psd_dims = {n_dets_dim, n_psd_dim};

            // psd variable
            netCDF::NcVar psd_var = fo.addVar("scan_psd_"+std::to_string(in.index.data),netCDF::ncDouble, psd_dims);
            psd_var.putAtt("Units","V * s^(1/2)");
            // psd freq variable
            NcVar psd_freq_var = fo.addVar("scan_psd_freq_"+std::to_string(in.index.data),netCDF::ncDouble, n_psd_dim);
            psd_freq_var.putAtt("Units","Hz");

            Eigen::VectorXd psd_freq = d_freq * Eigen::VectorXd::LinSpaced(n_freqs, 0, n_pts / 2);
            psd_freq_var.putVar(psd_freq.data());

            // start index for hist
            std::vector<std::size_t> start_index_hist = {0,0};
            // size for hist
            std::vector<std::size_t> size_hist = {1,n_hist_bins};

            // size for psd
            std::vector<std::size_t> size_psd = {1,TULA_SIZET(n_freqs)};

            // loop through detectors
            for (Eigen::Index i=0; i<in.scans.data.cols(); i++) {
                // increment starting row
                start_index_hist[0] = i;
                // get data for detector
                Eigen::VectorXd scan = in.scans.data.col(i);
                // calculate histogram
                auto [h, h_bins] = engine_utils::calc_hist(scan, n_hist_bins);
                // add data to histogram variable
                hist_var.putVar(start_index_hist, size_hist, h.data());
                // add data to histogram bins variable
                hist_bin_var.putVar(start_index_hist, size_hist, h_bins.data());

                // do fft
                Eigen::FFT<double> fft;
                fft.SetFlag(Eigen::FFT<double>::HalfSpectrum);
                fft.SetFlag(Eigen::FFT<double>::Unscaled);

                // vector to hold fft data
                Eigen::VectorXcd freqdata;

                // do fft
                fft.fwd(freqdata, scan.head(n_pts));
                // calc psd
                Eigen::VectorXd psd = freqdata.cwiseAbs2() / d_freq;
                // account for negative freqs
                psd.segment(1, n_freqs - 2) *= 2.;

                // put detector's psd into variable
                psd_var.putVar(start_index_hist, size_psd, psd.data());
            }
        }

        fo.sync();
        fo.close();

    } catch (NcException &e) {
        SPDLOG_ERROR("{}", e.what());
    }
}

} // namespace timestream
