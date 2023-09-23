#pragma once

#include <string>
#include <map>

#include <Eigen/Core>

#include <citlali/core/utils/utils.h>

#include <citlali/core/timestream/timestream.h>

namespace engine {

class Diagnostics {
public:
    std::map<std::string, Eigen::MatrixXd> stats;

    // adc snap data
    std::vector<Eigen::Matrix<short, Eigen::Dynamic, Eigen::Dynamic>> adc_snap_data;

    // sample rate
    double fsmp;

    // write evals?
    bool write_evals;

    // number of histogram bins
    int n_hist_bins = 50;

    // header for det stats data
    std::vector<std::string> det_stats_header = {
        "rms",
        "stddev",
        "median",
        "flagged_frac",
        "weights",
    };

    // header for group stats data
    std::vector<std::string> grp_stats_header = {
        "median_weights"
    };

    // store eigenvalues
    std::map<Eigen::Index, std::vector<std::vector<Eigen::VectorXd>>> evals;

    // calc basic stats
    template <timestream::TCDataKind tcdata_kind>
    void calc_stats(timestream::TCData<tcdata_kind, Eigen::MatrixXd> &);

    // calc detector histograms
    template <timestream::TCDataKind tcdata_kind>
    void calc_tod_hist(timestream::TCData<tcdata_kind, Eigen::MatrixXd> &);

    // calc detector psds
    template <timestream::TCDataKind tcdata_kind>
    void calc_tod_psd(timestream::TCData<tcdata_kind, Eigen::MatrixXd> &);
};

template <timestream::TCDataKind tcdata_kind>
void Diagnostics::calc_stats(timestream::TCData<tcdata_kind, Eigen::MatrixXd> &in) {
    Eigen::Index n_pts = in.scans.data.rows();
    Eigen::Index n_dets = in.scans.data.cols();

    for (Eigen::Index i=0; i<n_dets; i++) {
        // make Eigen::Maps for each detector's scan
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> scans(
            in.scans.data.col(i).data(), in.scans.data.rows());
        Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> flags(
            in.flags.data.col(i).data(), in.flags.data.rows());
        // calc rms
        stats["rms"](i,in.index.data) = engine_utils::calc_rms(scans);
        // calc stddev
        stats["stddev"](i,in.index.data) = engine_utils::calc_std_dev(scans);
        // calc median
        stats["median"](i,in.index.data) = tula::alg::median(scans);
        // fraction of detectors that were flagged
        stats["flagged_frac"](i,in.index.data) = flags.cast<double>().sum()/static_cast<double>(n_pts);
    }

    if (in.weights.data.size()!=0) {
        // add weights
        stats["weights"].col(in.index.data) = in.weights.data;
        //stats["median_weights"].col(in.index.data) = Eigen::Map<Eigen::VectorXd>(in.median_weights.data(),in.median_weights.size());
    }

    // add eigenvalues
    if (write_evals) {
        evals[in.index.data] = in.evals.data;
    }
}

template <timestream::TCDataKind tcdata_kind>
void Diagnostics::calc_tod_hist(timestream::TCData<tcdata_kind, Eigen::MatrixXd> &in) {
    for (Eigen::Index i=0; i<in.scans.data.cols(); i++) {
        // get data for detector
        Eigen::VectorXd scan = in.scans.data.col(i);
        // calculate histogram
        auto [h, h_bins] = engine_utils::calc_hist(scan, n_hist_bins);
    }
}

template <timestream::TCDataKind tcdata_kind>
void Diagnostics::calc_tod_psd(timestream::TCData<tcdata_kind, Eigen::MatrixXd> &in) {
    /*
    // number of samples
    unsigned long n_pts = in.scans.data.rows();

    // make sure its even
    if (n_pts % 2 == 1) {
        n_pts--;
    }

    // containers for frequency domain
    Eigen::Index n_freqs = n_pts / 2 + 1; // number of one sided freq bins
    double d_freq = telescope.d_fsmp / n_pts;

    Eigen::VectorXd psd_freq = d_freq * Eigen::VectorXd::LinSpaced(n_freqs, 0, n_pts / 2);

    // hanning window
    Eigen::VectorXd hanning = (0.5 - 0.5 * Eigen::ArrayXd::LinSpaced(n_pts, 0, 2.0 * pi / n_pts * (n_pts - 1)).cos());

    // loop through detectors
    for (Eigen::Index i=0; i<in.scans.data.cols(); i++) {

        // get data for detector
        Eigen::VectorXd scan = in.scans.data.col(i).array()*hanning.array();

        //scan.noalias() = (scan.array()*hanning.array()).matrix();

        // setup fft
        // set up fftw
        fftw_complex *a;
        fftw_complex *b;
        fftw_plan pf, pr;

        // allocate space for 2d ffts
        a = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n_rows*n_cols);
        b = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n_rows*n_cols);

        // fftw plans
        pf = fftw_plan_dft_2d(n_rows, n_cols, a, b, FFTW_FORWARD, FFTW_ESTIMATE);
        pr = fftw_plan_dft_2d(n_rows, n_cols, a, b, FFTW_BACKWARD, FFTW_ESTIMATE);

        // vector to hold fft data
        Eigen::VectorXcd freqdata;

        // do fft
        fft.fwd(freqdata, scan.head(n_pts));
        // calc psd
        Eigen::VectorXd psd = freqdata.cwiseAbs2() / d_freq / n_pts / telescope.d_fsmp;
        // account for negative freqs
        psd.segment(1, n_freqs - 2) *= 2.;

        Eigen::VectorXd smoothed_psd(psd.size());
        engine_utils::smooth<engine_utils::SmoothType::edge_truncate>(psd, smoothed_psd, omb.smooth_window);
        psd = std::move(smoothed_psd);

    }*/
}
} // namespace engine
