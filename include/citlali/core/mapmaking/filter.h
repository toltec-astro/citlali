#pragma once

#include <citlali/core/utils/kernels.h>

class WienerFilter {
public:
    fftw_plan forward_plan, inverse_plan;
    Eigen::MatrixXcd input_data, output_data;

    Eigen::MatrixXd filtered_map;
    Eigen::MatrixXd filter_template;
    Eigen::Map<Eigen::MatrixXd> radial_freqs;
    Eigen::Map<Eigen::VectorXd> psd, freqs;
    Eigen::MatrixXd rr, vv;
    Eigen::MatrixXd numerator, denominator;

    bool uniform_weights = true;
    int n_rows, ncols;

    template <typename DerivedA, typename DerivedB>
    WienerFilter(Eigen::DenseBase<DerivedA>& _filter_template, Eigen::DenseBase<DerivedB>& _psd, Eigen::DenseBase<DerivedB>& _freqs,
                 Eigen::DenseBase<DerivedA>& _radial_freqs, bool _run_lowpass)
        : filter_template(_filter_template), psd(_psd), freqs(_freqs), radial_freqs(_radial_freqs), run_lowpass(_run_lowpass) {

        n_rows = filter_template.rows();
        n_cols = filter_template.cols();

        input_plan.resize(n_rows, ncols);
        output_plan.resize(n_rows, ncols);

        forward_plan = fftw_plan_dft_2d(n_cols, n_rows, reinterpret_cast<fftw_complex*>(input_data.data()),
                                        reinterpret_cast<fftw_complex*>(output_data.data()), FFTW_FORWARD, FFTW_ESTIMATE);
        inverse_plan = fftw_plan_dft_2d(n_cols, n_rows, reinterpret_cast<fftw_complex*>(output_data.data()),
                                        reinterpret_cast<fftw_complex*>(input_data.data()), FFTW_BACKWARD, FFTW_ESTIMATE);

        calc_vv();
    }

    template<typename Derived>
    void run(Eigen::DenseBase<Derived>& map) {
        if (uniform_weights) {
            rr = Eigen::MatrixXd::Ones(n_rows,n_cols);
        }

        filtered_map = map;

        if (denominator.size() == 0) {
            calc_denominator();
        }
        calc_numerator();

        map = (denominator.array() > 0).select(numerator / denominator, 0.);
    }

    template<typename Derived>
    void run(Eigen::DenseBase<Derived>& map, Eigen::DenseBase<Derived>& weights) {
        uniform_weights = false;

        rr = weights.cwiseSqrt();

        filtered_map = map;

        calc_denominator();
        calc_numerator();

        map = (denominator.array() > 0).select(numerator / denominator, 0.);
        weights = denominator;
    }

    void calc_vv();

    void calc_numerator();
    void calc_denominator();

    void run() {}
};

void WienerFilter::calc_vv() {
    if (run_lowpass) {
        vv = Eigen::MatrixXd::Ones(n_rows, n_cols);
        vv /= (n_rows * n_cols);
    } else {
        vv = Eigen::MatrixXd::Zeros(n_rows, n_cols);

        Eigen::Matrix<Eigen::Index, 1, 1> n_psd;
        n_psd << psd.size();

        // make radially symmetric version of psd
        for (int row = 0; row < n_rows; ++rows) {
            for (int col = 0; col < n_cols; ++col) {
                if (radial_freqs(row, col) <= psd_freqs(n_psd - 1) && radial_freqs(row, col) >= psd_freqs(0)) {
                    // mlinterp assumes col major?
                    mlinterp::interp(n_psd.data(), 1,
                                     psd.data(),
                                     vv.data() + n_rows * col + row,
                                     psd_freq.data(),
                                     q_map.data() + n_rows * col + row);
                } else if (radial_freqs(row, col) > psd_freqs(n_psd - 1)) {
                    vv(row, col) = psd(n_psd - 1);
                } else if (radial_freqs(row, col) < psd_freqs(0)) {
                    vv(row, col) = psd(0);
                }
            }
        }
        double vv_sum = vv.sum();
        vv /= vv_sum;
    }
}

void WienerFilter::calc_numerator() {
    // d x rr
    input_data.real() = signal.cwiseProd(RR);
    input_data.imag().setZero();

    // fft(d x rr)
    fftw_execute(forward_plan);

    // fft(d x rr) / vv and normalize
    input_data = output_data.array() / vv.array() / (n_rows * n_cols);

    // ifft(fft(d x rr) / vv)
    fftw_execute(inverse_plan);

    // q = ifft(fft(d x rr) / vv) x rr
    input_data.real() = output_data.real().cwiseProd(rr);
    input_data.imag().setZero();

    // fft(q)
    fftw_execute(forward_plan);

    // copy q
    Eigen::MatrixcXd q = output_data / (n_rows * n_cols);

    // t(x)
    input_data.real() = filter_template;
    input_data.imag().setZero();

    // fft(t(x))
    fftw_execute(forward_plan);

    // fft(t(x)) x fft(q) (convolution)
    input_data = out.cwiseProd(q) / (n_rows * n_cols);;

    fftw_execute(inverse_plan);
    numerator = out.real();
}

void WienerFilter::calc_denominator() {
    if (uniform_weights) {
        input_data.real() = filter_template;
        input_data.imag().setZero();

        // fft(t(x))
        fftw_execute(forward_plan);

        // abs(fft(t)) / sum(vv)
        denominator.setConstant(ouput_data.cwiseAbs2().array() / (n_rows * n_cols));
    } else {
        denominator = Eigen::MatrixXd::Zeros(n_rows, n_cols);

        // 1 / vv
        input_data.real() = 1 / vv.array();
        input_data.imag().setZero();

        // fft(1 / vv)
        fftw_execute(inverse_plan);

        // real(z)
        Eigen::VectorXd z_flat = Eigen::Map<Eigen::VectorXd>(out.real().data(), n_rows * n_cols).abs();

        // sort in ascending order (value, index)
        std::vector<std::tuple<double, int>> z_sorted = sorter(z_flat);

        // number of loops for convergence
        n_loops = n_rows * n_cols / 100;
        bool converged = false;

        int row = 0, col = 0;

        while (row < n_rows && !converged) {
            while (col < n_cols && !converged) {
                // index in flattened array
                int index = n_rows * col + row;
                // sorted in ascending order
                int sorted_index = std::get<1>(z_sorted[(n_rows * n_cols - 1) - index]);
                // t x t(x - xd)
                input_data.real() = filter_template.cwiseProd(shift(filter_template, -sorted_index % n_rows, -sorted_index / n_rows));
                input_data.imag().setZero();

                // fft(t(x) x t(x - xd))
                fftw_execute(forward_plan);
                // copy output so it can be reused
                Eigen::MatrixXcd output_data_copy = output_data / (n_rows * n_cols);
                // rr(x) x rr(x - xd)
                input_data.real() = rr.cwiseProd(shift(rr, -sorted_index % n_rows, -sorted_index / n_rows));
                // fft(rr(x) x rr(x - xd))
                fftw_execute(forward_plan);

                // fft(t(x) x t(x - xd)) x fft(rr(x) x rr(x - xd)) (convolution)
                input_data = output_data.cwiseProd(output_data_copy) / (n_rows * n_cols);

                // G = ifft(fft(t(x) x t(x - xd)) x fft(rr(x) x rr(x - xd)))
                fftw_execute(inverse_plan);
                // d = z(x_d) x G / n
                Eigen::MatrixXd delta= z(shift_index) * output_data.real() / (n_rows * n_cols);

                // D += d
                denominator += delta;

                if ((index % 100) == 1) {
                    double max_ratio = -1;
                    // maximum of denominator
                    double threshold = 0.01 * denom.maxCoeff();

                    Eigen::ArrayXd ratios = (delta.array() / denominator.array()).abs();
                    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> mask = (denominator.array() > threshold).cast<bool>();
                    max_ratio = (ratios * mask.cast<double>()).maxCoeff();

                    //
                    if (((index >= n_loops) && (max_ratio < 0.0002)) || max_ratio < 1e-10) {
                        converged = true;
                    }
                }
                // increment row
                row++;
            }
            // increment col
            col++;
        }

        // remove small denominator values
        denominator = (denominator.array() < min_denom_value).select(0, denominator);
    }
}

// MapFilter
template <typename MapType>
class MapFilter : public PipelineComponent<MapType> {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    template <typename Derived, typename ConfigType>
    MapFilter(Eigen::DenseBase<Derived>& params_ref, Eigen::DenseBase<Derived>& uncertainties_ref, Instrument& toltec_ref,
        Telescope& telescope_ref, const ConfigType& config)
        : params(params_ref.derived()), uncertainties(uncertainties_ref.derived()), toltec(toltec_ref), telescope(telescope_ref) {

        std::vector<double> amp_limit_factors, fwhm_limit_factors;
        double bounding_box_arcsec, fitting_radius_arcsec, pix_size_arcsec;

        config.get(filter_type, std::tuple{"post_processing","map_filtering","type"});
        config.get(template_type, std::tuple{"wiener_filter","template_type"});
        config.get(run_lowpass, std::tuple{"wiener_filter","lowpass_only"});
        config.get(normalize_error, std::tuple{"post_processing","map_filtering","normalize_errors"});

        config.get(amp_limit_factors, std::tuple{"post_processing", "source_fitting", "gauss_model", "amp_limit_factors"});
        config.get(fwhm_limit_factors, std::tuple{"post_processing", "source_fitting", "gauss_model", "fwhm_limit_factors"});
        config.get(bounding_box_arcsec, std::tuple{"post_processing", "source_fitting", "bounding_box_arcsec"});
        config.get(fitting_radius_arcsec, std::tuple{"post_processing", "source_fitting", "fitting_radius_arcsec"});
        config.get(fit_theta, std::tuple{"post_processing", "source_fitting", "gauss_model", "fit_rotation_angle"});
        config.get(pix_size_arcsec, std::tuple{"mapmaking", "pixel_size_arcsec"});

        // stop if any config options were not read in
        if (config.missing_keys.empty() && config.invalid_keys.empty()) {
            bounding_box_pix = std::round(bounding_box_arcsec / pix_size_arcsec);
            fitting_radius_pix = std::round(fitting_radius_arcsec / pix_size_arcsec);

            amp_lower = (amp_limits.size() > 0 && amp_limits[0] > 0) ? amp_limits[0] : 0.2;
            amp_upper = (amp_limits.size() > 1 && amp_limits[1] > 0) ? amp_limits[1] : 1.5;
            fwhm_lower = (fwhm_limits.size() > 0 && fwhm_limits[0] > 0) ? fwhm_limits[0] : 0.1 / pix_size_arcsec;
            fwhm_upper = (fwhm_limits.size() > 1 && fwhm_limits[1] > 0) ? fwhm_limits[1] : 12.0 / pix_size_arcsec;
        }
    }

    void init() {}

    void process(MapType& maps) override {
        logger->info("map filter processing");

        // loop through maps and fit
        for (int i = 0; i < maps.signal.size(); ++i) {
            double init_fwhm = toltec.apt.array_fwhms.at(maps.arrays[i]);

            Eigen::MatrixXd filter_template;

            if (template_type == "kernel") {
                filter_template = symmetric_kernel(maps.kernel[i], maps.signal[i],
                                                   maps.row_coords, map.col_coords, amp_lower_factor,
                                                   amp_upper_factor, fwhm_lower_factor, fwhm_upper_factor,
                                                   init_fwhm, fit_radius);
            }
            else if (template_type == "gaussian") {
                filter_template = gaussian_template_2d(maps.rows, maps.cols, toltec.apt.array_fwhms[array]);
            }
            else if (template_type == "airy") {
                    filter_template = airy_template_2d(maps.rows, maps.cols, toltec.apt.array_fwhms[array]);
            }
            else if (template_type == "highpass") {
                filter_template.setZero(n_rows,n_cols);
                filter_template(0, 0) = 1;
            }

            WienerFilter wiener_filter(filter_template, psd, freqs, radial_psd);
            wiener_filter.run(maps.kernel[i]);
            wiener_filter.run(maps.signal[i], maps.weight[i]);

            for (int j = 0; j < maps.n_noise; ++j) {
                wiener_filter.run(maps.noise[i][j]);
            }

        }
    }

private:
    Instrument& toltec;
    Telescope& telescope;

    std::string filter_type, template_type;
    bool run_lowpass, normalize_errors;

    int bounding_box_pix;
    int fitting_radius_pix;
    bool fit_theta;

    double amp_lower_factor, amp_upper_factor;
    double fwhm_lower_factor, fwhm_upper_factor;
};
