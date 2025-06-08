# pragma once

#include <citlali/core/utils/utils.h>
#include <citlali/core/utils/models.h>
#include <citlali/core/utils/ceres.h>

template <typename Derived>
auto find_peak(const Eigen::DenseBase<Derived> &data, double radius, int bounding_box_pix) {
    // find max of array and setup coordinates
    auto [max_y, max_x] = find_max_in_radius(data, radius);
    auto [x_lower, x_upper] = find_bounding_limits(max_x, bounding_box_pix, data.cols());
    auto [y_lower, y_upper] = find_bounding_limits(max_y, bounding_box_pix, data.rows());
    auto x_coords = Eigen::VectorXd::LinSpaced(x_upper - x_lower + 1, x_lower, x_upper);
    auto y_coords = Eigen::VectorXd::LinSpaced(y_upper - y_lower + 1, y_lower, y_upper);

    // create the grid of coordinates using the generated x and y coordinates
    auto XY = meshgrid(x_coords, y_coords);

    return std::make_tuple(max_x, max_y, x_lower, x_upper, y_lower, y_upper, std::move(XY));
}

template <typename DerivedA, typename DerivedB>
auto fit_to_gaussian(const Eigen::DenseBase<DerivedA> &signal,
                     const Eigen::DenseBase<DerivedB> &weight,
                     const double fitting_radius_pix,
                     const int bounding_box_pix,
                     const double amp_lower, const double amp_upper,
                     const double fwhm_lower, const double fwhm_upper,
                     const double init_fwhm, const bool fit_theta) {

    using model = citlali::utils::models::Gaussian2DModel;
    using citlali::utils::fitting::fit_model;

    auto sig2noise = signal.derived().binaryExpr(weight.derived(), [](double num, double denom) {
        return (denom != 0) ? (num / denom) : 0.0;
    });

    // define box around peak
    auto [max_x, max_y, x_lower, x_upper, y_lower, y_upper, XY] = find_peak(sig2noise, fitting_radius_pix, bounding_box_pix);

    // extract the subarray of signal and weight within the bounding box
    auto signal_block = signal.block(y_lower, x_lower, y_upper - y_lower + 1, x_upper - x_lower + 1);
    auto weight_block = weight.block(y_lower, x_lower, y_upper - y_lower + 1, x_upper - x_lower + 1);

    Eigen::VectorXd params_initial(model::nparams);
    Eigen::MatrixXd bounds(model::nparams, 2);
    Eigen::Matrix<bool, Eigen::Dynamic, 1> fixed_params =
        Eigen::Matrix<bool, Eigen::Dynamic, 1>::Constant(model::nparams, false);

    // set initial parameters and bounds
    params_initial << signal(max_y, max_x), max_x, max_y, init_fwhm, init_fwhm, 0;

    bounds.col(0) << amp_lower * signal(max_y, max_x), x_lower, y_lower,
        fwhm_lower * init_fwhm * FWHM_TO_STD, fwhm_lower * init_fwhm * FWHM_TO_STD, -pi / 2;
    bounds.col(1) << amp_upper * signal(max_y, max_x), x_upper, y_upper,
        fwhm_upper * init_fwhm * FWHM_TO_STD, fwhm_upper * init_fwhm * FWHM_TO_STD, pi / 2;

    // fit rotation angle?
    if (fit_theta) {
        fixed_params(model::nparams - 1) = true;
    }

    return fit_model<model>(XY.col(0), XY.col(1), signal_block, weight_block, params_initial,
                            bounds.col(0), bounds.col(1), fixed_params);
}

template <typename Derived>
auto fit_to_airy(const Eigen::DenseBase<Derived> &signal,
                 const Eigen::DenseBase<Derived> &weight,
                 const double fitting_radius_pix,
                 const int bounding_box_pix,
                 const double amp_lower, const double amp_upper,
                 const double fwhm_lower, const double fwhm_upper,
                 const double init_fwhm) {

    using model = citlali::utils::models::Airy2DModel;
    using citlali::utils::fitting::fit_model;

    // calculate S/N map
    auto sig2noise = signal.derived().binaryExpr(weight.derived(), [](double num, double denom) {
        return (denom != 0) ? (num / denom) : 0.0;
    });

    // define box around peak
    auto [max_x, max_y, x_lower, x_upper, y_lower, y_upper, XY] = find_peak(sig2noise, fitting_radius_pix, bounding_box_pix);

    // extract the subarray of signal and weight within the bounding box
    auto signal_block = signal.block(y_lower, x_lower, y_upper - y_lower + 1, x_upper - x_lower + 1);
    auto weight_block = weight.block(y_lower, x_lower, y_upper - y_lower + 1, x_upper - x_lower + 1);

    Eigen::VectorXd params_initial(model::nparams);
    Eigen::MatrixXd bounds(model::nparams, 2);
    Eigen::Matrix<bool, Eigen::Dynamic, 1> fixed_params =
        Eigen::Matrix<bool, Eigen::Dynamic, 1>::Constant(model::nparams, false);

    // set initial parameters and bounds
    params_initial << signal(max_y, max_x), max_x, max_y, init_fwhm;

    bounds.col(0) << amp_lower * signal(max_y, max_x), x_lower, y_lower,
        fwhm_lower * init_fwhm * FWHM_TO_STD;
    bounds.col(1) << amp_upper * signal(max_y, max_x), x_upper, y_upper,
        fwhm_upper * init_fwhm * FWHM_TO_STD;

    return fit_model<model>(XY.col(0), XY.col(1), signal_block, weight_block, params_initial,
                            bounds.col(0), bounds.col(1), fixed_params);
}

// Fit
template <typename MapType>
class Fit : public PipelineComponent<MapType> {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    Instrument& toltec;
    Telescope& telescope;

    int bounding_box_pix;
    int fitting_radius_pix;
    bool fit_theta;

    double amp_lower, amp_upper;
    double fwhm_lower, fwhm_upper;
    double pix_size_arcsec;

    std::string redu_type;

    template <typename ConfigType>
    Fit(Instrument& toltec_, Telescope& telescope_, ConfigType& config)
        : toltec(toltec_), telescope(telescope_) {

        // get amplitude and FWHM limit factors
        std::vector<double> amp_limits, fwhm_limits;
        double bounding_box_arcsec, fitting_radius_arcsec;

        config.get(amp_limits, std::tuple{"post_processing", "source_fitting", "gauss_model", "amp_limit_factors"});
        config.get(fwhm_limits, std::tuple{"post_processing", "source_fitting", "gauss_model", "fwhm_limit_factors"});
        config.get(bounding_box_arcsec, std::tuple{"post_processing", "source_fitting", "bounding_box_arcsec"});
        config.get(fitting_radius_arcsec, std::tuple{"post_processing", "source_fitting", "fitting_radius_arcsec"});
        config.get(fit_theta, std::tuple{"post_processing", "source_fitting", "gauss_model", "fit_rotation_angle"});
        config.get(pix_size_arcsec, std::tuple{"mapmaking", "pixel_size_arcsec"});
        config.get(redu_type, std::tuple{"runtime", "reduction_type"});

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

    void init() override {}

    void process(MapType& maps) override {
        logger->info("fit processing");

        maps.params.resize(citlali::utils::models::Gaussian2DModel::nparams, maps.n_maps);
        maps.errors.resize(citlali::utils::models::Gaussian2DModel::nparams, maps.n_maps);

        auto [in, out] = citlali::utils::threads::get_grppi_vectors(maps.signal.size());
        auto exec_mode = citlali::utils::threads::get_map_exec_mode();

        // loop through maps and fit
        grppi::map(exec_mode, in, out, [&](int i) {
            if (!converged(i)) {
                auto map_key = maps.signal[i].key;
                double init_fwhm = toltec.array_index_to_fwhm.at(map_key.array_index) / pix_size_arcsec;

                Eigen::MatrixXd weight;

                if (redu_type == "beammap") {
                    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> mask = (maps.weight[i].data.array() == 0);
                    weight.resize(maps.wcs.naxis[1], maps.wcs.naxis[0]);
                    weight.setConstant(1. / std::sqrt(flagged_variance(maps.signal[i].data, mask)));
                } else {
                    weight = maps.weight[i].data.cwiseSqrt();
                }

                auto [p, err] = fit_to_gaussian(maps.signal[i].data, weight, fitting_radius_pix, bounding_box_pix,
                                                amp_lower, amp_upper, fwhm_lower, fwhm_upper, init_fwhm,
                                                fit_theta);

                // rescale fit params from pixel to arcsec
                p(1) = pix_size_arcsec * (p(1) - (maps.wcs.naxis[0]) / 2);
                p(2) = pix_size_arcsec * (p(2) - (maps.wcs.naxis[1]) / 2);
                p(3) = STD_TO_FWHM * pix_size_arcsec*(p(3));
                p(4) = STD_TO_FWHM * pix_size_arcsec*(p(4));

                // rescale fit errors from pixel to on-sky units
                err(1) = pix_size_arcsec * err(1);
                err(2) = pix_size_arcsec * err(2);
                err(3) = STD_TO_FWHM * pix_size_arcsec * err(3);
                err(4) = STD_TO_FWHM * pix_size_arcsec * err(4);

                maps.params.col(i) = p;
                maps.errors.col(i) = err;
            }

            return 0;
        });
    }
};
