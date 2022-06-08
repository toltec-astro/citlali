#pragma once

#include <Eigen/Core>

#include <citlali/core/utils/constants.h>
#include <citlali/core/utils/gaussfit.h>
#include <citlali/core/utils/pointing.h>

namespace gaussfit {

class MapFitter {
public:
    enum FitMode {
        peakValue = 0,
        centerValue = 1,
        aptTable = 2,
    };

    int nparams = 6;
    double bounding_box_pix;

    double flux0;

    double fwhm0 = 10;
    double ang0 = 0;

    double flux_low = 0.5;
    double flux_high = 5.0;

    double fwhm_low = 0;
    double fwhm_high = 20;

    double ang_low = -pi/2;
    double ang_high = pi/2;

    Eigen::MatrixXd covariance;
    Eigen::VectorXd error;

    Eigen::MatrixXd::Index row0, col0;
    Eigen::VectorXd p0;
    Eigen::MatrixXd limits;
    Eigen::VectorXd x, y;

    template<FitMode fit_mode, typename DerivedA, typename DerivedB, typename C>
    void setup(Eigen::DenseBase<DerivedA> &data, Eigen::DenseBase<DerivedB> &weight, C &calib) {

        p0.resize(nparams);

        // use the peak value of the map as the starting guess
        if constexpr (fit_mode == peakValue) {
            auto sig2noise = data.derived().array()*sqrt(weight.derived().array());
            sig2noise.maxCoeff(&row0, &col0);
            flux0 = data(row0,col0);

            // set initial guess
            p0 << flux0, col0, row0, fwhm0, fwhm0, ang0;
        }

        else if constexpr (fit_mode == centerValue) {
            row0 = data.rows()/2;
            col0 = data.cols()/2;

            flux0 = data(row0, col0);

            p0 << flux0, col0, row0, fwhm0, fwhm0, ang0;
        }

        //else if constexpr (fit_mode == aptTable) {

        //}


        // get the maximum bounding box size that is contained in the map
        if ((row0 - bounding_box_pix) < 0) {
            bounding_box_pix = row0;
        }
        if ((row0 + bounding_box_pix + 1) >= data.rows()) {
            bounding_box_pix = data.rows() - row0 - 1;
        }
        if ((col0 - bounding_box_pix) < 0) {
            bounding_box_pix = col0;
        }
        if ((col0 + bounding_box_pix + 1) >= data.cols()) {
            bounding_box_pix = data.cols() - col0 - 1;
        }

        SPDLOG_INFO("bounding_box_pix {}", bounding_box_pix);

        // get limits
        limits.resize(nparams, 2);
        limits.col(0) << flux_low*flux0, col0 - bounding_box_pix, row0 - bounding_box_pix, fwhm_low,
                fwhm_low, ang_low;
        limits.col(1) << flux_high*flux0, col0 + bounding_box_pix + 1, row0 + bounding_box_pix + 1,
                fwhm_high, fwhm_high, ang_high;

        SPDLOG_INFO("param limits {}", limits);
        SPDLOG_INFO("param init values {}",p0);

        // axes coordinate vectors for meshgrid
        x = Eigen::VectorXd::LinSpaced(2*bounding_box_pix+1, col0-bounding_box_pix, col0+bounding_box_pix);
        y = Eigen::VectorXd::LinSpaced(2*bounding_box_pix+1, row0-bounding_box_pix, row0+bounding_box_pix);
    }

    template <FitMode fit_mode, typename DerivedA, typename DerivedB, typename C>
    auto fit(Eigen::DenseBase<DerivedA> &data, Eigen::DenseBase<DerivedB> &weight, C &calib_data) {

        // get bounding initial conditions, bounding box, and limits
        setup<fit_mode>(data, weight, calib_data);

        // setup gauss model with initial conditions
        auto g = gaussfit::modelgen<gaussfit::Gaussian2D>(p0);
        auto _p = g.params;

        // meshgrid for coordinates
        auto xy = g.meshgrid(x, y);

        // calculate sigma matrix
        Eigen::MatrixXd sigma = weight;
        // set 1/weight=0 to 0
        //sigma = (weight.derived().array() !=0).select(1./sqrt(weight.derived().array()),0.);

        for (Eigen::Index nr=0; nr<weight.rows(); nr++) {
            for (Eigen::Index nc=0; nc<weight.cols(); nc++) {
                if (weight(nr,nc)!=0) {
                    sigma(nr,nc) = 1./sqrt(weight(nr,nc));
                }
                else {
                    sigma(nr,nc) = 0;
                }
            }
        }

        // copy data and sigma within bounding box region
        auto _data = data.block(row0-bounding_box_pix, col0-bounding_box_pix, 2*bounding_box_pix+1, 2*bounding_box_pix+1);
        auto _sigma = sigma.block(row0-bounding_box_pix, col0-bounding_box_pix, 2*bounding_box_pix+1, 2*bounding_box_pix+1);

        // do the fit with ceres-solver
        auto [g_fit, covariance] = gaussfit::curvefit_ceres(g, _p, xy, _data, _sigma, limits);

        error = covariance.diagonal().cwiseSqrt();
        SPDLOG_INFO("source fit error {}", error);

        // return the parameters
        return g_fit.params;
    }
};

} // namespace

template <class Engine, typename Derived, typename tel_meta_t>
void add_gaussian(Engine engine, Eigen::DenseBase<Derived> &scan, tel_meta_t &tel_meta_data) {

    // loop through detectors
    for (Eigen::Index d = 0; d < scan.cols(); d++) {
        double azoff, eloff;

        // use apt table for science and pointing mode offsets
        if (engine->map_grouping == "array_name" || engine->map_grouping == "pointing") {
            azoff = engine->calib_data["x_t"](d);
            eloff = engine->calib_data["y_t"](d);
        }

        // set offsets to 0 for beammap mode
        else if (engine->map_grouping == "beammap") {
            azoff = 0;
            eloff = 0;
        }

        // get detector pointing (lat/lon = rows/cols -> dec/ra or el/az)
        auto [lat, lon] = engine_utils::get_det_pointing(tel_meta_data, azoff, eloff, engine->map_type, engine->pointing_offsets);

        // get parameters for current detector
        auto amplitude = engine->mb.pfit(0,d);
        auto off_lat = engine->mb.pfit(2,d);
        auto off_lon = engine->mb.pfit(1,d);
        auto sigma_lat = engine->mb.pfit(4,d);
        auto sigma_lon = engine->mb.pfit(3,d);

        // rescale offsets and stddev to on-sky units
        off_lat = engine->pixel_size*(off_lat - (engine->mb.nrows)/2);
        off_lon = engine->pixel_size*(off_lon - (engine->mb.ncols)/2);

        sigma_lon = engine->pixel_size*sigma_lon;
        sigma_lat = engine->pixel_size*sigma_lat;

        // calculate gaussian
        auto gauss = amplitude*exp(-0.5*(pow(lat.array() - off_lat, 2) / (pow(sigma_lat,2))
                                        + pow(lon.array() - off_lon, 2) / (pow(sigma_lon,2))));

        // add gaussian to detector scan
        scan.col(d) = scan.col(d).array() + gauss;

        // check speed vs vectorized routine
        /*for (Eigen::Index i=0; i<scan.rows(); i++) {
            double gauss = amplitude*exp(-0.5*(pow(lat(i) - off_lat, 2) / (pow(sigma_lat,2))
                                              + pow(lon(i) - off_lon, 2) / (pow(sigma_lon,2))));

            scan(i,d) = scan(i,d) + gauss;
        }*/
    }
}

template <class Engine, typename Derived, typename tel_meta_t>
void add_gaussian_2(Engine engine, Eigen::DenseBase<Derived> &scan, tel_meta_t &tel_meta_data) {

    // loop through detectors
    for (Eigen::Index d = 0; d < scan.cols(); d++) {
        double azoff, eloff;

        // use apt table for science and pointing mode offsets
        if (engine->map_grouping == "array_name" || engine->map_grouping == "pointing") {
            azoff = engine->calib_data["x_t"](d);
            eloff = engine->calib_data["y_t"](d);
        }

        // set offsets to 0 for beammap mode
        else if (engine->map_grouping == "beammap") {
            azoff = 0;
            eloff = 0;
        }

        // get detector pointing (lat/lon = rows/cols -> dec/ra or el/az)
        auto [lat, lon] = engine_utils::get_det_pointing(tel_meta_data, azoff, eloff, engine->map_type, engine->pointing_offsets);

        // get parameters for current detector
        auto amp = engine->mb.pfit(0,d);
        auto off_lat = engine->mb.pfit(2,d);
        auto off_lon = engine->mb.pfit(1,d);
        auto sigma_lat = engine->mb.pfit(4,d);
        auto sigma_lon = engine->mb.pfit(3,d);
        auto rot_ang = engine->mb.pfit(5,d);

        SPDLOG_INFO("amp {}, off_lat {}, off_lon {}, sigma_lat {}, sigma_lon {}, rot_ang {}", amp, off_lat, off_lon, sigma_lat, sigma_lon,
                    rot_ang);


        // rescale offsets and stddev to on-sky units
        off_lat = engine->pixel_size*(off_lat - (engine->mb.nrows)/2);
        off_lon = engine->pixel_size*(off_lon - (engine->mb.ncols)/2);

        SPDLOG_INFO("off_lat {}, off_lon {}", off_lat, off_lon);

        sigma_lon = engine->pixel_size*sigma_lon;
        sigma_lat = engine->pixel_size*sigma_lat;

        SPDLOG_INFO("sigma_lat {}, sigma_lon {}", sigma_lat, sigma_lon);


        // calculate gaussian
        //auto gauss = amplitude*exp(-0.5*(pow(lat.array() - off_lat, 2) / (pow(sigma_lat,2))
        //                                     + pow(lon.array() - off_lon, 2) / (pow(sigma_lon,2))));


        auto cost2 = cos(rot_ang) * cos(rot_ang);
        auto sint2 = sin(rot_ang) * sin(rot_ang);
        auto sin2t = sin(2. * rot_ang);
        auto xstd2 = sigma_lon * sigma_lon;
        auto ystd2 = sigma_lat * sigma_lat;
        auto a = - 0.5 * ((cost2 / xstd2) + (sint2 / ystd2));
        auto b = - 0.5 * ((sin2t / xstd2) - (sin2t / ystd2));
        auto c = - 0.5 * ((sint2 / xstd2) + (cost2 / ystd2));

        Eigen::VectorXd gauss(scan.rows());

        for (Eigen::Index i=0; i<scan.rows(); i++) {
            gauss(i) = amp*exp(pow(lat(i) - off_lat, 2) * a +
                           (lon(i) - off_lon) * (lat(i) - off_lat) * b +
                           pow(lon(i) - off_lon, 2) * c);
        }

        // add gaussian to detector scan
        scan.col(d) = scan.col(d).array() + gauss.array();

         // check speed vs vectorized routine
        /*for (Eigen::Index i=0; i<scan.rows(); i++) {
            double gauss = amplitude*exp(-0.5*(pow(lat(i) - off_lat, 2) / (pow(sigma_lat,2))
                                              + pow(lon(i) - off_lon, 2) / (pow(sigma_lon,2))));

        scan(i,d) = scan(i,d) + gauss;
    }*/
}
}
