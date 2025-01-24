#pragma once

#include <boost/math/special_functions/bessel.hpp>

// base beam class with scalar x, y and eigen::vectorxd beam_params
struct BaseBeam {
    virtual ~BaseBeam() = default;
    virtual double calculate(double x, double y, const Eigen::VectorXd& beam_params, double max_distance) const = 0;
};

// elliptical gaussian beam
struct EllipticalGaussianBeam : public BaseBeam {
    double calculate(double x, double y, const Eigen::VectorXd& beam_params, double max_distance) const override {
        // extract beam parameters
        double std_dev_x = beam_params(0);
        double std_dev_y = beam_params(1);
        double theta = beam_params(2);
        double mean_x = beam_params(3);
        double mean_y = beam_params(4);
        double amplitude = beam_params(5);

        // calculate distance from the center
        double dx = x - mean_x;
        double dy = y - mean_y;
        double distance = std::sqrt(dx * dx + dy * dy);

        // return 0 if distance exceeds max_distance
        if (distance > max_distance) {
            return 0.0;
        }

        // rotate coordinates
        double dx_rot = dx * std::cos(theta) + dy * std::sin(theta);
        double dy_rot = -dx * std::sin(theta) + dy * std::cos(theta);

        // calculate the elliptical gaussian beam
        double exponent = -((dx_rot * dx_rot) / (2 * std_dev_x * std_dev_x)
                            + (dy_rot * dy_rot) / (2 * std_dev_y * std_dev_y));
        return amplitude*std::exp(exponent);
    }
};

// airy beam
struct AiryBeam : public BaseBeam {
    double calculate(double x, double y, const Eigen::VectorXd& beam_params, double max_distance) const override {
        double airy_scale = pi * 1.028 / beam_params(0);
        double mean_x = beam_params(1);
        double mean_y = beam_params(2);
        double amplitude = beam_params(3);

        double r = std::sqrt((x - mean_x) * (x - mean_x) + (y - mean_y) * (y - mean_y));

        // return 0 if distance exceeds max_distance
        if (r > max_distance) {
            return 0.0;
        }

        if (r == 0.0) {
            return 1.0; // airy beam has a maximum at the center
        } else {
            double kr = airy_scale * r;
            return amplitude * std::pow(2.0 * boost::math::cyl_bessel_j(1, kr) / kr, 2);
        }
    }
};

// beam class with specific calculation methods for gaussian and airy beams
class Beam {
public:
    // constructor to initialize both gaussian and airy beams
    Beam() {
        gaussian_beam_ = std::make_shared<EllipticalGaussianBeam>();
        airy_beam_ = std::make_shared<AiryBeam>();
    }

    // calculate the gaussian beam for given x, y, and beam parameters, with a max distance limit
    double calculate_gaussian(double x, double y, const Eigen::VectorXd& beam_params, double max_distance) const {
        return gaussian_beam_->calculate(x, y, beam_params, max_distance);
    }

    // calculate the airy beam for given x, y, and beam parameters, with a max distance limit
    double calculate_airy(double x, double y, const Eigen::VectorXd& beam_params, double max_distance) const {
        return airy_beam_->calculate(x, y, beam_params, max_distance);
    }

private:
    std::shared_ptr<BaseBeam> gaussian_beam_;
    std::shared_ptr<BaseBeam> airy_beam_;
};
