#pragma once

#include <citlali/core/timestream/atmosphere_operator_nodes_generated.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace timestream {

class FixedDjf25AtmosphereOperator {
public:
    static constexpr double minimum_tau225 = 0.0;
    static constexpr double maximum_tau225 = 0.25;
    static constexpr double minimum_elevation_deg = 25.0;
    static constexpr double maximum_elevation_deg = 80.0;
    static constexpr double science_regime_maximum_tau225 = 0.15;
    static constexpr std::size_t array_count = 3;
    static constexpr std::size_t anchor_count = 6;
    static constexpr std::size_t maximum_elevation_nodes = 31;

    FixedDjf25AtmosphereOperator() {
        select_reference_spectral_index(std::nullopt);
    }

    void select_reference_spectral_index(
        std::optional<double> requested_alpha) {
        const double alpha = requested_alpha.value_or(0.0);
        if (!supported_alpha(alpha)) {
            throw std::invalid_argument(
                "calibration.reference_spectral_index_alpha must be finite and exactly one of {-1,0,2,4}");
        }
        requested_alpha_ = requested_alpha;
        effective_alpha_ = alpha;
        default_applied_ = !requested_alpha.has_value();
        prepare_selected_surface(static_cast<int>(alpha));
    }

    static bool supported_alpha(double alpha) {
        return std::isfinite(alpha)
            && (alpha == -1.0 || alpha == 0.0
                || alpha == 2.0 || alpha == 4.0);
    }

    std::optional<double> requested_alpha() const {
        return requested_alpha_;
    }

    double effective_alpha() const {
        return effective_alpha_;
    }

    bool alpha_default_applied() const {
        return default_applied_;
    }

    static constexpr std::string_view operator_id() {
        return atmosphere_nodes::operator_id;
    }

    static constexpr std::string_view contract_sha256() {
        return atmosphere_nodes::contract_sha256;
    }

    static constexpr std::string_view nodes_sha256() {
        return atmosphere_nodes::nodes_sha256;
    }

    static constexpr std::string_view passband_set_id() {
        return atmosphere_nodes::passband_set_id;
    }

    static constexpr std::string_view reference_profile_id() {
        return atmosphere_nodes::reference_profile_id;
    }

    static std::string_view quality_regime(double tau225) {
        require_tau_support(tau225);
        return tau225 <= science_regime_maximum_tau225
            ? std::string_view{"science_qualification_regime"}
            : std::string_view{"engineering_availability_regime"};
    }

    double line_of_sight_optical_depth(
        int array_index, double tau225, double elevation_deg) const {
        if (array_index < 0
            || array_index >= static_cast<int>(array_count)) {
            throw std::out_of_range("unsupported TolTEC array index");
        }
        require_tau_support(tau225);
        require_elevation_support(elevation_deg);
        if (tau225 == 0.0) {
            return 0.0;
        }

        const auto &surface = prepared_[static_cast<std::size_t>(array_index)];
        std::size_t upper = 0;
        while (upper < anchor_count && surface[upper].tau225 < tau225) {
            ++upper;
        }
        if (upper >= anchor_count) {
            throw std::domain_error("missing upper opacity anchor");
        }

        const double upper_los = evaluate_pchip(surface[upper], elevation_deg);
        double lower_tau = 0.0;
        double lower_los = 0.0;
        if (upper > 0) {
            lower_tau = surface[upper - 1].tau225;
            lower_los = evaluate_pchip(surface[upper - 1], elevation_deg);
        }
        const double width = surface[upper].tau225 - lower_tau;
        if (!std::isfinite(width) || width <= 0.0) {
            throw std::domain_error("invalid opacity anchor interval");
        }
        const double fraction = (tau225 - lower_tau) / width;
        const double los = lower_los + fraction * (upper_los - lower_los);
        if (!std::isfinite(los) || los < 0.0) {
            throw std::domain_error("invalid line-of-sight optical depth");
        }
        return los;
    }

    double transmission(
        int array_index, double tau225, double elevation_deg) const {
        const double los = line_of_sight_optical_depth(
            array_index, tau225, elevation_deg);
        const double value = std::exp(-los);
        if (!std::isfinite(value) || value <= 0.0 || value > 1.0) {
            throw std::domain_error("invalid atmospheric transmission");
        }
        return value;
    }

    double extinction_correction(
        int array_index, double tau225, double elevation_deg) const {
        const double los = line_of_sight_optical_depth(
            array_index, tau225, elevation_deg);
        const double value = std::exp(los);
        if (!std::isfinite(value) || value < 1.0) {
            throw std::domain_error("invalid atmospheric extinction correction");
        }
        return value;
    }

private:
    struct PreparedSeries {
        double tau225 = 0.0;
        std::size_t count = 0;
        std::array<double, maximum_elevation_nodes> x{};
        std::array<double, maximum_elevation_nodes> y{};
        std::array<double, maximum_elevation_nodes> derivative{};
    };

    static void require_tau_support(double tau225) {
        if (!std::isfinite(tau225)
            || tau225 < minimum_tau225 || tau225 > maximum_tau225) {
            throw std::domain_error(
                "tau225 is outside the supported closed interval [0,0.25]");
        }
    }

    static void require_elevation_support(double elevation_deg) {
        if (!std::isfinite(elevation_deg)
            || elevation_deg < minimum_elevation_deg
            || elevation_deg > maximum_elevation_deg) {
            throw std::domain_error(
                "elevation is outside the supported closed interval [25,80] deg");
        }
    }

    static double endpoint_derivative(
        double h0, double h1, double delta0, double delta1) {
        double derivative =
            ((2.0 * h0 + h1) * delta0 - h0 * delta1) / (h0 + h1);
        if (std::signbit(derivative) != std::signbit(delta0)) {
            derivative = 0.0;
        }
        else if (std::signbit(delta0) != std::signbit(delta1)
                 && std::abs(derivative) > 3.0 * std::abs(delta0)) {
            derivative = 3.0 * delta0;
        }
        return derivative;
    }

    static void prepare_pchip(PreparedSeries &series) {
        if (series.count < 2 || series.count > maximum_elevation_nodes) {
            throw std::logic_error("invalid atmosphere elevation series size");
        }
        std::array<double, maximum_elevation_nodes - 1> h{};
        std::array<double, maximum_elevation_nodes - 1> delta{};
        for (std::size_t i = 0; i + 1 < series.count; ++i) {
            h[i] = series.x[i + 1] - series.x[i];
            if (!std::isfinite(h[i]) || h[i] <= 0.0) {
                throw std::logic_error("atmosphere elevation nodes are not ordered");
            }
            delta[i] = (series.y[i + 1] - series.y[i]) / h[i];
            if (!std::isfinite(delta[i])) {
                throw std::logic_error("invalid atmosphere node slope");
            }
        }
        if (series.count == 2) {
            series.derivative[0] = delta[0];
            series.derivative[1] = delta[0];
            return;
        }
        series.derivative[0] = endpoint_derivative(
            h[0], h[1], delta[0], delta[1]);
        for (std::size_t i = 1; i + 1 < series.count; ++i) {
            if (delta[i - 1] == 0.0 || delta[i] == 0.0
                || std::signbit(delta[i - 1]) != std::signbit(delta[i])) {
                series.derivative[i] = 0.0;
                continue;
            }
            const double w1 = 2.0 * h[i] + h[i - 1];
            const double w2 = h[i] + 2.0 * h[i - 1];
            series.derivative[i] =
                (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
        }
        const std::size_t last = series.count - 1;
        series.derivative[last] = endpoint_derivative(
            h[last - 1], h[last - 2], delta[last - 1], delta[last - 2]);
    }

    static double evaluate_pchip(
        const PreparedSeries &series, double elevation_deg) {
        require_elevation_support(elevation_deg);
        const auto begin = series.x.begin();
        const auto end = begin + static_cast<std::ptrdiff_t>(series.count);
        const auto upper_it = std::lower_bound(begin, end, elevation_deg);
        if (upper_it != end && *upper_it == elevation_deg) {
            return series.y[static_cast<std::size_t>(upper_it - begin)];
        }
        if (upper_it == begin || upper_it == end) {
            throw std::domain_error("missing elevation interpolation bracket");
        }
        const std::size_t upper = static_cast<std::size_t>(upper_it - begin);
        const std::size_t lower = upper - 1;
        const double h = series.x[upper] - series.x[lower];
        const double t = (elevation_deg - series.x[lower]) / h;
        const double t2 = t * t;
        const double t3 = t2 * t;
        const double value =
            (2.0 * t3 - 3.0 * t2 + 1.0) * series.y[lower]
            + (t3 - 2.0 * t2 + t) * h * series.derivative[lower]
            + (-2.0 * t3 + 3.0 * t2) * series.y[upper]
            + (t3 - t2) * h * series.derivative[upper];
        if (!std::isfinite(value) || value <= 0.0) {
            throw std::domain_error("invalid elevation-interpolated LOS optical depth");
        }
        return value;
    }

    void prepare_selected_surface(int alpha) {
        std::array<std::size_t, array_count> counts{};
        for (const auto &descriptor : atmosphere_nodes::series) {
            if (descriptor.alpha != alpha) {
                continue;
            }
            if (descriptor.array_index < 0
                || descriptor.array_index >= static_cast<int>(array_count)) {
                throw std::logic_error("invalid generated array index");
            }
            auto &count = counts[static_cast<std::size_t>(descriptor.array_index)];
            if (count >= anchor_count
                || descriptor.count > maximum_elevation_nodes
                || descriptor.offset + descriptor.count
                    > atmosphere_nodes::elevation_deg.size()) {
                throw std::logic_error("invalid generated series descriptor");
            }
            auto &prepared = prepared_[static_cast<std::size_t>(descriptor.array_index)][count++];
            prepared = {};
            prepared.tau225 = descriptor.tau225;
            prepared.count = descriptor.count;
            for (std::size_t i = 0; i < descriptor.count; ++i) {
                prepared.x[i] = atmosphere_nodes::elevation_deg[descriptor.offset + i];
                prepared.y[i] = atmosphere_nodes::los_optical_depth[descriptor.offset + i];
            }
            prepare_pchip(prepared);
        }
        if (std::any_of(
                counts.begin(), counts.end(),
                [](std::size_t count) { return count != anchor_count; })) {
            throw std::logic_error("selected atmosphere surface is incomplete");
        }
    }

    std::optional<double> requested_alpha_;
    double effective_alpha_ = 0.0;
    bool default_applied_ = true;
    std::array<std::array<PreparedSeries, anchor_count>, array_count> prepared_{};
};

}  // namespace timestream
