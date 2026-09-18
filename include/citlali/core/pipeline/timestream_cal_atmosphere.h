#pragma once

#include <array>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace citlali::pipeline {

// Reduction-owned, immutable realization of the exact approved node table.
// No runtime AM/profile choice, inferred airmass law, or alpha interpolation.
class CalAtmosphereSurface {
public:
    struct Node { int array, alpha; double tau225, elevation_deg, los_tau, correction; };
    static std::shared_ptr<const CalAtmosphereSurface> frozen(int alpha = 0);
    std::optional<double> correction(int array, double tau225, double elevation_deg) const;
    const auto &nodes() const noexcept { return nodes_; }
    int alpha() const noexcept { return alpha_; }
    static constexpr std::string_view operator_id = "am12_fixed_djf25_piecewise_linear_los_tau_v1";
    static constexpr std::string_view contract_sha256 = "7a064ff768a3de4f427f1338d94ef6cb9026d248f3c3c816fc3dfc96d156e36a";
    static constexpr std::string_view nodes_sha256 = "fd688a4cd3f46585b08631bc63a562aed482feb9b24ec9ee0071b70db7eb8a5f";
    static constexpr std::string_view passband_sha256 = "5e6f38f14bcae93a29ffe8362c52b15209f51aee4e48373b23aaa5ec2f8a6433";
private:
    struct Curve { double tau; std::vector<double> elevation, ordinate, slope; };
    explicit CalAtmosphereSurface(int alpha);
    int alpha_;
    std::vector<Node> nodes_;
    std::array<std::vector<Curve>, 3> curves_;
};

} // namespace citlali::pipeline
