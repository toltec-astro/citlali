#pragma once

#include <citlali/core/pipeline/timestream_rtc_output_grid.h>
#include <citlali/core/pipeline/timestream_native_pointing.h>

namespace citlali::pipeline {

enum class AstRtcCoordinateCause { available, missing_detector_geometry, outside_telescope_support };

struct AstRtcDetectorGeometry {
    NativeReadoutDetectorBinding detector;
    std::string selected_apt_row;
    double x_t_arcsec, y_t_arcsec;
    int array;
};

// The bounded existing V2 radec tangent convention, evaluated on the exact
// RTC schedule by AST. Coordinates are never processed by RTC filter taps.
class AstRtcCoordinates {
public:
    struct Direction {
        double tangent_lon_deg, tangent_lat_deg, telescope_elevation_deg;
    };
    static std::shared_ptr<const AstRtcCoordinates> realize_v2(
        std::shared_ptr<const RtcOutputGrid>, NativeObservationScope,
        std::string telescope_identity, std::shared_ptr<const RawTelescopeTrajectory>,
        double center_ra_rad, double center_dec_rad,
        std::string offset_identity, std::shared_ptr<const NativePointingOffsetModel>,
        std::vector<AstRtcDetectorGeometry>);
    const auto &grid_handle() const noexcept { return grid_; }
    const auto &telescope_identity() const noexcept { return telescope_identity_; }
    const auto &offset_identity() const noexcept { return offset_identity_; }
    const auto &geometry() const noexcept { return geometry_; }
    const auto &trajectory_handle() const noexcept { return trajectory_; }
    std::optional<Direction> at(std::size_t detector, std::size_t slot) const;
    AstRtcCoordinateCause cause(std::size_t detector, std::size_t slot) const;
    std::optional<double> telescope_elevation_deg(std::size_t detector, std::size_t slot) const;
    std::size_t logical_owned_numeric_bytes() const noexcept {
        std::size_t n=0;for(const auto &d:directions_)n+=d.size()*sizeof(double);
        for(const auto &e:elevations_)n+=e.size()*sizeof(double);return n;
    }
    static constexpr std::string_view role = "SCI-AST:rtc_output_grid_coordinates@1";
    static constexpr std::string_view method = "bounded-v2-radec-gnomonic-detector-offset-v1";
private:
    std::shared_ptr<const RtcOutputGrid> grid_;
    std::string telescope_identity_, offset_identity_;
    std::shared_ptr<const RawTelescopeTrajectory> trajectory_;
    std::shared_ptr<const NativePointingOffsetModel> offsets_;
    double center_ra_, center_dec_;
    std::vector<AstRtcDetectorGeometry> geometry_;
    std::vector<Eigen::Matrix<double,Eigen::Dynamic,2>> directions_;
    std::vector<std::vector<double>> elevations_;
};

} // namespace citlali::pipeline
