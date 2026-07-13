#pragma once

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/config/post_processing_config.h>
#include <citlali/core/config/runtime_config.h>
#include <citlali/core/pipeline/config_parse_tracking.h>

#include <string>
#include <tuple>

namespace citlali::pipeline {

template <class Config, class Diagnostics>
void read_mapmaking_output_request_config(
    Config &config, citlali::config::MapmakingConfig &mapmaking,
    citlali::config::PostProcessingConfig &post_processing,
    Diagnostics &diagnostics) {
    read_config_value(
        config, mapmaking.coverage_cut, diagnostics,
        std::tuple{"mapmaking", "coverage_cut"});
    read_config_value(
        config, post_processing.map_histogram_n_bins, diagnostics,
        std::tuple{"post_processing", "map_histogram_n_bins"}, {}, {0});
    read_config_value(
        config, mapmaking.pixel_size_arcsec, diagnostics,
        std::tuple{"mapmaking", "pixel_size_arcsec"}, {}, {0.0});
    read_config_value(
        config, mapmaking.unit, diagnostics,
        std::tuple{"mapmaking", "cunit"},
        {"mJy/beam", "MJy/sr", "uK", "Jy/pixel"});
    read_config_value(
        config, mapmaking.x_size_pix, diagnostics,
        std::tuple{"mapmaking", "x_size_pix"});
    read_config_value(
        config, mapmaking.y_size_pix, diagnostics,
        std::tuple{"mapmaking", "y_size_pix"});
    read_config_value(
        config, mapmaking.crpix1, diagnostics,
        std::tuple{"mapmaking", "crpix1"});
    read_config_value(
        config, mapmaking.crpix2, diagnostics,
        std::tuple{"mapmaking", "crpix2"});
    read_config_value(
        config, mapmaking.crval1_j2000, diagnostics,
        std::tuple{"mapmaking", "crval1_J2000"});
    read_config_value(
        config, mapmaking.crval2_j2000, diagnostics,
        std::tuple{"mapmaking", "crval2_J2000"});
    read_config_value(
        config, mapmaking.tan_ra, diagnostics,
        std::tuple{"mapmaking", "tan_ra"});
    read_config_value(
        config, mapmaking.tan_dec, diagnostics,
        std::tuple{"mapmaking", "tan_dec"});
}

template <class OutputMapBlock>
void adapt_mapmaking_output_config_one_way(
    const citlali::config::MapmakingConfig &source,
    const citlali::config::PostProcessingConfig &post_processing,
    citlali::config::MapPixelAxes pixel_axes,
    citlali::config::ReductionType reduction_type,
    double arcsec_to_rad, double rad_to_deg, double rad_to_arcsec,
    OutputMapBlock &target) {
    target.cov_cut = source.coverage_cut;
    target.hist_n_bins = post_processing.map_histogram_n_bins;
    target.pixel_size_rad = source.pixel_size_arcsec;
    target.pixel_size_rad *= arcsec_to_rad;
    target.sig_unit = source.unit;

    target.wcs.cdelt.clear();
    target.wcs.naxis.clear();
    target.wcs.crpix.clear();
    target.wcs.crval.clear();
    target.wcs.cunit.clear();
    target.wcs.ctype.clear();
    target.crval_config.clear();

    target.wcs.cdelt.push_back(-target.pixel_size_rad);
    target.wcs.cdelt.push_back(target.pixel_size_rad);
    target.wcs.naxis.push_back(source.x_size_pix);
    target.wcs.naxis.push_back(source.y_size_pix);
    target.wcs.crpix.push_back(source.crpix1);
    target.wcs.crpix.push_back(source.crpix2);
    target.crval_config.push_back(source.crval1_j2000);
    target.crval_config.push_back(source.crval2_j2000);

    if (citlali::config::is_radec_map_pixel_axes(pixel_axes)) {
        target.wcs.ctype.insert(
            target.wcs.ctype.end(), {"RA---TAN", "DEC--TAN"});
        target.wcs.cunit.insert(
            target.wcs.cunit.end(), {"deg", "deg"});
        target.wcs.cdelt[0] *= rad_to_deg;
        target.wcs.cdelt[1] *= rad_to_deg;
    } else if (citlali::config::is_altaz_map_pixel_axes(pixel_axes)) {
        target.wcs.ctype.insert(
            target.wcs.ctype.end(), {"AZOFFSET", "ELOFFSET"});
        if (citlali::config::is_science_reduction_type(reduction_type)) {
            target.wcs.cunit.insert(
                target.wcs.cunit.end(), {"deg", "deg"});
            target.wcs.cdelt[0] *= rad_to_deg;
            target.wcs.cdelt[1] *= rad_to_deg;
        } else {
            target.wcs.cunit.insert(
                target.wcs.cunit.end(), {"arcsec", "arcsec"});
            target.wcs.cdelt[0] *= rad_to_arcsec;
            target.wcs.cdelt[1] *= rad_to_arcsec;
        }
    } else if (citlali::config::is_galactic_map_pixel_axes(pixel_axes)) {
        target.wcs.ctype.insert(
            target.wcs.ctype.end(), {"GLON-TAN", "GLAT-TAN"});
        target.wcs.cunit.insert(
            target.wcs.cunit.end(), {"deg", "deg"});
        target.wcs.cdelt[0] *= rad_to_deg;
        target.wcs.cdelt[1] *= rad_to_deg;
    }

    target.wcs.cdelt.insert(target.wcs.cdelt.end(), {1, 1});
    target.wcs.crpix.insert(target.wcs.crpix.end(), {0, 0});
    target.wcs.crval.resize(4, 0.0);
    target.wcs.naxis.insert(target.wcs.naxis.end(), {1, 1});
    target.wcs.ctype.insert(
        target.wcs.ctype.end(), {"FREQ", "STOKES"});
    target.wcs.cunit.insert(target.wcs.cunit.end(), {"Hz", ""});
}

}  // namespace citlali::pipeline
