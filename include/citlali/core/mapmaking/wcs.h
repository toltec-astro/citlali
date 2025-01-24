#pragma once

// WCS class
class WCS {
public:
    WCS() = default;
    // WCS reference values (CRVAL)
    std::vector<double> crval = std::vector<double>(2);
    // WCS reference pixel (CRPIX)
    std::vector<double> crpix = std::vector<double>(2);
    // pixel scale (CDELT)
    std::vector<double> cdelt = std::vector<double>(2);
    // coordinate types (CTYPE)
    std::vector<std::string> ctype = std::vector<std::string>(2);
    // units of the coordinates (CUNIT)
    std::vector<std::string> cunit = std::vector<std::string>(2);
    // dimensions of the map (NAXIS)
    std::vector<double> naxis = std::vector<double>(2);
    // epoch
    double epoch;
};
