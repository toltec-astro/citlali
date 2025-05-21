#pragma once

#include <iostream>
#include <vector>
#include <string>

#include <citlali/core/utils/constants.h>

struct CRVAL {};
struct CRPIX {};
struct CDELT {};
struct CTYPE {};
struct CUNIT {};
struct NAXIS {};
struct EPOCH {};

// WCS class
class WCS {
public:
    WCS() = default;
    // WCS reference values (CRVAL)
    std::vector<double> crval;
    // WCS reference pixel (CRPIX)
    std::vector<double> crpix;
    // pixel scale (CDELT)
    std::vector<double> cdelt;
    // coordinate types (CTYPE)
    std::vector<std::string> ctype;
    // units of the coordinates (CUNIT)
    std::vector<std::string> cunit;
    // dimensions of the map (NAXIS)
    std::vector<int> naxis;
    // epoch
    double epoch;

    void set_keys() {}

    template <typename Key, typename Value, typename... Rest>
    void set_keys(Key&& key, Value&& val, Rest&&... rest) {
        apply(std::forward<Key>(key), std::forward<Value>(val));
        set_keys(std::forward<Rest>(rest)...);
    }

    void set(std::string pixel_axes, double x, double y, int n_rows,
             int n_cols, double pix_size_radians, double epoch) {
        if (pixel_axes == "radec") {
            set_radec(x, y, pix_size_radians);
        } else if (pixel_axes == "altaz") {
            set_altaz(x, y, pix_size_radians);
        }
        else if (pixel_axes == "galactic") {
            set_galactic(x, y, pix_size_radians);
        }

        set_keys(
            NAXIS{}, std::vector<int>{n_cols, n_rows},
            CRPIX{}, std::vector<double>{static_cast<double>((n_cols - 1)) / 2.0 + 1.0, static_cast<double>((n_rows - 1)) / 2.0 + 1.0},
            EPOCH{}, epoch
            );

    }

    // radec coords
    void set_radec(double ra, double dec, double pix_size_radians) {
        set_keys(
            CDELT{}, std::vector<double>{pix_size_radians * RAD_TO_DEG, pix_size_radians * RAD_TO_DEG},
            CTYPE{}, std::vector<std::string>{"RA---TAN", "DEC--TAN"},
            CUNIT{}, std::vector<std::string>{"deg", "deg"},
            CRVAL{}, std::vector<double>{ra * RAD_TO_DEG, dec * RAD_TO_DEG}
            );
    }

    // altaz coords
    void set_altaz(double az, double alt, double pix_size_radians) {
        set_keys(
            CDELT{}, std::vector<double>{pix_size_radians * RAD_TO_DEG, pix_size_radians * RAD_TO_DEG},
            CTYPE{}, std::vector<std::string>{"AZOFFSET", "ELOFFSET"},
            CUNIT{}, std::vector<std::string>{"deg", "deg"},
            CRVAL{}, std::vector<double>{az * RAD_TO_ASEC, alt * RAD_TO_ASEC}
            );
    }

    //galactic coords
    void set_galactic(double l, double b, double pix_size_radians) {
        set_keys(
            CDELT{}, std::vector<double>{pix_size_radians * RAD_TO_DEG, pix_size_radians * RAD_TO_DEG},
            CTYPE{}, std::vector<std::string>{"GLON--TAN", "GLAT-TAN"},
            CUNIT{}, std::vector<std::string>{"deg", "deg"},
            CRVAL{}, std::vector<double>{l * RAD_TO_DEG, b * RAD_TO_DEG}
            );
    }

private:
    void apply(struct CRVAL tag, const std::vector<double>& val) { crval = val; }
    void apply(struct CRPIX tag, const std::vector<double>& val) { crpix = val; }
    void apply(struct CDELT tag, const std::vector<double>& val) { cdelt = val; }
    void apply(struct CTYPE tag, const std::vector<std::string>& val) { ctype = val; }
    void apply(struct CUNIT tag, const std::vector<std::string>& val) { cunit = val; }
    void apply(struct NAXIS tag, const std::vector<int>& val) { naxis = val; }
    void apply(struct EPOCH tag, double val) { epoch = val; }
};
