#pragma once

#include <string>

namespace citlali::pipeline {

template <class Hdu>
void add_image_unit_description_keys(Hdu &hdu, const std::string &unit,
                                     const std::string &description) {
    hdu.addKey("UNIT", unit, "Unit of map");
    hdu.addKey("BUNIT", unit, "Physical unit of image values");
    hdu.addKey("DESCRIP", description, "Image product description");
}

}  // namespace citlali::pipeline
