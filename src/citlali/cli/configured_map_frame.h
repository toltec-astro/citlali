#pragma once
#include <citlali/core/pipeline/ast_rtc_coordinates.h>
#include <yaml-cpp/yaml.h>

namespace citlali::cli::detail {
inline const char *configured_frame_name(bool science) {
    return science ? "J2000-radec-gnomonic-tangent" : "delta-source-AltAz";
}
// Boundary serialization only. AST owns coordinate evaluation and grid values.
inline YAML::Node configured_frame_record(const pipeline::AstRtcTanGrid &wcs,bool science) {
    YAML::Node frame;
    frame["ASTFRAME"]=configured_frame_name(science);
    frame["CTYPE1"]=science?"RA---TAN":"AZOFFSET";
    frame["CTYPE2"]=science?"DEC--TAN":"ELOFFSET";
    frame["CUNIT1"]=science?"deg":"arcsec";frame["CUNIT2"]=science?"deg":"arcsec";
    frame["CRPIX1"]=wcs.crpix1;frame["CRPIX2"]=wcs.crpix2;
    frame["CDELT1"]=(science?1.:3600.)*wcs.cdelt1_deg;
    frame["CDELT2"]=(science?1.:3600.)*wcs.cdelt2_deg;
    frame["CRVAL1"]=wcs.crval1_deg;frame["CRVAL2"]=wcs.crval2_deg;
    frame["rows"]=wcs.rows;frame["columns"]=wcs.columns;
    if(science) { frame["RADESYS"]="FK5";frame["EQUINOX"]=2000.; }
    return frame;
}
}
