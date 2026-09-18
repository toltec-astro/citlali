#pragma once

#include <netcdf>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace citlali::cli::detail {
// Private ingress for the currently known TEL header layout, not a scientific
// opacity evaluator. Never silently ignore a newly populated second source.
struct CalOpacityHeader {
    std::optional<double> tau225;
    std::string update_text;
};
inline CalOpacityHeader read_cal_opacity_header(const netCDF::NcFile &file) {
    auto require=[](bool ok,const std::string &message) {
        if(!ok)throw std::invalid_argument(message);
    };
    auto scalar=[&](const std::string &name)->std::optional<double> {
        const auto variable=file.getVar(name);if(variable.isNull())return {};
        require(variable.getDimCount()==0,"CAL opacity header requires scalar layout: "+name);
        double value;variable.getVar(&value);return value;
    };
    auto text=[&](const std::string &name) {
        const auto variable=file.getVar(name);if(variable.isNull())return std::string{};
        require(variable.getDimCount()==1,"CAL opacity update requires text layout: "+name);
        std::vector<char> bytes(variable.getDim(0).getSize());variable.getVar(bytes.data());
        std::string result(bytes.begin(),bytes.end());const auto nul=result.find('\0');
        if(nul!=std::string::npos)result.erase(nul);
        const auto last=result.find_last_not_of(" \t\r\n");
        if(last==std::string::npos)result.clear();else result.erase(last+1);
        return result;
    };
    for(const auto &[name,variable]:file.getVars())
        require(!name.starts_with("Data.Radiometer.") && !name.starts_with("Data.WVR."),
                "CAL WVR producer series layout requires an explicit time/validity adapter: "+name);
    const auto secondary=scalar("Header.Radiometer.Tau2");
    require(!secondary || *secondary==0.,"CAL non-placeholder second WVR value requires an explicit source-time adapter");
    require(text("Header.Radiometer.UpdateDate2").empty(),
            "CAL second WVR header reading requires an explicit source-time adapter");
    return {scalar("Header.Radiometer.Tau"),text("Header.Radiometer.UpdateDate")};
}
} // namespace citlali::cli::detail
