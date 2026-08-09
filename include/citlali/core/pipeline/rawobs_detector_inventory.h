#pragma once

#include <citlali/core/utils/netcdf_io.h>
#include <citlali/core/utils/sha256.h>

#include <fmt/core.h>
#include <netcdf>
#include <tula/eigen.h>

#include <string>
#include <sstream>
#include <vector>

namespace citlali::pipeline {

struct RawObsDetectorInventory {
    Eigen::Index n_dets = 0;
    std::vector<Eigen::Index> dets;
    std::vector<Eigen::Index> nws;
    std::vector<Eigen::Index> arrays;
};

inline Eigen::Index detector_count_from_rawobs_file(netCDF::NcFile &fo) {
    return static_cast<Eigen::Index>(
        fo.getVar("Data.Toltec.Is").getDim(1).getSize());
}

inline Eigen::Index rawobs_interface_id(const std::string &interface_name) {
    return static_cast<Eigen::Index>(std::stoi(interface_name.substr(6)));
}

template <class RawObs, class Logger>
Eigen::Index read_rawobs_detector_count(const RawObs &rawobs,
                                        const Logger &logger) {
    using namespace netCDF;
    using namespace netCDF::exceptions;

    Eigen::Index n_dets = 0;
    for (const typename RawObs::DataItem &data_item : rawobs.kidsdata()) {
        try {
            NcFile fo(data_item.filepath(), NcFile::read);
            n_dets += detector_count_from_rawobs_file(fo);
            fo.close();
        }
        catch (NcException &e) {
            logger->error("{}", e.what());
            throw ::DataIOError{fmt::format(
                "failed to load data from netCDF file {}",
                data_item.filepath())};
        }
    }
    return n_dets;
}

template <class RawObs, class NetworkToArrayMap, class Logger>
RawObsDetectorInventory read_rawobs_detector_inventory(
    const RawObs &rawobs, NetworkToArrayMap &nw_to_array_map,
    const Logger &logger) {
    using namespace netCDF;
    using namespace netCDF::exceptions;

    RawObsDetectorInventory inventory;
    for (const typename RawObs::DataItem &data_item : rawobs.kidsdata()) {
        try {
            NcFile fo(data_item.filepath(), NcFile::read);
            const auto interface_id =
                rawobs_interface_id(data_item.interface());
            const auto n_file_dets = detector_count_from_rawobs_file(fo);
            inventory.n_dets += n_file_dets;
            inventory.dets.push_back(n_file_dets);
            inventory.nws.push_back(interface_id);
            inventory.arrays.push_back(nw_to_array_map[interface_id]);
            fo.close();
        }
        catch (NcException &e) {
            logger->error("{}", e.what());
            throw ::DataIOError{fmt::format(
                "failed to load data from netCDF file {}",
                data_item.filepath())};
        }
    }
    return inventory;
}

template <class Calib>
void populate_internal_apt_from_detector_inventory(
    Calib &calib, const RawObsDetectorInventory &inventory) {
    calib.apt.clear();

    for (auto const& key : calib.apt_header_keys) {
        if (key=="x_t" || key=="y_t") {
            calib.apt[key].setZero(inventory.n_dets);
        }
        else {
            calib.apt[key].setOnes(inventory.n_dets);
        }
    }

    calib.apt["flag"].setZero(inventory.n_dets);

    Eigen::Index j = 0;
    for (Eigen::Index i=0; i<inventory.nws.size(); ++i) {
        calib.apt["nw"].segment(j,inventory.dets[i])
            .setConstant(inventory.nws[i]);
        calib.apt["array"].segment(j,inventory.dets[i])
            .setConstant(inventory.arrays[i]);
        j = j + inventory.dets[i];
    }

    calib.apt["uid"] = Eigen::VectorXd::LinSpaced(
        inventory.n_dets,0,inventory.n_dets-1);
    calib.setup();
    calib.apt_filepath = "internally generated for beammap";
    std::ostringstream identity;
    identity << "beammap-internal-raw-detector-inventory-v1";
    for (Eigen::Index index = 0;
         index < static_cast<Eigen::Index>(inventory.nws.size()); ++index) {
        identity << '|' << inventory.nws[static_cast<std::size_t>(index)]
                 << ':' << inventory.dets[static_cast<std::size_t>(index)]
                 << ':' << inventory.arrays[static_cast<std::size_t>(index)];
    }
    calib.apt_acquisition_binding.available = true;
    calib.apt_acquisition_binding.valid = true;
    calib.apt_acquisition_binding.mode =
        "internal_raw_network_local_row_v1";
    calib.apt_acquisition_binding.key_schema =
        "raw_observation_artifact+network+network_local_row";
    calib.apt_acquisition_binding.detail =
        "Beammap internal APT constructed directly from the admitted raw detector inventory";
    calib.apt_acquisition_binding.artifact_sha256 =
        citlali::utils::sha256(identity.str());
    calib.apt_acquisition_binding.binding_sha256 =
        calib.apt_acquisition_binding.artifact_sha256;
    calib.apt_acquisition_binding.raw_observation_identity = identity.str();
    calib.apt_acquisition_binding.detector_count = inventory.n_dets;
    calib.apt_acquisition_binding.network_count =
        static_cast<Eigen::Index>(inventory.nws.size());
}

}  // namespace citlali::pipeline
