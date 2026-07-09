#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/config/config_value.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

bool Beammap::load_soft_priors() {
    auto &priors_config =
        citlali::pipeline::beammap_config(*this).priors;
    beammap_soft_prior_slots.clear();
    beammap_soft_priors_loaded = false;
    beammap_soft_priors_are_centered = false;
    beammap_soft_priors_are_derotated = false;

    if (!priors_config.enabled) {
        return false;
    }

    if (citlali::config::is_empty_or_null_config_value(
            priors_config.filepath)) {
        logger->warn("beammap priors filepath is empty/null");
        return false;
    }
    const auto resolved_priors_filepath = resolve_soft_priors_filepath();
    if (resolved_priors_filepath.empty()) {
        logger->warn("beammap priors file does not exist: {}", priors_config.filepath);
        return false;
    }
    if (resolved_priors_filepath.string() != priors_config.filepath) {
        logger->info("beammap priors resolved {} -> {}", priors_config.filepath, resolved_priors_filepath.string());
        priors_config.filepath = resolved_priors_filepath.string();
    }

    auto [priors_table, priors_header, priors_meta] =
        to_map_from_ecsv_mixted_type(priors_config.filepath);
    static_cast<void>(priors_header);

    auto prior_frame_it = priors_meta.find("prior_frame");
    if (prior_frame_it != priors_meta.end()) {
        std::string prior_frame = prior_frame_it->second;
        std::transform(prior_frame.begin(), prior_frame.end(), prior_frame.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        beammap_soft_priors_are_centered = (prior_frame.find("center") != std::string::npos);
        beammap_soft_priors_are_derotated = (prior_frame.find("derot") != std::string::npos);
    }

    const std::vector<std::string> required_columns = {
        "array",
        "nw",
        "slot_index",
        "x_rel_med_arcsec",
        "y_rel_med_arcsec",
        "x_rel_sigma_soft_arcsec",
        "y_rel_sigma_soft_arcsec"
    };

    for (const auto &col : required_columns) {
        if (priors_table.find(col) == priors_table.end()) {
            logger->warn("beammap priors missing required column '{}': {}", col, priors_config.filepath);
            return false;
        }
    }

    const Eigen::Index n_rows = priors_table.at("array").size();
    for (const auto &col : required_columns) {
        if (priors_table.at(col).size() != n_rows) {
            logger->warn("beammap priors column '{}' has wrong size {} (expected {})",
                         col, priors_table.at(col).size(), n_rows);
            return false;
        }
    }
    if (n_rows <= 0) {
        logger->warn("beammap priors table has no rows: {}", priors_config.filepath);
        return false;
    }

    constexpr double sigma_floor_arcsec = 1e-3;
    Eigen::Index n_valid_rows = 0;
    Eigen::Index n_dropped_rows = 0;
    for (Eigen::Index i = 0; i < n_rows; ++i) {
        const double array_d = priors_table.at("array")(i);
        const double nw_d = priors_table.at("nw")(i);
        const double slot_d = priors_table.at("slot_index")(i);
        const double x_d = priors_table.at("x_rel_med_arcsec")(i);
        const double y_d = priors_table.at("y_rel_med_arcsec")(i);
        const double sx_d = priors_table.at("x_rel_sigma_soft_arcsec")(i);
        const double sy_d = priors_table.at("y_rel_sigma_soft_arcsec")(i);

        if (!(std::isfinite(array_d) && std::isfinite(nw_d) && std::isfinite(slot_d) &&
              std::isfinite(x_d) && std::isfinite(y_d) && std::isfinite(sx_d) && std::isfinite(sy_d))) {
            n_dropped_rows++;
            continue;
        }

        const int array = static_cast<int>(std::lround(array_d));
        const int nw = static_cast<int>(std::lround(nw_d));

        SoftPriorSlot slot;
        slot.slot_index = static_cast<int>(std::lround(slot_d));
        slot.x_arcsec = x_d;
        slot.y_arcsec = y_d;
        slot.sx_arcsec = std::max(sigma_floor_arcsec, std::abs(sx_d));
        slot.sy_arcsec = std::max(sigma_floor_arcsec, std::abs(sy_d));

        beammap_soft_prior_slots[{array, nw}].push_back(slot);
        n_valid_rows++;
    }

    for (auto &entry : beammap_soft_prior_slots) {
        auto &slots = entry.second;
        std::sort(slots.begin(), slots.end(),
                  [](const SoftPriorSlot &a, const SoftPriorSlot &b) {
                      if (a.slot_index == b.slot_index) {
                          return a.y_arcsec < b.y_arcsec;
                      }
                      return a.slot_index < b.slot_index;
                  });
    }

    if (beammap_soft_prior_slots.empty()) {
        logger->warn("beammap priors produced no valid slots: {}", priors_config.filepath);
        return false;
    }

    Eigen::Index n_slots = 0;
    for (const auto &entry : beammap_soft_prior_slots) {
        n_slots += static_cast<Eigen::Index>(entry.second.size());
    }
    beammap_soft_priors_loaded = true;
    logger->info("loaded beammap soft priors: {} slot rows across {} (array,nw) groups from {}",
                 n_slots, beammap_soft_prior_slots.size(), priors_config.filepath);
    if (n_dropped_rows > 0) {
        logger->warn("dropped {} non-finite prior rows (kept {})", n_dropped_rows, n_valid_rows);
    }

    return true;
}
