#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/case_conv.hpp>

#include <tula/logging.h>
#include <tula/algorithm/ei_stats.h>

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/engine/detail/sci_align_netcdf_input_contract.h>
#include <citlali/core/engine/detail/sci_align_telescope_alias_contract.h>
#include <citlali/core/engine/telescope.h>
#include <citlali/core/error/error.h>
#include <citlali/core/pipeline/sci_align_field_registry.h>
#include <citlali/core/utils/utils.h>
#include <citlali/core/utils/pointing.h>

#include <limits>

namespace engine {

void Telescope::get_tel_data(
    std::string &filepath,
    const citlali::config::TimestreamChunkingConfig &chunking) {
    using namespace netCDF;
    using namespace netCDF::exceptions;

    // Telescope owns observation-scoped state. Replacement is complete even
    // when a later read fails; stale fields, scan identities, and Hold words
    // must never leak into the next observation.
    tel_data.clear();
    tel_header.clear();
    native_tel_header.clear();
    realized_compatibility_tel_header.clear();
    hold_raw_word.resize(0);
    scan_indices.resize(0, 0);
    scan_plan.clear();
    obs_pgm.clear();
    source_name.clear();
    obs_goal.clear();
    project_id.clear();
    map_coord.clear();
    std::fill_n(sim_job_key, sizeof(sim_job_key), '\0');
    sim_obs = false;
    exec_mode = true;

    try {
        // get telescope file
        NcFile fo(filepath, NcFile::read, NcFile::classic);
        namespace nc_contract =
            citlali::engine_detail::sci_align_netcdf;

        auto read_required_text = [&](const std::string &name) {
            const auto variable = nc_contract::require_variable(fo, name);
            return nc_contract::read_fixed_width_text(
                variable, name, 128);
        };
        auto remove_legacy_spaces = [](std::string value) {
            const auto end = std::remove(value.begin(), value.end(), ' ');
            value.erase(end, value.end());
            return value;
        };

        // check if simulation job key is found.
        const auto simulation_job_key = fo.getVar("Header.Sim.Jobkey");
        if (!simulation_job_key.isNull()) {
            const auto value = nc_contract::read_fixed_width_text(
                simulation_job_key, "Header.Sim.Jobkey", 128);
            std::copy(value.begin(), value.end(), sim_job_key);
            sim_job_key[value.size()] = '\0';
            logger->info("found Header.Sim.Jobkey");
            sim_obs = true;
        }
        else {
            logger->info("Header.Sim.Jobkey is absent; treating input as real data");
            sim_obs = false;
        }

        // get obs goal
        if (!sim_obs) {
            obs_goal = remove_legacy_spaces(
                read_required_text("Header.Dcs.ObsGoal"));
        }

        // get map pattern
        obs_pgm = remove_legacy_spaces(
            read_required_text("Header.Dcs.ObsPgm"));

        if (obs_pgm=="Map") {
            const auto exec_mode_var = nc_contract::require_variable(
                fo, "Header.Map.ExecMode");
            nc_contract::require_scalar(exec_mode_var,
                                        "Header.Map.ExecMode");
            nc_contract::require_type(exec_mode_var,
                                      "Header.Map.ExecMode",
                                      {NcType::nc_INT});
            int raw_exec_mode = -1;
            exec_mode_var.getVar(&raw_exec_mode);
            if (raw_exec_mode != 0 && raw_exec_mode != 1) {
                throw DataIOError{
                    "Header.Map.ExecMode must be the exact integer state 0 or 1"};
            }
            exec_mode = raw_exec_mode != 0;

            map_coord = remove_legacy_spaces(
                read_required_text("Header.Map.MapCoord"));
        }
        else {
            exec_mode = 1;
        }

        // cannot reduce in lissajous mode if chunk less than or equal to zero
        if ((obs_pgm=="Lissajous" || (obs_pgm=="Map" && exec_mode==1)) &&
            chunking.value <= 0) {
            throw citlali::error::invalid_config(
                "lissajous mapping requires a positive time chunk size");
        }

        // get source name
        source_name = remove_legacy_spaces(
            read_required_text("Header.Source.SourceName"));

        // get project id
        if (!sim_obs) {
            project_id = remove_legacy_spaces(
                read_required_text("Header.Dcs.ProjectId"));
        }
        else {
            project_id = "simu";
        }

        std::vector<std::string> missing_data_keys;
        std::vector<std::string> missing_header_keys;

        // Load the two RA/Dec schema alternatives without allowing map
        // insertion order to choose the canonical identity.
        std::map<std::string, Eigen::VectorXd> alias_data;

        const auto tel_time_var = nc_contract::require_variable(
            fo, "Data.TelescopeBackend.TelTime");
        nc_contract::require_type(
            tel_time_var, "Data.TelescopeBackend.TelTime",
            {NcType::nc_DOUBLE});
        const auto authoritative_time_dimension =
            nc_contract::require_nonempty_vector(
                tel_time_var, "Data.TelescopeBackend.TelTime");
        const auto authoritative_time_count =
            nc_contract::require_eigen_index_size(
                authoritative_time_dimension.getSize(),
                "Data.TelescopeBackend.TelTime");
        nc_contract::require_units(
            tel_time_var, "Data.TelescopeBackend.TelTime", {"sec"});

        auto active_registry_entry_for_raw = [](std::string_view raw_name)
            -> const citlali::pipeline::sci_align::ActiveFieldRegistryEntry * {
            using namespace citlali::pipeline::sci_align;
            for (const auto &entry : active_field_registry) {
                if (entry.raw_name == raw_name) {
                    return &entry;
                }
            }
            for (const auto &alias : active_field_aliases) {
                if (alias.raw_alias != raw_name) {
                    continue;
                }
                for (const auto &entry : active_field_registry) {
                    if (entry.field_id == alias.canonical_field_id) {
                        return &entry;
                    }
                }
            }
            return nullptr;
        };

        // loop through telescope data keys and populate vectors
        for (const auto& pair : tel_data_keys) {
            logger->info("tel_data key {}",pair.first);
            const auto variable = fo.getVar(pair.first);
            if (variable.isNull()) {
                missing_data_keys.push_back(pair.first);
                logger->debug("optional telescope data variable is absent: {}",
                              pair.first);
                continue;
            }

            const auto *registry_entry =
                active_registry_entry_for_raw(pair.first);
            if (registry_entry == nullptr) {
                throw DataIOError{fmt::format(
                    "configured telescope data variable '{}' lacks an active field-registry identity",
                    pair.first)};
            }
            nc_contract::require_type(
                variable, pair.first, {NcType::nc_DOUBLE});
            nc_contract::require_vector_on_dimension(
                variable, pair.first, authoritative_time_dimension);
            if (registry_entry->canonical_name == "Hold") {
                // Preserve the producer's historical schema label as an
                // input-contract fact only.  It does not assign boolean or
                // per-bit physical meaning to the retained raw word.
                nc_contract::require_units(variable, pair.first,
                                           {"boolean"});
            }
            else {
                nc_contract::require_units(
                    variable, pair.first,
                    {citlali::pipeline::sci_align::active_field_raw_unit(
                        *registry_entry)});
            }

            Eigen::VectorXd data_temp(authoritative_time_count);
            variable.getVar(data_temp.data());
            if (pair.first == "Data.TelescopeBackend.SourceRaAct" ||
                pair.first == "Data.TelescopeBackend.SourceDecAct" ||
                pair.first == "Data.TelescopeBackend.TelRaAct" ||
                pair.first == "Data.TelescopeBackend.TelDecAct") {
                alias_data.emplace(pair.first, std::move(data_temp));
            }
            else {
                tel_data[pair.second] = std::move(data_temp);
            }
        }

        try {
            auto equatorial = citlali::engine_detail::
                resolve_equatorial_aliases(alias_data);
            tel_data["TelRa"] = std::move(equatorial.right_ascension);
            tel_data["TelDec"] = std::move(equatorial.declination);
        }
        catch (const std::runtime_error &error) {
            throw DataIOError{error.what()};
        }

        const std::vector<std::string> required_active_fields = {
            "TelTime", "ActGalAng", "ActParAng", "SourceAz", "TelL",
            "TelRa", "TelAzAct", "TelAzDes", "TelB", "TelDec",
            "SourceEl", "TelAzCor", "TelAzMap", "TelElAct",
            "TelElCor", "TelElDes", "TelElMap", "Hold",
        };
        const auto tel_time_it = tel_data.find("TelTime");
        if (tel_time_it == tel_data.end() || tel_time_it->second.size() == 0) {
            throw DataIOError{"required telescope coordinate TelTime is absent or empty"};
        }
        const Eigen::Index native_count = tel_time_it->second.size();
        for (const auto &field : required_active_fields) {
            const auto it = tel_data.find(field);
            if (it == tel_data.end() || it->second.size() != native_count) {
                throw DataIOError{fmt::format(
                    "required telescope field '{}' has unavailable or inconsistent time support",
                    field)};
            }
            if (!it->second.allFinite()) {
                throw DataIOError{fmt::format(
                    "required telescope field '{}' contains nonfinite values",
                    field)};
            }
        }
        for (const auto &optional_exact : {"TelUTC", "PpsTime"}) {
            const auto it = tel_data.find(optional_exact);
            if (it != tel_data.end() &&
                (it->second.size() != native_count ||
                 !it->second.allFinite())) {
                throw DataIOError{fmt::format(
                    "optional exact telescope diagnostic '{}' has inconsistent or nonfinite native support",
                    optional_exact)};
            }
        }
        for (Eigen::Index i = 1; i < native_count; ++i) {
            if (!(tel_time_it->second(i) > tel_time_it->second(i - 1))) {
                throw DataIOError{fmt::format(
                    "native telescope TelTime is not strictly increasing at row {}",
                    i)};
            }
        }
        const auto &hold = tel_data.at("Hold");
        hold_raw_word.resize(hold.size());
        constexpr double max_exact_integer_double = 9007199254740991.0;
        for (Eigen::Index i = 0; i < hold.size(); ++i) {
            const double value = hold(i);
            if (value < 0.0 || std::floor(value) != value ||
                value > max_exact_integer_double) {
                throw DataIOError{fmt::format(
                    "native Hold word is not finite, nonnegative, integral, and losslessly representable at row {}",
                    i)};
            }
            const auto word = static_cast<
                citlali::pipeline::sci_align::TelescopeHoldWord>(value);
            if (static_cast<double>(word) != value) {
                throw DataIOError{fmt::format(
                    "native Hold word loses integer identity at row {}", i)};
            }
            hold_raw_word(i) = word;
        }

        // loop through telescope header keys and populate vectors
        for (const auto& pair : tel_header_keys) {
            const auto variable = fo.getVar(pair.first);
            if (variable.isNull()) {
                if (!sim_obs) {
                    missing_header_keys.push_back(pair.first);
                    logger->debug("optional telescope header is absent: {}",
                                  pair.first);
                }
                continue;
            }

            nc_contract::require_type(
                variable, pair.first,
                {NcType::nc_BYTE, NcType::nc_SHORT, NcType::nc_INT,
                 NcType::nc_FLOAT, NcType::nc_DOUBLE, NcType::nc_UBYTE,
                 NcType::nc_USHORT, NcType::nc_UINT, NcType::nc_INT64,
                 NcType::nc_UINT64});
            const auto snapshot =
                nc_contract::read_numeric_telescope_header(
                    variable, pair.first);
            const auto element_count =
                citlali::pipeline::sci_align::
                    telescope_header_element_count(snapshot);
            if (element_count > static_cast<std::size_t>(
                                    std::numeric_limits<Eigen::Index>::max())) {
                throw DataIOError{fmt::format(
                    "telescope header '{}' exceeds the Eigen index range",
                    pair.first)};
            }
            std::vector<double> legacy_values;
            try {
                legacy_values = citlali::pipeline::sci_align::
                    telescope_header_legacy_double_view(snapshot,
                                                        pair.first);
            }
            catch (const std::invalid_argument &error) {
                throw DataIOError{error.what()};
            }
            Eigen::VectorXd header_temp(
                static_cast<Eigen::Index>(element_count));
            std::copy(legacy_values.begin(), legacy_values.end(),
                      header_temp.data());
            tel_header[pair.second] = std::move(header_temp);
            native_tel_header[pair.second] = snapshot;
        }

        auto require_scalar_double_header = [&](const std::string &name) {
            const auto variable = nc_contract::require_variable(fo, name);
            nc_contract::require_scalar(variable, name);
            nc_contract::require_type(variable, name,
                                      {NcType::nc_DOUBLE});
            const auto loaded = tel_header.find(name);
            if (loaded == tel_header.end() || loaded->second.size() != 1 ||
                !loaded->second.allFinite()) {
                throw DataIOError{fmt::format(
                    "required scalar telescope header '{}' is unavailable or nonfinite",
                    name)};
            }
            return loaded;
        };
        auto require_two_element_radian_header =
            [&](const std::string &name) {
                const auto variable = nc_contract::require_variable(fo, name);
                nc_contract::require_type(variable, name,
                                          {NcType::nc_DOUBLE});
                const auto dimension =
                    nc_contract::require_nonempty_vector(variable, name);
                if (dimension.getSize() != 2) {
                    throw DataIOError{fmt::format(
                        "required telescope header '{}' must have exact shape (2)",
                        name)};
                }
                nc_contract::require_units(variable, name, {"rad"});
                const auto loaded = tel_header.find(name);
                if (loaded == tel_header.end() ||
                    loaded->second.size() != 2 ||
                    !loaded->second.allFinite()) {
                    throw DataIOError{fmt::format(
                        "required telescope header '{}' is unavailable or nonfinite",
                        name)};
                }
                return loaded;
            };

        const auto tau_header =
            require_scalar_double_header("Header.Radiometer.Tau");
        for (const auto &name : {"Header.Source.Ra", "Header.Source.Dec",
                                 "Header.Source.L", "Header.Source.B"}) {
            (void)require_two_element_radian_header(name);
        }
        if (tel_data.find("TelUTC") != tel_data.end()) {
            const auto ut_date = require_scalar_double_header(
                "Header.TimePlace.UTDate");
            const auto ut_date_var = nc_contract::require_variable(
                fo, "Header.TimePlace.UTDate");
            nc_contract::require_units(
                ut_date_var, "Header.TimePlace.UTDate", {"year"});
            (void)ut_date;
        }
        if (obs_pgm == "Map" && !exec_mode) {
            for (const auto &name : {"Header.Map.XLength",
                                     "Header.Map.YLength",
                                     "Header.Map.ScanAngle"}) {
                const auto loaded = require_scalar_double_header(name);
                const auto variable = nc_contract::require_variable(fo, name);
                nc_contract::require_units(variable, name, {"radian"});
                const double value = loaded->second(0);
                if (!std::isfinite(value) ||
                    ((std::string{name} == "Header.Map.XLength" ||
                      std::string{name} == "Header.Map.YLength") &&
                     !(value > 0.0))) {
                    throw DataIOError{fmt::format(
                        "raster geometry header '{}' is nonfinite or has invalid extent",
                        name)};
                }
            }
        }

        if (!missing_data_keys.empty() || !missing_header_keys.empty()) {
            logger->warn(
                "telescope input {} omits {} configured data variables and {} configured header values; individual optional names are available at debug level",
                filepath, missing_data_keys.size(), missing_header_keys.size());
        }

        // set tau 225 GHz
        tau_225_GHz = tau_header->second(0);

        // close netcdf file
        fo.close();

    } catch (NcException &e) {
        logger->warn("{}", e.what());
        throw DataIOError{fmt::format(
            "failed to load data from netCDF file {}", filepath)};
    }

    if (!sim_obs) {
        // TelUTC is an optional exact diagnostic. When present, its legacy
        // conversion requires the corresponding UTDate header; absence does
        // not create an empty field through map insertion.
        const auto tel_utc = tel_data.find("TelUTC");
        if (tel_utc != tel_data.end()) {
            const auto ut_date = tel_header.find("Header.TimePlace.UTDate");
            if (ut_date == tel_header.end() || ut_date->second.size() != 1 ||
                !ut_date->second.allFinite()) {
                throw DataIOError{
                    "TelUTC is present but Header.TimePlace.UTDate is unavailable or invalid"};
            }
            engine_utils::utc_to_unix(tel_utc->second, ut_date->second);
        }

        // calculate galactic l and b for source
        engine_utils::equatorial_to_galactic(
            tel_header.at("Header.Source.Ra")(0),
            tel_header.at("Header.Source.Dec")(0),
            tel_header.at("Header.Source.L")(0),
            tel_header.at("Header.Source.B")(0));
    }

    // manually set epoch to J2000 for simulations
    else {
        tel_header["Header.Source.Epoch"] =
            Eigen::VectorXd::Constant(1, 2000.0);
        // This is a governing simulation compatibility fact, not a native
        // producer snapshot. Keep the authorities distinct while preserving
        // the exact legacy output value and one-element shape.
        native_tel_header.erase("Header.Source.Epoch");
        realized_compatibility_tel_header["Header.Source.Epoch"] =
            citlali::engine_detail::
                simulation_j2000_compatibility_header_snapshot();
    }
}

void Telescope::calc_tan_pointing() {
    // get radec tangent pointing
    calc_tan_radec();
    // get altaz tangent pointing
    calc_tan_altaz();

    if (!sim_obs) {
        // get galactic tangent pointing
        calc_tan_galactic();
    }

    // set tangential projection to radec
    if (citlali::config::is_radec_map_pixel_axes(pixel_axes)) {
        logger->info("using radec frame");
        tel_data["lat_phys"] = tel_data["dec_phys"];
        tel_data["lon_phys"] = tel_data["ra_phys"];
    }
    // set tangential projection to altaz
    else if (citlali::config::is_altaz_map_pixel_axes(pixel_axes)) {
        logger->info("using altaz frame");
        tel_data["lat_phys"] = tel_data["alt_phys"];
        tel_data["lon_phys"] = tel_data["az_phys"];
    }
    // set tangential projection to galactic
    else if (citlali::config::is_galactic_map_pixel_axes(pixel_axes)) {
        logger->info("using galactic frame");
        tel_data["lat_phys"] = tel_data["b_phys"];
        tel_data["lon_phys"] = tel_data["l_phys"];
    }

    // apply corrections
    tel_data["TelElAct"] -= tel_data["TelElCor"];
    tel_data["TelAzAct"] -= tel_data["TelAzCor"];
}

void Telescope::calc_tan_radec() {
    // size of data
    Eigen::Index n_pts = tel_data["TelRa"].size();

    // vectors to hold physical (tangent plane) coordinates
    tel_data["dec_phys"].resize(n_pts);
    tel_data["ra_phys"].resize(n_pts);

    // copy radec
    Eigen::VectorXd ra = tel_data["TelRa"];
    auto& dec = tel_data["TelDec"];

    // rescale ra
    ra = (ra.array() > pi).select(ra.array() - 2.0*pi, ra.array());

    // center positions
    double ra0 = tel_header["Header.Source.Ra"](0);
    double dec0 = tel_header["Header.Source.Dec"](0);

    // rescale center ra
    ra0 = (ra0 > pi) ? ra0 - (2.0*pi) : ra0;

    // calculate gnomonic projection
    engine_utils::gnomonic_projection(ra, dec, ra0, dec0, tel_data["ra_phys"], tel_data["dec_phys"]);
}

void Telescope::calc_tan_altaz() {
    // use loop to avoid annoying eigen aliasing issues with select
    for (Eigen::Index i=0; i<tel_data["TelAzAct"].size(); ++i) {
        if ((tel_data["TelAzAct"](i) - tel_data["SourceAz"](i)) > 0.9*2.0*pi) {
            tel_data["TelAzAct"](i) = tel_data["TelAzAct"](i) - 2.0*pi;
        }
    }

    // subtract source az
    auto az_diff = tel_data["TelAzAct"].array() - tel_data["SourceAz"].array();

    // tangent plane lat (alt)
    tel_data["alt_phys"] = (tel_data["TelElAct"].array() - tel_data["SourceEl"].array() - tel_data["TelElCor"].array()).matrix();

    // tangent plane lon (az)
    tel_data["az_phys"] = (cos(tel_data["TelElAct"].array() - tel_data["TelElCor"].array()) * az_diff - tel_data["TelAzCor"].array()).matrix();
}

void Telescope::calc_tan_galactic() {
    // size of data
    Eigen::Index n_pts = tel_data["TelL"].size();

    // vectors to hold physical (tangent plane) coordinates
    tel_data["l_phys"].resize(n_pts);
    tel_data["b_phys"].resize(n_pts);

    // copy lb
    Eigen::VectorXd l = tel_data["TelL"];
    auto b = tel_data["TelB"];

    // rescale l
    l = (l.array() > pi).select(l.array() - 2.0*pi, l.array());

    // center positions
    double l0 = tel_header["Header.Source.L"](0);
    double b0 = tel_header["Header.Source.B"](0);

    // rescale center l
    l0 = (l0 > pi) ? l0 - (2.0*pi) : l0;

    // calculate gnomonic projection
    engine_utils::gnomonic_projection(l, b, l0, b0, tel_data["l_phys"], tel_data["b_phys"]);
}

void Telescope::calc_scan_indices(
    const citlali::config::TimestreamChunkingConfig &chunking) {
    const auto hold_it = tel_data.find("Hold");
    if (hold_it == tel_data.end() || hold_it->second.size() <= 0) {
        throw std::runtime_error(
            "cannot calculate scan indices: telescope series 'Hold' is missing or empty");
    }
    calc_scan_indices(
        chunking, {0, static_cast<Eigen::Index>(hold_it->second.size())});
}

void Telescope::calc_scan_indices(
    const citlali::config::TimestreamChunkingConfig &chunking,
    citlali::pipeline::sci_align::HalfOpenInterval governing_support) {
    auto require_tel_series = [&](const std::string &key) -> Eigen::VectorXd & {
        auto it = tel_data.find(key);
        if (it == tel_data.end() || it->second.size() == 0) {
            throw std::runtime_error(fmt::format(
                "cannot calculate scan indices: telescope series '{}' is missing or empty", key));
        }
        return it->second;
    };

    auto require_header_scalar = [&](const std::string &key) -> double {
        auto it = tel_header.find(key);
        if (it == tel_header.end() || it->second.size() == 0) {
            throw std::runtime_error(fmt::format(
                "cannot calculate scan indices: telescope header '{}' is missing or empty", key));
        }
        return it->second(0);
    };

    const auto &sample_axis = require_tel_series("Hold");
    const Eigen::Index n_total_samples = sample_axis.size();
    if (n_total_samples <= 0 || !std::isfinite(fsmp) || fsmp <= 0.0) {
        throw std::runtime_error(
            "cannot calculate scan indices: invalid detector support or sample rate");
    }
    if (!governing_support.valid() || governing_support.empty() ||
        governing_support.stop > n_total_samples) {
        throw std::runtime_error(
            "cannot calculate scan indices: invalid governing consumer support");
    }
    const Eigen::Index context_samples =
        std::max<Eigen::Index>(0, outer_scans_chunk);

    // Existing-use-only raster compatibility: keep the named whole-word
    // linear/nonzero view separate from the outside-map-box condition, then
    // construct stable half-open false runs. The processor adapter admits the
    // same >=2-second cohort while retaining short identities in scan_plan.
    if ((obs_pgm=="Map" && exec_mode==0) && !chunking.force) {
        logger->info("calculating scans for raster mode");
        auto &hold = require_tel_series("Hold");
        std::string coord1_key, coord2_key;
        const auto map_coord_lower = boost::algorithm::to_lower_copy(map_coord);
        if (map_coord_lower == "ra" || map_coord_lower == "dec") {
            coord1_key = "ra_phys";
            coord2_key = "dec_phys";
        }
        else if (map_coord_lower == "az" || map_coord_lower == "el" || map_coord_lower == "alt") {
            coord1_key = "az_phys";
            coord2_key = "alt_phys";
        }
        else if (map_coord_lower == "gal" || map_coord_lower == "l" || map_coord_lower == "b") {
            coord1_key = "l_phys";
            coord2_key = "b_phys";
        }
        else {
            throw std::runtime_error(fmt::format(
                "cannot calculate scans for raster mode: unsupported Header.Map.MapCoord='{}'", map_coord));
        }

        auto &coord1 = require_tel_series(coord1_key);
        auto &coord2 = require_tel_series(coord2_key);
        if (coord1.size() != hold.size() || coord2.size() != hold.size()) {
            throw std::runtime_error(fmt::format(
                "cannot calculate scans for raster mode: coordinate sizes do not match Hold size "
                "(Hold={}, {}={}, {}={})",
                hold.size(), coord1_key, coord1.size(), coord2_key, coord2.size()));
        }

        const double x_length = require_header_scalar("Header.Map.XLength");
        const double y_length = require_header_scalar("Header.Map.YLength");
        const double scan_angle = require_header_scalar("Header.Map.ScanAngle");

        std::vector<unsigned char> outside_map_box(
            static_cast<std::size_t>(hold.size()), 0);
        for (Eigen::Index i = 0; i < hold.size(); ++i) {
            if (!engine_utils::is_point_in_box(coord1(i), coord2(i),
                                              x_length, y_length, scan_angle)) {
                outside_map_box[static_cast<std::size_t>(i)] = 1;
            }
        }
        const auto composite =
            citlali::pipeline::sci_align::compose_legacy_hold_and_outside(
                hold, outside_map_box);
        scan_plan =
            citlali::pipeline::sci_align::make_raster_compatibility_scan_plan(
                composite, governing_support, fsmp, context_samples, 2.0,
                std::max<Eigen::Index>(0, inner_scans_chunk));
    }
    else if (obs_pgm=="Lissajous" || (obs_pgm=="Map" && exec_mode==1) ||
             chunking.force) {
        logger->info("calculating scans for lissajous/rastajous mode");
        if (chunking.mode == "duration") {
            scan_plan =
                citlali::pipeline::sci_align::make_fixed_duration_scan_plan(
                    n_total_samples, governing_support, chunking.value,
                    1.0 / fsmp,
                    context_samples);
        }
        else if (chunking.mode == "number") {
            const Eigen::Index requested_count =
                citlali::pipeline::sci_align::checked_number_scan_count(
                    chunking.value, governing_support.size());
            scan_plan = citlali::pipeline::sci_align::make_number_scan_plan(
                n_total_samples, governing_support, requested_count, 1.0 / fsmp,
                context_samples);
        }
        else {
            throw std::runtime_error(fmt::format(
                "cannot calculate scans for lissajous/rastajous mode: unsupported chunk_mode='{}'",
                chunking.mode));
        }

    }
    else {
        throw std::runtime_error(fmt::format(
            "cannot calculate scan indices: unsupported observation pattern '{}'",
            obs_pgm));
    }
    scan_indices =
        citlali::pipeline::sci_align::compatibility_scan_indices(scan_plan);
    if (scan_indices.cols() <= 0) {
        throw std::runtime_error(
            "cannot calculate scan indices: no compatibility-admitted scan support");
    }
    logger->info(
        "scan plan policy={} identities={} admitted={} scan_indices {}",
        scan_plan.policy, scan_plan.records.size(), scan_indices.cols(),
        scan_indices);
}

} // namespace engine
