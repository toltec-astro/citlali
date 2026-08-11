#include <citlali/core/utils/constants.h>
#include <citlali/core/utils/utils.h>
#include <citlali/core/engine/calib.h>
#include <citlali/core/error/error.h>
#include <citlali/core/utils/toltec_io.h>
#include <citlali/core/utils/sha256.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace {

constexpr const char *tolapt_run_contract_version = "tolapt.run.v1";

std::string join_column_names(const std::vector<std::string> &names) {
    std::string result;
    for (const auto &name : names) {
        if (!result.empty()) {
            result += ", ";
        }
        result += name;
    }
    return result;
}

bool same_frequency_identity(double left, double right) {
    if (!std::isfinite(left) || !std::isfinite(right)) {
        return false;
    }
    const double scale = std::max({1.0, std::abs(left), std::abs(right)});
    return std::abs(left - right) <=
           64.0 * std::numeric_limits<double>::epsilon() * scale;
}

int interface_network_id(const std::string &interface_name) {
    constexpr std::string_view prefix{"toltec"};
    if (interface_name.rfind(prefix, 0) != 0 ||
        interface_name.size() == prefix.size()) {
        throw citlali::error::io(
            "invalid TolTEC interface name for APT acquisition binding: " +
            interface_name);
    }
    const auto suffix = std::string_view{interface_name}.substr(prefix.size());
    int network = -1;
    const auto parsed =
        std::from_chars(suffix.data(), suffix.data() + suffix.size(), network);
    if (parsed.ec != std::errc{} ||
        parsed.ptr != suffix.data() + suffix.size() || network < 0) {
        throw citlali::error::io(
            "invalid TolTEC interface name for APT acquisition binding: " +
            interface_name);
    }
    return network;
}

void collect_metadata_scalars(const YAML::Node &node,
                              const std::string &target,
                              std::vector<std::string> &values) {
    if (!node) {
        return;
    }
    if (node.IsMap()) {
        for (const auto &entry : node) {
            if (entry.first.IsScalar() &&
                entry.first.as<std::string>() == target &&
                entry.second.IsScalar()) {
                values.push_back(entry.second.Scalar());
            }
            collect_metadata_scalars(entry.second, target, values);
        }
        return;
    }
    if (node.IsSequence()) {
        for (const auto &entry : node) {
            collect_metadata_scalars(entry, target, values);
        }
    }
}

std::optional<std::string> unique_metadata_scalar(
    const YAML::Node &metadata, const std::string &key) {
    std::vector<std::string> values;
    collect_metadata_scalars(metadata, key, values);
    if (values.empty()) {
        return std::nullopt;
    }
    const auto &value = values.front();
    if (std::any_of(values.begin() + 1, values.end(),
                    [&](const auto &candidate) {
                        return candidate != value;
                    })) {
        throw citlali::error::io(
            "selected APT has conflicting metadata values for " + key);
    }
    return value;
}

YAML::Node read_ecsv_metadata(const std::string &filepath) {
    std::ifstream stream(filepath);
    if (!stream) {
        throw citlali::error::io(
            "cannot open selected APT metadata " + filepath);
    }
    try {
        const auto header = tula::ecsv::ECSVHeader::read(stream);
        return YAML::Clone(header.meta());
    }
    catch (const std::exception &error) {
        throw citlali::error::io(
            "cannot read selected APT metadata " + filepath + ": " +
            error.what());
    }
}

bool contains_parent_component(const std::filesystem::path &path) {
    return std::any_of(path.begin(), path.end(), [](const auto &component) {
        return component == "..";
    });
}

YAML::Node require_unique_map_entry(const YAML::Node &mapping,
                                    const std::string &key,
                                    const std::string &context) {
    if (!mapping.IsMap()) {
        throw citlali::error::io(context + " must be a mapping");
    }
    std::vector<YAML::Node> matches;
    for (const auto &entry : mapping) {
        if (entry.first.IsScalar() &&
            entry.first.as<std::string>() == key) {
            matches.push_back(YAML::Clone(entry.second));
        }
    }
    if (matches.empty()) {
        throw citlali::error::io(
            context + " is missing required field " + key);
    }
    if (matches.size() != 1) {
        throw citlali::error::io(
            context + " has ambiguous duplicate field " + key);
    }
    return matches.front();
}

std::string require_scalar_text(const YAML::Node &mapping,
                                const std::string &key,
                                const std::string &context) {
    const auto value = require_unique_map_entry(mapping, key, context);
    if (!value.IsScalar() || value.Scalar().empty()) {
        throw citlali::error::io(
            context + "." + key + " must be a non-empty scalar");
    }
    return value.Scalar();
}

bool is_lower_hex_sha256(const std::string &value) {
    return value.size() == 64 &&
        std::all_of(value.begin(), value.end(), [](unsigned char ch) {
            return std::isdigit(ch) != 0 || (ch >= 'a' && ch <= 'f');
        });
}

bool is_tolapt_utc_timestamp(const std::string &value) {
    if (value.size() != 20 || value[4] != '-' || value[7] != '-' ||
        value[10] != 'T' || value[13] != ':' || value[16] != ':' ||
        value[19] != 'Z') {
        return false;
    }
    for (const std::size_t index :
         {0U, 1U, 2U, 3U, 5U, 6U, 8U, 9U,
          11U, 12U, 14U, 15U, 17U, 18U}) {
        if (std::isdigit(static_cast<unsigned char>(value[index])) == 0) {
            return false;
        }
    }
    return true;
}

engine::Calib::TolaptInputRecord parse_tolapt_input_record(
    const YAML::Node &inputs, const std::string &name) {
    const std::string context = "TolAPT manifest inputs." + name;
    const auto record = require_unique_map_entry(
        inputs, name, "TolAPT manifest inputs");
    if (!record.IsMap()) {
        throw citlali::error::io(context + " must be a mapping");
    }

    engine::Calib::TolaptInputRecord result;
    result.path = require_scalar_text(record, "path", context);
    if (!std::filesystem::path(result.path).is_absolute()) {
        throw citlali::error::io(context + ".path must be absolute");
    }
    result.sha256 = require_scalar_text(record, "sha256", context);
    if (!is_lower_hex_sha256(result.sha256)) {
        throw citlali::error::io(
            context + ".sha256 must be a lowercase SHA-256 digest");
    }
    const std::string bytes = require_scalar_text(record, "bytes", context);
    const auto parsed = std::from_chars(
        bytes.data(), bytes.data() + bytes.size(), result.bytes);
    if (parsed.ec != std::errc{} ||
        parsed.ptr != bytes.data() + bytes.size()) {
        throw citlali::error::io(
            context + ".bytes must be a non-negative integer");
    }
    result.mtime_utc = require_scalar_text(record, "mtime_utc", context);
    if (!is_tolapt_utc_timestamp(result.mtime_utc)) {
        throw citlali::error::io(
            context + ".mtime_utc must use YYYY-MM-DDTHH:MM:SSZ");
    }
    return result;
}

void append_identity_field(std::ostringstream &identity,
                           const std::string &name,
                           const std::string &type,
                           const std::string &value) {
    identity << '|' << name.size() << ':' << name
             << ':' << type.size() << ':' << type
             << ':' << value.size() << ':' << value;
}

void append_input_record_identity(
    std::ostringstream &identity, const std::string &prefix,
    const engine::Calib::TolaptInputRecord &record) {
    append_identity_field(identity, prefix + ".path", "string", record.path);
    append_identity_field(
        identity, prefix + ".sha256", "sha256", record.sha256);
    append_identity_field(
        identity, prefix + ".bytes", "uint64",
        std::to_string(record.bytes));
    append_identity_field(
        identity, prefix + ".mtime_utc", "utc_timestamp",
        record.mtime_utc);
}

struct SelectedAptAssociationColumns {
    std::map<std::string, std::string> datatypes;
    std::map<std::string, std::vector<std::string>> string_values;
};

SelectedAptAssociationColumns read_selected_apt_association_columns(
    const std::string &filepath, Eigen::Index expected_rows) {
    const std::set<std::string> retained_names{
        "uid", "flag", "det_id", "det_id_right", "meas_idx",
        "design_idx", "match_id", "measured_id", "matched_design_id",
        "match_status"};
    std::ifstream stream(filepath);
    if (!stream) {
        throw citlali::error::io(
            "cannot open selected APT row lineage " + filepath);
    }
    try {
        auto header = tula::ecsv::ECSVHeader::read(stream);
        SelectedAptAssociationColumns result;
        for (const auto &column : header.cols()) {
            if (retained_names.count(column.name) == 0U) {
                continue;
            }
            if (!result.datatypes.emplace(
                    column.name, column.datatype).second) {
                throw citlali::error::io(
                    "selected APT has ambiguous duplicate row-lineage column " +
                    column.name);
            }
        }
        auto table = tula::ecsv::ECSVTable(header);
        auto parser = aria::csv::CsvParser(stream).delimiter(
            table.header().delimiter());
        table.load_rows(parser);
        if (static_cast<Eigen::Index>(table.rows()) != expected_rows) {
            throw citlali::error::io(
                "selected APT row-lineage cardinality mismatch");
        }
        for (const auto &name :
             {std::string{"det_id"}, std::string{"measured_id"},
              std::string{"matched_design_id"},
              std::string{"match_status"}}) {
            const auto datatype = result.datatypes.find(name);
            if (datatype == result.datatypes.end()) {
                continue;
            }
            if (datatype->second != "string") {
                if (name == "det_id") {
                    continue;
                }
                throw citlali::error::io(
                    "selected APT row-lineage column " + name +
                    " must have ECSV string datatype");
            }
            const auto &string_data = table.array_data<std::string>();
            const auto &column = string_data.array().at(
                string_data.index(name));
            auto &values = result.string_values[name];
            values.reserve(static_cast<std::size_t>(expected_rows));
            for (Eigen::Index row = 0; row < expected_rows; ++row) {
                values.push_back(column.at(static_cast<std::size_t>(row)));
            }
        }
        return result;
    }
    catch (const citlali::error::Error &) {
        throw;
    }
    catch (const std::exception &error) {
        throw citlali::error::io(
            "cannot read selected APT row lineage " + filepath + ": " +
            error.what());
    }
}

void attach_contract_defined_tolapt_manifest(
    engine::Calib::AptLineage &lineage) {
    namespace fs = std::filesystem;
    const fs::path selected = fs::weakly_canonical(lineage.selected_apt_path);
    if (selected.parent_path().filename() != "tables") {
        return;
    }

    const fs::path run_root = selected.parent_path().parent_path();
    const fs::path manifest_path = run_root / "manifest.yaml";
    if (!fs::is_regular_file(manifest_path)) {
        return;
    }

    YAML::Node manifest;
    try {
        manifest = YAML::LoadFile(manifest_path.string());
    }
    catch (const std::exception &error) {
        throw citlali::error::io(
            "cannot read contract-defined TolAPT manifest " +
            manifest_path.string() + ": " + error.what());
    }
    if (!manifest.IsMap()) {
        throw citlali::error::io(
            "contract-defined TolAPT manifest must be a mapping");
    }
    const std::string contract_version = require_scalar_text(
        manifest, "contract_version", "TolAPT manifest");
    if (contract_version != tolapt_run_contract_version) {
        throw citlali::error::io(
            "unsupported TolAPT run output contract version " +
            contract_version);
    }
    const std::string run_id = require_scalar_text(
        manifest, "run_id", "TolAPT manifest");
    const auto inputs = require_unique_map_entry(
        manifest, "inputs", "TolAPT manifest");
    if (!inputs.IsMap()) {
        throw citlali::error::io("TolAPT manifest inputs must be a mapping");
    }
    const auto design_input = parse_tolapt_input_record(inputs, "design_apt");
    const auto measured_input =
        parse_tolapt_input_record(inputs, "measured_apt");
    const auto outputs = require_unique_map_entry(
        manifest, "outputs", "TolAPT manifest");
    if (!outputs.IsMap()) {
        throw citlali::error::io("TolAPT manifest outputs must be a mapping");
    }

    std::vector<std::pair<std::string, std::string>> matches;
    for (const auto &entry : outputs) {
        if (!entry.first.IsScalar() || !entry.second.IsScalar()) {
            throw citlali::error::io(
                "TolAPT manifest outputs must be scalar path associations");
        }
        const std::string key = entry.first.as<std::string>();
        const std::string relative_text = entry.second.as<std::string>();
        const fs::path relative_path{relative_text};
        if (relative_path.empty() || relative_path.is_absolute() ||
            contains_parent_component(relative_path)) {
            throw citlali::error::io(
                "TolAPT manifest output path is not a safe run-relative path: " +
                relative_text);
        }
        const fs::path declared =
            fs::weakly_canonical(run_root / relative_path);
        if (declared == selected) {
            matches.emplace_back(key, relative_path.generic_string());
        }
    }
    if (matches.size() != 1) {
        throw citlali::error::io(
            "contract-defined TolAPT manifest must associate the selected APT output exactly once");
    }

    lineage.modern_tolapt_manifest_available = true;
    lineage.modern_tolapt_manifest_path = manifest_path.string();
    lineage.modern_tolapt_manifest_sha256 =
        citlali::utils::sha256_file(manifest_path);
    lineage.modern_tolapt_contract_version =
        contract_version;
    lineage.modern_tolapt_run_id = run_id;
    lineage.modern_tolapt_output_key = matches.front().first;
    lineage.modern_tolapt_output_path = matches.front().second;
    lineage.modern_tolapt_design_input = design_input;
    lineage.modern_tolapt_measured_input = measured_input;
    std::ostringstream association;
    association << "tolapt-selected-output-association-v2";
    append_identity_field(
        association, "manifest_sha256", "sha256",
        lineage.modern_tolapt_manifest_sha256);
    append_identity_field(
        association, "contract_version", "string",
        lineage.modern_tolapt_contract_version);
    append_identity_field(
        association, "run_id", "string", lineage.modern_tolapt_run_id);
    append_input_record_identity(
        association, "inputs.design_apt", design_input);
    append_input_record_identity(
        association, "inputs.measured_apt", measured_input);
    append_identity_field(
        association, "output_key", "string",
        lineage.modern_tolapt_output_key);
    append_identity_field(
        association, "output_path", "run_relative_path",
        lineage.modern_tolapt_output_path);
    append_identity_field(
        association, "selected_output_sha256", "sha256",
        lineage.selected_apt_sha256);
    lineage.modern_tolapt_association_sha256 =
        citlali::utils::sha256(association.str());
}

engine::Calib::AptLineage read_selected_apt_lineage(
    const std::string &filepath, const std::string &artifact_sha256) {
    engine::Calib::AptLineage lineage;
    lineage.available = true;
    lineage.selected_apt_path =
        std::filesystem::weakly_canonical(filepath).string();
    lineage.selected_apt_sha256 = artifact_sha256;
    const auto metadata = read_ecsv_metadata(filepath);

    const auto obsnum = unique_metadata_scalar(metadata, "obsnum");
    const auto header_obsnum =
        unique_metadata_scalar(metadata, "Header ObsNum");
    if (obsnum && header_obsnum && *obsnum != *header_obsnum) {
        throw citlali::error::io(
            "selected APT obsnum conflicts with Header ObsNum");
    }
    lineage.observation_identity =
        obsnum.value_or(header_obsnum.value_or(""));
    lineage.matched_observation_identity =
        unique_metadata_scalar(metadata, "obsnum_matched").value_or("");
    lineage.selected_source =
        unique_metadata_scalar(metadata, "source").value_or("");
    lineage.legacy_metadata_available =
        !lineage.observation_identity.empty() ||
        !lineage.matched_observation_identity.empty() ||
        !lineage.selected_source.empty();

    attach_contract_defined_tolapt_manifest(lineage);
    return lineage;
}

std::string numeric_identity(double value) {
    if (!std::isfinite(value)) {
        return "nonfinite";
    }
    std::ostringstream stream;
    stream << std::setprecision(std::numeric_limits<double>::max_digits10)
           << value;
    return stream.str();
}

const Eigen::VectorXd *optional_apt_column(
    const std::map<std::string, Eigen::VectorXd> &table,
    const std::string &name, Eigen::Index row_count) {
    const auto found = table.find(name);
    if (found == table.end()) {
        return nullptr;
    }
    if (found->second.size() != row_count) {
        throw citlali::error::io(
            "APT optional-column cardinality mismatch for " + name);
    }
    return &found->second;
}

}  // namespace

namespace engine {

void Calib::get_apt(const std::string &filepath, std::vector<std::string> &raw_filenames, std::vector<std::string> &interfaces) {
    apt_acquisition_binding = {};
    apt_lineage = {};
    // store apt filepath
    apt_filepath = filepath;
    apt_acquisition_binding.artifact_sha256 =
        citlali::utils::sha256_file(filepath);
    apt_lineage = read_selected_apt_lineage(
        filepath, apt_acquisition_binding.artifact_sha256);
    // read in the apt table
    auto [apt_temp, header, map_with_strs] = to_map_from_ecsv_mixted_type(filepath);

    // vector to hold any missing header keys
    std::vector<std::string> missing_header_keys, empty_header_keys;

    // look for missing header keys by comparing to required keys
    for (auto &apt_header_key: apt_header_keys) {
        bool found = std::find(header.begin(), header.end(), apt_header_key) != header.end();
        if (!found) {
            missing_header_keys.push_back(apt_header_key);
        }
        // look for empty headers
        else if (apt_temp[apt_header_key].size()==0) {
            empty_header_keys.push_back(apt_header_key);
        }
    }

    // reject tables with missing required columns
    if (!missing_header_keys.empty()) {
        throw citlali::error::io(
            "APT table is missing required columns: [" +
            join_column_names(missing_header_keys) + "]");
    }

    // reject tables with empty required columns
    if (!empty_header_keys.empty()) {
        throw citlali::error::io(
            "APT table columns are empty: [" +
            join_column_names(empty_header_keys) + "]");
    }

    // pointing calculations require an altaz APT reference frame
    if (map_with_strs["Radesys"]!="altaz") {
        throw citlali::error::io(
            "APT table reference frame must be altaz");
    }

    if (raw_filenames.empty() || raw_filenames.size() != interfaces.size()) {
        throw citlali::error::io(
            "APT acquisition binding requires one raw file for every interface");
    }

    const Eigen::Index apt_row_count = apt_temp["nw"].size();
    const auto association_columns =
        read_selected_apt_association_columns(filepath, apt_row_count);
    for (const auto &key : apt_header_keys) {
        if (apt_temp[key].size() != apt_row_count) {
            throw citlali::error::io(
                "APT required-column cardinality mismatch for " + key);
        }
    }

    struct RawNetworkIdentity {
        int network = -1;
        Eigen::VectorXd tone_frequency_hz;
    };
    std::vector<RawNetworkIdentity> raw_networks;
    raw_networks.reserve(raw_filenames.size());
    apt_acquisition_binding.raw_artifacts.clear();
    apt_acquisition_binding.raw_artifacts.reserve(raw_filenames.size());
    std::set<int> seen_networks;
    Eigen::Index n_dets_temp = 0;

    for (std::size_t index = 0; index < raw_filenames.size(); ++index) {
        netCDF::NcFile fo(raw_filenames[index], netCDF::NcFile::read);
        const int interface_network = interface_network_id(interfaces[index]);
        int roach_index = -1;
        fo.getVar("Header.Toltec.RoachIndex").getVar(&roach_index);
        if (roach_index != interface_network) {
            throw citlali::error::io(
                "raw-file RoachIndex disagrees with interface identity for " +
                interfaces[index]);
        }
        if (!seen_networks.insert(roach_index).second) {
            throw citlali::error::io(
                "duplicate TolTEC network in APT acquisition binding: " +
                std::to_string(roach_index));
        }

        const auto signal = fo.getVar("Data.Toltec.Is");
        const auto tone_var = fo.getVar("Header.Toltec.ToneFreq");
        const auto lo_var = fo.getVar("Header.Toltec.LoCenterFreq");
        if (signal.isNull() || signal.getDimCount() < 2 || tone_var.isNull() ||
            tone_var.getDimCount() < 2 || lo_var.isNull()) {
            throw citlali::error::io(
                "raw acquisition identity is unavailable for network " +
                std::to_string(roach_index));
        }
        const Eigen::Index detector_count = static_cast<Eigen::Index>(
            signal.getDim(1).getSize());
        const Eigen::Index sweep_count = static_cast<Eigen::Index>(
            tone_var.getDim(0).getSize());
        const Eigen::Index tone_count = static_cast<Eigen::Index>(
            tone_var.getDim(1).getSize());
        if (detector_count <= 0 || sweep_count <= 0 ||
            tone_count != detector_count) {
            throw citlali::error::io(
                "raw detector/tone cardinality mismatch for network " +
                std::to_string(roach_index));
        }
        Eigen::MatrixXd tone_frequency(tone_count, sweep_count);
        tone_var.getVar(tone_frequency.data());
        double lo_frequency = std::numeric_limits<double>::quiet_NaN();
        lo_var.getVar(&lo_frequency);
        if (!std::isfinite(lo_frequency)) {
            throw citlali::error::io(
                "non-finite raw LO frequency for network " +
                std::to_string(roach_index));
        }
        RawNetworkIdentity identity;
        identity.network = roach_index;
        identity.tone_frequency_hz =
            tone_frequency.col(0).array() + lo_frequency;
        if (!identity.tone_frequency_hz.array().isFinite().all()) {
            throw citlali::error::io(
                "non-finite raw tone frequency for network " +
                std::to_string(roach_index));
        }
        for (Eigen::Index left = 0; left < detector_count; ++left) {
            for (Eigen::Index right = left + 1; right < detector_count; ++right) {
                if (same_frequency_identity(identity.tone_frequency_hz(left),
                                            identity.tone_frequency_hz(right))) {
                    throw citlali::error::io(
                        "duplicate raw acquisition tone key for network " +
                        std::to_string(roach_index));
                }
            }
        }
        n_dets_temp += detector_count;
        AptAcquisitionBinding::RawArtifact artifact;
        artifact.path = raw_filenames[index];
        artifact.sha256 = citlali::utils::sha256_file(raw_filenames[index]);
        artifact.interface = interfaces[index];
        artifact.network = roach_index;
        artifact.absolute_tone_frequency_hz.assign(
            identity.tone_frequency_hz.data(),
            identity.tone_frequency_hz.data() +
                identity.tone_frequency_hz.size());
        apt_acquisition_binding.raw_artifacts.push_back(
            std::move(artifact));
        raw_networks.push_back(std::move(identity));
        fo.close();
    }

    std::map<std::string, Eigen::VectorXd> ordered_apt;
    for (const auto &key : apt_header_keys) {
        ordered_apt[key].setZero(n_dets_temp);
    }
    ordered_apt["kids_tone"].setZero(n_dets_temp);

    const std::vector<std::string> optional_row_columns{
        "det_id", "det_id_right", "meas_idx", "design_idx", "match_id"};
    for (const auto &key : optional_row_columns) {
        if (optional_apt_column(apt_temp, key, apt_row_count) != nullptr) {
            ordered_apt[key].setZero(n_dets_temp);
        }
    }

    Eigen::Index output_row = 0;
    std::vector<Eigen::Index> selected_source_rows(
        static_cast<std::size_t>(n_dets_temp), -1);
    std::vector<bool> used(static_cast<std::size_t>(apt_temp["nw"].size()), false);
    for (const auto &raw : raw_networks) {
        const Eigen::Index raw_count = raw.tone_frequency_hz.size();
        const Eigen::Index apt_count =
            (apt_temp["nw"].array() == raw.network).count();
        if (apt_count != raw_count) {
            throw citlali::error::io(
                "APT/raw tone cardinality mismatch for network " +
                std::to_string(raw.network));
        }
        for (Eigen::Index local_tone = 0; local_tone < raw_count;
             ++local_tone) {
            std::vector<Eigen::Index> matches;
            for (Eigen::Index apt_row = 0; apt_row < apt_temp["nw"].size();
                 ++apt_row) {
                if (!used[static_cast<std::size_t>(apt_row)] &&
                    apt_temp["nw"](apt_row) == raw.network &&
                    same_frequency_identity(
                        apt_temp["tone_freq"](apt_row),
                        raw.tone_frequency_hz(local_tone))) {
                    matches.push_back(apt_row);
                }
            }
            if (matches.size() != 1) {
                throw citlali::error::io(
                    "APT acquisition key is missing or duplicated for network " +
                    std::to_string(raw.network) + " local tone " +
                    std::to_string(local_tone));
            }
            const Eigen::Index apt_row = matches.front();
            used[static_cast<std::size_t>(apt_row)] = true;
            for (const auto &key : apt_header_keys) {
                ordered_apt[key](output_row) = apt_temp[key](apt_row);
            }
            for (const auto &key : optional_row_columns) {
                const auto *column =
                    optional_apt_column(apt_temp, key, apt_row_count);
                if (column != nullptr) {
                    ordered_apt[key](output_row) = (*column)(apt_row);
                }
            }
            ordered_apt["kids_tone"](output_row) =
                static_cast<double>(local_tone);
            selected_source_rows[static_cast<std::size_t>(output_row)] =
                apt_row;
            ++output_row;
        }
    }
    if (output_row != apt_row_count ||
        std::find(used.begin(), used.end(), false) != used.end()) {
        throw citlali::error::io(
            "APT contains acquisition keys not present in the raw observation");
    }

    apt = std::move(ordered_apt);
    std::ostringstream row_associations;
    row_associations << "selected-apt-row-association-v2";
    append_identity_field(
        row_associations, "apt_sha256", "sha256",
        apt_acquisition_binding.artifact_sha256);
    apt_lineage.ordered_rows.clear();
    apt_lineage.ordered_rows.reserve(static_cast<std::size_t>(n_dets_temp));
    for (Eigen::Index row = 0; row < n_dets_temp; ++row) {
        AptRowLineage lineage_row;
        lineage_row.ordered_detector_index = row;
        lineage_row.selected_source_row_index =
            selected_source_rows[static_cast<std::size_t>(row)];
        const auto source_row = lineage_row.selected_source_row_index;
        const auto datatype_for = [&](const std::string &name) {
            const auto found = association_columns.datatypes.find(name);
            if (found == association_columns.datatypes.end()) {
                throw citlali::error::io(
                    "selected APT row-lineage datatype is unavailable for " +
                    name);
            }
            return found->second;
        };
        const auto retain = [&](const std::string &name,
                                const std::string &value) {
            lineage_row.retained_fields.push_back(
                {name, datatype_for(name), value});
        };
        const auto selected_string = [&](const std::string &name)
            -> std::optional<std::string> {
            const auto found = association_columns.string_values.find(name);
            if (found == association_columns.string_values.end()) {
                return std::nullopt;
            }
            if (source_row < 0 ||
                source_row >= static_cast<Eigen::Index>(found->second.size())) {
                throw citlali::error::io(
                    "selected APT string row-lineage index is out of range for " +
                    name);
            }
            return found->second[static_cast<std::size_t>(source_row)];
        };
        lineage_row.uid = numeric_identity(apt["uid"](row));
        retain("uid", lineage_row.uid);
        retain("flag", numeric_identity(apt["flag"](row)));
        if (apt.count("det_id") != 0U) {
            lineage_row.det_id = numeric_identity(apt["det_id"](row));
            retain("det_id", lineage_row.det_id);
        }
        else if (const auto value = selected_string("det_id")) {
            lineage_row.det_id = *value;
            retain("det_id", lineage_row.det_id);
        }
        if (apt.count("det_id_right") != 0U) {
            lineage_row.det_id_right =
                numeric_identity(apt["det_id_right"](row));
            retain("det_id_right", lineage_row.det_id_right);
        }
        if (apt.count("meas_idx") != 0U) {
            lineage_row.measured_row_id =
                numeric_identity(apt["meas_idx"](row));
            retain("meas_idx", lineage_row.measured_row_id);
        }
        if (apt.count("design_idx") != 0U) {
            lineage_row.design_row_id =
                numeric_identity(apt["design_idx"](row));
            retain("design_idx", lineage_row.design_row_id);
        }
        if (apt.count("match_id") != 0U) {
            lineage_row.modern_match_id =
                numeric_identity(apt["match_id"](row));
            retain("match_id", lineage_row.modern_match_id);
        }
        if (const auto value = selected_string("measured_id")) {
            lineage_row.measured_id = *value;
            retain("measured_id", lineage_row.measured_id);
        }
        if (const auto value = selected_string("matched_design_id")) {
            lineage_row.matched_design_id = *value;
            retain("matched_design_id", lineage_row.matched_design_id);
        }
        if (const auto value = selected_string("match_status")) {
            lineage_row.match_status = *value;
            retain("match_status", lineage_row.match_status);
        }
        lineage_row.eligible = apt["flag"](row) == 0.0;
        lineage_row.validity_basis = "selected_APT_flag_eq_0";
        if (apt.count("det_id_right") != 0U) {
            const bool tone_match_eligible =
                std::isfinite(apt["det_id_right"](row)) &&
                apt["det_id_right"](row) >= 0.0;
            lineage_row.eligible =
                lineage_row.eligible && tone_match_eligible;
            lineage_row.validity_basis +=
                ";det_id_right_nonnegative_when_present";
        }
        if (apt_lineage.modern_tolapt_manifest_available &&
            apt.count("match_id") != 0U) {
            const bool tolapt_match_eligible =
                std::isfinite(apt["match_id"](row)) &&
                apt["match_id"](row) >= 0.0;
            lineage_row.eligible =
                lineage_row.eligible && tolapt_match_eligible;
            lineage_row.validity_basis +=
                ";tolapt_match_id_nonnegative_when_present";
        }
        if (apt_lineage.modern_tolapt_manifest_available &&
            association_columns.string_values.count("match_status") != 0U) {
            if (lineage_row.match_status != "matched" &&
                lineage_row.match_status != "unmatched" &&
                lineage_row.match_status != "excluded_inactive") {
                throw citlali::error::io(
                    "selected TolAPT row has unsupported match_status " +
                    lineage_row.match_status);
            }
            const bool status_eligible =
                lineage_row.match_status == "matched";
            if (apt.count("match_id") != 0U) {
                const bool match_id_eligible =
                    std::isfinite(apt["match_id"](row)) &&
                    apt["match_id"](row) >= 0.0;
                if (status_eligible != match_id_eligible) {
                    throw citlali::error::io(
                        "selected TolAPT row has conflicting match_id and match_status validity");
                }
            }
            lineage_row.eligible =
                lineage_row.eligible && status_eligible;
            lineage_row.validity_basis +=
                ";tolapt_match_status_matched_when_present";
        }
        if (apt_lineage.modern_tolapt_manifest_available &&
            association_columns.string_values.count("match_status") != 0U &&
            association_columns.string_values.count("matched_design_id") !=
                0U) {
            const bool status_eligible =
                lineage_row.match_status == "matched";
            const bool design_id_eligible =
                !lineage_row.matched_design_id.empty();
            if (status_eligible != design_id_eligible) {
                throw citlali::error::io(
                    "selected TolAPT row has conflicting match_status and matched_design_id association");
            }
        }
        if (apt_lineage.modern_tolapt_manifest_available &&
            association_columns.string_values.count("matched_design_id") !=
                0U &&
            apt.count("match_id") != 0U) {
            const bool design_id_eligible =
                !lineage_row.matched_design_id.empty();
            const bool match_id_eligible =
                std::isfinite(apt["match_id"](row)) &&
                apt["match_id"](row) >= 0.0;
            if (design_id_eligible != match_id_eligible) {
                throw citlali::error::io(
                    "selected TolAPT row has conflicting match_id and matched_design_id association");
            }
        }
        std::ostringstream stable;
        stable << "selected-apt-ordered-row-v2";
        append_identity_field(
            stable, "ordered_detector_index", "index",
            std::to_string(lineage_row.ordered_detector_index));
        append_identity_field(
            stable, "selected_source_row_index", "index",
            std::to_string(lineage_row.selected_source_row_index));
        for (const auto &field : lineage_row.retained_fields) {
            append_identity_field(
                stable, field.name, field.ecsv_datatype, field.value);
        }
        append_identity_field(
            stable, "eligible", "bool",
            lineage_row.eligible ? "true" : "false");
        append_identity_field(
            stable, "validity_basis", "string",
            lineage_row.validity_basis);
        lineage_row.stable_association = stable.str();
        append_identity_field(
            row_associations, "ordered_row", "typed_row_association",
            lineage_row.stable_association);
        apt_lineage.ordered_rows.push_back(std::move(lineage_row));
    }
    apt_lineage.row_association_sha256 =
        citlali::utils::sha256(row_associations.str());
    apt_lineage.valid = true;
    apt_lineage.detail =
        apt_lineage.modern_tolapt_manifest_available
            ? "exact selected APT legacy facts and contract-associated TolAPT manifest retained"
            : "exact selected APT legacy facts retained; modern TolAPT lineage unavailable";
    apt_acquisition_binding.available = true;
    apt_acquisition_binding.valid = true;
    apt_acquisition_binding.mode =
        "explicit_network_local_tone_frequency_join_v1";
    apt_acquisition_binding.key_schema =
        "raw_observation_artifact+network+network_local_tone_frequency";
    apt_acquisition_binding.detail =
        "unique complete raw/APT acquisition-key join; APT row order is not authoritative";
    apt_acquisition_binding.detector_count = n_dets_temp;
    apt_acquisition_binding.network_count =
        static_cast<Eigen::Index>(raw_networks.size());
    std::ostringstream raw_identity;
    raw_identity << "raw-observation-acquisition-identity-v2";
    for (std::size_t index = 0; index < raw_networks.size(); ++index) {
        std::ostringstream artifact;
        artifact << "raw-artifact-v1";
        append_identity_field(
            artifact, "path", "string", raw_filenames[index]);
        append_identity_field(
            artifact, "sha256", "sha256",
            apt_acquisition_binding.raw_artifacts[index].sha256);
        append_identity_field(
            artifact, "interface", "string", interfaces[index]);
        append_identity_field(
            artifact, "network", "int",
            std::to_string(raw_networks[index].network));
        for (Eigen::Index tone = 0;
             tone < raw_networks[index].tone_frequency_hz.size(); ++tone) {
            append_identity_field(
                artifact, "absolute_tone_frequency_hz", "float64",
                numeric_identity(
                    raw_networks[index].tone_frequency_hz(tone)));
        }
        append_identity_field(
            raw_identity, "artifact", "typed_raw_artifact",
            artifact.str());
    }
    apt_acquisition_binding.raw_observation_identity = raw_identity.str();
    std::ostringstream binding_identity;
    binding_identity << "apt-acquisition-binding-v2";
    append_identity_field(
        binding_identity, "apt_sha256", "sha256",
        apt_acquisition_binding.artifact_sha256);
    append_identity_field(
        binding_identity, "raw_identity", "typed_raw_identity",
        raw_identity.str());
    append_identity_field(
        binding_identity, "selected_row_association_sha256", "sha256",
        apt_lineage.row_association_sha256);
    for (Eigen::Index row = 0; row < n_dets_temp; ++row) {
        std::ostringstream join;
        join << "apt-raw-ordered-join-v1";
        append_identity_field(
            join, "network", "int", numeric_identity(apt["nw"](row)));
        append_identity_field(
            join, "network_local_tone", "index",
            numeric_identity(apt["kids_tone"](row)));
        append_identity_field(
            join, "absolute_tone_frequency_hz", "float64",
            numeric_identity(apt["tone_freq"](row)));
        append_identity_field(
            join, "uid", "int64", numeric_identity(apt["uid"](row)));
        append_identity_field(
            binding_identity, "ordered_join", "typed_join", join.str());
    }
    apt_acquisition_binding.binding_sha256 =
        citlali::utils::sha256(binding_identity.str());

    // run setup on new apt table
    setup();
}

void Calib::get_hwpr(const std::string &filepath, bool sim_obs) {
    using namespace netCDF;
    using namespace netCDF::exceptions;

    try {
        // get hwp file
        NcFile fo(filepath, NcFile::read, NcFile::classic);

        // variable for whether or not hwpr is installed
        std::string hwpr_install_v;

        // get hwp install vector for sim or real obs
        if (!sim_obs) {
            hwpr_install_v = "Header.Toltec.HwpInstalled";
        }
        else {
            hwpr_install_v = "Header.Hwp.Installed";
        }

        // check if hwpr is enabled
        fo.getVar(hwpr_install_v).getVar(&run_hwpr);

        // if not enabled or running
        if (run_hwpr) {
            // get hwpr signal
            Eigen::Index n_pts = fo.getVar("Data.Hwp.").getDim(0).getSize();
            hwpr_angle.resize(n_pts);
            // hwpr signal
            fo.getVar("Data.Hwp.").getVar(hwpr_angle.data());

            // if real data
            if (!sim_obs) {
                // get hwpr time for interpolation
                hwpr_ts.resize(n_pts,6);

                // timing for hwpr (temporary)
                fo.getVar("Data.Hwp.Ts").getVar(hwpr_ts.data());
                hwpr_ts.transposeInPlace();

                // UT time for hwpr
                Eigen::Index recvt_n_pts = fo.getVar("Data.Hwp.Uts").getDim(0).getSize();
                hwpr_recvt.resize(recvt_n_pts);
                fo.getVar("Data.Hwp.Uts").getVar(hwpr_recvt.data());

                // fpga frequency
                fo.getVar("Header.Toltec.FpgaFreq").getVar(&hwpr_fpga_freq);
            }
        }

        fo.close();

    } catch (NcException &e) {
        logger->error("{}", e.what());
        throw DataIOError{fmt::format(
            "failed to load data from netCDF file {}", filepath)};
    }
}

void Calib::calc_flux_calibration(std::string units, double pixel_size_rad) {
    (void)pixel_size_rad;
    if (units != "mJy/beam") {
        throw citlali::error::invalid_config(
            "SCI-CAL-001 supports only top-of-atmosphere point-source-peak mJy/beam; unsupported unit " +
            units);
    }
    // flux conversion is per detector
    flux_conversion_factor.resize(n_dets);
    mean_flux_conversion_factor.clear();
    flux_conversion_factor.setOnes();

    // get mean flux conversion factor from all unflagged detectors
    for (Eigen::Index i=0; i<n_arrays; ++i) {
        auto array = arrays[i];
        // number of good detectors
        Eigen::Index n_good_dets = 0;
        // name of array
        std::string name = array_name_map[array];
        // loop through detectors in current array
        for (const auto &j: array_detector_indices[array]) {
            // if good
            if (apt["flag"](j)==0) {
                mean_flux_conversion_factor[name] += flux_conversion_factor(j);
                n_good_dets++;
            }
        }
        if (n_good_dets <= 0) {
            throw std::runtime_error(
                "cannot calculate mean flux conversion factor: array has no unflagged detectors");
        }
        // calculate mean flux conversion factor
        mean_flux_conversion_factor[name] = mean_flux_conversion_factor[name]/n_good_dets;
    }
}

void Calib::setup() {
    // get number of detectors
    n_dets = apt["uid"].size();

    if (n_dets <= 0) {
        throw std::runtime_error("APT table has no detectors");
    }

    const std::vector<std::string> setup_keys = {
        "uid", "nw", "array", "fg", "flag", "a_fwhm", "b_fwhm", "angle"
    };
    for (const auto &key: setup_keys) {
        auto it = apt.find(key);
        if (it == apt.end()) {
            throw std::runtime_error("APT table is missing required setup column " + key);
        }
        if (it->second.size() != n_dets) {
            throw std::runtime_error("APT column " + key + " length does not match uid length");
        }
    }

    auto read_index = [&](const std::string &key, Eigen::Index i) {
        const double value = apt[key](i);
        const double rounded = std::round(value);
        if (!std::isfinite(value) || std::abs(value - rounded) > 1e-6) {
            throw std::runtime_error("APT column " + key + " contains a non-integer group id");
        }
        return static_cast<Eigen::Index>(rounded);
    };

    auto validate_contiguous = [&](const auto &groups, const std::string &key) {
        for (const auto &[group_id, indices]: groups) {
            if (indices.empty()) {
                throw std::runtime_error("APT " + key + " group has no detector indices");
            }
            for (std::size_t i=1; i<indices.size(); ++i) {
                if (indices[i] != indices[i - 1] + 1) {
                    throw std::runtime_error(
                        "APT rows for " + key + " group are not contiguous; sort the APT before reduction");
                }
            }
        }
    };

    nw_detector_indices.clear();
    array_detector_indices.clear();
    for (Eigen::Index i=0; i<n_dets; ++i) {
        nw_detector_indices[read_index("nw", i)].push_back(i);
        array_detector_indices[read_index("array", i)].push_back(i);
    }
    validate_contiguous(nw_detector_indices, "nw");
    validate_contiguous(array_detector_indices, "array");

    // get number of networks and arrays
    n_nws = nw_detector_indices.size();
    n_arrays = array_detector_indices.size();

    // stores nw number
    nws.setZero(n_nws);
    // stores array number
    arrays.setZero(n_arrays);

    // set up network values
    nw_limits.clear();
    nw_fwhms.clear();
    nw_pas.clear();
    nw_beam_areas.clear();

    for (const auto &[key, indices]: nw_detector_indices) {
        nw_limits[key] = std::tuple<Eigen::Index, Eigen::Index>{indices.front(), indices.back() + 1};
    }

    // get average fwhms for networks
    Eigen::Index j = 0;
    for (auto const& [key, val] : nw_limits) {
        nws(j) = key;
        j++;
        nw_fwhms[key] = std::tuple<double,double>{0, 0};

        // number of good detectors
        Eigen::Index n_good_det = 0;

        // remove flagged dets
        for (const auto &idx: nw_detector_indices[key]) {
            if (apt["flag"](idx)==0) {
                std::get<0>(nw_fwhms[key]) = std::get<0>(nw_fwhms[key]) + apt["a_fwhm"](idx);
                std::get<1>(nw_fwhms[key]) = std::get<1>(nw_fwhms[key]) + apt["b_fwhm"](idx);
                n_good_det++;
            }
        }

        if (n_good_det <= 0) {
            throw std::runtime_error("APT nw group has no unflagged detectors");
        }

        std::get<0>(nw_fwhms[key]) = std::get<0>(nw_fwhms[key])/n_good_det;
        std::get<1>(nw_fwhms[key]) = std::get<1>(nw_fwhms[key])/n_good_det;

        // average of nw fwhms in both axes
        double avg_nw_fwhm = (std::get<0>(nw_fwhms[key]) + std::get<1>(nw_fwhms[key]))/2;
        // average nw beam area
        nw_beam_areas[key] = 2.*pi*pow(avg_nw_fwhm/STD_TO_FWHM,2);
    }

    // set up array values
    array_limits.clear();
    array_fwhms.clear();
    array_pas.clear();
    array_beam_areas.clear();

    for (const auto &[key, indices]: array_detector_indices) {
        array_limits[key] = std::tuple<Eigen::Index, Eigen::Index>{indices.front(), indices.back() + 1};
    }

    // get average fwhms for arrays
    j = 0;
    // loop through arrays
    for (auto const& [key, val] : array_limits) {
        arrays(j) = key;
        j++;
        array_fwhms[key] = std::tuple<double,double>{0, 0};
        array_pas[key] = 0;
        // number of good detectors
        Eigen::Index n_good_det = 0;

        // remove flagged dets
        for (const auto &idx: array_detector_indices[key]) {
            if (apt["flag"](idx)==0) {
                std::get<0>(array_fwhms[key]) = std::get<0>(array_fwhms[key]) + apt["a_fwhm"](idx);
                std::get<1>(array_fwhms[key]) = std::get<1>(array_fwhms[key]) + apt["b_fwhm"](idx);
                array_pas[key] = array_pas[key] + apt["angle"](idx);
                n_good_det++;
            }
        }

        if (n_good_det <= 0) {
            throw std::runtime_error("APT array group has no unflagged detectors");
        }

        // average fwhms and PA
        std::get<0>(array_fwhms[key]) = std::get<0>(array_fwhms[key])/n_good_det;
        std::get<1>(array_fwhms[key]) = std::get<1>(array_fwhms[key])/n_good_det;
        array_pas[key] = array_pas[key]/n_good_det;
        // average of array fwhms in both axes
        double avg_array_fwhm = (std::get<0>(array_fwhms[key]) + std::get<1>(array_fwhms[key]))/2;
        // average array beam area
        array_beam_areas[key] = 2.*pi*pow(avg_array_fwhm*FWHM_TO_STD,2);
    }

    // vector to hold unique fg's in apt
    std::vector<Eigen::Index> fg_temp;
    // init fg
    fg_temp.push_back(apt["fg"](0));

    // loop through detectors
    for (Eigen::Index i=1; i<apt["fg"].size(); ++i) {
        // map to Eigen::Vector to use any()
        Eigen::Map<Eigen::VectorXI> x(fg_temp.data(),fg_temp.size());
        // if current fg is not in fg_temp
        if (!(x.array() == apt["fg"](i)).any()) {
            // append to fg_temp
            fg_temp.push_back(apt["fg"](i));
        }
    }
    // allocate fg_temp to fg vector
    fg = Eigen::Map<Eigen::VectorXI>(fg_temp.data(),fg_temp.size());
}

} // namespace engine
