#pragma once

// Implementation detail included by io.h.

void RawObs::collect_data_items() {
    m_data_items.clear();
    std::vector<DataItem> data_items{};
    auto node_data_items = this->config().get_node("data_items");
    auto n_data_items = node_data_items.size();
    for (std::size_t i = 0; i < n_data_items; ++i) {
        SPDLOG_INFO("add data item {} of {}", i, n_data_items);
        data_items.emplace_back(
            config_t{node_data_items[i], this->config().filepath()});
    }
    m_data_items = std::move(data_items);

    sort(m_data_items.begin(), m_data_items.end(), [] (const RawObs::DataItem& a, const RawObs::DataItem& b) {
        auto key = [](const RawObs::DataItem& item) {
            const auto& iface = item.interface();
            if (iface.rfind("toltec", 0) == 0) {
                int idx = 0;
                try {
                    idx = std::stoi(iface.substr(6));
                } catch (...) {
                    idx = 0;
                }
                return std::tuple<int, int, std::string>{0, idx, iface};
            }
            if (iface == "lmt") {
                return std::tuple<int, int, std::string>{1, 0, iface};
            }
            if (iface == "hwpr") {
                return std::tuple<int, int, std::string>{2, 0, iface};
            }
            return std::tuple<int, int, std::string>{3, 0, iface};
        };
        return key(a) < key(b);
    });

    SPDLOG_DEBUG("collected n_data_items={}\n{}", this->n_data_items(),
                 this->data_items());
    // update the data indices
    m_kidsdata_indices.clear();
    m_teldata_index.reset();
    m_hwpdata_index.reset();
    std::smatch m;
    for (std::size_t i = 0; i < m_data_items.size(); ++i) {
        if (std::regex_match(m_data_items[i].interface(), m,
                             re_interface_kidsdata)) {
            m_kidsdata_indices.push_back(i);
        }
        if (std::regex_match(m_data_items[i].interface(), m,
                             re_interface_teldata)) {
            if (m_teldata_index.has_value()) {
                throw std::runtime_error("found too many telescope data items");
            }
            m_teldata_index = i;
        }
        if (std::regex_match(m_data_items[i].interface(), m,
                             re_interface_hwpdata)) {
            if (m_hwpdata_index.has_value()) {
                throw std::runtime_error(
                    "found too many halfwave plate data items");
            }
            m_hwpdata_index = i;
        }
    }
    if (!m_teldata_index) {
        throw std::runtime_error("no telescope data item found");
    }
    // The hwp data is optional
    if (!m_hwpdata_index) {
        SPDLOG_INFO("no hwp data item found");
    }
    SPDLOG_INFO("kidsdata_indices={} teldata_index={} hwpdata_index={}",
                 m_kidsdata_indices, m_teldata_index, m_hwpdata_index);
    SPDLOG_INFO("kidsdata={} teldata={} hwpdata={}", kidsdata(), teldata(),
                 hwpdata());
}

// collect cal items impl

void RawObs::collect_cal_items() {
    m_cal_items.clear();
    std::vector<CalItem> cal_items{};
    auto node_cal_items = this->config().get_node("cal_items");
    auto n_cal_items = node_cal_items.size();
    for (std::size_t i = 0; i < n_cal_items; ++i) {
        cal_items.emplace_back(
            config_t{node_cal_items[i], this->config().filepath()});
    }
    m_cal_items = std::move(cal_items);
    SPDLOG_DEBUG("collected n_cal_items={}\n{}", this->n_cal_items(),
                 this->cal_items());
    // update the data indices
    m_apt_index.reset();
    m_phot_cal_index.reset();
    m_astro_cal_index.reset();
    m_flxscale_corr_index.reset();
    for (std::size_t i = 0; i < m_cal_items.size(); ++i) {
        if (m_cal_items[i].is_type<CalItemType::array_prop_table>()) {
            if (m_apt_index.has_value()) {
                throw std::runtime_error("found too many array prop tables");
            }
            m_apt_index = i;
        }
        if (m_cal_items[i].is_type<CalItemType::photometry>()) {
            if (m_phot_cal_index.has_value()) {
                throw std::runtime_error("found too many photometry calib info.");
            }
            m_phot_cal_index = i;
        }
        if (m_cal_items[i].is_type<CalItemType::astrometry>()) {
            if (m_astro_cal_index.has_value()) {
                throw std::runtime_error("found too many astrometry calib info.");
            }
            m_astro_cal_index = i;
        }
        if (m_cal_items[i].is_type<CalItemType::flxscale_correction>()) {
            if (m_flxscale_corr_index.has_value()) {
                throw std::runtime_error(
                    "found too many flxscale correction items.");
            }
            m_flxscale_corr_index = i;
        }
    }
    if (!m_apt_index) {
        throw std::runtime_error("no array prop table found");
    }
    SPDLOG_INFO("apt_index={}", m_apt_index);
    SPDLOG_INFO("apt={}", array_prop_table());
    if (m_flxscale_corr_index) {
        SPDLOG_INFO("flxscale_correction={}", *flxscale_correction());
    }
}
/**
 * @brief The Coordinator struct
 * This wraps around the config object and provides
 * high level methods in various ways to setup the MPI runtime
 * with node-local and cross-node environment.
 */

