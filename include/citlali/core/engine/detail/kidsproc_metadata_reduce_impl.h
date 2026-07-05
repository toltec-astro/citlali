#pragma once

// Implementation detail included by kidsproc.h.

auto KidsDataProc::get_data_item_meta(const RawObs::DataItem &data_item) {
    namespace kidsdata = predefs::kidsdata;
    auto source = data_item.filepath();
    predefs::suppress_hdf5_diagnostics_for_this_thread();
    std::lock_guard<std::mutex> lock(predefs::netcdf_io_mutex());
    auto [kind, meta] = kidsdata::get_meta<>(source);
    return meta;
}

std::vector<kids::KidsData<>::meta_t> KidsDataProc::get_rawobs_meta(const RawObs &rawobs) {
    std::vector<kids::KidsData<>::meta_t> result;
    for (const auto &data_item : rawobs.kidsdata()) {
        result.push_back(get_data_item_meta(data_item));
    }
    return result;
}

auto KidsDataProc::populate_rtc_meta(const RawObs &rawobs) {
    std::vector<kids::KidsData<>::meta_t> result;
    for (const auto &data_item : rawobs.kidsdata()) {
        result.push_back(get_data_item_meta(data_item));
    }
    return result;
}

auto KidsDataProc::reduce_data_item(const RawObs::DataItem &data_item,
                                    const tula::container_utils::Slice<int> &slice) {
    logger->debug("kids reduce data_item {}", data_item);
    // read data
    namespace kidsdata = predefs::kidsdata;
    auto source = data_item.filepath();
    kids::KidsDataKind kind;
    kids::KidsData<>::meta_t meta;
    {
        predefs::suppress_hdf5_diagnostics_for_this_thread();
        std::lock_guard<std::mutex> lock(predefs::netcdf_io_mutex());
        auto km = kidsdata::get_meta<>(source);
        kind = km.first;
        meta = std::move(km.second);
    }
    if (!(kind & kids::KidsDataKind::TimeStream)) {
        throw std::runtime_error(
            fmt::format("wrong type of kids data {}", kind));
    }
    kids::KidsData<kids::KidsDataKind::RawTimeStream> rts;
    try {
        predefs::suppress_hdf5_diagnostics_for_this_thread();
        std::lock_guard<std::mutex> lock(predefs::netcdf_io_mutex());
        rts = kidsdata::read_data_slice<kids::KidsDataKind::RawTimeStream>(
            source, slice);
    } catch (const std::exception &e) {
        throw std::runtime_error(fmt::format(
            "failed to read raw timestream slice from {} slice {}: {}",
            source, slice, e.what()));
    }
    auto result = this->solver()(rts, Solver::Config{});
    return result;
}

auto KidsDataProc::reduce_rawobs(const RawObs &rawobs,
                                 const tula::container_utils::Slice<int> &slice) {
    logger->debug("kids reduce rawobs {}", rawobs);
    std::vector<kids::TimeStreamSolverResult> result;
    for (const auto &data_item : rawobs.kidsdata()) {
        result.push_back(reduce_data_item(data_item, slice));
    }
    return result;
}

auto KidsDataProc::load_data_item(const RawObs::DataItem &data_item,
                                  const tula::container_utils::Slice<int> &slice) {
    logger->debug("kids reduce data_item {}", data_item);
    // read data
    namespace kidsdata = predefs::kidsdata;
    auto source = data_item.filepath();
    kids::KidsDataKind kind;
    if (auto it = m_data_item_kind_cache.find(source); it != m_data_item_kind_cache.end()) {
        kind = it->second;
    }
    else {
        predefs::suppress_hdf5_diagnostics_for_this_thread();
        std::lock_guard<std::mutex> lock(predefs::netcdf_io_mutex());
        auto [kind_, meta] = kidsdata::get_meta<>(source);
        kind = kind_;
        m_data_item_kind_cache[source] = kind;
    }
    if (!(kind & kids::KidsDataKind::TimeStream)) {
        throw std::runtime_error(
            fmt::format("wrong type of kids data {}", kind));
    }
    kids::KidsData<kids::KidsDataKind::RawTimeStream> rts;
    try {
        predefs::suppress_hdf5_diagnostics_for_this_thread();
        std::lock_guard<std::mutex> lock(predefs::netcdf_io_mutex());
        rts = kidsdata::read_data_slice<kids::KidsDataKind::RawTimeStream>(
            source, slice);
    } catch (const std::exception &e) {
        throw std::runtime_error(fmt::format(
            "failed to read raw timestream slice from {} slice {}: {}",
            source, slice, e.what()));
    }
    return rts;
}

auto KidsDataProc::load_fit_report(const RawObs &rawobs) {
    std::vector<Eigen::MatrixXd> kids_models;
    std::vector<std::string> header;

    for (const auto &data_item : rawobs.kidsdata()) {
        auto meta = get_data_item_meta(data_item);
        //auto fitreport = this->solver().loadfitreport(this->config(),meta);

        namespace fs = std::filesystem;
        auto pattern = meta.get_str("cal_file");
        std::string filepath{};
        if (this->solver().config.has("fitreportfile")) {
            filepath = this->solver().config.get_str("fitreportfile");
        } else if (this->solver().config.has("fitreportdir")) {
            auto dir = this->solver().config.get_str("fitreportdir");
            logger->info("look for fitreport dir {} with pattern {}", dir, pattern);
            auto candidates = tula::filename_utils::find_regex(dir, pattern);
            if (!candidates.empty()) {
                filepath = candidates[0];
            } else {
                throw std::runtime_error(fmt::format(
                    "no fit report found in {} that matches {}", dir, pattern));
            }
        } else {
            throw std::runtime_error(
                fmt::format("no fit report location specified."));
        }
        logger->info("use fitreport file {}", filepath);
        //std::vector<std::string> header;
        header.clear();
        Eigen::MatrixXd table;
        using meta_t = kids::KidsData<>::meta_t;
        meta_t meta_cal{};

        try {
            YAML::Node meta_;
            table = datatable::read<double, datatable::Format::ecsv>(
                filepath, &header, &meta_);
            auto meta_map =
                tula::ecsv::meta_to_map<typename meta_t::storage_t::key_type,
                                        typename meta_t::storage_t::mapped_type>(
                    meta_, &meta_);
            meta_cal = meta_t{std::move(meta_map)};

            kids_models.push_back(std::move(table));
            if (!meta_.IsNull()) {
                logger->warn("un recongnized meta:\n{}", YAML::Dump(meta_));
            }
        } catch (datatable::ParseError &e) {
            logger->warn("unable to read fitreport file as ECSV {}: {}", filepath,
                        e.what());
            try {
                table = datatable::read<double, datatable::Format::ascii>(filepath,
                                                                          &header);
                kids_models.push_back(std::move(table));

            } catch (datatable::ParseError &e) {
                logger->warn("unable to read fitreport file as ASCII {}: {}",
                            filepath, e.what());
                throw e;
            }
        }
        logger->info("meta_cal: {}", meta_cal.pformat());
        logger->info("table {}",table);
        logger->info("header {}",header);

        //return std::tuple{
        //                  kids::ToneAxis(std::move(table).transpose(), std::move(header)),
        //                  std::move(meta_cal)};
    }

    return std::tuple{std::move(kids_models), std::move(header)};
}

