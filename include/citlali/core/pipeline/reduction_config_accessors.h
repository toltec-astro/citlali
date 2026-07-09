#pragma once

namespace citlali::pipeline {

template <class Engine>
auto &reduction_config(Engine &engine) {
    return engine.typed_config;
}

template <class Engine>
const auto &reduction_config(const Engine &engine) {
    return engine.typed_config;
}

template <class Engine>
auto &runtime_config(Engine &engine) {
    return reduction_config(engine).runtime;
}

template <class Engine>
const auto &runtime_config(const Engine &engine) {
    return reduction_config(engine).runtime;
}

template <class Engine>
auto &timestream_config(Engine &engine) {
    return reduction_config(engine).timestream;
}

template <class Engine>
const auto &timestream_config(const Engine &engine) {
    return reduction_config(engine).timestream;
}

template <class Engine>
auto &mapmaking_config(Engine &engine) {
    return reduction_config(engine).mapmaking;
}

template <class Engine>
const auto &mapmaking_config(const Engine &engine) {
    return reduction_config(engine).mapmaking;
}

template <class Engine>
auto &coadd_config(Engine &engine) {
    return reduction_config(engine).coadd;
}

template <class Engine>
const auto &coadd_config(const Engine &engine) {
    return reduction_config(engine).coadd;
}

template <class Engine>
auto &beammap_config(Engine &engine) {
    return reduction_config(engine).beammap;
}

template <class Engine>
const auto &beammap_config(const Engine &engine) {
    return reduction_config(engine).beammap;
}

template <class Engine>
auto &pointing_config(Engine &engine) {
    return reduction_config(engine).pointing;
}

template <class Engine>
const auto &pointing_config(const Engine &engine) {
    return reduction_config(engine).pointing;
}

template <class Engine>
auto &noise_config(Engine &engine) {
    return reduction_config(engine).noise;
}

template <class Engine>
const auto &noise_config(const Engine &engine) {
    return reduction_config(engine).noise;
}

template <class Engine>
auto &post_processing_config(Engine &engine) {
    return reduction_config(engine).post_processing;
}

template <class Engine>
const auto &post_processing_config(const Engine &engine) {
    return reduction_config(engine).post_processing;
}

}  // namespace citlali::pipeline
