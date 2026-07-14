#pragma once

#include <type_traits>
#include <utility>

namespace citlali::pipeline {

template <class Engine, class = void>
struct has_processed_timestream_plan : std::false_type {};

template <class Engine, class = void>
struct has_raw_timestream_plan : std::false_type {};

template <class Engine, class = void>
struct has_mapmaking_plan : std::false_type {};

template <class Engine, class = void>
struct has_coadd_plan : std::false_type {};

template <class Engine, class = void>
struct has_noise_plan : std::false_type {};

template <class Engine, class = void>
struct has_pointing_plan : std::false_type {};

template <class Engine>
struct has_raw_timestream_plan<
    Engine,
    std::void_t<decltype(std::declval<Engine &>().raw_timestream_plan)>>
    : std::true_type {};

template <class Engine>
inline constexpr bool has_raw_timestream_plan_v =
    has_raw_timestream_plan<Engine>::value;

template <class Engine>
struct has_processed_timestream_plan<
    Engine,
    std::void_t<decltype(
        std::declval<Engine &>().processed_timestream_plan)>>
    : std::true_type {};

template <class Engine>
inline constexpr bool has_processed_timestream_plan_v =
    has_processed_timestream_plan<Engine>::value;

template <class Engine>
struct has_mapmaking_plan<
    Engine,
    std::void_t<decltype(std::declval<Engine &>().mapmaking_plan)>>
    : std::true_type {};

template <class Engine>
inline constexpr bool has_mapmaking_plan_v =
    has_mapmaking_plan<Engine>::value;

template <class Engine>
struct has_coadd_plan<
    Engine,
    std::void_t<decltype(std::declval<Engine &>().coadd_plan)>>
    : std::true_type {};

template <class Engine>
inline constexpr bool has_coadd_plan_v = has_coadd_plan<Engine>::value;

template <class Engine>
struct has_noise_plan<
    Engine,
    std::void_t<decltype(std::declval<Engine &>().noise_plan)>>
    : std::true_type {};

template <class Engine>
inline constexpr bool has_noise_plan_v = has_noise_plan<Engine>::value;

template <class Engine>
struct has_pointing_plan<
    Engine,
    std::void_t<decltype(std::declval<Engine &>().pointing_plan)>>
    : std::true_type {};

template <class Engine>
inline constexpr bool has_pointing_plan_v =
    has_pointing_plan<Engine>::value;

template <class Engine>
auto &runtime_config_provenance(Engine &engine) {
    return engine.runtime_config_provenance;
}

template <class Engine>
const auto &runtime_config_provenance(const Engine &engine) {
    return engine.runtime_config_provenance;
}

template <class Engine>
const auto &requested_runtime_config(const Engine &engine) {
    return runtime_config_provenance(engine).requested;
}

template <class Engine>
const auto &effective_runtime_config(const Engine &engine) {
    return runtime_config_provenance(engine).effective;
}

template <class Engine>
const auto &effective_runtime_values(const Engine &engine) {
    return effective_runtime_config(engine).values;
}

template <class Engine>
const auto &realized_runtime_config(const Engine &engine) {
    return runtime_config_provenance(engine).realized;
}

template <class Engine>
auto &config_diagnostics(Engine &engine) {
    return engine.config_diagnostics;
}

template <class Engine>
const auto &config_diagnostics(const Engine &engine) {
    return engine.config_diagnostics;
}

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
auto &polarimetry_config(Engine &engine) {
    return timestream_config(engine).polarimetry;
}

template <class Engine>
const auto &polarimetry_config(const Engine &engine) {
    return timestream_config(engine).polarimetry;
}

template <class Engine>
auto &raw_time_chunk_config(Engine &engine) {
    return timestream_config(engine).raw_time_chunk;
}

template <class Engine>
const auto &raw_time_chunk_config(const Engine &engine) {
    return timestream_config(engine).raw_time_chunk;
}

template <class Engine>
auto &raw_timestream_plan(Engine &engine) {
    return engine.raw_timestream_plan;
}

template <class Engine>
const auto &raw_timestream_plan(const Engine &engine) {
    return engine.raw_timestream_plan;
}

template <class Engine>
auto &processed_time_chunk_config(Engine &engine) {
    if constexpr (has_processed_timestream_plan_v<Engine>) {
        if (engine.processed_timestream_plan.initialized) {
            return engine.processed_timestream_plan.effective
                .processed_time_chunk;
        }
    }
    return timestream_config(engine).processed_time_chunk;
}

template <class Engine>
const auto &processed_time_chunk_config(const Engine &engine) {
    if constexpr (has_processed_timestream_plan_v<Engine>) {
        if (engine.processed_timestream_plan.initialized) {
            return engine.processed_timestream_plan.effective
                .processed_time_chunk;
        }
    }
    return timestream_config(engine).processed_time_chunk;
}

template <class Engine>
auto &fruit_loops_config(Engine &engine) {
    if constexpr (has_processed_timestream_plan_v<Engine>) {
        if (engine.processed_timestream_plan.initialized) {
            return engine.processed_timestream_plan.effective.fruit_loops;
        }
    }
    return timestream_config(engine).fruit_loops;
}

template <class Engine>
const auto &fruit_loops_config(const Engine &engine) {
    if constexpr (has_processed_timestream_plan_v<Engine>) {
        if (engine.processed_timestream_plan.initialized) {
            return engine.processed_timestream_plan.effective.fruit_loops;
        }
    }
    return timestream_config(engine).fruit_loops;
}

template <class Engine>
auto &processed_timestream_plan(Engine &engine) {
    return engine.processed_timestream_plan;
}

template <class Engine>
const auto &processed_timestream_plan(const Engine &engine) {
    return engine.processed_timestream_plan;
}

template <class Engine>
auto &mapmaking_plan(Engine &engine) {
    return engine.mapmaking_plan;
}

template <class Engine>
const auto &mapmaking_plan(const Engine &engine) {
    return engine.mapmaking_plan;
}

template <class Engine>
auto &mapmaking_config(Engine &engine) {
    if constexpr (has_mapmaking_plan_v<Engine>) {
        if (engine.mapmaking_plan.initialized) {
            return engine.mapmaking_plan.effective;
        }
    }
    return reduction_config(engine).mapmaking;
}

template <class Engine>
const auto &mapmaking_config(const Engine &engine) {
    if constexpr (has_mapmaking_plan_v<Engine>) {
        if (engine.mapmaking_plan.initialized) {
            return engine.mapmaking_plan.effective;
        }
    }
    return reduction_config(engine).mapmaking;
}

template <class Engine>
auto &coadd_config(Engine &engine) {
    if constexpr (has_coadd_plan_v<Engine>) {
        if (engine.coadd_plan.initialized) {
            return engine.coadd_plan.effective;
        }
    }
    return reduction_config(engine).coadd;
}

template <class Engine>
const auto &coadd_config(const Engine &engine) {
    if constexpr (has_coadd_plan_v<Engine>) {
        if (engine.coadd_plan.initialized) {
            return engine.coadd_plan.effective;
        }
    }
    return reduction_config(engine).coadd;
}

template <class Engine>
auto &coadd_plan(Engine &engine) {
    return engine.coadd_plan;
}

template <class Engine>
const auto &coadd_plan(const Engine &engine) {
    return engine.coadd_plan;
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
    if constexpr (has_pointing_plan_v<Engine>) {
        if (engine.pointing_plan.initialized) {
            return engine.pointing_plan.effective;
        }
    }
    return reduction_config(engine).pointing;
}

template <class Engine>
const auto &pointing_config(const Engine &engine) {
    if constexpr (has_pointing_plan_v<Engine>) {
        if (engine.pointing_plan.initialized) {
            return engine.pointing_plan.effective;
        }
    }
    return reduction_config(engine).pointing;
}

template <class Engine>
auto &pointing_plan(Engine &engine) {
    return engine.pointing_plan;
}

template <class Engine>
const auto &pointing_plan(const Engine &engine) {
    return engine.pointing_plan;
}

template <class Engine>
auto &noise_config(Engine &engine) {
    if constexpr (has_noise_plan_v<Engine>) {
        if (engine.noise_plan.initialized) {
            return engine.noise_plan.effective;
        }
    }
    return reduction_config(engine).noise;
}

template <class Engine>
const auto &noise_config(const Engine &engine) {
    if constexpr (has_noise_plan_v<Engine>) {
        if (engine.noise_plan.initialized) {
            return engine.noise_plan.effective;
        }
    }
    return reduction_config(engine).noise;
}

template <class Engine>
auto &noise_plan(Engine &engine) {
    return engine.noise_plan;
}

template <class Engine>
const auto &noise_plan(const Engine &engine) {
    return engine.noise_plan;
}

template <class Engine>
auto &post_processing_config(Engine &engine) {
    return reduction_config(engine).post_processing;
}

template <class Engine>
const auto &post_processing_config(const Engine &engine) {
    return reduction_config(engine).post_processing;
}

template <class Engine>
auto &astrometry_config(Engine &engine) {
    return reduction_config(engine).astrometry;
}

template <class Engine>
const auto &astrometry_config(const Engine &engine) {
    return reduction_config(engine).astrometry;
}

}  // namespace citlali::pipeline
