# Production translation units and numerical compiler policy shared by both
# dependency entry points. Keep dependency discovery in each entry point.
include_guard(GLOBAL)

set(CITLALI_CORE_SOURCES
    src/citlali/core/config/timestream_enums.cpp
    src/citlali/core/pipeline/ast_scan_motion.cpp
    src/citlali/core/pipeline/ast_scan_motion_alignment.cpp
    src/citlali/core/pipeline/timestream_rtc_common_mode.cpp
    src/citlali/core/pipeline/timestream_cal_atmosphere.cpp
    src/citlali/core/pipeline/timestream_cal_wvr.cpp
    src/citlali/core/pipeline/ast_rtc_coordinates.cpp
    src/citlali/core/pipeline/timestream_cal_pipeline.cpp
    src/citlali/core/mapmaking/forward_map.cpp
    src/citlali/core/mapmaking/jinc_discrete.cpp
    src/citlali/core/mapmaking/jinc_certified.cpp
    src/citlali/core/pipeline/timestream_ptc_numerics.cpp
    src/citlali/core/pipeline/ptc_inverse_noise.cpp
    src/citlali/core/pipeline/ptc_empirical_map_handoff.cpp
    src/citlali/core/pipeline/point_optical_response.cpp
    src/citlali/core/pipeline/fruit_cycle.cpp
    src/citlali/core/pipeline/timestream_ptc_pipeline.cpp
    src/citlali/core/pipeline/ptc_map_handoff.cpp
    src/citlali/core/pipeline/timestream_identity_route_context.cpp
    src/citlali/core/pipeline/output_root_lease.cpp
    src/citlali/core/pipeline/reduction_restart_checkpoint.cpp
    src/citlali/core/engine/calib.cpp
    src/citlali/core/engine/telescope.cpp
    src/citlali/core/utils/gauss_models.cpp
    src/citlali/core/timestream/ptc/sensitivity.cpp
    src/citlali/core/mapmaking/map.cpp
)

set(CITLALI_STRICT_FP_SOURCES
    src/citlali/core/pipeline/timestream_rtc_common_mode.cpp
    src/citlali/core/pipeline/timestream_cal_atmosphere.cpp
    src/citlali/core/pipeline/timestream_cal_wvr.cpp
    src/citlali/core/pipeline/ast_rtc_coordinates.cpp
    src/citlali/core/pipeline/timestream_cal_pipeline.cpp
    src/citlali/core/pipeline/timestream_ptc_numerics.cpp
    src/citlali/core/pipeline/ptc_inverse_noise.cpp
    src/citlali/core/pipeline/ptc_empirical_map_handoff.cpp
    src/citlali/core/pipeline/point_optical_response.cpp
    src/citlali/core/mapmaking/jinc_discrete.cpp
    src/citlali/core/mapmaking/jinc_certified.cpp
    src/citlali/core/pipeline/fruit_cycle.cpp
    src/citlali/core/pipeline/timestream_ptc_pipeline.cpp
)

function(citlali_add_core_sources target source_root)
    foreach(source IN LISTS CITLALI_CORE_SOURCES)
        target_sources(${target} PRIVATE "${source_root}/${source}")
    endforeach()
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        foreach(source IN LISTS CITLALI_STRICT_FP_SOURCES)
            set_source_files_properties("${source_root}/${source}" PROPERTIES
                COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
        endforeach()
    endif()
endfunction()
