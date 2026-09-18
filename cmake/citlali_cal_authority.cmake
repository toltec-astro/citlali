# Embed exact approved bytes, never regenerate the scientific nodes.
function(citlali_bind_cal_authority source_root)
    set(root "${source_root}/doc/scientific_contracts/audits/WP7_TIMESTREAM_CLEAN_ROOM_F01E22F5F/REPAIR_AND_CLOSURE/RECOVERED_CAL_NUMERICAL_AUTHORITY_2026-08-25/sources")
    set(cal "${root}/citlali/validation/sci_cal_001_atmosphere_operator_2026-08-01")
    set(files
        "${cal}/sci_cal_001_fixed_djf25_full_domain_operator_contract.json"
        "${cal}/sci_cal_001_fixed_djf25_full_domain_operator_nodes.csv"
        "${root}/tolteca/tolteca/data/cal/toltec_passband/index.yaml"
        "${root}/tolteca/tolteca/data/cal/toltec_passband/data/a1100_passband.ecsv"
        "${root}/tolteca/tolteca/data/cal/toltec_passband/data/a1400_passband.ecsv"
        "${root}/tolteca/tolteca/data/cal/toltec_passband/data/a2000_passband.ecsv")
    set(hashes
        7a064ff768a3de4f427f1338d94ef6cb9026d248f3c3c816fc3dfc96d156e36a
        fd688a4cd3f46585b08631bc63a562aed482feb9b24ec9ee0071b70db7eb8a5f
        74465637294e536c44818099e4858a916fc6b9acbb1ea21b40427d15fb6532d5
        13b8fd009bb8d7c375d3c46d21e26d0a779f7f00a949a2a5ccd619d1fe56fd72
        a7b671d9f659cbc98dad99d3015ce81a3d7a3486c702819d9b3305703e7c682e
        77e4b33c7bbc2c345ef94d41480d5fee5cb096d789f4fe78e1b4f80a37e0d6ff)
    foreach(i RANGE 0 5)
        list(GET files ${i} path)
        list(GET hashes ${i} expected)
        file(SHA256 "${path}" actual)
        if(NOT actual STREQUAL expected)
            message(FATAL_ERROR "Approved CAL authority bytes differ: ${path}")
        endif()
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${path}")
    endforeach()
    list(GET files 1 nodes)
    file(READ "${nodes}" cal_nodes)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/cal_authority")
    configure_file("${source_root}/cmake/citlali_cal_authority.h.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cal_authority/citlali_cal_authority.h" @ONLY)
    target_include_directories(citlali PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/cal_authority")
endfunction()
