#include <gtest/gtest.h>
#include <utility>
#include "../src/citlali/cli/configured_point_scope.h"
#include "../src/citlali/cli/oof_development_scope.h"
using namespace citlali::cli::detail;
TEST(ConfiguredPointScope, AcquisitionIsDataAndOtherModesStayExcluded) {
    auto n=YAML::Load("{development_reference: configurable-point-v1, mode: POINT, observation: 152389, subobservation: 0, scan: 2}");
    for(int obs: {152389,152418,999999}) {
        n["observation"]=obs;
        EXPECT_TRUE(configured_point_scope(n,obs,0,2));
        EXPECT_FALSE(configured_point_scope(n,obs+1,0,2));
        EXPECT_FALSE(configured_point_scope(n,obs,1,2));
        EXPECT_FALSE(configured_point_scope(n,obs,0,1));
    }
    n["observation"]=0;EXPECT_FALSE(configured_point_scope(n,0,0,2));
    n["observation"]=152389;n["mode"]="SCIENCE";
    EXPECT_FALSE(configured_point_scope(n,152389,0,2));
    n["mode"]="OOF";EXPECT_FALSE(configured_point_scope(n,152389,0,2));
    n["mode"]="POINT";n["development_reference"]="bright-point-base-v1";
    EXPECT_FALSE(configured_point_scope(n,152389,0,2));
    n["development_reference"]="configurable-point-v1";n.remove("scan");
    EXPECT_FALSE(configured_point_scope(n,152389,0,2));
}

#include "../src/citlali/cli/configured_map_frame.h"
#include "../src/citlali/cli/point_execution_budget.h"
#include <numbers>
TEST(ConfiguredScienceScope, SingleAcquisitionAndModeRemainExplicit) {
    auto n=YAML::Load("{development_reference: configurable-science-compact-v1, mode: SCIENCE, observation: 152389, subobservation: 0, scan: 2}");
    EXPECT_TRUE(configured_map_scope(n,152389,0,2));
    EXPECT_TRUE(configured_science_scope(n,152389,0,2));
    EXPECT_FALSE(configured_point_scope(n,152389,0,2));
    EXPECT_FALSE(configured_science_scope(n,152389,0,1));
    n["observation"]=152418;EXPECT_FALSE(configured_science_scope(n,152418,0,2));
    n["observation"]=152389;n["mode"]="POINT";
    EXPECT_FALSE(configured_map_scope(n,152389,0,2));
    n["mode"]="SCIENCE";n["development_reference"]="configurable-point-v1";
    EXPECT_FALSE(configured_map_scope(n,152389,0,2));
}
TEST(ConfiguredScienceScope, SavedMapAdmissionRetainsScienceIdentity) {
    // This is the helper used by load_saved_map_network at CAL -> health,
    // not just the upstream configured_map_scope predicate.
    auto n=YAML::Load("{development_reference: configurable-science-compact-v1, mode: SCIENCE, observation: 152389, subobservation: 0, scan: 2, network: 0}");
    for (const auto [nw,array] : {std::pair{0,0},std::pair{7,1},std::pair{11,2}}) {
        n["network"]=nw;
        EXPECT_EQ(saved_map_observation_identity(n,152389,nw,array),"obs:152389_0_2");
        EXPECT_EQ(n["mode"].as<std::string>(),"SCIENCE");
    }
    n["development_reference"]="configurable-point-v1";n["mode"]="POINT";
    EXPECT_EQ(saved_map_observation_identity(n,152389,11,2),"obs:152389_0_2");
    // Historical POINT inputs without explicit acquisition fields remain valid.
    n.remove("mode");n.remove("scan");n.remove("subobservation");
    EXPECT_EQ(saved_map_observation_identity(n,152389,11,2),"obs:152389_0_2");
}
TEST(ConfiguredScienceScope, SavedMapAdmissionRejectsOtherAcquisitionsAndProfiles) {
    const auto valid=YAML::Load("{development_reference: configurable-science-compact-v1, mode: SCIENCE, observation: 152389, subobservation: 0, scan: 2, network: 0}");
    EXPECT_THROW(saved_map_observation_identity(valid,152418,0,0),std::invalid_argument);
    for (const char *field : {"observation","subobservation","scan","mode"}) {
        auto n=YAML::Clone(valid);n.remove(field);
        EXPECT_THROW(saved_map_observation_identity(n,152389,0,0),std::invalid_argument) << field;
    }
    for (const auto &[field,value] : {std::pair{"observation",152418},
                                    std::pair{"subobservation",1},std::pair{"scan",1}}) {
        auto n=YAML::Clone(valid);n[field]=value;
        EXPECT_THROW(saved_map_observation_identity(n,152389,0,0),std::invalid_argument) << field;
    }
    auto other=YAML::Clone(valid);other["observation"]=152418;
    EXPECT_THROW(saved_map_observation_identity(other,152418,0,0),std::invalid_argument);
    for (const char *mode : {"POINT","OOF","BEAMMAP"}) {
        auto n=YAML::Clone(valid);n["mode"]=mode;
        EXPECT_THROW(saved_map_observation_identity(n,152389,0,0),std::invalid_argument) << mode;
    }
    for (const char *profile : {"configurable-point-v1","bright-point-base-v1","oof-cold-m0-v1","unknown"}) {
        auto n=YAML::Clone(valid);n["development_reference"]=profile;
        EXPECT_THROW(saved_map_observation_identity(n,152389,0,0),std::invalid_argument) << profile;
    }
    auto missing=YAML::Clone(valid);missing.remove("development_reference");
    EXPECT_THROW(saved_map_observation_identity(missing,152389,0,0),std::invalid_argument);
}
TEST(ConfiguredScienceFrame, CelestialDegreesAndPointArcsecondsStayDistinct) {
    using citlali::pipeline::AstRtcTanGrid;
    const auto sky=AstRtcTanGrid::centered(std::numbers::pi, std::numbers::pi/6, 2., .01, .02);
    const auto celestial=configured_frame_record(sky,true);
    EXPECT_DOUBLE_EQ(celestial["CRVAL1"].as<double>(),180.);
    EXPECT_NEAR(celestial["CRVAL2"].as<double>(),30.,1e-13);
    EXPECT_DOUBLE_EQ(celestial["CDELT1"].as<double>(),-2./3600.);
    EXPECT_EQ(celestial["CTYPE1"].as<std::string>(),"RA---TAN");
    EXPECT_EQ(celestial["RADESYS"].as<std::string>(),"FK5");
    EXPECT_DOUBLE_EQ(celestial["EQUINOX"].as<double>(),2000.);
    const auto [column,row]=sky.continuous_pixel(-2./3600.,4./3600.);
    EXPECT_NEAR(row,sky.crpix2+2,1e-12);
    EXPECT_NEAR(column,sky.crpix1+1,1e-12);
    const auto point=configured_frame_record(AstRtcTanGrid::source_offset(2.,.01,.02),false);
    EXPECT_EQ(point["CTYPE1"].as<std::string>(),"AZOFFSET");
    EXPECT_DOUBLE_EQ(point["CRVAL1"].as<double>(),0.);
    EXPECT_DOUBLE_EQ(point["CDELT1"].as<double>(),-2.);
    EXPECT_FALSE(point["RADESYS"]);
}
TEST(ConfiguredScienceBudget, OneNodeCeilingDoesNotBroadenPointDefault) {
    EXPECT_THROW(point_execution_budget(13),std::invalid_argument);
    EXPECT_THROW(point_execution_budget(33,32),std::invalid_argument);
    EXPECT_THROW(point_execution_budget(0,32),std::invalid_argument);
    const auto science=point_execution_budget(32,32);
    EXPECT_EQ(science.requested,32u);
    EXPECT_GE(science.execution.workers,1u);
    EXPECT_LE(science.execution.workers,32u);
}
