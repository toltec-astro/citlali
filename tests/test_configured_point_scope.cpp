#include <gtest/gtest.h>
#include "../src/citlali/cli/configured_point_scope.h"
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
