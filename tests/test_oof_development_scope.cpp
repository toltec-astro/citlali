#include <gtest/gtest.h>
#include "../src/citlali/cli/oof_development_scope.h"
using namespace citlali::cli::detail;
TEST(OofDevelopmentScope, ExactModeScanAndDetectorNetwork) {
  auto n=YAML::Load("{development_reference: oof-cold-m0-v1, mode: OOF, observation: 152385, subobservation: 0, scan: 1, network: 1}");
  for(int obs=152385;obs<=152387;++obs) {n["observation"]=obs; EXPECT_TRUE(oof_network_scope(n,obs,0,1,1));}
  EXPECT_FALSE(oof_network_scope(n,152387,0,2,1));
  EXPECT_FALSE(oof_network_scope(n,152387,1,1,1));
  EXPECT_FALSE(oof_network_scope(n,152420,0,1,1));
  EXPECT_FALSE(oof_network_scope(n,152387,0,1,0));
  EXPECT_FALSE(oof_network_scope(n,152387,0,1,5));
  n["mode"]="POINT";EXPECT_FALSE(oof_network_scope(n,152387,0,1,1));
  n["mode"]="OOF";n["development_reference"]="bright-point-base-v1";
  EXPECT_FALSE(oof_network_scope(n,152387,0,1,1));
}
