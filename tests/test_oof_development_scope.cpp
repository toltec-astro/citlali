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

TEST(OofDevelopmentScope, A2000KeepsPopulationAndAcquisitionExplicit) {
  auto n=YAML::Load("{development_reference: oof-a2000-m0-v1, mode: OOF, observation: 152385, subobservation: 0, scan: 1, network: 11}");
  EXPECT_EQ(oof_array(n),2);
  for(int obs=152385;obs<=152387;++obs) for(int nw: {11,12}) {
    n["observation"]=obs;n["network"]=nw;EXPECT_TRUE(oof_network_scope(n,obs,0,1,nw));
    EXPECT_FALSE(oof_network_scope(n,obs,0,2,nw));
  }
  for(int nw: {0,1,4,10,13}) {n["network"]=nw;EXPECT_FALSE(oof_network_scope(n,152387,0,1,nw));}
  n["network"]=11;n["mode"]="POINT";EXPECT_FALSE(oof_network_scope(n,152387,0,1,11));
}
