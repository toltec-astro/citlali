#pragma once

namespace mapmaking {

class Wiener {

  void prepareTemplate();
  void prepareGaussianTemplate();
  void filterCoaddition();
  void filterNoiseMaps();
  void calcRr();
  void calcVvq();
  void calcNumerator();
  void calcDenominator();
};


} //namespace
