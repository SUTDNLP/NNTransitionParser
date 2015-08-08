/*
 * Feature.h
 *
 *  Created on: Mar 17, 2015
 *      Author: mszhang
 */

#ifndef SRC_FEATURE_H_
#define SRC_FEATURE_H_

#include <vector>

using namespace std;
class Feature {

public:
  vector<int> wneural_features;
  vector<int> aneural_features;

  vector<int> linear_features;

public:
  Feature()
  {
    clear();
  }

  /*virtual ~Feature()
  {

  }*/

  void clear()
  {
    wneural_features.clear();
    aneural_features.clear();

    linear_features.clear();
  }
};

#endif /* SRC_FEATURE_H_ */
