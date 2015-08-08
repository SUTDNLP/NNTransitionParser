/*
 * Example.h
 *
 *  Created on: Mar 17, 2015
 *      Author: mszhang
 */

#ifndef SRC_EXAMPLE_H_
#define SRC_EXAMPLE_H_

#include "Feature.h"

using namespace std;

class Example {

public:
  vector<int> m_labels;
  Feature m_feature;

public:
  Example()
  {

  }

  /*~Example()
  {

  }*/

  void clear()
  {
    m_labels.clear();
    m_feature.clear();
  }


};

#endif /* SRC_EXAMPLE_H_ */
