/*
 * LinearClassifier.h
 *
 *  Created on: Mar 25, 2015
 *      Author: mszhang
 */

#ifndef SRC_LINEARNNCLASSIFIER_H_
#define SRC_LINEARNNCLASSIFIER_H_

#include <hash_set>
#include <iostream>
#include <armadillo>
#include <assert.h>
#include "Example.h"
#include "Feature.h"
#include "MyLib.h"
#include "Metric.h"
#include "LinearHidderLayer.h"
#include "SparseLinearHidderLayer.h"
#include "NRMat.h"

using namespace nr;
using namespace std;
using namespace arma;

class LinearClassifier {
public:
  LinearClassifier();
  virtual ~LinearClassifier();

public:
  SparseLinearHidderLayer _layer_linear;

  int _actionSize;

  int _linearfeatSize;

  Metric _eval;

  int _lossFunc;

  double _dropOut;


public:

  void init(int labelSize, int linearfeatSize);

  double process(const vector<Example>& examples, int iter);

  void predict(const Feature& features, vector<double>& results);

  double computeScore(const Example& example);

  void updateParams(double regularizationWeight, double adaAlpha, double adaEps);

  void writeModel();

  void loadModel();

  void checkgrads(const vector<Example>& examples, int iter);

  void checkgrad(const vector<Example>& examples, mat& Wd, const mat& gradWd, const string& mark, int iter, const hash_set<int>& sparseRowIndexes,  const mat& ft);

public:

  inline void resetEval()
  {
    _eval.reset();
  }

  inline void setLossFunc(int lossFunc)
  {
    _lossFunc = lossFunc;
  }

  inline void setDropValue(double dropOut)
  {
    _dropOut = dropOut;
  }

};

#endif /* SRC_LINEARNNCLASSIFIER_H_ */
