/*
 * NNClassifier.h
 *
 *  Created on: Mar 25, 2015
 *      Author: mszhang
 */

#ifndef SRC_NNCLASSIFIER_H_
#define SRC_NNCLASSIFIER_H_

#include <hash_set>
#include <iostream>
#include <armadillo>
#include <assert.h>
#include "Example.h"
#include "Feature.h"
#include "MyLib.h"
#include "Metric.h"
#include "LinearHidderLayer.h"
#include "NRMat.h"

using namespace nr;
using namespace std;
using namespace arma;

class NNClassifier {
public:
  NNClassifier();
  virtual ~NNClassifier();

public:
  double* _wordEmb;
  double* _grad_wordEmb;
  double* _eg2_wordEmb;


  double* _atomEmb;
  double* _grad_atomEmb;
  double* _eg2_atomEmb;

  double* _layer1W;
  mat _layer1b;

  double* _gradlayer1W;
  mat _gradlayer1b;

  double* _eg2layer1W;
  mat _eg2layer1b;

  int _lay1InputDim;
  int _layer1OutDim;
  LinearHidderLayer _layer2;

  int _actionSize;
  int _wordcontext;
  int _wordSize;
  int _wordDim;

  int _atomcontext;
  int _atomSize;
  int _atomDim;

  Metric _eval;

  int _lossFunc;

  double _dropOut;

  bool _b_wordEmb_finetune;

  hash_set<int> _wordPreComputed;
  hash_set<int> _atomPreComputed;

  hash_set<int> _curWordPreComputed;
  hash_set<int> _curAtomPreComputed;

  hash_map<int, int> _curWordPreComputedId;
  hash_map<int, int> _curAtomPreComputedId;

  double* _wordPreComputedForward, *_atomPreComputedForward;  // results
  double* _wordPreComputedBackward, *_atomPreComputedBackward; // grad
  int _curWordPreComputedNum, _curAtomPreComputedNum;

public:

  void init(int wordDim, int wordSize, int wordcontext, int atomDim, int atomSize, int atomcontext, int lay1OutDim, int labelSize);

  void init(const mat& wordEmb, int wordcontext, int atomDim, int atomSize, int atomcontext, int lay1OutDim, int labelSize);

  double process(const vector<Example>& examples, int iter);

  void predict(const Feature& features, vector<double>& results);

  double computeScore(const Example& example);

  void updateParams(double regularizationWeight, double adaAlpha, double adaEps);

  void writeModel();

  void loadModel();

  void preCompute();

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

  inline void setWordEmbFinetune(bool b_wordEmb_finetune)
  {
    _b_wordEmb_finetune = b_wordEmb_finetune;
  }

  inline void setPreComputed(const hash_set<int>& wordPreComputed, const hash_set<int>& atomPreComputed)
  {
    static hash_set<int>::iterator it;
    for (it = wordPreComputed.begin(); it != wordPreComputed.end(); ++it)
    {
      _wordPreComputed.insert(*it);
    }
    for (it = atomPreComputed.begin(); it != atomPreComputed.end(); ++it)
    {
      _atomPreComputed.insert(*it);
    }
  }

  void checkgrads(const vector<Example>& examples, int iter);

  void checkgradColSparse(const vector<Example>& examples, mat& Wd, const mat& gradWd, const string& mark, int iter, const hash_set<int>& sparseColIndexes,  const mat& ft);

  void checkgradRowSparse(const vector<Example>& examples, mat& Wd, const mat& gradWd, const string& mark, int iter, const hash_set<int>& sparseRowIndexes,  const mat& ft);
  void checkgrad(const vector<Example>& examples, double* Wd, const double* gradWd, const string& mark, int iter, int rowSize, int colSize);
  void checkgrad(const vector<Example>& examples, mat& Wd, const mat& gradWd, const string& mark, int iter);


};

#endif /* SRC_NNCLASSIFIER_H_ */
