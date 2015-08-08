/*
 * SparseLinearHidderLayer.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_SPARSELINEARHIDDERLAYER_H_
#define SRC_SPARSELINEARHIDDERLAYER_H_
#include <armadillo>
#include "MyLib.h"

using namespace arma;

// do not consider b
class SparseLinearHidderLayer {

public:
  mat _W;
  mat _gradW;
  mat _eg2W;
  mat _ftW;
  hash_set<int> _indexers;

  mat _b;
  mat _gradb;
  mat _eg2b;

  bool _bUseB;

public:

  SparseLinearHidderLayer() {}

  void initial(int nOSize, int nISize, bool bUseB = false) {
    //double bound = sqrt(6.0 / (nOSize + nISize+1));
    double bound = 0.01;

    _W.randu(nOSize, nISize); _W = _W * 2.0 * bound - bound;
    _gradW.zeros(nOSize, nISize);
    _eg2W.zeros(nOSize, nISize);
    _ftW.ones(nOSize, nISize);
    _indexers.clear();

    _b.randu(nOSize, 1); _b = _b * 2.0 * bound - bound;
    _gradb.zeros(nOSize, 1);
    _eg2b.zeros(nOSize, 1);

    _bUseB = bUseB;

  }



  void initial(const mat& W) {
    static int nOSize, nISize;
    _W = W;

    nOSize = _W.n_rows;
    nISize = _W.n_cols;

    _gradW.zeros(nOSize, nISize);
    _eg2W.zeros(nOSize, nISize);
    _ftW.ones(nOSize, nISize);
    _indexers.clear();

    _b.zeros(nOSize, 1);
    _gradb.zeros(nOSize, 1);
    _eg2b.zeros(nOSize, 1);
    _bUseB = false;
  }

  void initial(const mat& W, const mat&b) {
    static int nOSize, nISize;
    _W = W;

    nOSize = _W.n_rows;
    nISize = _W.n_cols;

    _gradW.zeros(nOSize, nISize);
    _eg2W.zeros(nOSize, nISize);
    _ftW.ones(nOSize, nISize);
    _indexers.clear();

    _b = b;
    _gradb.zeros(nOSize, 1);
    _eg2b.zeros(nOSize, 1);
    _bUseB = true;
  }



  virtual ~SparseLinearHidderLayer() {
    // TODO Auto-generated destructor stub
  }


public:
  void ComputeForwardScore(const std::vector<int>& x, mat& y)
  {
    static int featNum, featId, outDim;
    featNum = x.size();
    outDim = _W.n_rows;
    y.zeros(outDim, 1);
    for(int idx = 0; idx < featNum; idx++)
    {
      featId = x[idx];
      for(int idy = 0; idy < outDim; idy++)
      {
        y(idy, 0) += _W(idy, featId)/_ftW(idy, featId);
      }
    }

    if(_bUseB) y = y + _b;

  }

  // loss is stopped at this layer, since the input is one-hold alike
  void ComputeBackwardLoss(const std::vector<int>& x, const mat& ly)
  {
    //_gradW
    static int featNum, featId, outDim;
    featNum = x.size();
    outDim = _W.n_rows;
    for(int idx = 0; idx < featNum; idx++)
    {
      featId = x[idx];
      _indexers.insert(featId);
      for(int idy = 0; idy < outDim; idy++)
      {
        _gradW(idy, featId) += ly[idy];
      }
    }

    if(_bUseB)_gradb = _gradb +ly;
  }

  void randomprint(int num)
  {
    static int nOSize, nISize;
    nOSize = _W.n_rows;
    nISize = _W.n_cols;
    int count = 0;
    while(count < num)
    {
      int idx = rand()%nOSize;
      int idy = rand()%nISize;


      std::cout << "_W[" << idx << "," << idy << "]=" << _W(idx, idy) << " ";

      if(_bUseB)
      {
          int idz = rand()%nOSize;
          std::cout << "_b[" << idz << "]=" << _b(idz, 0) << " ";
      }

      count++;
    }

    std::cout << std::endl;
  }

  void updateAdaGrad(double regularizationWeight, double adaAlpha, double adaEps)
  {

    static int outDim;
    outDim = _W.n_rows;
    static hash_set<int>::iterator it;

    for(it = _indexers.begin();  it!=_indexers.end(); ++it)
    {
      int index = *it;
      for(int idx = 0; idx < outDim; idx++)
      {
        double _grad_wordEmb_ij = _gradW(idx, index) +  regularizationWeight * _W(idx, index) / _ftW(idx, index);
        _eg2W(idx, index) += _grad_wordEmb_ij * _grad_wordEmb_ij;
        double tmp_normaize_alpha = sqrt(_eg2W(idx, index)+adaEps);
        double tmp_alpha = adaAlpha/tmp_normaize_alpha;

        double _ft_wordEmb_ij = _ftW(idx, index)*tmp_alpha*regularizationWeight;
        _ftW(idx, index) -= _ft_wordEmb_ij;
        _W(idx, index) -= tmp_alpha * _gradW(idx, index) / _ftW(idx, index);
      }
    }

    if(_bUseB)
    {
      _gradb = _gradb + _b * regularizationWeight;
      _eg2b = _eg2b + _gradb % _gradb;
      _b = _b - _gradb * adaAlpha / sqrt(_eg2b + adaEps);
    }


    clearGrad();
  }

  void clearGrad()
  {
    static int outDim;
    outDim = _W.n_rows;
    static hash_set<int>::iterator it;

    for(it = _indexers.begin();  it!=_indexers.end(); ++it)
    {
      int index = *it;
      for(int idx = 0; idx < outDim; idx++)
      {
        _gradW(idx, index)  = 0.0;
      }
    }

    _indexers.clear();
    if(_bUseB)_gradb.zeros();
  }
};

#endif /* SRC_LINEARHIDDERLAYER_H_ */
