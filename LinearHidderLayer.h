/*
 * LinearHidderLayer.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_LINEARHIDDERLAYER_H_
#define SRC_LINEARHIDDERLAYER_H_

#include <armadillo>

using namespace arma;

class LinearHidderLayer {

public:
  mat _W;
  mat _b;

  mat _gradW;
  mat _gradb;

  mat _eg2W;
  mat _eg2b;

  bool _bUseB;



public:

  LinearHidderLayer() {}

  void initial(int nOSize, int nISize, bool bUseB=true) {
    //double bound = sqrt(6.0 / (nOSize + nISize+1));
    double bound = 0.01;

    _W.randu(nOSize, nISize); _W = _W * 2.0 * bound - bound;
    _b.randu(nOSize, 1); _b = _b * 2.0 * bound - bound;

    _gradW.zeros(nOSize, nISize);
    _gradb.zeros(nOSize, 1);

    _eg2W.zeros(nOSize, nISize);
    _eg2b.zeros(nOSize, 1);

    _bUseB = bUseB;
  }

  void initial(const mat& W, const mat& b) {
    static int nOSize, nISize;
    _W = W; _b = b;

    nOSize = _W.n_rows;
    nISize = _W.n_cols;



    _gradW.zeros(nOSize, nISize);
    _gradb.zeros(nOSize, 1);

    _eg2W.zeros(nOSize, nISize);
    _eg2b.zeros(nOSize, 1);

    _bUseB = false;
  }

  void initial(const mat& W) {
    static int nOSize, nISize;
    _W = W;

    nOSize = _W.n_rows;
    nISize = _W.n_cols;

    _b.zeros(nOSize, 1);

    _gradW.zeros(nOSize, nISize);
    _gradb.zeros(nOSize, 1);

    _eg2W.zeros(nOSize, nISize);
    _eg2b.zeros(nOSize, 1);

    _bUseB = true;
  }


  virtual ~LinearHidderLayer() {
    // TODO Auto-generated destructor stub
  }


public:
  void ComputeForwardScore(const mat& x, mat& y)
  {
    y = _W * x;
    if(_bUseB)y = y + _b;
  }

  void ComputeForwardScorePreCompute(const mat& x, mat& y, int start_offset)
  {
    assert(x.n_rows + start_offset <=  _W.n_cols);
    y.zeros(_W.n_rows, 1);
    for(int idk = 0; idk < _W.n_rows; idk++)
    {
      for(int idx = 0 ;idx < x.n_rows; idx++)
      {
        y(idk, 0) += _W(idk, start_offset + idx) * x(idx, 0);
      }
    }
  }

  void ComputeBackwardLoss(const mat& x, const mat& y, const mat& ly, mat& lx)
  {
    //_gradW
    _gradW = _gradW + ly*x.t();

    //_gradb
    if(_bUseB)_gradb = _gradb +ly;

    //lx
    lx = _W.t()*ly;
  }

  void updateAdaGrad(double regularizationWeight, double adaAlpha, double adaEps)
  {
    _gradW = _gradW + _W * regularizationWeight;
    _eg2W = _eg2W + _gradW % _gradW;
    _W = _W - _gradW * adaAlpha / sqrt(_eg2W + adaEps);


    if(_bUseB)
    {
      _gradb = _gradb + _b * regularizationWeight;
      _eg2b = _eg2b + _gradb % _gradb;
      _b = _b - _gradb * adaAlpha / sqrt(_eg2b + adaEps);
    }


    clearGrad();
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

  void clearGrad()
  {
    _gradW.zeros();
    if(_bUseB)_gradb.zeros();
  }
};

#endif /* SRC_LINEARHIDDERLAYER_H_ */
