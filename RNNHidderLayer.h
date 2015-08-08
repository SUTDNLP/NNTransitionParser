/*
 * RNNHidderLayer.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_RNNHIDDERLAYER_H_
#define SRC_RNNHIDDERLAYER_H_

#include <armadillo>

using namespace arma;
class RNNHidderLayer {

public:
  mat _WL;
  mat _WR;
  mat _b;

  mat _gradWL;
  mat _gradWR;
  mat _gradb;

  mat _eg2WL;
  mat _eg2WR;
  mat _eg2b;

  bool _bUseB;

  int _funcType; // 0: tanh, 1: sigmod, 2: f(x)=x



public:
  RNNHidderLayer(){}

  void initial(int nOSize, int nISize, bool bUseB=true) {
    //double bound = 2.0*sqrt(6.0 / (2*nOSize + nISize+1));
    double bound = 0.01;

    _WL.randu(nOSize, nOSize); _WL = _WL * 2.0 * bound - bound;
    _WR.randu(nOSize, nISize); _WR = _WR * 2.0 * bound - bound;
    _b.randu(nOSize, 1); _b = _b * 2.0 * bound - bound;

    _gradWL.zeros(nOSize, nOSize);
    _gradWR.zeros(nOSize, nISize);
    _gradb.zeros(nOSize, 1);

    _eg2WL.zeros(nOSize, nOSize);
    _eg2WR.zeros(nOSize, nISize);
    _eg2b.zeros(nOSize, 1);

    _bUseB = bUseB;
    _funcType = 0;
  }

  void initial(const mat& WL, const mat& WR, const mat& b) {
    static int nOSize, nISize;
    _WL = WL; _WR = WR; _b = b;

    nOSize = _WR.n_rows;
    nISize = _WR.n_cols;

    _gradWL.zeros(nOSize, nOSize);
    _gradWR.zeros(nOSize, nISize);
    _gradb.zeros(nOSize, 1);

    _eg2WL.zeros(nOSize, nOSize);
    _eg2WR.zeros(nOSize, nISize);
    _eg2b.zeros(nOSize, 1);

    _bUseB = false;
    _funcType = 0;
  }

  void initial(const mat& WL, const mat& WR) {
    static int nOSize, nISize;
    _WL = WL; _WR = WR;

    nOSize = _WR.n_rows;
    nISize = _WR.n_cols;

    _b.zeros(nOSize, 1);

    _gradWL.zeros(nOSize, nOSize);
    _gradWR.zeros(nOSize, nISize);
    _gradb.zeros(nOSize, 1);

    _eg2WL.zeros(nOSize, nOSize);
    _eg2WR.zeros(nOSize, nISize);
    _eg2b.zeros(nOSize, 1);

    _bUseB = true;
    _funcType = 0;
  }

  virtual ~RNNHidderLayer() {
    // TODO Auto-generated destructor stub
  }

  void setFunc(int funcType)
  {
    _funcType = funcType;
  }


public:
  void ComputeForwardScore(const mat& py, const mat& x, mat& y)
  {
    y = _WL * py + _WR *x;
    if(_bUseB)y = y + _b;
    if(_funcType == 0)y = tanh(y);
    else if(_funcType == 1) y = 1.0/(1.0+exp(-y));

  }

  void ComputeForwardScore(const mat& x, mat& y)
  {
    y = _WR *x;
    if(_bUseB)y = y + _b;
    if(_funcType == 0)y = tanh(y);
    else if(_funcType == 1) y = 1.0/(1.0+exp(-y));
  }

  void ComputeBackwardLoss(const mat& py, const mat& x, const mat& y, const mat& ly, mat& lpy, mat& lx)
  {

    static mat deri_yx, cly;

    if(_funcType == 0)
    {
      deri_yx = 1 - y%y;
      cly = ly % deri_yx;
    }
    else if(_funcType == 1)
    {
      deri_yx = y - y % y;
      cly = ly % deri_yx;
    }
    else
    {
      cly = ly;
    }

    //_gradWL, _gradWR
    _gradWL = _gradWL + cly * py.t();
    _gradWR = _gradWR + cly * x.t();

    //_gradb
    if(_bUseB)_gradb = _gradb +cly;

    //lx
    lpy = _WL.t() * cly;
    lx = _WR.t() * cly;
  }

  void ComputeBackwardLoss(const mat& x, const mat& y, const mat& ly, mat& lx)
   {

     static mat deri_yx, cly;
     deri_yx = 1 - y%y;
     cly = ly % deri_yx;

     //_gradWL, _gradWR
     _gradWR = _gradWR + cly * x.t();

     //_gradb
     if(_bUseB)_gradb = _gradb +cly;

     //lx
     lx = _WR.t() * cly;
   }

  void randomprint(int num)
  {
    static int nOSize, nISize;
    nOSize = _WR.n_rows;
    nISize = _WR.n_cols;
    int count = 0;
    while(count < num)
    {
      int idx = rand()%nOSize;
      int idy = rand()%nOSize;


      std::cout << "_WL[" << idx << "," << idy << "]=" << _WL(idx, idy) << " ";

      int idm = rand()%nOSize;
      int idn = rand()%nISize;


      std::cout << "_WR[" << idm << "," << idn << "]=" << _WR(idm, idn) << " ";

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
    _gradWL = _gradWL + _WL * regularizationWeight;
    _eg2WL = _eg2WL + _gradWL % _gradWL;
    _WL = _WL - _gradWL * adaAlpha / sqrt(_eg2WL + adaEps);

    _gradWR = _gradWR + _WR * regularizationWeight;
    _eg2WR = _eg2WR + _gradWR % _gradWR;
    _WR = _WR - _gradWR * adaAlpha / sqrt(_eg2WR + adaEps);


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
    _gradWL.zeros();
    _gradWR.zeros();
    if(_bUseB)_gradb.zeros();
  }
};

#endif /* SRC_RNNHIDDERLAYER_H_ */
