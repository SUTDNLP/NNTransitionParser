/*
 * TanhHidderLayer.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_TANHHIDDERLAYER_H_
#define SRC_TANHHIDDERLAYER_H_
#include <armadillo>

using namespace arma;

class TanhHidderLayer {

public:
  mat _W;
  mat _b;

  mat _gradW;
  mat _gradb;

  mat _eg2W;
  mat _eg2b;

  bool _bzerob;

  int _funcType; // 0: tanh, 1: sigmod, 2: f(x) = x*x*x


public:
  TanhHidderLayer(){}

  void initial(int nOSize, int nISize, bool bzerob=false) {
     //double bound = sqrt(6.0 / (nOSize + nISize+1));
     double bound = 0.01;

     _W.randu(nOSize, nISize); _W = _W * 2.0 * bound - bound;
     _b.randu(nOSize, 1); _b = _b * 2.0 * bound - bound;

     _gradW.zeros(nOSize, nISize);
     _gradb.zeros(nOSize, 1);

     _eg2W.zeros(nOSize, nISize);
     _eg2b.zeros(nOSize, 1);

     _bzerob = bzerob;

     _funcType = 0;
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

     _bzerob = false;

     _funcType = 0;
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

     _bzerob = true;

     _funcType = 0;
   }


  virtual ~TanhHidderLayer() {
    // TODO Auto-generated destructor stub
  }

  void setFunc(int funcType)
  {
    _funcType = funcType;
  }


public:
  void ComputeForwardScore(const mat& x, mat& mid_y, mat& y)
  {
    mid_y = _W * x;
    if(!_bzerob)mid_y = mid_y + _b;
    if(_funcType == 2)y = mid_y % mid_y % mid_y;
    else if(_funcType == 1) y = 1.0/(1.0+exp(-mid_y));
    else y = tanh(mid_y);
  }

  void ComputeBackwardLoss(const mat& x, const mat& mid_y, const mat& y, const mat& ly, mat& lx)
  {
    //_gradW
    static mat deri_yx, cly;

    if(_funcType == 2)deri_yx =  3 * ( mid_y % mid_y);
    else if(_funcType == 1) deri_yx = y - y % y;
    else deri_yx = 1 - y%y;

    cly = ly % deri_yx;

    //_gradW
    _gradW = _gradW + cly*x.t();

    //_gradb
    if(!_bzerob)_gradb = _gradb +cly;

    //lx
    lx = _W.t()*cly;

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

      if(!_bzerob)
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
    _gradW = _gradW + _W * regularizationWeight;
    _eg2W = _eg2W + _gradW % _gradW;
    _W = _W - _gradW * adaAlpha / sqrt(_eg2W + adaEps);


    if(!_bzerob)
    {
      _gradb = _gradb + _b * regularizationWeight;
      _eg2b = _eg2b + _gradb % _gradb;
      _b = _b - _gradb * adaAlpha / sqrt(_eg2b + adaEps);
    }


    clearGrad();
  }

  void clearGrad()
  {
    _gradW.zeros();
    if(!_bzerob)_gradb.zeros();
  }
};

#endif /* SRC_TANHHIDDERLAYER_H_ */
