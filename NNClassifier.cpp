/*
 * NNClassifier.cpp
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

// did not judge memory allocate fail!
#include "NNClassifier.h"

NNClassifier::NNClassifier() {
  // TODO Auto-generated constructor stub
  _b_wordEmb_finetune = true;
  _lossFunc = 0;
  _wordEmb = NULL;
  _grad_wordEmb = NULL;
  _eg2_wordEmb = NULL;
  _atomEmb = NULL;
  _grad_atomEmb = NULL;
  _eg2_atomEmb = NULL;
  _layer1W = NULL;
  _gradlayer1W = NULL;
  _eg2layer1W = NULL;
  _wordPreComputedForward = NULL;
  _atomPreComputedForward = NULL;
  _wordPreComputedBackward = NULL;
  _atomPreComputedBackward = NULL;
  _dropOut = 0.0;
}

NNClassifier::~NNClassifier() {
  // TODO Auto-generated destructor stub
  Free(&_wordEmb);
  Free(&_grad_wordEmb);
  Free(&_eg2_wordEmb);
  Free(&_atomEmb);
  Free(&_grad_atomEmb);
  Free(&_eg2_atomEmb);
  Free(&_layer1W);
  Free(&_gradlayer1W);
  Free(&_eg2layer1W);
  Free(&_wordPreComputedForward);
  Free(&_atomPreComputedForward);
  Free(&_wordPreComputedBackward);
  Free(&_atomPreComputedBackward);
}

// did not judge memory allocate fail!
void NNClassifier::init(int wordDim, int wordSize, int wordcontext, int atomDim, int atomSize, int atomcontext, int lay1OutDim, int labelSize) {
  _b_wordEmb_finetune = true;
  _actionSize = labelSize;
  _wordcontext = wordcontext;
  _wordSize = wordSize;
  _wordDim = wordDim;
  _atomcontext = atomcontext;
  _atomSize = atomSize;
  _atomDim = atomDim;
  _layer1OutDim = lay1OutDim;

  _wordEmb = (double *) calloc(_wordSize * _wordDim, sizeof(double));
  _grad_wordEmb = (double *) calloc(_wordSize * _wordDim, sizeof(double));
  _eg2_wordEmb = (double *) calloc(_wordSize * _wordDim, sizeof(double));

  randomMatAssign(_wordEmb, _wordSize * _wordDim, 1.0, 0);
  //normalize to a unit sphere
  //for (int idx = 0; idx < _wordDim; idx++) {
   // normalize_mat_onecol(_wordEmb, idx, _wordSize, _wordDim);
  //}
  for(int idx = 0; idx < _wordSize; idx++)
  {
    normalize_mat_onerow(_wordEmb, idx, _wordSize, _wordDim);
  }

  _atomEmb = (double *) calloc(_atomSize * _atomDim, sizeof(double));
  _grad_atomEmb = (double *) calloc(_atomSize * _atomDim, sizeof(double));
  _eg2_atomEmb = (double *) calloc(_atomSize * _atomDim, sizeof(double));
  randomMatAssign(_atomEmb, _atomSize * _atomDim, 1.0, 1);
  //normalize to a unit sphere
  //for (int idx = 0; idx < _atomDim; idx++) {
    //normalize_mat_onecol(_atomEmb, idx, _atomSize, _atomDim);
  //}
  for (int idx = 0; idx < _atomSize; idx++) {
    normalize_mat_onerow(_atomEmb, idx, _atomSize, _atomDim);
  }

  _lay1InputDim = _wordDim * _wordcontext + _atomDim * _atomcontext;
  double bound = 0.01;
  //double bound = sqrt(6.0 / (_layer1OutDim + _lay1InputDim + 1));
  _layer1W = (double *) calloc(_layer1OutDim * _lay1InputDim, sizeof(double));
  _gradlayer1W = (double *) calloc(_layer1OutDim * _lay1InputDim, sizeof(double));
  _eg2layer1W = (double *) calloc(_layer1OutDim * _lay1InputDim, sizeof(double));
  randomMatAssign(_layer1W, _layer1OutDim * _lay1InputDim, bound, 2);

  _layer1b.randu(_layer1OutDim, 1);
  _layer1b = _layer1b * 2.0 * bound - bound;
  _gradlayer1b.zeros(_layer1OutDim, 1);
  _eg2layer1b.zeros(_layer1OutDim, 1);

  _layer2.initial(_actionSize, _layer1OutDim, false);

  _eval.reset();

}

// did not judge memory allocate fail!
void NNClassifier::init(const mat& wordEmb, int wordcontext, int atomDim, int atomSize, int atomcontext, int lay1OutDim, int labelSize) {

  _actionSize = labelSize;
  _wordcontext = wordcontext;
  _wordSize = wordEmb.n_rows;
  _wordDim = wordEmb.n_cols;
  _atomcontext = atomcontext;
  _atomSize = atomSize;
  _atomDim = atomDim;
  _layer1OutDim = lay1OutDim;

  _wordEmb = (double *) calloc(_wordSize * _wordDim, sizeof(double));
  _grad_wordEmb = (double *) calloc(_wordSize * _wordDim, sizeof(double));
  _eg2_wordEmb = (double *) calloc(_wordSize * _wordDim, sizeof(double));

  assign(_wordEmb, wordEmb);
  //normalize to a unit sphere
  //for (int idx = 0; idx < _wordDim; idx++) {
  //  normalize_mat_onecol(_wordEmb, idx, _wordSize, _wordDim);
  //}
  for(int idx = 0; idx < _wordSize; idx++)
  {
    normalize_mat_onerow(_wordEmb, idx, _wordSize, _wordDim);
  }

  _atomEmb = (double *) calloc(_atomSize * _atomDim, sizeof(double));
  _grad_atomEmb = (double *) calloc(_atomSize * _atomDim, sizeof(double));
  _eg2_atomEmb = (double *) calloc(_atomSize * _atomDim, sizeof(double));
  randomMatAssign(_atomEmb, _atomSize * _atomDim, 1.0, 1);
  //normalize to a unit sphere
  //for (int idx = 0; idx < _atomDim; idx++) {
  //  normalize_mat_onecol(_atomEmb, idx, _atomSize, _atomDim);
  //}
  for (int idx = 0; idx < _atomSize; idx++) {
    normalize_mat_onerow(_atomEmb, idx, _atomSize, _atomDim);
  }

  _lay1InputDim = _wordDim * _wordcontext + _atomDim * _atomcontext;
  //double bound = sqrt(6.0 / (_layer1OutDim + _lay1InputDim + 1));
  double bound = 0.01;
  _layer1W = (double *) calloc(_layer1OutDim * _lay1InputDim, sizeof(double));
  _gradlayer1W = (double *) calloc(_layer1OutDim * _lay1InputDim, sizeof(double));
  _eg2layer1W = (double *) calloc(_layer1OutDim * _lay1InputDim, sizeof(double));
  randomMatAssign(_layer1W, _layer1OutDim * _lay1InputDim, bound, 2);

  _layer1b.randu(_layer1OutDim, 1);
  _layer1b = _layer1b * 2.0 * bound - bound;
  _gradlayer1b.zeros(_layer1OutDim, 1);
  _eg2layer1b.zeros(_layer1OutDim, 1);

  _layer2.initial(_actionSize, _layer1OutDim, false);

  _eval.reset();

}

void NNClassifier::preCompute() {
  static int count, tmpI, tmpJ, tmpK;
  static hash_set<int>::iterator it;
  static double temp;
  for (it = _wordPreComputed.begin(); it != _wordPreComputed.end(); ++it) {
    _curWordPreComputed.insert(*it);
  }

  for (it = _atomPreComputed.begin(); it != _atomPreComputed.end(); ++it) {
    _curAtomPreComputed.insert(*it);
  }

  _curWordPreComputedId.clear();
  _curAtomPreComputedId.clear();
  Free(&_wordPreComputedForward);
  Free(&_atomPreComputedForward);
  Free(&_wordPreComputedBackward);
  Free(&_atomPreComputedBackward);
  //initial
  _curWordPreComputedNum = _curWordPreComputed.size();
  _wordPreComputedForward = (double *) calloc(_layer1OutDim * _curWordPreComputedNum, sizeof(double));
  count = 0;
  for (it = _curWordPreComputed.begin(); it != _curWordPreComputed.end(); ++it) {
    _curWordPreComputedId[*it] = count;
    int offset = (*it) % _wordcontext;
    int wordId = (*it) / _wordcontext;
    tmpJ = wordId * _wordDim;
    tmpI = offset * _wordDim;
    tmpK = count;
    for (int idk = 0; idk < _layer1OutDim; idk++) {
      temp = 0.0;
      for (int idy = 0; idy < _wordDim; idy++) {
        temp += _layer1W[tmpI + idy] * _wordEmb[tmpJ + idy];
      }
      _wordPreComputedForward[tmpK] = temp;
      tmpI += _lay1InputDim;
      tmpK += _curWordPreComputedNum;
    }
    count++;
  }

  _curAtomPreComputedNum = _curAtomPreComputed.size();
  _atomPreComputedForward = (double *) calloc(_layer1OutDim * _curAtomPreComputedNum, sizeof(double));
  count = 0;
  for (it = _curAtomPreComputed.begin(); it != _curAtomPreComputed.end(); ++it) {
    _curAtomPreComputedId[*it] = count;
    int offset = (*it) % _atomcontext;
    int atomId = (*it) / _atomcontext;
    tmpJ = atomId * _atomDim;
    tmpI = _wordDim * _wordcontext + offset * _atomDim;
    tmpK = count;
    for (int idk = 0; idk < _layer1OutDim; idk++) {
      temp = 0.0;
      for (int idy = 0; idy < _atomDim; idy++) {
        temp += _layer1W[tmpI + idy] * _atomEmb[tmpJ + idy];
      }
      _atomPreComputedForward[tmpK] = temp;
      tmpI += _lay1InputDim;
      tmpK += _curAtomPreComputedNum;
    }
    count++;
  }

}

double NNClassifier::process(const vector<Example>& examples, int iter) {
  _eval.reset();

  _curWordPreComputed.clear();
  _curAtomPreComputed.clear();

  static hash_set<int>::iterator it;
  static int count, tmpI, tmpJ, tmpK;
  static double temp;
  int example_num = examples.size();
  for (count = 0; count < example_num; count++) {
    const Feature& feature = examples[count].m_feature;
    const vector<int>& wneural_features = feature.wneural_features;
    const vector<int>& aneural_features = feature.aneural_features;

    for (int idk = 0; idk < wneural_features.size(); idk++) {
      int curFeatId = wneural_features[idk] * wneural_features.size() + idk;
      if (_wordPreComputed.find(curFeatId) != _wordPreComputed.end()) {
        _curWordPreComputed.insert(curFeatId);
      }
    }

    for (int idk = 0; idk < aneural_features.size(); idk++) {
      int curFeatId = aneural_features[idk] * aneural_features.size() + idk;
      if (_atomPreComputed.find(curFeatId) != _atomPreComputed.end()) {
        _curAtomPreComputed.insert(curFeatId);
      }
    }
  }

  _curWordPreComputedId.clear();
  _curAtomPreComputedId.clear();
  Free(&_wordPreComputedForward);
  Free(&_atomPreComputedForward);
  Free(&_wordPreComputedBackward);
  Free(&_atomPreComputedBackward);
  //initial
  _curWordPreComputedNum = _curWordPreComputed.size();
  _wordPreComputedForward = (double *) calloc(_layer1OutDim * _curWordPreComputedNum, sizeof(double));
  count = 0;
  for (it = _curWordPreComputed.begin(); it != _curWordPreComputed.end(); ++it) {
    _curWordPreComputedId[*it] = count;
    int offset = (*it) % _wordcontext;
    int wordId = (*it) / _wordcontext;
    tmpJ = wordId * _wordDim;
    tmpI = offset * _wordDim;
    tmpK = count;
    for (int idk = 0; idk < _layer1OutDim; idk++) {
      temp = 0.0;
      for (int idy = 0; idy < _wordDim; idy++) {
        temp += _layer1W[tmpI + idy] * _wordEmb[tmpJ + idy];
      }
      _wordPreComputedForward[tmpK] = temp;
      tmpI += _lay1InputDim;
      tmpK += _curWordPreComputedNum;
    }
    count++;
  }

  _curAtomPreComputedNum = _curAtomPreComputed.size();
  _atomPreComputedForward = (double *) calloc(_layer1OutDim * _curAtomPreComputedNum, sizeof(double));
  count = 0;
  for (it = _curAtomPreComputed.begin(); it != _curAtomPreComputed.end(); ++it) {
    _curAtomPreComputedId[*it] = count;
    int offset = (*it) % _atomcontext;
    int atomId = (*it) / _atomcontext;
    tmpJ = atomId * _atomDim;
    tmpI = _wordDim * _wordcontext + offset * _atomDim;
    tmpK = count;
    for (int idk = 0; idk < _layer1OutDim; idk++) {
      temp = 0.0;
      for (int idy = 0; idy < _atomDim; idy++) {
        temp += _layer1W[tmpI + idy] * _atomEmb[tmpJ + idy];
      }
      _atomPreComputedForward[tmpK] = temp;
      tmpI += _lay1InputDim;
      tmpK += _curAtomPreComputedNum;
    }
    count++;
  }

  _wordPreComputedBackward = (double *) calloc(_layer1OutDim * _curWordPreComputedNum, sizeof(double));
  _atomPreComputedBackward = (double *) calloc(_layer1OutDim * _curAtomPreComputedNum, sizeof(double));

  double cost = 0.0;
  for (count = 0; count < example_num; count++) {
    const Example& example = examples[count];



    NRVec<double> mid_layer1out(_layer1OutDim), mid_layer1outLoss(_layer1OutDim);
    mat layer1out, layer1outLoss;
    mat layer2out, layer2outLoss;

    int offset;
    //forward propagation
    const Feature& feature = example.m_feature;
    const vector<int>& wneural_features = feature.wneural_features;
    const vector<int>& aneural_features = feature.aneural_features;

    assert(wneural_features.size() == _wordcontext && aneural_features.size() == _atomcontext);

    srand(iter*example_num + count);
    NRVec<bool> indexes_layer1(_layer1OutDim);
    for (int i = 0; i < _layer1OutDim; ++i)
    {
      if(1.0*rand()/RAND_MAX >= _dropOut)
      {
        indexes_layer1[i] = true;
      }
      else
      {
        indexes_layer1[i] = false;
      }
    }

    offset = 0;

    mid_layer1out = 0.0;
    for (int i = 0; i < _wordcontext; i++) {
      int curFeatId = wneural_features[i] * wneural_features.size() + i;
      tmpJ = wneural_features[i] * _wordDim;
      if (_curWordPreComputed.find(curFeatId) == _curWordPreComputed.end())
      {
        tmpI = offset + i * _wordDim;
        for (int idk = 0; idk < _layer1OutDim; idk++) {
          if(indexes_layer1[idk])
          {
            temp = 0.0;
            for (int j = 0; j < _wordDim; j++) {
              temp += _layer1W[tmpI + j] * _wordEmb[tmpJ + j];
            }
            mid_layer1out[idk] += temp;
          }
          tmpI += _lay1InputDim;
        }
      }
      else {
        tmpI = _curWordPreComputedId[curFeatId];
        for (int idk = 0; idk < _layer1OutDim; idk++) {
          if(indexes_layer1[idk])
          {
            mid_layer1out[idk] += _wordPreComputedForward[tmpI];
          }
          tmpI += _curWordPreComputedNum;
        }
      }
    }

    offset = _wordDim * _wordcontext;
    for (int i = 0; i < _atomcontext; i++) {
      int curFeatId = aneural_features[i] * aneural_features.size() + i;
      tmpJ = aneural_features[i] * _atomDim;
      if (_curAtomPreComputed.find(curFeatId) == _curAtomPreComputed.end())
      {
        tmpI = offset + i * _atomDim;
        for (int idk = 0; idk < _layer1OutDim; idk++) {
          if(indexes_layer1[idk])
          {
            temp = 0.0;
            for (int j = 0; j < _atomDim; j++) {
              temp += _layer1W[tmpI + j] * _atomEmb[tmpJ + j];
            }
            mid_layer1out[idk] += temp;
          }
          tmpI += _lay1InputDim;
        }
      }
      else {
        tmpI = _curAtomPreComputedId[curFeatId];
        for (int idk = 0; idk < _layer1OutDim; idk++) {
          if(indexes_layer1[idk])
          {
            mid_layer1out[idk] += _atomPreComputedForward[tmpI];
          }
          tmpI += _curAtomPreComputedNum;
        }
      }
    }

    layer1out.zeros(_layer1OutDim, 1);
    for (int idx = 0; idx < _layer1OutDim; idx++) {
      if(indexes_layer1[idx])
      {
        mid_layer1out[idx] = mid_layer1out[idx] + _layer1b(idx, 0);
        layer1out(idx, 0) = mid_layer1out[idx] * mid_layer1out[idx] * mid_layer1out[idx];
      }
    }

    _layer2.ComputeForwardScore(layer1out, layer2out);

    // get delta for each output

    // Feed forward to softmax layer (no activation yet)

    const vector<int>& labels = example.m_labels;

    if (_lossFunc == 1) {
      NRVec<double> scores(_actionSize);
      double sum1 = -1e10, sum2 = -1e10;
      int optLabel1 = -1, optLabel2 = -1;
      for (int i = 0; i < _actionSize; ++i) {
        scores[i] = -1e10;
        if (labels[i] >= 0) {
          scores[i] = layer2out(i, 0);
          if (labels[i] == 1) {
            if (optLabel1 == -1 || sum1 < scores[i]) {
              sum1 = scores[i];
              optLabel1 = i;
            }
          }
          if (optLabel2 == -1 || sum2 < scores[i]) {
            sum2 = scores[i];
            optLabel2 = i;
          }
        }
      }

      double loss = sum2 - sum1 + 1;

      cost += (sum2 - sum1) / example_num;

      _eval.overall_label_count++;
      if (optLabel1 == optLabel2) {
        _eval.correct_label_count++;
        continue; // need no update
      }

      layer2outLoss.zeros(_actionSize, 1);

      if (optLabel1 != optLabel2) {
        layer2outLoss(optLabel1, 0) = -loss / example_num;
        layer2outLoss(optLabel2, 0) = loss / example_num;
      }
    } else {
      int optLabel = -1;
      for (int i = 0; i < _actionSize; ++i) {
        //std::cout << layer2out(i, 0) << " ";
        if (labels[i] >= 0) {
          if (optLabel < 0 || layer2out(i, 0) > layer2out(optLabel, 0))
            optLabel = i;
        }
      }

      NRVec<double> scores(_actionSize);
      double sum1 = 0.0;
      double sum2 = 0.0;
      double maxScore = layer2out(optLabel, 0);
      for (int i = 0; i < _actionSize; ++i) {
        scores[i] = -1e10;
        if (labels[i] >= 0) {
          scores[i] = exp(layer2out(i, 0) - maxScore);
          if (labels[i] == 1)
            sum1 += scores[i];
          sum2 += scores[i];
        }
      }

      cost += (log(sum2) - log(sum1)) / example_num;
      if (labels[optLabel] == 1)
        _eval.correct_label_count++;
      _eval.overall_label_count++;

      layer2outLoss.resize(_actionSize, 1);
      for (int i = 0; i < _actionSize; ++i) {
        layer2outLoss(i, 0) = 0.0;
        if (labels[i] >= 0) {
          layer2outLoss(i, 0) = (scores[i] / sum2 - labels[i]) / example_num;
        }
      }
    }

    // loss backward propagation

    _layer2.ComputeBackwardLoss(layer1out, layer2out, layer2outLoss, layer1outLoss);

    mid_layer1outLoss = 0.0;
    for (int idx = 0; idx < _layer1OutDim; idx++) {
      if(indexes_layer1[idx])
      {
        mid_layer1outLoss[idx] = 3 * layer1outLoss(idx, 0) * mid_layer1out[idx] * mid_layer1out[idx];
        _gradlayer1b(idx, 0) = _gradlayer1b(idx, 0) + mid_layer1outLoss[idx];
      }
    }

    offset = 0;
    for (int i = 0; i < _wordcontext; i++) {
      int curFeatId = wneural_features[i] * wneural_features.size() + i;
      tmpJ = wneural_features[i] * _wordDim;
      if (_curWordPreComputed.find(curFeatId) == _curWordPreComputed.end())
      {
        tmpI = offset + i * _wordDim;
        for (int idk = 0; idk < _layer1OutDim; idk++) {
          if(indexes_layer1[idk])
          {
            temp = mid_layer1outLoss[idk];
            for (int j = 0; j < _wordDim; j++) {
              _gradlayer1W[tmpI + j] += temp * _wordEmb[tmpJ + j];
              if (_b_wordEmb_finetune)
                _grad_wordEmb[tmpJ + j] += temp * _layer1W[tmpI + j];
            }
          }
          tmpI += _lay1InputDim;
        }
      }
      else {
        tmpI = _curWordPreComputedId[curFeatId];
        for (int idk = 0; idk < _layer1OutDim; idk++) {
          if(indexes_layer1[idk])
          {
            _wordPreComputedBackward[tmpI] += mid_layer1outLoss[idk];
          }
          tmpI += _curWordPreComputedNum;
        }
      }
    }

    offset += _wordDim * _wordcontext;
    for (int i = 0; i < _atomcontext; i++) {
      int curFeatId = aneural_features[i] * aneural_features.size() + i;
      tmpJ = aneural_features[i] * _atomDim;
      if (_curAtomPreComputed.find(curFeatId) == _curAtomPreComputed.end())
      {
        tmpI = offset + i * _atomDim;
        for (int idk = 0; idk < _layer1OutDim; idk++) {
          if(indexes_layer1[idk])
          {
            temp = mid_layer1outLoss[idk];
            for (int j = 0; j < _atomDim; j++) {
              _gradlayer1W[tmpI + j] += temp * _atomEmb[tmpJ + j];
              _grad_atomEmb[tmpJ + j] += temp * _layer1W[tmpI + j];
            }
          }
          tmpI += _lay1InputDim;
        }
      }
      else {
        tmpI = _curAtomPreComputedId[curFeatId];
        for (int idk = 0; idk < _layer1OutDim; idk++) {
          if(indexes_layer1[idk])
          {
            _atomPreComputedBackward[tmpI] += mid_layer1outLoss[idk];
          }
          tmpI += _curAtomPreComputedNum;
        }
      }
    }

  }

  //backward feed back
  for (it = _curWordPreComputed.begin(); it != _curWordPreComputed.end(); ++it) {
    count = _curWordPreComputedId[*it];
    int offset = (*it) % _wordcontext;
    int wordId = (*it) / _wordcontext;
    tmpI = offset * _wordDim;
    tmpJ = wordId * _wordDim;
    tmpK = count;
    for (int idk = 0; idk < _layer1OutDim; idk++) {
      temp = _wordPreComputedBackward[tmpK];
      for (int idy = 0; idy < _wordDim; idy++) {
        _gradlayer1W[tmpI + idy] += temp * _wordEmb[tmpJ + idy];
        if (_b_wordEmb_finetune)
          _grad_wordEmb[tmpJ + idy] += temp * _layer1W[tmpI + idy];
      }
      tmpI += _lay1InputDim;
      tmpK += _curWordPreComputedNum;
    }
  }

  for (it = _curAtomPreComputed.begin(); it != _curAtomPreComputed.end(); ++it) {
    count = _curAtomPreComputedId[*it];
    int offset = (*it) % _atomcontext;
    int atomId = (*it) / _atomcontext;
    tmpI = _wordDim * _wordcontext + offset * _atomDim;
    tmpJ = atomId * _atomDim;
    tmpK = count;
    for (int idk = 0; idk < _layer1OutDim; idk++) {
      temp = _atomPreComputedBackward[tmpK];
      for (int idy = 0; idy < _atomDim; idy++) {
        _gradlayer1W[tmpI + idy] += temp * _atomEmb[tmpJ + idy];
        _grad_atomEmb[tmpJ + idy] += temp * _layer1W[tmpI + idy];
      }
      tmpI += _lay1InputDim;
      tmpK += _curAtomPreComputedNum;
    }
  }

  Free(&_wordPreComputedBackward);
  Free(&_atomPreComputedBackward);
  return cost;
}

void NNClassifier::predict(const Feature& feature, vector<double>& results) {

  NRVec<double> mid_layer1out(_layer1OutDim);
  mat layer1out;

  mat layer2out;

  int offset;

  static double temp;
  //forward propagation

  static int tmpI, tmpJ, tmpK;

  const vector<int>& wneural_features = feature.wneural_features;
  const vector<int>& aneural_features = feature.aneural_features;

  assert(wneural_features.size() == _wordcontext && aneural_features.size() == _atomcontext);

  offset = 0;

  mid_layer1out = 0.0;
  for (int i = 0; i < _wordcontext; i++) {
    int curFeatId = wneural_features[i] * wneural_features.size() + i;
    tmpJ = wneural_features[i] * _wordDim;
    if (_curWordPreComputed.find(curFeatId) == _curWordPreComputed.end()) {
      tmpI = offset + i * _wordDim;
      for (int idk = 0; idk < _layer1OutDim; idk++) {
        temp = 0.0;
        for (int j = 0; j < _wordDim; j++) {
          temp += _layer1W[tmpI + j] * _wordEmb[tmpJ + j];
        }
        mid_layer1out[idk] += temp;
        tmpI += _lay1InputDim;
      }
    } else {
      tmpI = _curWordPreComputedId[curFeatId];
      for (int idk = 0; idk < _layer1OutDim; idk++) {
        mid_layer1out[idk] += _wordPreComputedForward[tmpI];
        tmpI += _curWordPreComputedNum;
      }
    }
  }

  offset = _wordDim * _wordcontext;
  for (int i = 0; i < _atomcontext; i++) {
    int curFeatId = aneural_features[i] * aneural_features.size() + i;
    tmpJ = aneural_features[i] * _atomDim;
    if (_curAtomPreComputed.find(curFeatId) == _curAtomPreComputed.end()) {
      tmpI = offset + i * _atomDim;
      for (int idk = 0; idk < _layer1OutDim; idk++) {
        temp = 0.0;
        for (int j = 0; j < _atomDim; j++) {
          temp += _layer1W[tmpI + j] * _atomEmb[tmpJ + j];
        }
        mid_layer1out[idk] += temp;
        tmpI += _lay1InputDim;
      }
    } else {
      tmpI = _curAtomPreComputedId[curFeatId];
      for (int idk = 0; idk < _layer1OutDim; idk++) {
        mid_layer1out[idk] += _atomPreComputedForward[tmpI];
        tmpI += _curAtomPreComputedNum;
      }
    }
  }

  layer1out.resize(_layer1OutDim, 1);
  for (int idx = 0; idx < _layer1OutDim; idx++) {
    mid_layer1out[idx] = mid_layer1out[idx] + _layer1b(idx, 0);
    layer1out(idx, 0) = mid_layer1out[idx] * mid_layer1out[idx] * mid_layer1out[idx];
  }

  _layer2.ComputeForwardScore(layer1out, layer2out);

  results.resize(_actionSize);
  for (int i = 0; i < _actionSize; i++) {
    results[i] = layer2out(i, 0);
  }

}

double NNClassifier::computeScore(const Example& example) {
  NRVec<double> mid_layer1out(_layer1OutDim);
  mat layer1out;
  mat layer2out;

  int offset;
  static double temp;
  static int tmpI, tmpJ, tmpK;
  //forward propagation

  const Feature& feature = example.m_feature;
  const vector<int>& wneural_features = feature.wneural_features;
  const vector<int>& aneural_features = feature.aneural_features;

  assert(wneural_features.size() == _wordcontext && aneural_features.size() == _atomcontext);

  offset = 0;

  mid_layer1out = 0.0;
  for (int i = 0; i < _wordcontext; i++) {
    int curFeatId = wneural_features[i] * wneural_features.size() + i;
    tmpJ = wneural_features[i] * _wordDim;
    tmpI = offset + i * _wordDim;
    for (int idk = 0; idk < _layer1OutDim; idk++) {
      temp = 0.0;
      for (int j = 0; j < _wordDim; j++) {
        temp += _layer1W[tmpI + j] * _wordEmb[tmpJ + j];
      }
      mid_layer1out[idk] += temp;
      tmpI += _lay1InputDim;
    }

  }

  offset = _wordDim * _wordcontext;
  for (int i = 0; i < _atomcontext; i++) {
    int curFeatId = aneural_features[i] * aneural_features.size() + i;
    tmpJ = aneural_features[i] * _atomDim;

    tmpI = offset + i * _atomDim;
    for (int idk = 0; idk < _layer1OutDim; idk++) {
      temp = 0.0;
      for (int j = 0; j < _atomDim; j++) {
        temp += _layer1W[tmpI + j] * _atomEmb[tmpJ + j];
      }
      mid_layer1out[idk] += temp;
      tmpI += _lay1InputDim;
    }
  }

  layer1out.resize(_layer1OutDim, 1);
  for (int idx = 0; idx < _layer1OutDim; idx++) {
    mid_layer1out[idx] = mid_layer1out[idx] + _layer1b(idx, 0);
    layer1out(idx, 0) = mid_layer1out[idx] * mid_layer1out[idx] * mid_layer1out[idx];
  }

  _layer2.ComputeForwardScore(layer1out, layer2out);

  // get delta for each output

  // Feed forward to softmax layer (no activation yet)

  int optLabel = -1;
  const vector<int>& labels = example.m_labels;
  for (int i = 0; i < _actionSize; ++i) {
    if (labels[i] >= 0) {
      if (optLabel < 0 || layer2out(i, 0) > layer2out(optLabel, 0))
        optLabel = i;
    }
  }

  NRVec<double> scores(_actionSize);
  double sum1 = 0.0;
  double sum2 = 0.0;
  double maxScore = layer2out(optLabel, 0);
  for (int i = 0; i < _actionSize; ++i) {
    scores[i] = -1e10;
    if (labels[i] >= 0) {
      scores[i] = exp(layer2out(i, 0) - maxScore);
      if (labels[i] == 1)
        sum1 += scores[i];
      sum2 += scores[i];
    }
  }

  return log(sum2) - log(sum1);
}

void NNClassifier::updateParams(double regularizationWeight, double adaAlpha, double adaEps) {

  _layer2.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);

  static mat tmp;
  static hash_set<int>::iterator it;
  static int totalSize;

  if (_b_wordEmb_finetune) {
    totalSize = _wordSize * _wordDim;
    for (int index = 0; index < totalSize; index++) {
      double _grad_wordEmb_ij = _grad_wordEmb[index] + regularizationWeight * _wordEmb[index];
      _eg2_wordEmb[index] += _grad_wordEmb_ij * _grad_wordEmb_ij;
      double tmp_normaize_alpha = sqrt(_eg2_wordEmb[index] + adaEps);
      double tmp_alpha = adaAlpha / tmp_normaize_alpha;
      _wordEmb[index] -= tmp_alpha * _grad_wordEmb[index];
      _grad_wordEmb[index] = 0.0;
    }
  }

  totalSize = _atomSize * _atomDim;
  for (int index = 0; index < totalSize; index++) {
      double _grad_atomEmb_ij = _grad_atomEmb[index] + regularizationWeight * _atomEmb[index];
      _eg2_atomEmb[index] += _grad_atomEmb_ij * _grad_atomEmb_ij;
      double tmp_normaize_alpha = sqrt(_eg2_atomEmb[index] + adaEps);
      double tmp_alpha = adaAlpha / tmp_normaize_alpha;
      _atomEmb[index] -= tmp_alpha * _grad_atomEmb[index];
      _grad_atomEmb[index] = 0.0;
  }


  totalSize = _layer1OutDim * _lay1InputDim;
  for (int index = 0; index < totalSize; index++) {
    double _grad_atomEmb_ij = _gradlayer1W[index] + regularizationWeight * _layer1W[index];
    _eg2layer1W[index] += _grad_atomEmb_ij * _grad_atomEmb_ij;
    double tmp_normaize_alpha = sqrt(_eg2layer1W[index] + adaEps);
    double tmp_alpha = adaAlpha / tmp_normaize_alpha;
    _layer1W[index] -= tmp_alpha * _gradlayer1W[index];
    _gradlayer1W[index] = 0.0;
  }

  _gradlayer1b = _gradlayer1b + _layer1b * regularizationWeight;
  _eg2layer1b = _eg2layer1b + _gradlayer1b % _gradlayer1b;
  _layer1b = _layer1b - _gradlayer1b * adaAlpha / sqrt(_eg2layer1b + adaEps);
  _gradlayer1b.zeros();
}

// This is for sparse layers only, the cols are sparse
void NNClassifier::checkgradColSparse(const vector<Example>& examples, mat& Wd, const mat& gradWd, const string& mark, int iter,
    const hash_set<int>& sparseColIndexes, const mat& ft) {
  //Random randWdRowcheck = new Random(iter + "Row".hashCode() + hash));
  int charseed = mark.length();
  for (int i = 0; i < mark.length(); i++) {
    charseed = (int) (mark[i]) * 5 + charseed;
  }
  srand(iter + charseed);
  std::vector<int> idRows, idCols;
  idRows.clear();
  idCols.clear();
  if (sparseColIndexes.empty()) {
    for (int i = 0; i < Wd.n_cols; ++i)
      idCols.push_back(i);
  } else {
    hash_set<int>::const_iterator it;
    for (it = sparseColIndexes.begin(); it != sparseColIndexes.end(); ++it)
      idCols.push_back(*it);
  }

  for (int idx = 0; idx < Wd.n_rows; idx++)
    idRows.push_back(idx);

  random_shuffle(idRows.begin(), idRows.end());
  random_shuffle(idCols.begin(), idCols.end());

  int check_i = idRows[0], check_j = idCols[0];

  double orginValue = Wd(check_i, check_j);

  Wd(check_i, check_j) = orginValue + 0.001;
  double lossAdd = 0.0;
  for (int i = 0; i < examples.size(); i++) {
    Example oneExam = examples[i];
    lossAdd += computeScore(oneExam);
  }

  Wd(check_i, check_j) = orginValue - 0.001;
  double lossPlus = 0.0;
  for (int i = 0; i < examples.size(); i++) {
    Example oneExam = examples[i];
    lossPlus += computeScore(oneExam);
  }

  double mockGrad = (lossAdd - lossPlus) / (0.002 * ft(check_i, check_j));
  mockGrad = mockGrad / examples.size();
  double computeGrad = gradWd(check_i, check_j);

  printf("Iteration %d, Checking gradient for %s[%d][%d]:\t", iter, mark.c_str(), check_i, check_j);
  printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

  Wd(check_i, check_j) = orginValue;
}

void NNClassifier::checkgradRowSparse(const vector<Example>& examples, mat& Wd, const mat& gradWd, const string& mark, int iter,
    const hash_set<int>& sparseRowIndexes, const mat& ft) {
  //Random randWdRowcheck = new Random(iter + "Row".hashCode() + hash));
  int charseed = mark.length();
  for (int i = 0; i < mark.length(); i++) {
    charseed = (int) (mark[i]) * 5 + charseed;
  }
  srand(iter + charseed);
  std::vector<int> idRows, idCols;
  idRows.clear();
  idCols.clear();
  if (sparseRowIndexes.empty()) {
    for (int i = 0; i < Wd.n_rows; ++i)
      idRows.push_back(i);
  } else {
    hash_set<int>::iterator it;
    for (it = sparseRowIndexes.begin(); it != sparseRowIndexes.end(); ++it)
      idRows.push_back(*it);
  }

  for (int idx = 0; idx < Wd.n_cols; idx++)
    idCols.push_back(idx);

  random_shuffle(idRows.begin(), idRows.end());
  random_shuffle(idCols.begin(), idCols.end());

  int check_i = idRows[0], check_j = idCols[0];

  double orginValue = Wd(check_i, check_j);

  Wd(check_i, check_j) = orginValue + 0.001;
  double lossAdd = 0.0;
  for (int i = 0; i < examples.size(); i++) {
    Example oneExam = examples[i];
    lossAdd += computeScore(oneExam);
  }

  Wd(check_i, check_j) = orginValue - 0.001;
  double lossPlus = 0.0;
  for (int i = 0; i < examples.size(); i++) {
    Example oneExam = examples[i];
    lossPlus += computeScore(oneExam);
  }

  double mockGrad = (lossAdd - lossPlus) / (0.002 * ft(check_i, check_j));
  mockGrad = mockGrad / examples.size();
  double computeGrad = gradWd(check_i, check_j);

  printf("Iteration %d, Checking gradient for %s[%d][%d]:\t", iter, mark.c_str(), check_i, check_j);
  printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

  Wd(check_i, check_j) = orginValue;
}

void NNClassifier::checkgrad(const vector<Example>& examples, mat& Wd, const mat& gradWd, const string& mark, int iter) {
  //Random randWdRowcheck = new Random(iter + "Row".hashCode() + hash));
  int charseed = mark.length();
  for (int i = 0; i < mark.length(); i++) {
    charseed = (int) (mark[i]) * 5 + charseed;
  }
  srand(iter + charseed);
  std::vector<int> idRows, idCols;
  idRows.clear();
  idCols.clear();
  for (int i = 0; i < Wd.n_rows; ++i)
    idRows.push_back(i);
  for (int idx = 0; idx < Wd.n_cols; idx++)
    idCols.push_back(idx);

  random_shuffle(idRows.begin(), idRows.end());
  random_shuffle(idCols.begin(), idCols.end());

  int check_i = idRows[0], check_j = idCols[0];

  double orginValue = Wd(check_i, check_j);

  Wd(check_i, check_j) = orginValue + 0.001;
  double lossAdd = 0.0;
  for (int i = 0; i < examples.size(); i++) {
    Example oneExam = examples[i];
    lossAdd += computeScore(oneExam);
  }

  Wd(check_i, check_j) = orginValue - 0.001;
  double lossPlus = 0.0;
  for (int i = 0; i < examples.size(); i++) {
    Example oneExam = examples[i];
    lossPlus += computeScore(oneExam);
  }

  double mockGrad = (lossAdd - lossPlus) / 0.002;
  mockGrad = mockGrad / examples.size();
  double computeGrad = gradWd(check_i, check_j);

  printf("Iteration %d, Checking gradient for %s[%d][%d]:\t", iter, mark.c_str(), check_i, check_j);
  printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

  Wd(check_i, check_j) = orginValue;
}

void NNClassifier::checkgrad(const vector<Example>& examples, double* Wd, const double* gradWd, const string& mark, int iter, int rowSize, int colSize) {
  //Random randWdRowcheck = new Random(iter + "Row".hashCode() + hash));
  int charseed = mark.length();
  for (int i = 0; i < mark.length(); i++) {
    charseed = (int) (mark[i]) * 5 + charseed;
  }
  srand(iter + charseed);
  std::vector<int> idRows, idCols;
  idRows.clear();
  idCols.clear();
  for (int i = 0; i < rowSize; ++i)
    idRows.push_back(i);
  for (int idx = 0; idx < colSize; idx++)
    idCols.push_back(idx);

  random_shuffle(idRows.begin(), idRows.end());
  random_shuffle(idCols.begin(), idCols.end());

  int check_i = idRows[0], check_j = idCols[0];

  double orginValue = Wd[check_i* colSize + check_j];

  Wd[check_i* colSize + check_j] = orginValue + 0.001;
  double lossAdd = 0.0;
  for (int i = 0; i < examples.size(); i++) {
    Example oneExam = examples[i];
    lossAdd += computeScore(oneExam);
  }

  Wd[check_i* colSize + check_j] = orginValue - 0.001;
  double lossPlus = 0.0;
  for (int i = 0; i < examples.size(); i++) {
    Example oneExam = examples[i];
    lossPlus += computeScore(oneExam);
  }

  double mockGrad = (lossAdd - lossPlus) / 0.002;
  mockGrad = mockGrad / examples.size();
  double computeGrad = gradWd[check_i* colSize + check_j];

  printf("Iteration %d, Checking gradient for %s[%d][%d]:\t", iter, mark.c_str(), check_i, check_j);
  printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

  Wd[check_i* colSize + check_j] = orginValue;
}

void NNClassifier::checkgrads(const vector<Example>& examples, int iter) {
  checkgrad(examples, _layer1W, _gradlayer1W, "_layer1W", iter, _layer1OutDim, _lay1InputDim);
  checkgrad(examples, _layer1b, _gradlayer1b, "_layer1b", iter);
  checkgrad(examples, _layer2._W, _layer2._gradW, "_layer2._W", iter);
}
