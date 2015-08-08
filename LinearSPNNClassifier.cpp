/*
 * LinearSPNNClassifier.cpp
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

// did not judge memory allocate fail!
#include "LinearSPNNClassifier.h"

LinearSPNNClassifier::LinearSPNNClassifier() {
  // TODO Auto-generated constructor stub
  _b_wordEmb_finetune = true;
  _lossFunc = 0;
  _wordEmb = NULL;
  _grad_wordEmb = NULL;
  _eg2_wordEmb = NULL;

  _layer1W = NULL;
  _gradlayer1W = NULL;
  _eg2layer1W = NULL;
  _wordPreComputedForward = NULL;
  _wordPreComputedBackward = NULL;
  _dropOut = 0.0;
}

LinearSPNNClassifier::~LinearSPNNClassifier() {
  // TODO Auto-generated destructor stub
  Free(&_wordEmb);
  Free(&_grad_wordEmb);
  Free(&_eg2_wordEmb);

  Free(&_layer1W);
  Free(&_gradlayer1W);
  Free(&_eg2layer1W);
  Free(&_wordPreComputedForward);
  Free(&_wordPreComputedBackward);
}

// did not judge memory allocate fail!
void LinearSPNNClassifier::init(int wordDim, int wordSize, int wordcontext, int labelSize,
    int linearfeatSize) {
  _b_wordEmb_finetune = true;
  _actionSize = labelSize;
  _wordcontext = wordcontext;
  _wordSize = wordSize;
  _wordDim = wordDim;

  _linearfeatSize = linearfeatSize;

  _wordEmb = (double *) calloc(_wordSize * _wordDim, sizeof(double));
  _grad_wordEmb = (double *) calloc(_wordSize * _wordDim, sizeof(double));
  _eg2_wordEmb = (double *) calloc(_wordSize * _wordDim, sizeof(double));

  randomMatAssign(_wordEmb, _wordSize * _wordDim, 1.0, 0);
  //normalize to a unit sphere
  //for (int idx = 0; idx < _wordDim; idx++) {
  // normalize_mat_onecol(_wordEmb, idx, _wordSize, _wordDim);
  //}
  for (int idx = 0; idx < _wordSize; idx++) {
    normalize_mat_onerow(_wordEmb, idx, _wordSize, _wordDim);
  }


  _lay1InputDim = _wordDim * _wordcontext;
  double bound = 0.01;
  //double bound = sqrt(6.0 / (_actionSize + _lay1InputDim + 1));
  _layer1W = (double *) calloc(_actionSize * _lay1InputDim, sizeof(double));
  _gradlayer1W = (double *) calloc(_actionSize * _lay1InputDim, sizeof(double));
  _eg2layer1W = (double *) calloc(_actionSize * _lay1InputDim, sizeof(double));
  randomMatAssign(_layer1W, _actionSize * _lay1InputDim, bound, 2);


  _layer_linear.initial(_actionSize, _linearfeatSize);

  _eval.reset();

}

// did not judge memory allocate fail!
void LinearSPNNClassifier::init(const mat& wordEmb, int wordcontext, int labelSize,
    int linearfeatSize) {

  _actionSize = labelSize;
  _wordcontext = wordcontext;
  _wordSize = wordEmb.n_rows;
  _wordDim = wordEmb.n_cols;
  _linearfeatSize = linearfeatSize;

  _wordEmb = (double *) calloc(_wordSize * _wordDim, sizeof(double));
  _grad_wordEmb = (double *) calloc(_wordSize * _wordDim, sizeof(double));
  _eg2_wordEmb = (double *) calloc(_wordSize * _wordDim, sizeof(double));

  assign(_wordEmb, wordEmb);
  //normalize to a unit sphere
  //for (int idx = 0; idx < _wordDim; idx++) {
  //  normalize_mat_onecol(_wordEmb, idx, _wordSize, _wordDim);
  //}
  for (int idx = 0; idx < _wordSize; idx++) {
    normalize_mat_onerow(_wordEmb, idx, _wordSize, _wordDim);
  }



  _lay1InputDim = _wordDim * _wordcontext;
  //double bound = sqrt(6.0 / (_actionSize + _lay1InputDim + 1));
  double bound = 0.01;
  _layer1W = (double *) calloc(_actionSize * _lay1InputDim, sizeof(double));
  _gradlayer1W = (double *) calloc(_actionSize * _lay1InputDim, sizeof(double));
  _eg2layer1W = (double *) calloc(_actionSize * _lay1InputDim, sizeof(double));
  randomMatAssign(_layer1W, _actionSize * _lay1InputDim, bound, 2);



  _layer_linear.initial(_actionSize, _linearfeatSize);

  _eval.reset();

}

void LinearSPNNClassifier::preCompute() {
  static int count, tmpI, tmpJ, tmpK;
  static hash_set<int>::iterator it;
  static double temp;
  for (it = _wordPreComputed.begin(); it != _wordPreComputed.end(); ++it) {
    _curWordPreComputed.insert(*it);
  }



  _curWordPreComputedId.clear();
  Free(&_wordPreComputedForward);
  Free(&_wordPreComputedBackward);
  //initial
  _curWordPreComputedNum = _curWordPreComputed.size();
  _wordPreComputedForward = (double *) calloc(_actionSize * _curWordPreComputedNum, sizeof(double));
  count = 0;
  for (it = _curWordPreComputed.begin(); it != _curWordPreComputed.end(); ++it) {
    _curWordPreComputedId[*it] = count;
    int offset = (*it) % _wordcontext;
    int wordId = (*it) / _wordcontext;
    tmpJ = wordId * _wordDim;
    tmpI = offset * _wordDim;
    tmpK = count;
    for (int idk = 0; idk < _actionSize; idk++) {
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


}

double LinearSPNNClassifier::process(const vector<Example>& examples, int iter) {
  _eval.reset();

  _curWordPreComputed.clear();

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
  }

  _curWordPreComputedId.clear();
  Free(&_wordPreComputedForward);
  Free(&_wordPreComputedBackward);
  //initial
  _curWordPreComputedNum = _curWordPreComputed.size();
  _wordPreComputedForward = (double *) calloc(_actionSize * _curWordPreComputedNum, sizeof(double));
  count = 0;
  for (it = _curWordPreComputed.begin(); it != _curWordPreComputed.end(); ++it) {
    _curWordPreComputedId[*it] = count;
    int offset = (*it) % _wordcontext;
    int wordId = (*it) / _wordcontext;
    tmpJ = wordId * _wordDim;
    tmpI = offset * _wordDim;
    tmpK = count;
    for (int idk = 0; idk < _actionSize; idk++) {
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



  _wordPreComputedBackward = (double *) calloc(_actionSize * _curWordPreComputedNum, sizeof(double));

  double cost = 0.0;
  for (count = 0; count < example_num; count++) {
    const Example& example = examples[count];

    NRVec<double> mid_layer1out(_actionSize), mid_layer1outLoss(_actionSize);
    mat layer1out, layer1outLoss;
    mat layer_linear_output, output;

    int offset;
    //forward propagation
    const Feature& feature = example.m_feature;
    const vector<int>& wneural_features = feature.wneural_features;
    const vector<int>& aneural_features = feature.aneural_features;

    assert(wneural_features.size() == _wordcontext);
    srand(iter * example_num + count);
    int randnum = rand();

    NRVec<bool> indexes_layer1(_actionSize);
    for (int i = 0; i < _actionSize; ++i) {
      if (1.0 * rand() / RAND_MAX >= _dropOut) {
        indexes_layer1[i] = true;
      } else {
        indexes_layer1[i] = false;
      }
    }

    offset = 0;

    mid_layer1out = 0.0;
    for (int i = 0; i < _wordcontext; i++) {
      int curFeatId = wneural_features[i] * wneural_features.size() + i;
      tmpJ = wneural_features[i] * _wordDim;
      if (_curWordPreComputed.find(curFeatId) == _curWordPreComputed.end()) {
        tmpI = offset + i * _wordDim;
        for (int idk = 0; idk < _actionSize; idk++) {
          if (indexes_layer1[idk]) {
            temp = 0.0;
            for (int j = 0; j < _wordDim; j++) {
              temp += _layer1W[tmpI + j] * _wordEmb[tmpJ + j];
            }
            mid_layer1out[idk] += temp;
          }
          tmpI += _lay1InputDim;
        }
      } else {
        tmpI = _curWordPreComputedId[curFeatId];
        for (int idk = 0; idk < _actionSize; idk++) {
          if (indexes_layer1[idk]) {
            mid_layer1out[idk] += _wordPreComputedForward[tmpI];
          }
          tmpI += _curWordPreComputedNum;
        }
      }
    }

    layer1out.resize(_actionSize, 1);
    for (int idx = 0; idx < _actionSize; idx++) {
      if (indexes_layer1[idx]) {
        layer1out[idx] = mid_layer1out[idx];
      }
    }

    //const vector<int>& linear_features = feature.linear_features;
    //linear features should not be dropped out

    vector<int> linear_features;
    for (int idx = 0; idx < feature.linear_features.size(); idx++) {
      if (1.0 * rand() / RAND_MAX >= _dropOut) {
        linear_features.push_back(feature.linear_features[idx]);
      }
    }

    _layer_linear.ComputeForwardScore(linear_features, layer_linear_output);

    output = layer1out + layer_linear_output;

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
          scores[i] = output(i, 0);
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

      layer1outLoss.zeros(_actionSize, 1);

      if (optLabel1 != optLabel2) {
        layer1outLoss(optLabel1, 0) = -loss / example_num;
        layer1outLoss(optLabel2, 0) = loss / example_num;
      }
    } else {
      int optLabel = -1;
      for (int i = 0; i < _actionSize; ++i) {
        //std::cout << output(i, 0) << " ";
        if (labels[i] >= 0) {
          if (optLabel < 0 || output(i, 0) > output(optLabel, 0))
            optLabel = i;
        }
      }

      NRVec<double> scores(_actionSize);
      double sum1 = 0.0;
      double sum2 = 0.0;
      double maxScore = output(optLabel, 0);
      for (int i = 0; i < _actionSize; ++i) {
        scores[i] = -1e10;
        if (labels[i] >= 0) {
          scores[i] = exp(output(i, 0) - maxScore);
          if (labels[i] == 1)
            sum1 += scores[i];
          sum2 += scores[i];
        }
      }

      cost += (log(sum2) - log(sum1)) / example_num;
      if (labels[optLabel] == 1)
        _eval.correct_label_count++;
      _eval.overall_label_count++;

      layer1outLoss.resize(_actionSize, 1);
      for (int i = 0; i < _actionSize; ++i) {
        layer1outLoss(i, 0) = 0.0;
        if (labels[i] >= 0) {
          layer1outLoss(i, 0) = (scores[i] / sum2 - labels[i]) / example_num;
        }
      }
    }

    // loss backward propagation

    _layer_linear.ComputeBackwardLoss(linear_features, layer1outLoss);


    mid_layer1outLoss = 0.0;
    for (int idx = 0; idx < _actionSize; idx++) {
      if (indexes_layer1[idx]) {
        mid_layer1outLoss[idx] = layer1outLoss(idx, 0);
      }
    }

    offset = 0;
    for (int i = 0; i < _wordcontext; i++) {
      int curFeatId = wneural_features[i] * wneural_features.size() + i;
      tmpJ = wneural_features[i] * _wordDim;
      if (_curWordPreComputed.find(curFeatId) == _curWordPreComputed.end()) {
        tmpI = offset + i * _wordDim;
        for (int idk = 0; idk < _actionSize; idk++) {
          if (indexes_layer1[idk]) {
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
        for (int idk = 0; idk < _actionSize; idk++) {
          if (indexes_layer1[idk]) {
            _wordPreComputedBackward[tmpI] += mid_layer1outLoss[idk];
          }
          tmpI += _curWordPreComputedNum;
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
    for (int idk = 0; idk < _actionSize; idk++) {
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


  Free(&_wordPreComputedBackward);
  return cost;
}

void LinearSPNNClassifier::predict(const Feature& feature, vector<double>& results) {
  NRVec<double> mid_layer1out(_actionSize);
  mat layer1out;
  mat layer_linear_output, output;

  int offset;

  static double temp;
  //forward propagation

  static int tmpI, tmpJ, tmpK;

  const vector<int>& wneural_features = feature.wneural_features;
  const vector<int>& aneural_features = feature.aneural_features;

  assert(wneural_features.size() == _wordcontext);

  offset = 0;

  mid_layer1out = 0.0;
  for (int i = 0; i < _wordcontext; i++) {
    int curFeatId = wneural_features[i] * wneural_features.size() + i;
    tmpJ = wneural_features[i] * _wordDim;
    if (_curWordPreComputed.find(curFeatId) == _curWordPreComputed.end()) {
      tmpI = offset + i * _wordDim;
      for (int idk = 0; idk < _actionSize; idk++) {
        temp = 0.0;
        for (int j = 0; j < _wordDim; j++) {
          temp += _layer1W[tmpI + j] * _wordEmb[tmpJ + j];
        }
        mid_layer1out[idk] += temp;
        tmpI += _lay1InputDim;
      }
    } else {
      tmpI = _curWordPreComputedId[curFeatId];
      for (int idk = 0; idk < _actionSize; idk++) {
        mid_layer1out[idk] += _wordPreComputedForward[tmpI];
        tmpI += _curWordPreComputedNum;
      }
    }
  }

  layer1out.resize(_actionSize, 1);
  for (int idx = 0; idx < _actionSize; idx++) {
    layer1out(idx, 0) = mid_layer1out[idx];
  }


  const vector<int>& linear_features = feature.linear_features;
  _layer_linear.ComputeForwardScore(linear_features, layer_linear_output);
  output = layer1out + layer_linear_output;

  results.resize(_actionSize);
  for (int i = 0; i < _actionSize; i++) {
    results[i] = output(i, 0);
  }

}

double LinearSPNNClassifier::computeScore(const Example& example) {
  NRVec<double> mid_layer1out(_actionSize);
  mat layer1out;
  mat layer_linear_output, output;

  int offset;
  static double temp;
  static int tmpI, tmpJ, tmpK;
  //forward propagation

  const Feature& feature = example.m_feature;
  const vector<int>& wneural_features = feature.wneural_features;
  const vector<int>& aneural_features = feature.aneural_features;

  assert(wneural_features.size() == _wordcontext);

  offset = 0;

  mid_layer1out = 0.0;
  for (int i = 0; i < _wordcontext; i++) {
    int curFeatId = wneural_features[i] * wneural_features.size() + i;
    tmpJ = wneural_features[i] * _wordDim;
    tmpI = offset + i * _wordDim;
    for (int idk = 0; idk < _actionSize; idk++) {
      temp = 0.0;
      for (int j = 0; j < _wordDim; j++) {
        temp += _layer1W[tmpI + j] * _wordEmb[tmpJ + j];
      }
      mid_layer1out[idk] += temp;
      tmpI += _lay1InputDim;
    }

  }


  layer1out.resize(_actionSize, 1);
  for (int idx = 0; idx < _actionSize; idx++) {
    layer1out(idx, 0) = mid_layer1out[idx];
  }


  const vector<int>& linear_features = feature.linear_features;
  _layer_linear.ComputeForwardScore(linear_features, layer_linear_output);
  output = layer1out + layer_linear_output;

  // get delta for each output

  // Feed forward to softmax layer (no activation yet)

  int optLabel = -1;
  const vector<int>& labels = example.m_labels;
  for (int i = 0; i < _actionSize; ++i) {
    if (labels[i] >= 0) {
      if (optLabel < 0 || output(i, 0) > output(optLabel, 0))
        optLabel = i;
    }
  }

  NRVec<double> scores(_actionSize);
  double sum1 = 0.0;
  double sum2 = 0.0;
  double maxScore = output(optLabel, 0);
  for (int i = 0; i < _actionSize; ++i) {
    scores[i] = -1e10;
    if (labels[i] >= 0) {
      scores[i] = exp(output(i, 0) - maxScore);
      if (labels[i] == 1)
        sum1 += scores[i];
      sum2 += scores[i];
    }
  }

  return log(sum2) - log(sum1);
}

void LinearSPNNClassifier::updateParams(double regularizationWeight, double adaAlpha, double adaEps) {
  _layer_linear.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);

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


  totalSize = _actionSize * _lay1InputDim;
  for (int index = 0; index < totalSize; index++) {
    double _grad_atomEmb_ij = _gradlayer1W[index] + regularizationWeight * _layer1W[index];
    _eg2layer1W[index] += _grad_atomEmb_ij * _grad_atomEmb_ij;
    double tmp_normaize_alpha = sqrt(_eg2layer1W[index] + adaEps);
    double tmp_alpha = adaAlpha / tmp_normaize_alpha;
    _layer1W[index] -= tmp_alpha * _gradlayer1W[index];
    _gradlayer1W[index] = 0.0;
  }

}

// This is for sparse layers only, the cols are sparse
void LinearSPNNClassifier::checkgradColSparse(const vector<Example>& examples, mat& Wd, const mat& gradWd, const string& mark, int iter,
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

void LinearSPNNClassifier::checkgradRowSparse(const vector<Example>& examples, mat& Wd, const mat& gradWd, const string& mark, int iter,
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

void LinearSPNNClassifier::checkgrad(const vector<Example>& examples, mat& Wd, const mat& gradWd, const string& mark, int iter) {
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

void LinearSPNNClassifier::checkgrad(const vector<Example>& examples, double* Wd, const double* gradWd, const string& mark, int iter, int rowSize, int colSize) {
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

  double orginValue = Wd[check_i * colSize + check_j];

  Wd[check_i * colSize + check_j] = orginValue + 0.001;
  double lossAdd = 0.0;
  for (int i = 0; i < examples.size(); i++) {
    Example oneExam = examples[i];
    lossAdd += computeScore(oneExam);
  }

  Wd[check_i * colSize + check_j] = orginValue - 0.001;
  double lossPlus = 0.0;
  for (int i = 0; i < examples.size(); i++) {
    Example oneExam = examples[i];
    lossPlus += computeScore(oneExam);
  }

  double mockGrad = (lossAdd - lossPlus) / 0.002;
  mockGrad = mockGrad / examples.size();
  double computeGrad = gradWd[check_i * colSize + check_j];

  printf("Iteration %d, Checking gradient for %s[%d][%d]:\t", iter, mark.c_str(), check_i, check_j);
  printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

  Wd[check_i * colSize + check_j] = orginValue;
}

void LinearSPNNClassifier::checkgrads(const vector<Example>& examples, int iter) {
  checkgrad(examples, _layer1W, _gradlayer1W, "_layer1W", iter, _actionSize, _lay1InputDim);

}
