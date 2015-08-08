/*
 * LinearClassifier.cpp
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#include "LinearClassifier.h"

LinearClassifier::LinearClassifier() {
  // TODO Auto-generated constructor stub
  _lossFunc = 0;
  _dropOut = 0.0;
}

LinearClassifier::~LinearClassifier() {
  // TODO Auto-generated destructor stub
}

void LinearClassifier::init(int labelSize, int linearfeatSize) {

  _actionSize = labelSize;
  _linearfeatSize = linearfeatSize;

  _layer_linear.initial(_actionSize, _linearfeatSize);
  _eval.reset();

}



double LinearClassifier::process(const vector<Example>& examples, int iter) {
  _eval.reset();

  int example_num = examples.size();
  double cost = 0.0;
  for (int count = 0; count < example_num; count++) {
    const Example& example = examples[count];

    mat output, outputLoss;

    int offset;
    //forward propagation

    const Feature& feature = example.m_feature;

    //const vector<int>& linear_features = feature.linear_features;
    srand(iter*example_num + count);
    vector<int> linear_features;
    for(int idx = 0; idx < feature.linear_features.size(); idx++)
    {
      if(1.0*rand()/RAND_MAX >= _dropOut)
      {
        linear_features.push_back(feature.linear_features[idx]);
      }
    }

    _layer_linear.ComputeForwardScore(linear_features, output);


    // get delta for each output

    // Feed forward to softmax layer (no activation yet)


    const vector<int>& labels = example.m_labels;

    if(_lossFunc == 1)
    {
      NRVec<double> scores(_actionSize);
      double sum1 = -1e10, sum2 = -1e10;
      int optLabel1 = -1, optLabel2 = -1;
      for (int i = 0; i < _actionSize; ++i) {
        scores[i] = -1e10;
        if (labels[i] >= 0) {
          scores[i] = output(i, 0);
          if (labels[i] == 1)
          {
            if(optLabel1 == -1 || sum1 < scores[i])
            {
              sum1 = scores[i]; optLabel1 = i;
            }
          }
          if(optLabel2 == -1 || sum2 < scores[i])
          {
            sum2 = scores[i]; optLabel2 = i;
          }
        }
      }

      double loss = sum2 - sum1 + 1;

      cost += (sum2 - sum1) / example_num;

      _eval.overall_label_count++;
      if(optLabel1 == optLabel2)
      {
        _eval.correct_label_count++;
        continue; // need no update
      }

      outputLoss.zeros(_actionSize, 1);

      if(optLabel1 != optLabel2)
      {
        outputLoss(optLabel1, 0) = -loss / example_num;
        outputLoss(optLabel2, 0) = loss / example_num;
      }
    }
    else
    {
      int optLabel = -1;
      for (int i = 0; i < _actionSize; ++i) {
        //std::cout << output(i, 0) << std::endl;
        if(isnan(output(i, 0)))
        {
          std::cout << "debug please, error occurs" << std::endl;
        }
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
        scores[i] = 0.0;
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

      outputLoss.resize(_actionSize, 1);
      for (int i = 0; i < _actionSize; ++i) {
        outputLoss(i, 0) = 0.0;
        if (labels[i] >= 0) {
          outputLoss(i, 0) = (scores[i] / sum2 - labels[i]) / example_num;
        }
        if(isnan(outputLoss(i, 0)))
        {
          std::cout << "debug please, error occurs" << std::endl;
        }
      }
    }

    // loss backward propagation
    _layer_linear.ComputeBackwardLoss(linear_features, outputLoss);

  }

  return cost;
}

void LinearClassifier::predict(const Feature& feature, vector<double>& results) {

  mat output;

  int offset;
  //forward propagation

  const vector<int>& linear_features = feature.linear_features;

  _layer_linear.ComputeForwardScore(linear_features, output);


  results.resize(_actionSize);
  for (int i = 0; i < _actionSize; i++) {
    results[i] = output(i, 0);
  }

}

double LinearClassifier::computeScore(const Example& example) {
  assert(_lossFunc != 1);
  mat output;

  int offset;
  //forward propagation

  const Feature& feature = example.m_feature;
  const vector<int>& linear_features = feature.linear_features;

  _layer_linear.ComputeForwardScore(linear_features, output);

  // get delta for each output


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
  double maxScore = output(optLabel,0);
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

void LinearClassifier::updateParams(double regularizationWeight, double adaAlpha, double adaEps) {
  _layer_linear.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
}

// This is for sparse layers only, the cols are sparse
void LinearClassifier::checkgrad(const vector<Example>& examples, mat& Wd, const mat& gradWd, const string& mark, int iter, const hash_set<int>& sparseColIndexes, const mat& ft)
 {
  //Random randWdRowcheck = new Random(iter + "Row".hashCode() + hash));
  int charseed = mark.length();
  for(int i = 0; i < mark.length();  i++)
  {
    charseed = (int)(mark[i]) * 5 + charseed;
  }
  srand(iter+charseed);
  std::vector<int> idRows, idCols;
  idRows.clear();
  idCols.clear();
  if (sparseColIndexes.empty()) {
    for (int i = 0; i < Wd.n_cols; ++i)
      idCols.push_back(i);
  } else {
    hash_set<int>::iterator it;
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

  Wd(check_i, check_j) =  orginValue;
}


 void LinearClassifier::checkgrads(const vector<Example>& examples, int iter)
 {
   static mat ft;
   hash_set<int> fakedset;
   fakedset.clear();

   checkgrad(examples, _layer_linear._W, _layer_linear._gradW, "_layer_linear.W", iter, _layer_linear._indexers, _layer_linear._ftW);

 }
