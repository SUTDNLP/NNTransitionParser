#ifndef _PARSER_OPTIONS_
#define _PARSER_OPTIONS_

#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include <hash_set>
using namespace std;


class Options
{
public:
  /**
   * Refuse to train on words which have a corpus frequency less than
   * this number.
   */
  int wordCutOff;

  /**
   * Refuse to train on features which have a corpus frequency less than
   * this number.
   */
  int featCutOff;

  /**
   * Refuse to train on atoms which have a corpus frequency less than
   * this number.
   */
  int atomCutOff;

  /**
   * Model weights will be initialized to random values within the
   * range {@code [-initRange, initRange]}.
   */
  double initRange;

  /**
   * Maximum number of iterations for training
   */
  int maxIter;

  /**
   * Size of mini-batch for training. A random subset of training
   * examples of this size will be used to train the classifier on each
   * iteration.
   */
  int batchSize;

  /**
   * An epsilon value added to the denominator of the AdaGrad
   * expression for numerical stability
   */
  double adaEps;

  /**
   * Initial global learning rate for AdaGrad training
   */
  double adaAlpha;

  /**
   * Regularization parameter. All weight updates are scaled by this
   * single parameter.
   */
  double regParameter;

  /**
   * Dropout probability. For each training example we randomly choose
   * some amount of units to disable in the neural network classifier.
   * This probability controls the proportion of units "dropped out."
   */
  double dropProb;

  /**
   * Size of the neural network hidden layer.
   */
  int wrnnhiddenSize;
  int crnnhiddenSize;
  int wordEmbSize;
  int context;
  int atomcontext;
  int atomEmbSize;


  bool wordEmbFineTune;
  bool atomEmbFineTune;



  int verboseIter;
  bool saveIntermediate;
  bool train;
  int maxInstance;

  int numPreComputed;

  int lossFunc;


  vector<string> testFiles;

  string outBest;

  string clusterFile;

  int dropnum;

  Options()
  {
    wordCutOff = 0;
    featCutOff = 1;
    atomCutOff = 10;
    initRange = 0.01;
    maxIter = 1000;
    batchSize = 100;
    adaEps = 1e-6;
    adaAlpha = 0.01;
    regParameter = 1e-8;
    dropProb = 0.0;
    wrnnhiddenSize = 200;
    crnnhiddenSize = 50;
    wordEmbSize = 50;
    context = 2;
    atomcontext = 2;
    atomEmbSize = 50;
    wordEmbFineTune = true;
    atomEmbFineTune = true;
    verboseIter = 100;
    saveIntermediate = true;
    train = false;
    maxInstance = -1;

    numPreComputed = 100000;

    lossFunc = 0; //default 0: max likelihood, 1: max margin

    testFiles.clear();
    outBest = "";

    clusterFile = "";

    dropnum = -1;
  }

  virtual ~Options()
  {

  }



  void setOptions(const vector<string> &vecOption)
  {
    int i = 0;
    for (; i < vecOption.size(); ++i) {
      pair<string, string> pr;
      string2pair(vecOption[i], pr, '=');
      if (pr.first == "wordCutOff") wordCutOff = atoi(pr.second.c_str());
      if (pr.first == "featCutOff") featCutOff = atoi(pr.second.c_str());
      if (pr.first == "atomCutOff") atomCutOff = atoi(pr.second.c_str());
      if (pr.first == "initRange") initRange = atof(pr.second.c_str());
      if (pr.first == "maxIter") maxIter = atoi(pr.second.c_str());

      if (pr.first == "batchSize") batchSize = atoi(pr.second.c_str());
      if (pr.first == "adaEps") adaEps = atof(pr.second.c_str());
      if (pr.first == "adaAlpha") adaAlpha = atof(pr.second.c_str());
      if (pr.first == "regParameter") regParameter = atof(pr.second.c_str());
      if (pr.first == "dropProb") dropProb = atof(pr.second.c_str());


      if (pr.first == "wrnnhiddenSize") wrnnhiddenSize = atoi(pr.second.c_str());
      if (pr.first == "crnnhiddenSize") crnnhiddenSize = atoi(pr.second.c_str());
      if (pr.first == "context") context = atoi(pr.second.c_str());
      if (pr.first == "atomcontext") atomcontext = atoi(pr.second.c_str());
      if (pr.first == "atomEmbSize") atomEmbSize = atoi(pr.second.c_str());
      if (pr.first == "wordEmbSize") wordEmbSize = atoi(pr.second.c_str());

      if (pr.first == "wordEmbFineTune")
      {
        if(pr.second == "true") wordEmbFineTune = true;
        else wordEmbFineTune = false;
      }

      if (pr.first == "atomEmbFineTune")
      {
        if(pr.second == "true") atomEmbFineTune = true;
        else atomEmbFineTune = false;
      }

      if (pr.first == "verboseIter") verboseIter = atoi(pr.second.c_str());

      if (pr.first == "train")
      {
        if(pr.second == "true") train = true;
        else train = false;
      }
      if (pr.first == "saveIntermediate")
      {
        if(pr.second == "true") saveIntermediate = true;
        else saveIntermediate = false;
      }

      if (pr.first == "maxInstance") maxInstance = atoi(pr.second.c_str());

      if (pr.first == "numPreComputed") numPreComputed = atoi(pr.second.c_str());

      if (pr.first == "lossFunc") lossFunc = atoi(pr.second.c_str());

      if (pr.first == "testFile") testFiles.push_back(pr.second);

      if (pr.first == "outBest") outBest = pr.second;

      if (pr.first == "dropnum") dropnum = atoi(pr.second.c_str());

      if (pr.first == "clusterFile") clusterFile = pr.second;

    }
  }


  void showOptions() {
    std::cout << "wordCutOff = " << wordCutOff << std::endl;
    std::cout << "featCutOff = " << featCutOff << std::endl;
    std::cout << "atomCutOff = " << atomCutOff << std::endl;
    std::cout << "initRange = " << initRange << std::endl;
    std::cout << "maxIter = " << maxIter << std::endl;
    std::cout << "batchSize = " << batchSize << std::endl;
    std::cout << "adaEps = " << adaEps << std::endl;
    std::cout << "adaAlpha = " << adaAlpha << std::endl;
    std::cout << "regParameter = " << regParameter << std::endl;
    std::cout << "dropProb = " << dropProb << std::endl;
    std::cout << "wrnnhiddenSize = " << wrnnhiddenSize << std::endl;
    std::cout << "crnnhiddenSize = " << crnnhiddenSize << std::endl;
    std::cout << "wordEmbSize = " << wordEmbSize << std::endl;
    std::cout << "context = " << context << std::endl;
    std::cout << "atomcontext = " << atomcontext << std::endl;
    std::cout << "atomEmbSize = " << atomEmbSize << std::endl;

    std::cout << "wordEmbFineTune = " << wordEmbFineTune << std::endl;
    std::cout << "atomEmbFineTune = " << atomEmbFineTune << std::endl;

    std::cout << "verboseIter = " << verboseIter << std::endl;
    std::cout << "saveItermediate = " << saveIntermediate << std::endl;
    std::cout << "train = " << train << std::endl;
    std::cout << "maxInstance = " << maxInstance << std::endl;

    std::cout << "numPreComputed = " << numPreComputed << std::endl;

    std::cout << "lossFunc = " << lossFunc << std::endl;

    for(int idx = 0; idx < testFiles.size(); idx++)
    {
      std::cout << "testFile = " << testFiles[idx] << std::endl;
    }
    std::cout << "outBest = " << outBest << std::endl;
    std::cout << "dropnum = " << dropnum << std::endl;
    std::cout << "clusterFile = " << clusterFile << std::endl;
  }

  void load(const std::string& infile)
  {
    ifstream inf;
    inf.open(infile.c_str());
    vector<string> vecLine;
    while (1)
    {
        string strLine;
        if (!my_getline(inf, strLine)) {
              break;
        }
        if (strLine.empty()) continue;
        vecLine.push_back(strLine);
    }
    inf.close();
    setOptions(vecLine);
  }
};

#endif

