/*
 * Parser.h
 *
 *  Created on: Mar 25, 2015
 *      Author: mszhang
 */

#ifndef SRC_PARSER_H_
#define SRC_PARSER_H_

#define CLASSFIER 2


#include "Alphabet.h"
#include "Pipe.h"
#include "Utf.h"
#include "Options.h"
#include "State.h"

#if CLASSFIER==1
#include "NNClassifier.h"
#elif CLASSFIER==2
#include "LinearNNClassifier.h"
#elif CLASSFIER==3
#include "LinearSNNClassifier.h"
#elif CLASSFIER==4
#include "LinearSPNNClassifier.h"
#else
#include "LinearClassifier.h"
#endif

using namespace std;
using namespace arma;

class Parser {
public:
  std::string nullkey;
  std::string rootdepkey;
  std::string unknownkey;
  std::string paddingtag;
  std::string seperateKey;

public:
  Parser();
  virtual ~Parser();

public:
  Alphabet m_featAlphabet;
  Alphabet m_labelAlphabet;
  Alphabet m_atomAlphabet;  // include pos, dep embedding
  Alphabet m_wordAlphabet;  // include word embedding
  int m_maxActionNum;

  int m_atom_context;
  int m_word_context;

  hash_set<int> m_wordPreComputed;
  hash_set<int> m_atomPreComputed;

  NRMat<string> m_wordClusters;

public:
#if CLASSFIER==1
  NNClassifier m_classifier;
#elif CLASSFIER==2
  LinearNNClassifier m_classifier;
#elif CLASSFIER==3
  LinearSNNClassifier m_classifier;
#elif CLASSFIER==4
  LinearSPNNClassifier m_classifier;
#else
  LinearClassifier m_classifier;
#endif

  Options m_options;

  Pipe m_pipe;

public:
  void readWordEmbeddings(const string& inFile, mat& wordEmb);

  void readWordClusters(const string& inFile);

  int createAlphabet(const vector<CDependencyTree>& vecInsts);

  int createLinearFeatAlphabet(const vector<CDependencyTree>& vecInsts);

  int addTestWordAlpha(const vector<CDependencyTree>& vecInsts);

  void extractLinearFeatures(vector<string>& features, const CStateItem& item, const CDependencyTree& inst);

  void extractNeuralFeatures(vector<string>& wneural_features, vector<string>& aneural_features, const CStateItem& item, const CDependencyTree& inst);

  void extractFeature(Feature& feat, const CStateItem& item, const CDependencyTree& inst);

public:
  void train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile,
      const string& wordEmbFile);
  float predict(const CDependencyTree& inputTree, CDependencyTree& outputTree);
  void test(const string& testFile, const string& outputFile, const string& modelFile);

  // static training
  void getTrainExamples(const vector<CDependencyTree>& vecInsts, vector<Example>& vecExamples);


public:

  void proceedOneStepForDecode(const CDependencyTree& inputTree, CStateItem& state, int& outlab); //may be merged with train in the future

  void writeModelFile(const string& outputModelFile);
  void loadModelFile(const string& inputModelFile);

public:
  inline void getCandidateActions(const CStateItem &item, vector<int>& actions) {
    actions.clear();
    if (item.terminated()) {
      return;
    }

    if (item.stacksize() == 0 && item.size() < item.m_length) {
      actions.push_back(1);
    }
    else if (item.stacksize() == 1) {
      if (item.size() == item.m_length) {
        actions.push_back(2);
      } else if (item.size() < item.m_length) {
        actions.push_back(1);
      }
    }
    else {
      if (item.size() < item.m_length) { actions.push_back(1); }
      for (int l = 0; l < m_labelAlphabet.size(); l++) {
        actions.push_back(2 * l + 3);
        actions.push_back(2 * l + 4);
      }
    }
  }

  inline void evaluate(const CDependencyTree &gold, const CDependencyTree &pred, Metric &eval)
  {
    assert(gold.size() == pred.size());
    for(int idx = 0; idx < gold.size(); idx++)
    {
      if(!isPunc(gold[idx].tag))
      {
        if(gold[idx].head == pred[idx].head)
        {
          eval.correct_uas_count++;
          if(gold[idx].label.compare(pred[idx].label) == 0)
          {
            eval.correct_label_count++;
          }
        }
        eval.overall_label_count++;
      }
    }
  }

  // normalise link size and the direction
  inline string encodeLinkDistance(const int &head_index, const int &dep_index) {
     static int diff;
     diff = head_index - dep_index;
     assert(diff != 0);
     if (diff<0)
        diff=-diff;
     if (diff>10) diff = 6;
     else if (diff>5) diff = 5;

     stringstream ss;
     ss << diff;

     return ss.str();
  }

};

#endif /* SRC_PARSER_H_ */
