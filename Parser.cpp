/*
 * Parser.cpp
 *
 *  Created on: Mar 25, 2015
 *      Author: mszhang
 */

#include "Parser.h"

Parser::Parser() {
  // TODO Auto-generated constructor stub
  nullkey = "-null-";
  unknownkey = "-unknown-";
  paddingtag = "-padding-";
  seperateKey = "#";

  m_word_context = 18;
  m_atom_context = 30;
}

Parser::~Parser() {
  // TODO Auto-generated destructor stub
}

// all linear features are extracted from positive examples
int Parser::createAlphabet(const vector<CDependencyTree>& vecInsts) {
  cout << "Creating Alphabet..." << endl;

  int numInstance = vecInsts.size();

  hash_map<string, int> atom_stat;
  hash_map<string, int> word_stat;

  assert(numInstance > 0);

  for (int idx = 0; idx < vecInsts[0].size(); idx++) {
    if (vecInsts[0][idx].head == -1) {
      rootdepkey = vecInsts[0][idx].label;
      std::cout << "rootdepkey = " << rootdepkey << std::endl;
      break;
    }
  }

  m_labelAlphabet.clear();

  static Metric eval;
  static CStateItem state;
  static CDependencyTree output;
  static int answer;
  eval.reset();
  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const CDependencyTree &instance = vecInsts[numInstance];

    for (int idx = 0; idx < instance.size(); idx++) {
      word_stat[normalize_to_lowerwithdigit(instance[idx].word)]++;
      atom_stat["p=" + instance[idx].tag]++;
      atom_stat["d=" + instance[idx].label]++;
    }

    state.clear(instance.size());
    while (1) {
      answer = state.StandardMove(instance, m_labelAlphabet);
      state.Move(answer, m_labelAlphabet.size());

      if (answer == 2)
        break;
    }

    state.GenerateTree(instance, output, m_labelAlphabet, rootdepkey);

    evaluate(instance, output, eval);

    if (!eval.bIdentical()) {
      std::cout << "error state conversion!" << std::endl;
      exit(0);
    }

    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }

  m_maxActionNum = 2 * m_labelAlphabet.size() + 2;
  cout << numInstance << " " << endl;
  cout << "Label num: " << m_labelAlphabet.size() << endl;
  cout << "Action num: " << m_maxActionNum << endl;
  cout << "Total word num: " << word_stat.size() << endl;
  cout << "Total atom num: " << atom_stat.size() << endl;


  m_wordAlphabet.clear();
  m_wordAlphabet.from_string(nullkey);
  m_wordAlphabet.from_string(unknownkey);

  m_atomAlphabet.clear();
  m_atomAlphabet.from_string("p=" + nullkey);
  m_atomAlphabet.from_string("d=" + nullkey);

  hash_map<string, int>::iterator feat_iter;
  for (feat_iter = word_stat.begin(); feat_iter != word_stat.end(); feat_iter++) {
    if (!m_options.wordEmbFineTune || feat_iter->second > m_options.wordCutOff) {
      m_wordAlphabet.from_string(feat_iter->first);
    }
  }

  for (feat_iter = atom_stat.begin(); feat_iter != atom_stat.end(); feat_iter++) {
    m_atomAlphabet.from_string(feat_iter->first);
  }

  cout << "Remain words num: " << m_wordAlphabet.size() << endl;
  cout << "Remain atom num: " << m_atomAlphabet.size() << endl;

  m_labelAlphabet.set_fixed_flag(true);
  m_wordAlphabet.set_fixed_flag(true);
  m_atomAlphabet.set_fixed_flag(true);

  return 0;
}

// all linear features are extracted from positive examples
int Parser::createLinearFeatAlphabet(const vector<CDependencyTree>& vecInsts) {
  cout << "Creating Feature Alphabet..." << endl;

  int numInstance = vecInsts.size();

  hash_map<string, int> feature_stat;


  assert(numInstance > 0);

  static Metric eval;
  static CStateItem state;
  static CDependencyTree output;
  static int answer;
  static vector<string> features; // wneural_features, aneural_features;
  eval.reset();
  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const CDependencyTree &instance = vecInsts[numInstance];

    state.clear(instance.size());
    while (1) {
      extractLinearFeatures(features, state, instance);
      //extractNeuralFeatures(wneural_features, aneural_features, state, instance);
      for (int j = 0; j < features.size(); j++)
        feature_stat[features[j]]++;

      answer = state.StandardMove(instance, m_labelAlphabet);
      state.Move(answer, m_labelAlphabet.size());

      if (answer == 2)
        break;
    }

    state.GenerateTree(instance, output, m_labelAlphabet, rootdepkey);

    evaluate(instance, output, eval);

    if (!eval.bIdentical()) {
      std::cout << "error state conversion!" << std::endl;
      exit(0);
    }

    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }

  cout << numInstance << " " << endl;
  cout << "Total feature num: " << feature_stat.size() << endl;

  m_featAlphabet.clear();

  hash_map<string, int>::iterator feat_iter;
  for (feat_iter = feature_stat.begin(); feat_iter != feature_stat.end(); feat_iter++) {
    if (feat_iter->second > m_options.featCutOff) {
      m_featAlphabet.from_string(feat_iter->first);
    }
  }
  cout << "Remain feature num: " << m_featAlphabet.size() << endl;
  m_featAlphabet.set_fixed_flag(true);

  return 0;
}

int Parser::addTestWordAlpha(const vector<CDependencyTree>& vecInsts) {
  cout << "Add atom Alphabet..." << endl;

  m_wordAlphabet.set_fixed_flag(false);
  int numInstance;


  hash_map<string, int> word_stat;

  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const CDependencyTree &instance = vecInsts[numInstance];
    for (int idx = 0; idx < instance.size(); idx++) {
      word_stat[normalize_to_lowerwithdigit(instance[idx].word)]++;
    }

    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }

  hash_map<string, int>::iterator feat_iter;
  for (feat_iter = word_stat.begin(); feat_iter != word_stat.end(); feat_iter++) {
    if (!m_options.wordEmbFineTune || feat_iter->second > m_options.wordCutOff) {
      m_wordAlphabet.from_string(feat_iter->first);
    }
  }

  m_wordAlphabet.set_fixed_flag(true);

  return 0;
}

void Parser::extractLinearFeatures(vector<string>& features, const CStateItem& item, const CDependencyTree& inst) {
  features.clear();

  const int & S0id = item.stacktop();
  const int & S0l1did = (S0id == -1 ? -1 : item.leftdep(S0id));
  const int & S0r1did = (S0id == -1 ? -1 : item.rightdep(S0id));
  const int & S0l2did = (S0id == -1 ? -1 : item.left2dep(S0id));
  const int & S0r2did = (S0id == -1 ? -1 : item.right2dep(S0id));
  const int & S1id = item.stack2top();
  const int & S1l1did = (S1id == -1 ? -1 : item.leftdep(S1id));
  const int & S1r1did = (S1id == -1 ? -1 : item.rightdep(S1id));
  const int & S1l2did = (S1id == -1 ? -1 : item.left2dep(S1id));
  const int & S1r2did = (S1id == -1 ? -1 : item.right2dep(S1id));
  const int & N0id = item.size() >= inst.size() ? -1 : item.size();
  const int & N1id = item.size() + 1 >= inst.size() ? -1 : item.size() + 1;

  const string &s0_word = S0id == -1 ? nullkey : normalize_to_lowerwithdigit(inst[S0id].word);
  const string &s0l1d_word = S0l1did == -1 ? nullkey : normalize_to_lowerwithdigit(inst[S0l1did].word);
  const string &s0r1d_word = S0r1did == -1 ? nullkey : normalize_to_lowerwithdigit(inst[S0r1did].word);
  const string &s0l2d_word = S0l2did == -1 ? nullkey : normalize_to_lowerwithdigit(inst[S0l2did].word);
  const string &s0r2d_word = S0r2did == -1 ? nullkey : normalize_to_lowerwithdigit(inst[S0r2did].word);
  const string &s1_word = S1id == -1 ? nullkey : normalize_to_lowerwithdigit(inst[S1id].word);
  const string &s1l1d_word = S1l1did == -1 ? nullkey : normalize_to_lowerwithdigit(inst[S1l1did].word);
  const string &s1r1d_word = S1r1did == -1 ? nullkey : normalize_to_lowerwithdigit(inst[S1r1did].word);
  const string &s1l2d_word = S1l2did == -1 ? nullkey : normalize_to_lowerwithdigit(inst[S1l2did].word);
  const string &s1r2d_word = S1r2did == -1 ? nullkey : normalize_to_lowerwithdigit(inst[S1r2did].word);
  const string &n0_word = N0id == -1 ? nullkey : normalize_to_lowerwithdigit(inst[N0id].word);
  const string &n1_word = N1id == -1 ? nullkey : normalize_to_lowerwithdigit(inst[N1id].word);

  const string &s0_tag = S0id == -1 ? nullkey : inst[S0id].tag;
  const string &s0l1d_tag = S0l1did == -1 ? nullkey : inst[S0l1did].tag;
  const string &s0r1d_tag = S0r1did == -1 ? nullkey : inst[S0r1did].tag;
  const string &s0l2d_tag = S0l2did == -1 ? nullkey : inst[S0l2did].tag;
  const string &s0r2d_tag = S0r2did == -1 ? nullkey : inst[S0r2did].tag;
  const string &s1_tag = S1id == -1 ? nullkey : inst[S1id].tag;
  const string &s1l1d_tag = S1l1did == -1 ? nullkey : inst[S1l1did].tag;
  const string &s1r1d_tag = S1r1did == -1 ? nullkey : inst[S1r1did].tag;
  const string &s1l2d_tag = S1l2did == -1 ? nullkey : inst[S1l2did].tag;
  const string &s1r2d_tag = S1r2did == -1 ? nullkey : inst[S1r2did].tag;
  const string &n0_tag = N0id == -1 ? nullkey : inst[N0id].tag;
  const string &n1_tag = N1id == -1 ? nullkey : inst[N1id].tag;

  const int &S0l1d_labelId = S0l1did == -1 ? -1 : item.label(S0l1did);
  const int &S0r1d_labelId = S0r1did == -1 ? -1 : item.label(S0r1did);
  const int &S0l2d_labelId = S0l2did == -1 ? -1 : item.label(S0l2did);
  const int &S0r2d_labelId = S0r2did == -1 ? -1 : item.label(S0r2did);
  const int &S1l1d_labelId = S1l1did == -1 ? -1 : item.label(S1l1did);

  const int &S1r1d_labelId = S1r1did == -1 ? -1 : item.label(S1r1did);
  const int &S1l2d_labelId = S1l2did == -1 ? -1 : item.label(S1l2did);
  const int &S1r2d_labelId = S1r2did == -1 ? -1 : item.label(S1r2did);

  const string& s0l1d_label = S0l1d_labelId == -1 ? nullkey : (S0l1d_labelId == 0 ? rootdepkey : m_labelAlphabet.from_id(S0l1d_labelId - 1));
  const string& s0r1d_label = S0r1d_labelId == -1 ? nullkey : (S0r1d_labelId == 0 ? rootdepkey : m_labelAlphabet.from_id(S0r1d_labelId - 1));
  const string& s0l2d_label = S0l2d_labelId == -1 ? nullkey : (S0l2d_labelId == 0 ? rootdepkey : m_labelAlphabet.from_id(S0l2d_labelId - 1));
  const string& s0r2d_label = S0r2d_labelId == -1 ? nullkey : (S0r2d_labelId == 0 ? rootdepkey : m_labelAlphabet.from_id(S0r2d_labelId - 1));
  const string& s1l1d_label = S1l1d_labelId == -1 ? nullkey : (S1l1d_labelId == 0 ? rootdepkey : m_labelAlphabet.from_id(S1l1d_labelId - 1));
  const string& s1r1d_label = S1r1d_labelId == -1 ? nullkey : (S1r1d_labelId == 0 ? rootdepkey : m_labelAlphabet.from_id(S1r1d_labelId - 1));
  const string& s1l2d_label = S1l2d_labelId == -1 ? nullkey : (S1l2d_labelId == 0 ? rootdepkey : m_labelAlphabet.from_id(S1l2d_labelId - 1));
  const string& s1r2d_label = S1r2d_labelId == -1 ? nullkey : (S1r2d_labelId == 0 ? rootdepkey : m_labelAlphabet.from_id(S1r2d_labelId - 1));

  static string s0_n0_dist;
  if (S0id != -1 && S1id != -1)
    s0_n0_dist = encodeLinkDistance(S0id, S1id);
  else
    s0_n0_dist = "0";

  const int s0_rarity = S0id == -1 ? -1 : item.rightarity(S0id);
  const int s0_larity = S0id == -1 ? -1 : item.leftarity(S0id);
  const int s1_rarity = S1id == -1 ? -1 : item.rightarity(S1id);
  const int s1_larity = S1id == -1 ? -1 : item.leftarity(S1id);

  static string feat;

  int unknownWordId = m_wordAlphabet.from_string(unknownkey);
  int wordId;

  // single
  if (S0id != -1) {
    feat = "F1=" + s0_word;
    features.push_back(feat);
    feat = "F2=" + s0_tag;
    features.push_back(feat);
    feat = "F3=" + s0_word + seperateKey + s0_tag;
    features.push_back(feat);

    if(!m_options.clusterFile.empty())
    {
      int wordId = m_wordAlphabet.from_string(s0_word);
      if (wordId < 0)
      {
        wordId = unknownWordId;
      }
      for(int idx = 0; idx < m_wordClusters.ncols(); idx++)
      {
        stringstream cids;
        cids << idx;
        feat = "CF1=" + cids.str() + seperateKey + m_wordClusters[wordId][idx];
        features.push_back(feat);
        feat = "CF3=" + cids.str() + seperateKey  + m_wordClusters[wordId][idx] + seperateKey + s0_tag;
        features.push_back(feat);
      }
    }
  }

  if (S1id != -1) {
    feat = "F4=" + s1_word;
    features.push_back(feat);
    feat = "F5=" + s1_tag;
    features.push_back(feat);
    feat = "F6=" + s1_word + seperateKey + s1_tag;
    features.push_back(feat);

    if(!m_options.clusterFile.empty())
    {
      int wordId = m_wordAlphabet.from_string(s1_word);
      if (wordId < 0)
      {
        wordId = unknownWordId;
      }
      for(int idx = 0; idx < m_wordClusters.ncols(); idx++)
      {
        stringstream cids;
        cids << idx;
        feat = "CF4=" + cids.str() + seperateKey + m_wordClusters[wordId][idx];
        features.push_back(feat);
        feat = "CF6=" + cids.str() + seperateKey  + m_wordClusters[wordId][idx] + seperateKey + s1_tag;
        features.push_back(feat);
      }
    }
  }

  if (N0id != -1) {
    feat = "F77=" + n0_word;
    features.push_back(feat);
    feat = "F78=" + n0_tag;
    features.push_back(feat);
    feat = "F79=" + n0_word + seperateKey + n0_tag;
    features.push_back(feat);

    if(!m_options.clusterFile.empty())
    {
      int wordId = m_wordAlphabet.from_string(n0_word);
      if (wordId < 0)
      {
        wordId = unknownWordId;
      }
      for(int idx = 0; idx < m_wordClusters.ncols(); idx++)
      {
        stringstream cids;
        cids << idx;
        feat = "CF77=" + cids.str() + seperateKey + m_wordClusters[wordId][idx];
        features.push_back(feat);
        feat = "CF79=" + cids.str() + seperateKey  + m_wordClusters[wordId][idx] + seperateKey + n0_tag;
        features.push_back(feat);
      }
    }
  }

  if (N1id != -1) {
    feat = "F7=" + n1_word;
    features.push_back(feat);
    feat = "F8=" + n1_tag;
    features.push_back(feat);
    feat = "F9=" + n1_word + seperateKey + n1_tag;
    features.push_back(feat);

    if(!m_options.clusterFile.empty())
    {
      int wordId = m_wordAlphabet.from_string(n1_word);
      if (wordId < 0)
      {
        wordId = unknownWordId;
      }
      for(int idx = 0; idx < m_wordClusters.ncols(); idx++)
      {
        stringstream cids;
        cids << idx;
        feat = "CF7=" + cids.str() + seperateKey + m_wordClusters[wordId][idx];
        features.push_back(feat);
        feat = "CF9=" + cids.str() + seperateKey  + m_wordClusters[wordId][idx] + seperateKey + n1_tag;
        features.push_back(feat);
      }
    }
  }

  //S0, S1
  if (S0id != -1 && S0id != -1) {
    feat = "F10=" + s0_word + seperateKey + s1_word;
    features.push_back(feat);
    feat = "F11=" + s0_tag + seperateKey + s1_word;
    features.push_back(feat);
    feat = "F12=" + s0_word + seperateKey + s1_tag;
    features.push_back(feat);
    feat = "F13=" + s0_tag + seperateKey + s1_tag;
    features.push_back(feat);

    feat = "F14=" + s0_word + seperateKey + s0_tag + seperateKey + s1_word;
    features.push_back(feat);
    feat = "F15=" + s0_word + seperateKey + s0_tag + seperateKey + s1_tag;
    features.push_back(feat);
    feat = "F16=" + s0_word + seperateKey + s1_word + seperateKey + s1_tag;
    features.push_back(feat);
    feat = "F17=" + s0_tag + seperateKey + s1_word + seperateKey + s1_tag;
    features.push_back(feat);

    feat = "F18=" + s0_word + seperateKey + s0_tag + seperateKey + s1_word + seperateKey + s1_tag;
    features.push_back(feat);

    if(!m_options.clusterFile.empty())
    {
      int word1Id = m_wordAlphabet.from_string(n1_word);
      if (word1Id < 0)
      {
        word1Id = unknownWordId;
      }
      int word0Id = m_wordAlphabet.from_string(n0_word);
      if (word0Id < 0)
      {
        word0Id = unknownWordId;
      }
      for(int idx = 0; idx < m_wordClusters.ncols(); idx++)
      {
        stringstream cids;
        cids << idx;
	feat = "CF10=" + cids.str() + seperateKey  + m_wordClusters[word0Id][idx] + seperateKey + m_wordClusters[word1Id][idx];
	features.push_back(feat);
	feat = "CF11=" + cids.str() + seperateKey  + s0_tag + seperateKey + m_wordClusters[word1Id][idx];
	features.push_back(feat);
	feat = "CF12=" + cids.str() + seperateKey  + m_wordClusters[word0Id][idx] + seperateKey + s1_tag;
	features.push_back(feat);
      }
    }
  }

  if (S0l1did != -1) {
    feat = "F19=" + s0l1d_word;
    features.push_back(feat);
    feat = "F20=" + s0l1d_tag;
    features.push_back(feat);
    feat = "F21=" + s0l1d_word + seperateKey + s0l1d_tag;
    features.push_back(feat);
    feat = "F22=" + s0l1d_label;
    features.push_back(feat);

    //if(!m_options.clusterFile.empty())
    if(false)
    {
      int wordId = m_wordAlphabet.from_string(s0l1d_word);
      if (wordId < 0)
      {
        wordId = unknownWordId;
      }
      for(int idx = 0; idx < m_wordClusters.ncols(); idx++)
      {
        stringstream cids;
        cids << idx;
        feat = "CF19=" + cids.str() + seperateKey + m_wordClusters[wordId][idx];
        features.push_back(feat);
        feat = "CF21=" + cids.str() + seperateKey  + m_wordClusters[wordId][idx] + seperateKey + s0l1d_tag;
        features.push_back(feat);
      }
    }
  }

  if (S0r1did != -1) {
    feat = "F23=" + s0r1d_word;
    features.push_back(feat);
    feat = "F24=" + s0r1d_tag;
    features.push_back(feat);
    feat = "F25=" + s0r1d_word + seperateKey + s0r1d_tag;
    features.push_back(feat);
    feat = "F26=" + s0r1d_label;
    features.push_back(feat);

   // if(!m_options.clusterFile.empty())
    if(false)
    {
      int wordId = m_wordAlphabet.from_string(s0r1d_word);
      if (wordId < 0)
      {
        wordId = unknownWordId;
      }
      for(int idx = 0; idx < m_wordClusters.ncols(); idx++)
      {
        stringstream cids;
        cids << idx;
        feat = "CF23=" + cids.str() + seperateKey + m_wordClusters[wordId][idx];
        features.push_back(feat);
        feat = "CF25=" + cids.str() + seperateKey  + m_wordClusters[wordId][idx] + seperateKey + s0r1d_tag;
        features.push_back(feat);
      }
    }
  }

  if (S0l2did != -1) {
    feat = "F27=" + s0l2d_word;
    features.push_back(feat);
    feat = "F28=" + s0l2d_tag;
    features.push_back(feat);
    feat = "F29=" + s0l2d_word + seperateKey + s0l2d_tag;
    features.push_back(feat);
    feat = "F30=" + s0l2d_label;
    features.push_back(feat);

    //if(!m_options.clusterFile.empty())
    if(false)
    {
      int wordId = m_wordAlphabet.from_string(s0l2d_word);
      if (wordId < 0)
      {
        wordId = unknownWordId;
      }
      for(int idx = 0; idx < m_wordClusters.ncols(); idx++)
      {
        stringstream cids;
        cids << idx;
        feat = "CF27=" + cids.str() + seperateKey + m_wordClusters[wordId][idx];
        features.push_back(feat);
        feat = "CF29=" + cids.str() + seperateKey  + m_wordClusters[wordId][idx] + seperateKey + s0l2d_tag;
        features.push_back(feat);
      }
    }
  }

  if (S0r2did != -1) {
    feat = "F31=" + s0r2d_word;
    features.push_back(feat);
    feat = "F32=" + s0r2d_tag;
    features.push_back(feat);
    feat = "F33=" + s0r2d_word + seperateKey + s0r2d_tag;
    features.push_back(feat);
    feat = "F34=" + s0r2d_label;
    features.push_back(feat);

    //if(!m_options.clusterFile.empty())
    if(false)
    {
      int wordId = m_wordAlphabet.from_string(s0r2d_word);
      if (wordId < 0)
      {
        wordId = unknownWordId;
      }
      for(int idx = 0; idx < m_wordClusters.ncols(); idx++)
      {
        stringstream cids;
        cids << idx;
        feat = "CF31=" + cids.str() + seperateKey + m_wordClusters[wordId][idx];
        features.push_back(feat);
        feat = "CF33=" + cids.str() + seperateKey  + m_wordClusters[wordId][idx] + seperateKey + s0r2d_tag;
        features.push_back(feat);
      }
    }
  }

  if (S1l1did != -1) {
    feat = "F35=" + s1l1d_word;
    features.push_back(feat);
    feat = "F36=" + s1l1d_tag;
    features.push_back(feat);
    feat = "F37=" + s1l1d_word + seperateKey + s1l1d_tag;
    features.push_back(feat);
    feat = "F38=" + s1l1d_label;
    features.push_back(feat);

    //if(!m_options.clusterFile.empty())
    if(false)
    {
      int wordId = m_wordAlphabet.from_string(s1l1d_word);
      if (wordId < 0)
      {
        wordId = unknownWordId;
      }
      for(int idx = 0; idx < m_wordClusters.ncols(); idx++)
      {
        stringstream cids;
        cids << idx;
        feat = "CF35=" + cids.str() + seperateKey + m_wordClusters[wordId][idx];
        features.push_back(feat);
        feat = "CF37=" + cids.str() + seperateKey  + m_wordClusters[wordId][idx] + seperateKey + s1l1d_tag;
        features.push_back(feat);
      }
    }
  }

  if (S1r1did != -1) {
    feat = "F39=" + s1r1d_word;
    features.push_back(feat);
    feat = "F40=" + s1r1d_tag;
    features.push_back(feat);
    feat = "F41=" + s1r1d_word + seperateKey + s1r1d_tag;
    features.push_back(feat);
    feat = "F42=" + s1r1d_label;
    features.push_back(feat);

    //if(!m_options.clusterFile.empty())
    if(false)
    {
      int wordId = m_wordAlphabet.from_string(s1r1d_word);
      if (wordId < 0)
      {
        wordId = unknownWordId;
      }
      for(int idx = 0; idx < m_wordClusters.ncols(); idx++)
      {
        stringstream cids;
        cids << idx;
        feat = "CF39=" + cids.str() + seperateKey + m_wordClusters[wordId][idx];
        features.push_back(feat);
        feat = "CF41=" + cids.str() + seperateKey  + m_wordClusters[wordId][idx] + seperateKey + s1r1d_tag;
        features.push_back(feat);
      }
    }
  }

  if (S1l2did != -1) {
    feat = "F43=" + s1l2d_word;
    features.push_back(feat);
    feat = "F44=" + s1l2d_tag;
    features.push_back(feat);
    feat = "F45=" + s1l2d_word + seperateKey + s1l2d_tag;
    features.push_back(feat);
    feat = "F46=" + s1l2d_label;
    features.push_back(feat);

    //if(!m_options.clusterFile.empty())
    if(false)
    {
      int wordId = m_wordAlphabet.from_string(s1l2d_word);
      if (wordId < 0)
      {
        wordId = unknownWordId;
      }
      for(int idx = 0; idx < m_wordClusters.ncols(); idx++)
      {
        stringstream cids;
        cids << idx;
        feat = "CF43=" + cids.str() + seperateKey + m_wordClusters[wordId][idx];
        features.push_back(feat);
        feat = "CF45=" + cids.str() + seperateKey  + m_wordClusters[wordId][idx] + seperateKey + s1l2d_tag;
        features.push_back(feat);
      }
    }
  }

  if (S1r2did != -1) {
    feat = "F47=" + s1r2d_word;
    features.push_back(feat);
    feat = "F48=" + s1r2d_tag;
    features.push_back(feat);
    feat = "F49=" + s1r2d_word + seperateKey + s1r2d_tag;
    features.push_back(feat);
    feat = "F50=" + s1r2d_label;
    features.push_back(feat);

    //if(!m_options.clusterFile.empty())
    if(false)
    {
      int wordId = m_wordAlphabet.from_string(s1r2d_word);
      if (wordId < 0)
      {
        wordId = unknownWordId;
      }
      for(int idx = 0; idx < m_wordClusters.ncols(); idx++)
      {
        stringstream cids;
        cids << idx;
        feat = "CF47=" + cids.str() + seperateKey + m_wordClusters[wordId][idx];
        features.push_back(feat);
        feat = "CF49=" + cids.str() + seperateKey  + m_wordClusters[wordId][idx] + seperateKey + s1r2d_tag;
        features.push_back(feat);
      }
    }
  }

  if (-1 != S0l2did) {
    feat = "F51=" + s0_tag + seperateKey + s0l1d_tag + seperateKey + s0l2d_tag;
    features.push_back(feat);
  }

  if (-1 != S0r2did) {
    feat = "F52=" + s0_tag + seperateKey + s0r1d_tag + seperateKey + s0r2d_tag;
    features.push_back(feat);
  }

  if (-1 != S1l2did) {
    feat = "F53=" + s1_tag + seperateKey + s1l1d_tag + seperateKey + s1l2d_tag;
    features.push_back(feat);
  }

  if (-1 != S1r2did) {
    feat = "F54=" + s1_tag + seperateKey + s1r1d_tag + seperateKey + s1r2d_tag;
    features.push_back(feat);
  }

  if (-1 != S0id && -1 != S1id) {
    if (-1 != S0l1did) {
      feat = "F55=" + s0_tag + seperateKey + s1_tag + seperateKey + s0l1d_tag;
      features.push_back(feat);
    }
    if (-1 != S0r1did) {
      feat = "F56=" + s0_tag + seperateKey + s1_tag + seperateKey + s0r1d_tag;
      features.push_back(feat);
    }
    if (-1 != S1l1did) {
      feat = "F57=" + s0_tag + seperateKey + s1_tag + seperateKey + s1l1d_tag;
      features.push_back(feat);
    }
    if (-1 != S1r1did) {
      feat = "F58=" + s0_tag + seperateKey + s1_tag + seperateKey + s1r1d_tag;
      features.push_back(feat);
    }
    if (-1 != S0l2did) {
      feat = "F59=" + s0_tag + seperateKey + s1_tag + seperateKey + s0l2d_tag;
      features.push_back(feat);
    }
    if (-1 != S0r2did) {
      feat = "F60=" + s0_tag + seperateKey + s1_tag + seperateKey + s0r2d_tag;
      features.push_back(feat);
    }
    if (-1 != S1l2did) {
      feat = "F61=" + s0_tag + seperateKey + s1_tag + seperateKey + s1l2d_tag;
      features.push_back(feat);
    }
    if (-1 != S1r2did) {
      feat = "F62=" + s0_tag + seperateKey + s1_tag + seperateKey + s1r2d_tag;
      features.push_back(feat);
    }

    //distance
    feat = "F63=" + s0_word + seperateKey + s0_n0_dist;
    features.push_back(feat);
    feat = "F64=" + s0_tag + seperateKey + s0_n0_dist;
    features.push_back(feat);
    feat = "F65=" + n0_word + seperateKey + s0_n0_dist;
    features.push_back(feat);
    feat = "F66=" + n0_tag + seperateKey + s0_n0_dist;
    features.push_back(feat);

    feat = "F67=" + s0_word + seperateKey + n0_word + seperateKey + s0_n0_dist;
    features.push_back(feat);
    feat = "F68=" + s0_tag + seperateKey + n0_tag + seperateKey + s0_n0_dist;
    features.push_back(feat);
  }

  // s0 arity
  if (S0id != -1) {
    stringstream ssl;
    ssl << s0_larity;
    stringstream ssr;
    ssr << s0_rarity;
    feat = "F69=" + s0_word + seperateKey + ssl.str();
    features.push_back(feat);
    feat = "F70=" + s0_tag + seperateKey + ssl.str();
    features.push_back(feat);
    feat = "F71=" + s0_word + seperateKey + ssr.str();
    features.push_back(feat);
    feat = "F72=" + s0_tag + seperateKey + ssr.str();
    features.push_back(feat);
  }

  // s1 arity
  if (S1id != -1) {
    stringstream ssl;
    ssl << s1_larity;
    stringstream ssr;
    ssr << s1_rarity;
    feat = "F73=" + s1_word + seperateKey + ssl.str();
    features.push_back(feat);
    feat = "F74=" + s1_tag + seperateKey + ssl.str();
    features.push_back(feat);
    feat = "F75=" + s1_word + seperateKey + ssr.str();
    features.push_back(feat);
    feat = "F76=" + s1_tag + seperateKey + ssr.str();
    features.push_back(feat);
  }

  if(!m_options.clusterFile.empty())
  {

  }
}

void Parser::extractNeuralFeatures(vector<string>& wneural_features, vector<string>& aneural_features, const CStateItem& item, const CDependencyTree& inst) {
  wneural_features.clear();
  aneural_features.clear();

  const int & S0id = item.stacktop();
  const int & S0l1did = (S0id == -1 ? -1 : item.leftdep(S0id));
  const int & S0r1did = (S0id == -1 ? -1 : item.rightdep(S0id));
  const int & S0l2did = (S0id == -1 ? -1 : item.left2dep(S0id));
  const int & S0r2did = (S0id == -1 ? -1 : item.right2dep(S0id));
  const int & S1id = item.stack2top();
  const int & S1l1did = (S1id == -1 ? -1 : item.leftdep(S1id));
  const int & S1r1did = (S1id == -1 ? -1 : item.rightdep(S1id));
  const int & S1l2did = (S1id == -1 ? -1 : item.left2dep(S1id));
  const int & S1r2did = (S1id == -1 ? -1 : item.right2dep(S1id));
  const int & S2id = item.stack3top();
  const int & S2l1did = (S2id == -1 ? -1 : item.leftdep(S2id));
  const int & S2r1did = (S2id == -1 ? -1 : item.rightdep(S2id));
  const int & S2l2did = (S2id == -1 ? -1 : item.left2dep(S2id));
  const int & S2r2did = (S2id == -1 ? -1 : item.right2dep(S2id));
  const int & N0id = item.size() >= inst.size() ? -1 : item.size();
  const int & N1id = item.size() + 1 >= inst.size() ? -1 : item.size() + 1;
  const int & N2id = item.size() + 2 >= inst.size() ? -1 : item.size() + 2;

  const string &s0_word = S0id == -1 ? nullkey : normalize_to_lowerwithdigit(inst[S0id].word);
  const string &s0l1d_word = S0l1did == -1 ? nullkey : normalize_to_lowerwithdigit(inst[S0l1did].word);;
  const string &s0r1d_word = S0r1did == -1 ? nullkey : normalize_to_lowerwithdigit(inst[S0r1did].word);
  const string &s0l2d_word = S0l2did == -1 ? nullkey : normalize_to_lowerwithdigit(inst[S0l2did].word);
  const string &s0r2d_word = S0r2did == -1 ? nullkey : normalize_to_lowerwithdigit(inst[S0r2did].word);
  const string &s1_word = S1id == -1 ? nullkey : normalize_to_lowerwithdigit(inst[S1id].word);
  const string &s1l1d_word = S1l1did == -1 ? nullkey : normalize_to_lowerwithdigit(inst[S1l1did].word);
  const string &s1r1d_word = S1r1did == -1 ? nullkey : normalize_to_lowerwithdigit(inst[S1r1did].word);
  const string &s1l2d_word = S1l2did == -1 ? nullkey : normalize_to_lowerwithdigit(inst[S1l2did].word);
  const string &s1r2d_word = S1r2did == -1 ? nullkey : normalize_to_lowerwithdigit(inst[S1r2did].word);
  const string &s2_word = S2id == -1 ? nullkey : normalize_to_lowerwithdigit(inst[S2id].word);
  const string &s2l1d_word = S2l1did == -1 ? nullkey : normalize_to_lowerwithdigit(inst[S2l1did].word);
  const string &s2r1d_word = S2r1did == -1 ? nullkey : normalize_to_lowerwithdigit(inst[S2r1did].word);
  const string &s2l2d_word = S2l2did == -1 ? nullkey : normalize_to_lowerwithdigit(inst[S2l2did].word);
  const string &s2r2d_word = S2r2did == -1 ? nullkey : normalize_to_lowerwithdigit(inst[S2r2did].word);
  const string &n0_word = N0id == -1 ? nullkey : normalize_to_lowerwithdigit(inst[N0id].word);
  const string &n1_word = N1id == -1 ? nullkey : normalize_to_lowerwithdigit(inst[N1id].word);
  const string &n2_word = N2id == -1 ? nullkey : normalize_to_lowerwithdigit(inst[N2id].word);

  const string &s0_tag = S0id == -1 ? nullkey : inst[S0id].tag;
  const string &s0l1d_tag = S0l1did == -1 ? nullkey : inst[S0l1did].tag;
  ;
  const string &s0r1d_tag = S0r1did == -1 ? nullkey : inst[S0r1did].tag;
  const string &s0l2d_tag = S0l2did == -1 ? nullkey : inst[S0l2did].tag;
  const string &s0r2d_tag = S0r2did == -1 ? nullkey : inst[S0r2did].tag;
  const string &s1_tag = S1id == -1 ? nullkey : inst[S1id].tag;
  const string &s1l1d_tag = S1l1did == -1 ? nullkey : inst[S1l1did].tag;
  ;
  const string &s1r1d_tag = S1r1did == -1 ? nullkey : inst[S1r1did].tag;
  const string &s1l2d_tag = S1l2did == -1 ? nullkey : inst[S1l2did].tag;
  const string &s1r2d_tag = S1r2did == -1 ? nullkey : inst[S1r2did].tag;
  const string &s2_tag = S2id == -1 ? nullkey : inst[S2id].tag;
  const string &s2l1d_tag = S2l1did == -1 ? nullkey : inst[S2l1did].tag;
  ;
  const string &s2r1d_tag = S2r1did == -1 ? nullkey : inst[S2r1did].tag;
  const string &s2l2d_tag = S2l2did == -1 ? nullkey : inst[S2l2did].tag;
  const string &s2r2d_tag = S2r2did == -1 ? nullkey : inst[S2r2did].tag;
  const string &n0_tag = N0id == -1 ? nullkey : inst[N0id].tag;
  const string &n1_tag = N1id == -1 ? nullkey : inst[N1id].tag;
  const string &n2_tag = N2id == -1 ? nullkey : inst[N2id].tag;

  const int &S0l1d_labelId = S0l1did == -1 ? -1 : item.label(S0l1did);
  ;
  const int &S0r1d_labelId = S0r1did == -1 ? -1 : item.label(S0r1did);
  const int &S0l2d_labelId = S0l2did == -1 ? -1 : item.label(S0l2did);
  const int &S0r2d_labelId = S0r2did == -1 ? -1 : item.label(S0r2did);
  const int &S1l1d_labelId = S1l1did == -1 ? -1 : item.label(S1l1did);
  ;
  const int &S1r1d_labelId = S1r1did == -1 ? -1 : item.label(S1r1did);
  const int &S1l2d_labelId = S1l2did == -1 ? -1 : item.label(S1l2did);
  const int &S1r2d_labelId = S1r2did == -1 ? -1 : item.label(S1r2did);
  const int &S2l1d_labelId = S2l1did == -1 ? -1 : item.label(S2l1did);
  ;
  const int &S2r1d_labelId = S2r1did == -1 ? -1 : item.label(S2r1did);
  const int &S2l2d_labelId = S2l2did == -1 ? -1 : item.label(S2l2did);
  const int &S2r2d_labelId = S2r2did == -1 ? -1 : item.label(S2r2did);

  const string& s0l1d_label = S0l1d_labelId == -1 ? nullkey : (S0l1d_labelId == 0 ? rootdepkey : m_labelAlphabet.from_id(S0l1d_labelId - 1));
  const string& s0r1d_label = S0r1d_labelId == -1 ? nullkey : (S0r1d_labelId == 0 ? rootdepkey : m_labelAlphabet.from_id(S0r1d_labelId - 1));
  const string& s0l2d_label = S0l2d_labelId == -1 ? nullkey : (S0l2d_labelId == 0 ? rootdepkey : m_labelAlphabet.from_id(S0l2d_labelId - 1));
  const string& s0r2d_label = S0r2d_labelId == -1 ? nullkey : (S0r2d_labelId == 0 ? rootdepkey : m_labelAlphabet.from_id(S0r2d_labelId - 1));
  const string& s1l1d_label = S1l1d_labelId == -1 ? nullkey : (S1l1d_labelId == 0 ? rootdepkey : m_labelAlphabet.from_id(S1l1d_labelId - 1));
  const string& s1r1d_label = S1r1d_labelId == -1 ? nullkey : (S1r1d_labelId == 0 ? rootdepkey : m_labelAlphabet.from_id(S1r1d_labelId - 1));
  const string& s1l2d_label = S1l2d_labelId == -1 ? nullkey : (S1l2d_labelId == 0 ? rootdepkey : m_labelAlphabet.from_id(S1l2d_labelId - 1));
  const string& s1r2d_label = S1r2d_labelId == -1 ? nullkey : (S1r2d_labelId == 0 ? rootdepkey : m_labelAlphabet.from_id(S1r2d_labelId - 1));
  const string& s2l1d_label = S2l1d_labelId == -1 ? nullkey : (S2l1d_labelId == 0 ? rootdepkey : m_labelAlphabet.from_id(S2l1d_labelId - 1));
  const string& s2r1d_label = S2r1d_labelId == -1 ? nullkey : (S2r1d_labelId == 0 ? rootdepkey : m_labelAlphabet.from_id(S2r1d_labelId - 1));
  const string& s2l2d_label = S2l2d_labelId == -1 ? nullkey : (S2l2d_labelId == 0 ? rootdepkey : m_labelAlphabet.from_id(S2l2d_labelId - 1));
  const string& s2r2d_label = S2r2d_labelId == -1 ? nullkey : (S2r2d_labelId == 0 ? rootdepkey : m_labelAlphabet.from_id(S2r2d_labelId - 1));

  wneural_features.push_back(s0_word);
  wneural_features.push_back(s0l1d_word);
  wneural_features.push_back(s0r1d_word);
  wneural_features.push_back(s0l2d_word);
  wneural_features.push_back(s0r2d_word);
  wneural_features.push_back(s1_word);
  wneural_features.push_back(s1l1d_word);
  wneural_features.push_back(s1r1d_word);
  wneural_features.push_back(s1l2d_word);
  wneural_features.push_back(s1r2d_word);
  wneural_features.push_back(s2_word);
  wneural_features.push_back(s2l1d_word);
  wneural_features.push_back(s2r1d_word);
  wneural_features.push_back(s2l2d_word);
  wneural_features.push_back(s2r2d_word);
  wneural_features.push_back(n0_word);
  wneural_features.push_back(n1_word);
  wneural_features.push_back(n2_word);

  aneural_features.push_back("p=" + s0_tag);
  aneural_features.push_back("p=" + s0l1d_tag);
  aneural_features.push_back("p=" + s0r1d_tag);
  aneural_features.push_back("p=" + s0l2d_tag);
  aneural_features.push_back("p=" + s0r2d_tag);
  aneural_features.push_back("p=" + s1_tag);
  aneural_features.push_back("p=" + s1l1d_tag);
  aneural_features.push_back("p=" + s1r1d_tag);
  aneural_features.push_back("p=" + s1l2d_tag);
  aneural_features.push_back("p=" + s1r2d_tag);
  aneural_features.push_back("p=" + s2_tag);
  aneural_features.push_back("p=" + s2l1d_tag);
  aneural_features.push_back("p=" + s2r1d_tag);
  aneural_features.push_back("p=" + s2l2d_tag);
  aneural_features.push_back("p=" + s2r2d_tag);
  aneural_features.push_back("p=" + n0_tag);
  aneural_features.push_back("p=" + n1_tag);
  aneural_features.push_back("p=" + n2_tag);

  aneural_features.push_back("d=" + s0l1d_label);
  aneural_features.push_back("d=" + s0r1d_label);
  aneural_features.push_back("d=" + s0l2d_label);
  aneural_features.push_back("d=" + s0r2d_label);
  aneural_features.push_back("d=" + s1l1d_label);
  aneural_features.push_back("d=" + s1r1d_label);
  aneural_features.push_back("d=" + s1l2d_label);
  aneural_features.push_back("d=" + s1r2d_label);
  aneural_features.push_back("d=" + s2l1d_label);
  aneural_features.push_back("d=" + s2r1d_label);
  aneural_features.push_back("d=" + s2l2d_label);
  aneural_features.push_back("d=" + s2r2d_label);

}

void Parser::extractFeature(Feature& feat, const CStateItem& item, const CDependencyTree& inst) {
  feat.clear();

  static vector<string> features, wneural_features, aneural_features;

  extractLinearFeatures(features, item, inst);
  extractNeuralFeatures(wneural_features, aneural_features, item, inst);

  for (int idx = 0; idx < features.size(); idx++) {
    int featId = m_featAlphabet.from_string(features[idx]);
    if (featId >= 0)
      feat.linear_features.push_back(featId);
  }

  assert(wneural_features.size() == m_word_context);
  int unknownWordId = m_wordAlphabet.from_string(unknownkey);
  for (int idx = 0; idx < wneural_features.size(); idx++) {
    int wordId = m_wordAlphabet.from_string(wneural_features[idx]);
    if (wordId >= 0)
    {
      feat.wneural_features.push_back(wordId);
    }
    else
    {
      //std::cout << wneural_features[idx] << std::endl;
      feat.wneural_features.push_back(unknownWordId);
    }
  }

  //int nullAtomId = m_wordAlphabet.from_string(unknownkey);
  assert(aneural_features.size() == m_atom_context);
  for (int idx = 0; idx < aneural_features.size(); idx++) {
    int atomId = m_atomAlphabet.from_string(aneural_features[idx]);
    if (atomId >= 0)
      feat.aneural_features.push_back(atomId);
    else {
      //std::cout << "unknown atom features: " << aneural_features[idx] << std::endl;
      feat.aneural_features.push_back(0);
    }
  }

}

void Parser::getTrainExamples(const vector<CDependencyTree>& vecInsts, vector<Example>& vecExamples) {
  vecExamples.clear();
  static Example exam;
  static CStateItem state;
  static int answer;
  static vector<int> candidates;
  static CDependencyTree output;
  static Metric eval;
  hash_map<int, int> wordfeatFreq, atomfeatFreq;
  eval.reset();
  for (int idy = 0; idy < vecInsts.size(); idy++) {
    const CDependencyTree &inst = vecInsts[idy];
    state.clear(inst.size());
    exam.clear();
    while (1) {
      extractFeature(exam.m_feature, state, inst);
      answer = state.StandardMove(inst, m_labelAlphabet);
      exam.m_labels.resize(m_maxActionNum, -1);
      getCandidateActions(state, candidates);
      for (int c = 0; c < candidates.size(); c++)
        exam.m_labels[candidates[c] - 1] = 0;
      exam.m_labels[answer - 1] = 1;
      vecExamples.push_back(exam);

      // hash_set precomputed
      const vector<int>& wneural_features = exam.m_feature.wneural_features;
      for (int idk = 0; idk < wneural_features.size(); idk++) {
        int curFeatId = wneural_features[idk] * wneural_features.size() + idk;
        wordfeatFreq[curFeatId]++;
      }
      const vector<int>& aneural_features = exam.m_feature.aneural_features;
      for (int idk = 0; idk < aneural_features.size(); idk++) {
        int curFeatId = aneural_features[idk] * aneural_features.size() + idk;
        atomfeatFreq[curFeatId]++;
      }

      state.Move(answer, m_labelAlphabet.size());
      if (answer == 2)
        break;
      exam.clear();
    }

    state.GenerateTree(inst, output, m_labelAlphabet, rootdepkey);

    evaluate(inst, output, eval);

    if (!eval.bIdentical()) {
      std::cout << "error state conversion!" << std::endl;
      exit(0);
    }

    if ((idy + 1) % m_options.verboseIter == 0 || (idy + 1) == vecInsts.size()) {
      cout << idy + 1 << " ";
      if ((idy + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
  }
  std::cout << "Total generated example number: " << vecExamples.size() << std::endl;

  vector<pair<int, int> > sortedWordFeat, sortedAtomFeat;

  sortMapbyValue(wordfeatFreq, sortedWordFeat);

  m_wordPreComputed.clear();
  std::cout << sortedWordFeat[0].second << " " << sortedWordFeat[sortedWordFeat.size() - 1].second << std::endl;
  for (int idx = 0; idx < m_options.numPreComputed && idx < sortedWordFeat.size(); idx++) {
    m_wordPreComputed.insert(sortedWordFeat[idx].first);
  }

  sortMapbyValue(atomfeatFreq, sortedAtomFeat);
  m_atomPreComputed.clear();
  std::cout << sortedAtomFeat[0].second << " " << sortedAtomFeat[sortedAtomFeat.size() - 1].second << std::endl;
  for (int idx = 0; idx < sortedAtomFeat.size(); idx++) {
    m_atomPreComputed.insert(sortedAtomFeat[idx].first);
  }

}

void Parser::train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile,
    const string& wordEmbFile) {
  if (optionFile != "")
    m_options.load(optionFile);
  m_classifier.setLossFunc(m_options.lossFunc);
  m_options.showOptions();
  vector<CDependencyTree> trainInsts, devInsts, testInsts;
  m_pipe.readInstances(trainFile, trainInsts, m_options.maxInstance);
  if (devFile != "")
    m_pipe.readInstances(devFile, devInsts, m_options.maxInstance);
  if (testFile != "")
    m_pipe.readInstances(testFile, testInsts, m_options.maxInstance);

  vector<vector<CDependencyTree> > otherInsts(m_options.testFiles.size());
  for(int idx = 0; idx < m_options.testFiles.size();idx++)
  {
    m_pipe.readInstances(m_options.testFiles[idx], otherInsts[idx], m_options.maxInstance);
  }


  createAlphabet(trainInsts);

  if (!m_options.wordEmbFineTune) {
    addTestWordAlpha(devInsts);
    addTestWordAlpha(testInsts);
    for(int idx = 0; idx < otherInsts.size(); idx++)
    {
      addTestWordAlpha(otherInsts[idx]);
    }
    cout << "Remain word feature num: " << m_wordAlphabet.size() << endl;
  }

  if(!m_options.clusterFile.empty())
  {
    readWordClusters(m_options.clusterFile);
  }
  else
  {
    m_wordClusters.resize(1, 1);
  }

#if CLASSFIER != 1
  createLinearFeatAlphabet(trainInsts);
#endif

  arma_rng::set_seed(0);

  vector<Example> trainExamples;
  getTrainExamples(trainInsts, trainExamples);

#if CLASSFIER==1
  m_classifier.setPreComputed(m_wordPreComputed, m_atomPreComputed);
  if (wordEmbFile.length() > 0) {
    mat wordEmb;
    readWordEmbeddings(wordEmbFile, wordEmb);
    m_classifier.init(wordEmb, m_word_context, m_options.atomEmbSize, m_atomAlphabet.size(), m_atom_context, m_options.wrnnhiddenSize, m_maxActionNum);
  } else {
    m_classifier.init(m_options.wordEmbSize, m_wordAlphabet.size(), m_word_context, m_options.atomEmbSize, m_atomAlphabet.size(), m_atom_context,
        m_options.wrnnhiddenSize, m_maxActionNum);
  }
  m_classifier.setWordEmbFinetune(m_options.wordEmbFineTune);
#elif CLASSFIER==2
  m_classifier.setPreComputed(m_wordPreComputed, m_atomPreComputed);
  if (wordEmbFile.length() > 0) {
    mat wordEmb;
    readWordEmbeddings(wordEmbFile, wordEmb);
    m_classifier.init(wordEmb, m_word_context, m_options.atomEmbSize, m_atomAlphabet.size(), m_atom_context, m_options.wrnnhiddenSize, m_maxActionNum, m_featAlphabet.size());
  } else {
    m_classifier.init(m_options.wordEmbSize, m_wordAlphabet.size(), m_word_context, m_options.atomEmbSize, m_atomAlphabet.size(), m_atom_context, m_options.wrnnhiddenSize, m_maxActionNum, m_featAlphabet.size());
  }
  m_classifier.setWordEmbFinetune(m_options.wordEmbFineTune);
#elif CLASSFIER==3
  m_classifier.setPreComputed(m_wordPreComputed, m_atomPreComputed);
  if (wordEmbFile.length() > 0) {
    mat wordEmb;
    readWordEmbeddings(wordEmbFile, wordEmb);
    m_classifier.init(wordEmb, m_word_context, m_options.atomEmbSize, m_atomAlphabet.size(), m_atom_context, m_maxActionNum, m_featAlphabet.size());
  } else {
    m_classifier.init(m_options.wordEmbSize, m_wordAlphabet.size(), m_word_context, m_options.atomEmbSize, m_atomAlphabet.size(), m_atom_context, m_maxActionNum, m_featAlphabet.size());
  }
  m_classifier.setWordEmbFinetune(m_options.wordEmbFineTune);
#elif CLASSFIER==4
  m_classifier.setPreComputed(m_wordPreComputed, m_atomPreComputed);
  if (wordEmbFile.length() > 0) {
    mat wordEmb;
    readWordEmbeddings(wordEmbFile, wordEmb);
    m_classifier.init(wordEmb, m_word_context, m_maxActionNum, m_featAlphabet.size());
  } else {
    m_classifier.init(m_options.wordEmbSize, m_wordAlphabet.size(), m_word_context, m_maxActionNum, m_featAlphabet.size());
  }
  m_classifier.setWordEmbFinetune(m_options.wordEmbFineTune);
#else
  m_classifier.init(m_maxActionNum, m_featAlphabet.size());
#endif
  m_classifier.setDropValue(m_options.dropProb);


  double bestLAS = 0;

  int inputSize = trainExamples.size();
  //int batchBlock = inputSize / m_options.batchSize;
  //if (inputSize % m_options.batchSize != 0)
  //batchBlock++;

  std::vector<int> indexes;
  for (int i = 0; i < inputSize; ++i)
    indexes.push_back(i);

  static Metric eval, metric_dev, metric_test;
  static clock_t start_time, middle_time, end_time;
  vector<Example> subExamples;
  int maxIter = m_options.maxIter * (inputSize / m_options.batchSize + 1);
  std::cout << "maxIter = " << maxIter << std::endl;
  int devNum = devInsts.size(), testNum = testInsts.size();
  float gradcompute_time = 0.0, update_time = 0.0, decode_time = 0.0, feature_time = 0.0, other_time = 0.0;

  static vector<CDependencyTree> decodeInstResults;
  static CDependencyTree curDecodeInst;
  static bool bCurIterBetter;
  for (int iter = 0; iter < maxIter; ++iter) {
    start_time = clock();
    std::cout << "##### Iteration " << iter << std::endl;
    srand(iter);
    random_shuffle(indexes.begin(), indexes.end());
    std::cout << "random: " << indexes[0] << ", " << indexes[indexes.size()-1] << std::endl;
    eval.reset();
    //for (int updateIter = 0; updateIter < batchBlock; updateIter++) {
    subExamples.clear();
    //int start_pos = updateIter * m_options.batchSize;
    //int end_pos = (updateIter + 1) * m_options.batchSize;
    //if (end_pos > inputSize)
    //end_pos = inputSize;

    for (int idy = 0; idy < m_options.batchSize; idy++) {
      subExamples.push_back(trainExamples[indexes[idy]]);
    }
    end_time = clock();
    other_time += float( end_time - start_time ) /  CLOCKS_PER_SEC;
    //int curUpdateIter = iter * batchBlock + updateIter;
    start_time = clock();
    double cost = m_classifier.process(subExamples, iter);
    end_time = clock();
    gradcompute_time += float( end_time - start_time ) /  CLOCKS_PER_SEC;

    eval.overall_label_count += m_classifier._eval.overall_label_count;
    eval.correct_label_count += m_classifier._eval.correct_label_count;

    std::cout << "current: " << iter + 1 << ", Cost = " << cost << ", Correct(%) = " << eval.getAccuracy() << std::endl;

    //m_classifier.checkgrads(subExamples, iter+1);

    //if (m_options.batchSize > 1000 || (iter + 1) % m_options.verboseIter == 0)
    //{
    //m_classifier.checkgrads(subExamples, curUpdateIter+1);
    //std::cout << "current: " << iter + 1 << ", total block: " << batchBlock << std::endl;
    //std::cout << "Cost = " << cost << ", Correct(%) = " << eval.getAccuracy() << std::endl;
    //}

    start_time = clock();
    m_classifier.updateParams(m_options.regParameter, m_options.adaAlpha, m_options.adaEps);
    end_time = clock();
    update_time += float( end_time - start_time ) /  CLOCKS_PER_SEC;
    //}

    if (devNum > 0 && (iter + 1) % m_options.verboseIter == 0) {
#if CLASSFIER > 0
      m_classifier.preCompute();
#endif
      bCurIterBetter = false;
      if(!m_options.outBest.empty())decodeInstResults.clear();
      metric_dev.reset();
      start_time = clock();
      for (int idx = 0; idx < devInsts.size(); idx++) {
        feature_time += predict(devInsts[idx], curDecodeInst);
        evaluate(devInsts[idx], curDecodeInst, metric_dev);
        if(!m_options.outBest.empty())
        {
          decodeInstResults.push_back(curDecodeInst);
        }
      }
      std::cout << "dev:" << std::endl;
      metric_dev.print();

      if(!m_options.outBest.empty() && metric_dev.getAccuracy() > bestLAS)
      {
        m_pipe.outputAllInstances(devFile + m_options.outBest, decodeInstResults);
        bCurIterBetter = true;
      }

      if (testNum > 0) {
        if(!m_options.outBest.empty())decodeInstResults.clear();
        metric_test.reset();
        for (int idx = 0; idx < testInsts.size(); idx++) {
          feature_time += predict(testInsts[idx], curDecodeInst);
          evaluate(testInsts[idx], curDecodeInst, metric_test);
          if(bCurIterBetter && !m_options.outBest.empty())
          {
            decodeInstResults.push_back(curDecodeInst);
          }
        }
        std::cout << "test:" << std::endl;
        metric_test.print();

        if(!m_options.outBest.empty() && bCurIterBetter)
        {
          m_pipe.outputAllInstances(testFile + m_options.outBest, decodeInstResults);
        }
      }

      for(int idx = 0; idx < otherInsts.size(); idx++)
      {
        std::cout << "processing " << m_options.testFiles[idx] << std::endl;
        if(!m_options.outBest.empty())decodeInstResults.clear();
        metric_test.reset();
        for(int idy = 0; idy < otherInsts[idx].size(); idy++)
        {
          feature_time += predict(otherInsts[idx][idy], curDecodeInst);
          evaluate(otherInsts[idx][idy], curDecodeInst, metric_test);
          if(bCurIterBetter && !m_options.outBest.empty())
          {
            decodeInstResults.push_back(curDecodeInst);
          }
        }
        std::cout << "test:" << std::endl;
        metric_test.print();

        if(!m_options.outBest.empty() && bCurIterBetter)
        {
          m_pipe.outputAllInstances(m_options.testFiles[idx] + m_options.outBest, decodeInstResults);
        }
      }

      end_time = clock();
      decode_time += float( end_time - start_time ) /  CLOCKS_PER_SEC;

      if (m_options.saveIntermediate && metric_dev.getAccuracy() > bestLAS) {
        std::cout << "Exceeds best previous DIS of " << bestLAS << ". Saving model file.." << std::endl;
        bestLAS = metric_dev.getAccuracy();
        writeModelFile(modelFile);
      }
    }
    // Clear gradients
    std::cout << "gradcompute_time = " << gradcompute_time << ", update_time = " << update_time << ", decode_time = " << decode_time << ", feature_time = " << feature_time << ", other_time = " << other_time << std::endl;

    if( ((iter + 1) * m_options.batchSize) % 100000 == 0)
    {
      gradcompute_time = 0.0; update_time = 0.0; feature_time = 0.0; decode_time = 0.0;
    }
  }

  if (devNum > 0) {
#if CLASSFIER > 0
    m_classifier.preCompute();
#endif
    bCurIterBetter = false;
    if(!m_options.outBest.empty())decodeInstResults.clear();
    metric_dev.reset();
    start_time = clock();
    for (int idx = 0; idx < devInsts.size(); idx++) {
      predict(devInsts[idx], curDecodeInst);
      evaluate(devInsts[idx], curDecodeInst, metric_dev);
      if(!m_options.outBest.empty())
      {
        decodeInstResults.push_back(curDecodeInst);
      }
    }
    std::cout << "dev:" << std::endl;
    metric_dev.print();

    if(!m_options.outBest.empty() && metric_dev.getAccuracy() > bestLAS)
    {
      m_pipe.outputAllInstances(devFile + m_options.outBest, decodeInstResults);
      bCurIterBetter = true;
    }

    if (testNum > 0) {
      if(!m_options.outBest.empty())decodeInstResults.clear();
      metric_test.reset();
      for (int idx = 0; idx < testInsts.size(); idx++) {
        predict(testInsts[idx], curDecodeInst);
        evaluate(testInsts[idx], curDecodeInst, metric_test);
        if(bCurIterBetter && !m_options.outBest.empty())
        {
          decodeInstResults.push_back(curDecodeInst);
        }
      }
      std::cout << "test:" << std::endl;
      metric_test.print();

      if(!m_options.outBest.empty() && bCurIterBetter)
      {
        m_pipe.outputAllInstances(testFile + m_options.outBest, decodeInstResults);
      }
    }

    for(int idx = 0; idx < otherInsts.size(); idx++)
    {
      std::cout << "processing " << m_options.testFiles[idx] << std::endl;
      if(!m_options.outBest.empty())decodeInstResults.clear();
      metric_test.reset();
      for(int idy = 0; idy < otherInsts[idx].size(); idy++)
      {
        predict(otherInsts[idx][idy], curDecodeInst);
        evaluate(otherInsts[idx][idy], curDecodeInst, metric_test);
        if(bCurIterBetter && !m_options.outBest.empty())
        {
          decodeInstResults.push_back(curDecodeInst);
        }
      }
      std::cout << "test:" << std::endl;
      metric_test.print();

      if(!m_options.outBest.empty() && bCurIterBetter)
      {
        m_pipe.outputAllInstances(m_options.testFiles[idx] + m_options.outBest, decodeInstResults);
      }
    }

    if (m_options.saveIntermediate && metric_dev.getAccuracy() > bestLAS) {
      std::cout << "Exceeds best previous DIS of " << bestLAS << ". Saving model file.." << std::endl;
      bestLAS = metric_dev.getAccuracy();
      writeModelFile(modelFile);
    }
  } else {
    writeModelFile(modelFile);
  }
}

float Parser::predict(const CDependencyTree& input, CDependencyTree& output) {
  static CStateItem state;
  static vector<int> candidates;
  static vector<double> results;
  state.clear(input.size());
  static int predict;
  static Feature feat;
  predict = 0;
  //proceedOneStepForDecode();
  float feat_time = 0.0;
  static clock_t start_time;
  while(predict != 2) {
    start_time = clock();
    extractFeature(feat, state, input);
    feat_time += float(clock() - start_time)/CLOCKS_PER_SEC;
    m_classifier.predict(feat, results);

    getCandidateActions(state, candidates);
    predict = candidates[0];
    double maxscore = results[predict - 1];
    for (int idx = 1; idx < candidates.size(); idx++) {
      int curact = candidates[idx];
      if (results[curact - 1] > maxscore) {
        maxscore = results[curact - 1];
        predict = curact;
      }
    }
    state.Move(predict, m_labelAlphabet.size());
  }

  state.GenerateTree(input, output, m_labelAlphabet, rootdepkey);
  return feat_time;
}

void Parser::test(const string& testFile, const string& outputFile, const string& modelFile) {
  loadModelFile(modelFile);
  vector<CDependencyTree> testInsts;
  m_pipe.readInstances(testFile, testInsts, m_options.maxInstance);

  vector<CDependencyTree> testInstResults(testInsts.size());
  Metric metric_test;
  metric_test.reset();
  for (int idx = 0; idx < testInsts.size(); idx++) {
    vector<string> result_labels;
    predict(testInsts[idx], testInstResults[idx]);
    evaluate(testInsts[idx], testInstResults[idx], metric_test);
  }
  std::cout << "test:" << std::endl;
  metric_test.print();

  std::ofstream os(outputFile.c_str());

  for (int idx = 0; idx < testInsts.size(); idx++) {
    os << testInstResults[idx];
  }
  os.close();
}

void Parser::readWordEmbeddings(const string& inFile, mat& wordEmb) {
  static ifstream inf;
  if (inf.is_open()) {
    inf.close();
    inf.clear();
  }
  inf.open(inFile.c_str());

  static string strLine, curWord;
  static int wordId;

  //find the first line, decide the wordDim;
  while (1) {
    if (!my_getline(inf, strLine)) {
      break;
    }
    if (!strLine.empty())
      break;
  }

  int unknownId = m_wordAlphabet.from_string(unknownkey);

  static vector<string> vecInfo;
  split_bychar(strLine, vecInfo, ' ');
  int wordDim = vecInfo.size() - 1;

  std::cout << "word embedding dim is " << wordDim << std::endl;
  m_options.wordEmbSize = wordDim;

  wordEmb.zeros(m_wordAlphabet.size(), wordDim);
  curWord = normalize_to_lowerwithdigit(vecInfo[0]);
  wordId = m_wordAlphabet.from_string(curWord);
  hash_set<int> indexers;
  double sum[wordDim];
  int count = 0;
  bool bHasUnknown = false;
  if (wordId >= 0) {
    count++;
    if (unknownId == wordId)
      bHasUnknown = true;
    indexers.insert(wordId);
    for (int idx = 0; idx < wordDim; idx++) {
      double curValue = atof(vecInfo[idx + 1].c_str());
      sum[idx] = curValue;
      wordEmb(wordId, idx) = curValue;
    }

  } else {
    for (int idx = 0; idx < wordDim; idx++) {
      sum[idx] = 0.0;
    }
  }

  while (1) {
    if (!my_getline(inf, strLine)) {
      break;
    }
    if (strLine.empty())
      continue;
    split_bychar(strLine, vecInfo, ' ');
    if (vecInfo.size() != wordDim + 1) {
      std::cout << "error embedding file" << std::endl;
    }
    curWord = normalize_to_lowerwithdigit(vecInfo[0]);
    wordId = m_wordAlphabet.from_string(curWord);
    if (wordId >= 0) {
      count++;
      if (unknownId == wordId)
        bHasUnknown = true;
      indexers.insert(wordId);

      for (int idx = 0; idx < wordDim; idx++) {
        double curValue = atof(vecInfo[idx + 1].c_str());
        sum[idx] = curValue;
        wordEmb(wordId, idx) += curValue;
      }
    }

  }

  if (!bHasUnknown) {
    for (int idx = 0; idx < wordDim; idx++) {
      wordEmb(unknownId, idx) = sum[idx] / count;
    }
    count++;
    std::cout << unknownkey << " not found, using averaged value to initialize." << std::endl;
  }

  int oovWords = 0;
  int totalWords = 0;
  for (int id = 0; id < m_wordAlphabet.size(); id++) {
    if (indexers.find(id) == indexers.end()) {
      oovWords++;
      for (int idx = 0; idx < wordDim; idx++) {
        wordEmb(id, idx) = wordEmb(unknownId, idx);
      }
    }
    totalWords++;
  }

  std::cout << "OOV num is " << oovWords << ", total num is " << m_wordAlphabet.size() << ", embedding oov ratio is " << oovWords * 1.0 / m_wordAlphabet.size()
      << std::endl;

}


void Parser::readWordClusters(const string& inFile) {
  static ifstream inf;
  if (inf.is_open()) {
    inf.close();
    inf.clear();
  }
  inf.open(inFile.c_str());

  static string strLine, curWord;
  static int wordId;

  //find the first line, decide the wordDim;
  while (1) {
    if (!my_getline(inf, strLine)) {
      break;
    }
    if (!strLine.empty())
      break;
  }

  int unknownId = m_wordAlphabet.from_string(unknownkey);

  static vector<string> vecInfo;
  split_bychar(strLine, vecInfo, ' ');
  int wordClusterNum = vecInfo.size() - 1;

  std::cout << "word cluster number is " << wordClusterNum << std::endl;

  m_wordClusters.resize(m_wordAlphabet.size(), wordClusterNum);
  m_wordClusters = "0";
  curWord = normalize_to_lowerwithdigit(vecInfo[0]);
  wordId = m_wordAlphabet.from_string(curWord);
  hash_set<int> indexers;
  int count = 0;
  bool bHasUnknown = false;
  if (wordId >= 0) {
    count++;
    if (unknownId == wordId)
      bHasUnknown = true;
    indexers.insert(wordId);
    for (int idx = 0; idx < wordClusterNum; idx++) {
      m_wordClusters[wordId][idx] = vecInfo[idx + 1];
    }
  }

  while (1) {
    if (!my_getline(inf, strLine)) {
      break;
    }
    if (strLine.empty())
      continue;
    split_bychar(strLine, vecInfo, ' ');
    if (vecInfo.size() != wordClusterNum + 1) {
      std::cout << "error embedding file" << std::endl;
    }
    curWord = normalize_to_lowerwithdigit(vecInfo[0]);
    wordId = m_wordAlphabet.from_string(curWord);
    if (wordId >= 0) {
      count++;
      if (unknownId == wordId)
        bHasUnknown = true;
      indexers.insert(wordId);
      for (int idx = 0; idx < wordClusterNum; idx++) {
        m_wordClusters[wordId][idx] = vecInfo[idx + 1];
      }
    }

  }

  int oovWords = 0;
  int totalWords = 0;
  for (int id = 0; id < m_wordAlphabet.size(); id++) {
    if (indexers.find(id) == indexers.end()) {
      oovWords++;
      for (int idx = 0; idx < wordClusterNum; idx++) {
        m_wordClusters[id][idx] = m_wordClusters[unknownId][idx];
      }
    }
    totalWords++;
  }

  std::cout << "OOV num is " << oovWords << ", total num is " << m_wordAlphabet.size() << ", cluster oov ratio is " << oovWords * 1.0 / m_wordAlphabet.size()
      << std::endl;

}

void Parser::loadModelFile(const string& inputModelFile) {

}

void Parser::writeModelFile(const string& outputModelFile) {

}
