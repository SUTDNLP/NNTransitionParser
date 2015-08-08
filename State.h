// Copyright (C) University of Oxford 2010
#ifndef DEPPARSER_ARC_STANDARD_STATE_H
#define DEPPARSER_ARC_STANDARD_STATE_H
#include "Dependency.h"
#include "Alphabet.h"

// implicit action define:
// 0: no action, 1: shift, 2: pop_root,
// for dependency label l (0.... L-1), 2*l+3: arcleft, 2*l+4: arcright // maxactionnum = 2+2*L

class CStateItem {
protected:
  std::vector<int> m_Stack;
  // stack of words that are currently processed

  int m_nNextWord;
  // index for the next word

  int m_lHeads[256];
  // the lexical head for each word

  int m_lDepsL[256];
  // the leftmost dependency for each word (just for cache, temporary info)

  int m_lDepsR[256];
  // the rightmost dependency for each word (just for cache, temporary info)

  int m_lDepsL2[256];
  // the second-leftmost dependency for each word

  int m_lDepsR2[256];
  // the second-rightmost dependency for each word

  int m_lDepNumL[256];
  // the number of left dependencies

  int m_lDepNumR[256];
  // the number of right dependencies

  int m_lLabels[256];
  // the label of each dependency link

public:
  double score;
  // score of stack - predicting how potentially this is the correct one

  int m_length;
  // the length of the sentence, it's hash_set manually.

  const CStateItem * previous_;
  // Previous state of the current state

  unsigned long last_action;
  // the last stack action

public:
  // constructors and destructor
  CStateItem(int length = 0) {
    clear(length);
  }

  ~CStateItem() { }
public:
  // comparison
  inline bool operator < (const CStateItem &item) const {
    return score < item.score;
  }

  inline bool operator > (const CStateItem &item) const {
    return score > item.score;
  }

  inline bool operator == (const CStateItem &item) const {
    throw("equal operator should not be used");
  }

  inline bool operator != (const CStateItem &item) const {
    return ! ((*this)==item);
  }

  // propty
  inline int stacksize() const {
    return m_Stack.size();
  }

  inline bool stackempty() const {
    return m_Stack.empty();
  }

  inline int stacktop() const {
    if (m_Stack.empty()) { return -1; }
    return m_Stack.back();
  }

  inline int stack2top() const {
    if (m_Stack.size() < 2) { return -1; }
    return m_Stack[m_Stack.size() - 2];
  }

  inline int stack3top() const {
    if (m_Stack.size() < 3) { return -1; }
    return m_Stack[m_Stack.size() - 3];
  }

  inline int stackbottom() const {
    assert(!m_Stack.empty());
    return m_Stack.front();
  }

  inline int stackitem(const int & id) const {
    assert (id < m_Stack.size());
    return m_Stack[id];
  }

  inline int head(const int & id) const {
    assert (id < m_nNextWord);
    return m_lHeads[id];
  }

  inline int leftdep(const int & id) const {
    assert(id < m_nNextWord);
    return m_lDepsL[id];
  }

  inline int rightdep(const int & id) const {
    assert(id < m_nNextWord);
    return m_lDepsR[id];
  }

  inline int left2dep(const int & id) const {
    assert(id < m_nNextWord);
    return m_lDepsL2[id];
  }

  inline int right2dep(const int & id) const {
    assert(id < m_nNextWord);
    return m_lDepsR2[id];
  }

  inline int size() const {
    return m_nNextWord;
  }

  inline bool terminated() const {
    return (last_action == 2
            && m_Stack.empty()
            && m_nNextWord == m_length);
  }

  inline bool complete() const {
    return (m_Stack.size() == 1
            && m_nNextWord == m_length);
  }

  inline int label(const int & id) const {
    assert(id < m_nNextWord);
    return m_lLabels[id];
  }


  inline int leftarity(const int & id) const {
    assert(id < m_nNextWord);
    return m_lDepNumL[id];
  }

  inline int rightarity(const int & id) const {
    assert(id < m_nNextWord);
    return m_lDepNumR[id];
  }


  void clear(int length) {
    m_nNextWord = 0;
    m_Stack.clear();
    score = 0;
    previous_ = 0;
    last_action = 0;
    m_length = length;
    ClearNext();
  }

  void operator = (const CStateItem &item) {
    m_Stack = item.m_Stack;
    m_nNextWord = item.m_nNextWord;

    last_action = item.last_action;
    score       = item.score;
    m_length        = item.m_length;
    previous_   = item.previous_;

    for (int i = 0; i <= m_nNextWord; ++ i) { // only copy active word (including m_nNext)
      m_lHeads[i] = item.m_lHeads[i];
      m_lDepsL[i] = item.m_lDepsL[i];
      m_lDepsR[i] = item.m_lDepsR[i];
      m_lDepsL2[i] = item.m_lDepsL2[i];
      m_lDepsR2[i] = item.m_lDepsR2[i];
      m_lDepNumL[i] = item.m_lDepNumL[i];
      m_lDepNumR[i] = item.m_lDepNumR[i];
      m_lLabels[i] = item.m_lLabels[i];
    }
  }

//-----------------------------------------------------------------------------
public:
  // Perform Arc-Left operation in the arc-standard algorithm
  void ArcLeft(unsigned long lab) {
    // At least, there must be two elements in the stack.
    assert(m_Stack.size() > 1);
    assert(m_lHeads[m_Stack.back()] == DEPENDENCY_LINK_NO_HEAD);

    int stack_size = m_Stack.size();
    int top0 = m_Stack[stack_size - 1];
    int top1 = m_Stack[stack_size - 2];

    m_Stack.pop_back();
    m_Stack.back() = top0;

    m_lHeads[top1] = top0;
    m_lDepNumL[top0] ++;


    m_lLabels[top1] = lab +1;


    if (m_lDepsL[top0] == DEPENDENCY_LINK_NO_HEAD) {
      m_lDepsL[top0] = top1;
    } else if (top1 < m_lDepsL[top0]) {
      m_lDepsL2[top0] = m_lDepsL[top0];
      m_lDepsL[top0] = top1;
    } else if (top1 < m_lDepsL2[top0]) {
      m_lDepsL2[top0] = top1;
    }

    last_action = 2 * lab + 3;
  }

  // Perform the arc-right operation in arc-standard
  void ArcRight(
      unsigned long lab
      ) {
    assert(m_Stack.size() > 1);

    int stack_size = m_Stack.size();
    int top0 = m_Stack[stack_size - 1];
    int top1 = m_Stack[stack_size - 2];

    m_Stack.pop_back();
    m_lHeads[top0] = top1;
    m_lDepNumR[top1] ++;

    m_lLabels[top0] = lab + 1;

    if (m_lDepsR[top1] == DEPENDENCY_LINK_NO_HEAD) {
      m_lDepsR[top1] = top0;
    } else if (m_lDepsR[top1] < top0) {
      m_lDepsR2[top1] = m_lDepsR[top1];
      m_lDepsR[top1] = top0;
    } else if (m_lDepsR2[top1] < top0) {
      m_lDepsR2[top1] = top0;
    }

    last_action = 2 * lab + 4;
  }

  // the shift action does pushing
  void Shift() {
    m_Stack.push_back(m_nNextWord);
    m_nNextWord ++;
    ClearNext();
    last_action = 1;
  }

  // this is used for the convenience of scoring and updating
  void PopRoot() {
    assert(m_Stack.size() == 1
           && m_lHeads[m_Stack.back()] == DEPENDENCY_LINK_NO_HEAD);
    // make sure only one root item in stack
    m_lLabels[m_Stack.back()] = 0;

    last_action = 2;
    m_Stack.pop_back(); // pop it
  }

  // the clear next action is used to clear the next word, used
  // with forwarding the next word index
  void ClearNext() {
    m_lHeads[m_nNextWord]   = DEPENDENCY_LINK_NO_HEAD;
    m_lDepsL[m_nNextWord]   = DEPENDENCY_LINK_NO_HEAD;
    m_lDepsL2[m_nNextWord]  = DEPENDENCY_LINK_NO_HEAD;
    m_lDepsR[m_nNextWord]   = DEPENDENCY_LINK_NO_HEAD;
    m_lDepsR2[m_nNextWord]  = DEPENDENCY_LINK_NO_HEAD;
    m_lDepNumL[m_nNextWord] = 0;
    m_lDepNumR[m_nNextWord] = 0;
    m_lLabels[m_nNextWord] = -1;
  }

  // the move action is a simple call to do action according to the action code
  void Move (const unsigned long &ac, int maxlabel) {
    assert(ac >= 0);
    static int lab;
    if(ac == 0)
    {
      return;
    }
    else if (ac == 1)
    {
      Shift();
      return;
    }
    else if (ac == 2)
    {
      PopRoot();
      return;
    }
    else if (ac % 2 == 1)
    {
      lab = (ac - 3) / 2;
      ArcLeft(lab);
      return;
    }
    else if (ac % 2 == 0)
    {
      lab = (ac - 4) / 2;
      ArcRight(lab);
      return;
    }
  }

//-----------------------------------------------------------------------------

public:
  int StandardMove(const CDependencyTree & tree, Alphabet& labelAlphabet) {
    static int lab;
    if (terminated()) {
      return 0;
    }

    int stack_size = m_Stack.size();
    if (0 == stack_size) {
      return 1;
    }
    else if (1 == stack_size) {
      if (m_nNextWord == static_cast<int>(tree.size())) {
        return 2;
      } else {
        return 1;
      }
    }
    else {
      int top0 = m_Stack[stack_size - 1];
      int top1 = m_Stack[stack_size - 2];

      bool has_right_child = false;
      for (int i = m_nNextWord; i < tree.size(); ++ i) {
        if (tree[i].head == top0) { has_right_child = true; break; }
      }

      if (tree[top0].head == top1 && !has_right_child) {
        lab = labelAlphabet.from_string(tree[top0].label);
        return 2 * lab + 4;
      }
      else if (tree[top1].head == top0) {
        lab = labelAlphabet.from_string(tree[top1].label);
        return 2 * lab + 3;
      }
      else {
        return 1;
      }
    }
  }


  // we want to pop the root item after the whole tree done
  // on the one hand this seems more natural
  // on the other it is easier to score
  void StandardFinish() {
    assert(m_Stack.size() == 0);
  }

  void GenerateTree(const CDependencyTree &input, CDependencyTree &output, Alphabet& labelAlphabet, const string& rootdepkey) const {
    output.clear();
    for (int i = 0; i < size(); ++i)
    {
      if(m_lLabels[i] >= 1)
        output.push_back(CDependencyTreeNode(input[i].word, input[i].tag, labelAlphabet.from_id(m_lLabels[i]-1), m_lHeads[i]));
      else
      {
        assert(m_lHeads[i] == -1);
        output.push_back(CDependencyTreeNode(input[i].word, input[i].tag, rootdepkey, m_lHeads[i]));
      }
    }
  }
};

#endif  //  end for DEPPARSER_ARC_STANDARD_STATE_H
