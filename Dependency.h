// Copyright (C) University of Oxford 2010
/****************************************************************
 *                                                              *
 * dependency.h - the dependency tree                           *
 *                                                              *
 * Author: Yue Zhang                                            *
 *                                                              *
 * Computing Laboratory, Oxford. 2007.11                        *
 *                                                              *
 ****************************************************************/

#ifndef _DEPENDENCY_H
#define _DEPENDENCY_H

#include "MyLib.h"

const int DEPENDENCY_LINK_NO_HEAD = -1;

/*==============================================================
 *
 * CDependencyTreeNode
 *
 *==============================================================*/

class CDependencyTreeNode {

public:
   std::string word;
   std::string tag;
   std::string label;
   int head;

public:
   CDependencyTreeNode() : word(""), tag(""), label(""), head(DEPENDENCY_LINK_NO_HEAD) { }
   CDependencyTreeNode( const std::string &w, const std::string &t, const std::string &l, const int &h) : word(w), tag(t), label(l), head(h) { }
   virtual ~CDependencyTreeNode() {}

public:
   bool operator ==(const CDependencyTreeNode &item) const { 
      return word.compare(item.word) == 0 && tag.compare(item.tag) == 0 && label.compare(item.label) == 0 && head == item.head;
   }

};

//==============================================================

inline std::istream & operator >> (std::istream &is, CDependencyTreeNode &node) {
   (is) >> node.word >> node.tag >> node.head >> node.label ;
   return is ;
}

inline std::ostream & operator << (std::ostream &os, const CDependencyTreeNode &node) {
   os << node.word << "\t" << node.tag << "\t" << node.head << "\t" << node.label ;
   return os ;
}


typedef CSentenceTemplate<CDependencyTreeNode> CDependencyTree ;


//==============================================================

/*--------------------------------------------------------------
 *
 * IsValidDependencyTree - check well-formed
 *
 *--------------------------------------------------------------*/


inline
bool IsValidDependencyTree(const CDependencyTree &tree) {
   if ( tree.empty() ) return true;
   int nHead = 0;
   int nLoop = 0;
   int j;
   for ( int i=0; i<static_cast<int>(tree.size()); ++i ) {
      if ( tree.at(i).head == DEPENDENCY_LINK_NO_HEAD ) nHead++ ;
      j = i;
      while ( nLoop & (1<<j) == 0 ) {
         nLoop &= (1<<j); // mark to avoid duplicate checking for head
         j = tree.at(i).head;  // move to head
         if (j==DEPENDENCY_LINK_NO_HEAD) break; // head found
         if (j>=static_cast<int>(tree.size())) return false; // std::cout of the boundary of sentence
         if (j==i) return false; // loop found
      }
   }
   if (nHead==1) return true; return false;
}

/*--------------------------------------------------------------
 *
 * IsProjectiveDependencyTree - check projectivity
 *
 *--------------------------------------------------------------*/

inline
bool IsProjectiveDependencyTree(const CDependencyTree &tree) {
   if (!IsValidDependencyTree(tree)) return false;
   for ( int i=0; i<static_cast<int>(tree.size()); ++i ) {
      int mini = std::min(i, tree.at(i).head);
      int maxi = std::max(i, tree.at(i).head);
      for ( int j=mini+1; j<maxi; ++j )
         if (tree.at(j).head<mini||tree.at(j).head>maxi) return false;
   }
   return true;
}

/*---------------------------------------------------------------
 *
 * UnparseSentence - from dependency tree to raw sentence
 *
 *--------------------------------------------------------------*/

inline
void UnparseSentence(const CDependencyTree &parsed, std::vector<std::string>& retval) {
   retval.clear();
   CDependencyTree::const_iterator it;
   for (it=parsed.begin(); it!=parsed.end(); ++it)
      retval.push_back(it->word);
}

/*---------------------------------------------------------------
 *
 * UnparseSentence - from dependency tree to tagged sentence
 *
 *--------------------------------------------------------------*/

inline
void UnparseSentence(const CDependencyTree &parsed, CTwoStringVector& retval) {
   retval.clear();
   CDependencyTree::const_iterator it;
   for (it=parsed.begin(); it!=parsed.end(); ++it)
      retval.push_back(std::make_pair(it->word, it->tag));
}




#endif
