/*
 * Pipe.h
 *
 *  Created on: Mar 25, 2015
 *      Author: mszhang
 */

#ifndef SRC_PIPE_H_
#define SRC_PIPE_H_

#include "Dependency.h"

class Pipe {
public:
  Pipe() {

  }
  virtual ~Pipe() {

  }

public:
  void readInstances(const std::string &inputfile, std::vector<CDependencyTree>& trees, int maxNum) {
    trees.clear();
    std::ifstream is(inputfile.c_str());
    if(!is.is_open()) return;

    int nCount=0;

    while(is) {
      CDependencyTree ref_sent;
      is >> ref_sent;

      if(ref_sent.size() > 0 )
      {
        if(ref_sent.size() < 256 && IsProjectiveDependencyTree(ref_sent))
        {
          trees.push_back(ref_sent);
          nCount++;
        }
      }
      else
      {
        break;
      }

      if(maxNum > 0 && nCount == maxNum)
      {
        break;
      }
    }

    std::cout << "Total example number: " << nCount << std::endl;
  }


  void outputAllInstances(const std::string& outFile, const std::vector<CDependencyTree>& decodeInstResults)
  {
    std::ofstream os(outFile.c_str());

    for (int idx = 0; idx < decodeInstResults.size(); idx++) {
      os << decodeInstResults[idx];
    }
    os.close();
  }

};

#endif /* SRC_PIPE_H_ */
