/*
 * Metric.h
 *
 *  Created on: Mar 17, 2015
 *      Author: mszhang
 */

#ifndef SRC_METRIC_H_
#define SRC_METRIC_H_


using namespace std;

class Metric {

public:
  int overall_label_count;
  int correct_label_count;
  int correct_uas_count;

public:
  Metric() {
    overall_label_count = 0;
    correct_label_count = 0;
    correct_uas_count = 0;
  }

  ~Metric() {
  }

  void reset() {
    overall_label_count = 0;
    correct_label_count = 0;
    correct_uas_count = 0;
  }

  bool bIdentical() {

    if (overall_label_count == correct_label_count) {
      return true;
    }
    return false;

  }

  double getAccuracy() {
    return correct_label_count * 1.0 / overall_label_count;

  }

  void print() {
    if(correct_uas_count > 0)
    {
      std::cout << "UAS :\tP=" << correct_uas_count << "/" << overall_label_count << "=" << correct_uas_count * 1.0 / overall_label_count
          << "    LAS :\tP=" << correct_label_count << "/" << overall_label_count << "=" << correct_label_count * 1.0 / overall_label_count << std::endl;
    }
    else
    {
      std::cout << "Accuracy :\tP=" << correct_label_count << "/" << overall_label_count << "=" << correct_label_count * 1.0 / overall_label_count << std::endl;
    }
  }

};

#endif /* SRC_EXAMPLE_H_ */
