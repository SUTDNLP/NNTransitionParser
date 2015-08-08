/////////////////////////////////////////////////////////////////////////////////////
// File Name   : MyLib.h
// Project Name: IRLAS
// Author      : Huipeng Zhang (zhp@ir.hit.edu.cn)
// Environment : Microsoft Visual C++ 6.0
// Description : some utility functions
// Time        : 2005.9
// History     :
// CopyRight   : HIT-IRLab (c) 2001-2005, all rights reserved.
/////////////////////////////////////////////////////////////////////////////////////
#ifndef _MYLIB_H_
#define _MYLIB_H_

#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <deque>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <cmath>
#include <ctime>
#include <cfloat>
#include <cstring>
#include <sstream>
#include <armadillo>

#include "Hash_map.hpp"

using namespace arma;
using namespace std;

typedef std::vector<std::string> CStringVector;

typedef std::vector<std::pair<std::string, std::string> > CTwoStringVector;

class string_less {
public:
  bool operator()(const string &str1, const string &str2) const {
    int ret = strcmp(str1.c_str(), str2.c_str());
    if (ret < 0)
      return true;
    else
      return false;
  }
};

/*==============================================================
 *
 * CSentenceTemplate
 *
 *==============================================================*/

template<typename CSentenceNode>
class CSentenceTemplate: public std::vector<CSentenceNode> {

public:
  CSentenceTemplate() {
  }
  virtual ~CSentenceTemplate() {
  }
};

//==============================================================

template<typename CSentenceNode>
inline std::istream & operator >>(std::istream &is, CSentenceTemplate<CSentenceNode> &sent) {
  sent.clear();
  std::string line;
  while (is && line.empty())
    getline(is, line);

  //getline(is, line);

  while (is && !line.empty()) {
    CSentenceNode node;
    std::istringstream iss(line);
    iss >> node;
    sent.push_back(node);
    getline(is, line);
  }
  return is;
}

template<typename CSentenceNode>
inline std::ostream & operator <<(std::ostream &os, const CSentenceTemplate<CSentenceNode> &sent) {
  for (unsigned i = 0; i < sent.size(); ++i)
    os << sent.at(i) << std::endl;
  os << std::endl;
  return os;
}

inline void print_time() {

  time_t lt = time(NULL);
  cout << ctime(&lt) << endl;

}

inline char* mystrcat(char *dst, const char *src) {
  int n = (dst != 0 ? strlen(dst) : 0);
  dst = (char*) realloc(dst, n + strlen(src) + 1);
  strcat(dst, src);
  return dst;
}

inline char* mystrdup(const char *src) {
  char *dst = (char*) malloc(strlen(src) + 1);
  if (dst != NULL) {
    strcpy(dst, src);
  }
  return dst;
}

inline int message_callback(void *instance, const char *format, va_list args) {
  vfprintf(stdout, format, args);
  fflush(stdout);
  return 0;
}

inline void normalize_mat_onerow(mat &m, int row) {
  double sum = 0;
  for (int idx = 0; idx < m.n_cols; idx++)
    sum = sum + m(row, idx) * m(row, idx);
  sum = sum + 0.000001;
  double scale = sqrt(sum);
  for (int idx = 0; idx < m.n_cols; idx++)
    m(row, idx) = m(row, idx) / scale;
}

inline void normalize_mat_onecol(mat &m, int col) {
  double sum = 0;
  for (int idx = 0; idx < m.n_rows; idx++)
    sum = sum + m(idx, col);
  double avg = sum / m.n_rows;
  for (int idx = 0; idx < m.n_rows; idx++)
    m(idx, col) = m(idx, col) - avg;
  sum = 0.0;
  for (int idx = 0; idx < m.n_rows; idx++)
    sum = sum + m(idx, col) * m(idx, col);
  sum = sum + 0.000001;
  double scale = sqrt(sum);
  for (int idx = 0; idx < m.n_rows; idx++)
    m(idx, col) = m(idx, col) / scale;
}

inline void Free(double** p)
{
  if(*p != NULL) free(*p);
  *p = NULL;
}

//(-scale,scale)
inline void randomMatAssign(double* p, int length, double scale = 1.0, int seed = 0)
{
  srand(seed);
  for(int idx = 0; idx < length; idx++)
  {
    p[idx] = 2.0 * rand() * scale / RAND_MAX - scale;
  }
}

inline void assign(double* p, const mat &m)
{
  int count = 0;
  for(int idx = 0; idx < m.n_rows; idx++)
  {
    for(int idy = 0; idy < m.n_cols; idy++)
    {
      p[count] = m(idx, idy);
      count++;
    }
  }
}

inline int mod(int v1, int v2)
{
  if(v1 < 0 || v2 <= 0) return -1;
  else
  {
    return v1%v2;
  }
}

inline void ones(double* p, int length)
{
  for(int idx = 0; idx < length; idx++)
  {
    p[idx] = 1.0;
  }
}

inline void zeros(double* p, int length)
{
  for(int idx = 0; idx < length; idx++)
  {
    p[idx] = 0.0;
  }
}


inline void scaleMat(double* p, double scale, int length)
{
  for(int idx = 0; idx < length; idx++)
  {
    p[idx] = p[idx] * scale;
  }
}

inline void elemMulMat(double* p, double* q, int length)
{
  for(int idx = 0; idx < length; idx++)
  {
    p[idx] = p[idx] * q[idx];
  }
}

inline void elemMulMat(double* p, double* q, double *t, int length)
{
  for(int idx = 0; idx < length; idx++)
  {
    t[idx] = p[idx] * q[idx];
  }
}


inline void normalize_mat_onerow(double* p, int row, int rowSize, int colSize) {
  double sum = 0.000001;
  int start_pos = row * colSize;
  int end_pos = start_pos + colSize;
  for (int idx = start_pos; idx < end_pos; idx++)
    sum = sum + p[idx] * p[idx];
  double norm = sqrt(sum);
  for (int idx = start_pos; idx < end_pos; idx++)
    p[idx] = p[idx] / norm;
}

//shift to avg = 0, and then norm = 1
inline void normalize_mat_onecol(double* p, int col, int rowSize, int colSize)
{
  double sum = 0.0;
  int maxLength = rowSize * colSize;
  for(int idx = col; idx < maxLength; idx += rowSize)
  {
    sum += p[idx];
  }
  double avg = sum / colSize;

  sum = 0.000001;
  for(int idx = col; idx < maxLength; idx += rowSize)
  {
    p[idx] = p[idx] - avg;
    sum += p[idx] * p[idx];
  }

  double norm = sqrt(sum);
  for(int idx = col; idx < maxLength; idx += rowSize)
  {
    p[idx] = p[idx] / norm;
  }
}


inline bool isPunc(std::string thePostag) {

  if (thePostag.compare("PU") == 0 || thePostag.compare("``") == 0 || thePostag.compare("''") == 0 || thePostag.compare(",") == 0 || thePostag.compare(".") == 0
      || thePostag.compare(":") == 0 || thePostag.compare("-LRB-") == 0 || thePostag.compare("-RRB-") == 0 || thePostag.compare("$") == 0
      || thePostag.compare("#") == 0) {
    return true;
  } else {
    return false;
  }
}

// start some assumptions, "-*-" is a invalid label.
inline bool validlabels(const string& curLabel)
{
  if(curLabel[0]== '-' && curLabel[curLabel.length()-1] == '-')
  {
    return false;
  }

  return true;
}

inline bool is_start_label(const string& label)
{
  if(label.length() < 3) return false;
  return (label[0] == 'b' || label[0] == 'B' || label[0] == 's' || label[0] == 'S') && label[1] == '-';
}

inline bool is_end_label(const string& label, const string& nextlabel)
{
  if(label.length() < 3) return false;
  if((label[0] == 'e' || label[0] == 'E' || label[0] == 's' || label[0] == 'S') && label[1] == '-')
    return true;

  if(nextlabel.length() == 1 && (nextlabel[0] == 'o' || nextlabel[0] == 'O')) return true;

  return false;
}

// end some assumptions

inline int cmpPairByValue(const pair<int, int> &x, const pair<int, int> &y) {
  return x.second > y.second;
}


inline void sortMapbyValue(const hash_map<int, int> &t_map, vector<pair<int, int> > &t_vec) {
  t_vec.clear();

  for (hash_map<int,int>::const_iterator  iter = t_map.begin(); iter != t_map.end(); iter++) {
    t_vec.push_back(make_pair(iter->first, iter->second));
  }
  std::sort(t_vec.begin(), t_vec.end(), cmpPairByValue);
}

// split by each of the chars
void split_bychars(const string& str, vector<string> & vec, const char *sep = " ");

void replace_char_by_char(string &str, char c1, char c2);

// remove the blanks at the begin and end of string
void clean_str(string &str);
inline void remove_beg_end_spaces(string &str) {
  clean_str(str);
}

bool my_getline(ifstream &inf, string &line);

void int2str_vec(const vector<int> &vecInt, vector<string> &vecStr);

void str2uint_vec(const vector<string> &vecStr, vector<unsigned int> &vecInt);

void str2int_vec(const vector<string> &vecStr, vector<int> &vecInt);

void join_bystr(const vector<string> &vec, string &str, const string &sep);

void split_bystr(const string &str, vector<string> &vec, const string &sep);
inline void split_bystr(const string &str, vector<string> &vec, const char *sep) {
  split_bystr(str, vec, string(sep));
}

//split a sentence into a vector by separator which is a char
void split_bychar(const string& str, vector<string> & vec, const char separator = ' ');

//convert a string to a pair splited by separator which is '/' by default
void string2pair(const string& str, pair<string, string>& pairStr, const char separator = '/');

//convert every item separated by '/' in a vector to a pair
void convert_to_pair(vector<string>& vecString, vector<pair<string, string> >& vecPair);

//the combination of the two functions above
void split_to_pair(const string& str, vector<pair<string, string> >& vecPair);

void split_pair_vector(const vector<pair<int, string> > &vecPair, vector<int> &vecInt, vector<string> &vecStr);

//it is similar to split_bychar, except that the separator can be a string
void split_by_separator(const string& str, vector<string>& vec, const string separator);

//delete the white(space, Tab or a new line) on the two sides of a string
void chomp(string& str);

//get the length of the longest common string of two strings
int common_substr_len(string str1, string str2);

//compute the index of a Chinese character, the input
//can be any string whose length is larger than 2
int get_char_index(string& str);

//judge if a string is a Hanzi
bool is_chinese_char(string& str);

//find GB char which is two-char-width and the first char is negative
int find_GB_char(const string& str, string wideChar, int begPos);

string word(string& word_pos);

//judge if a string purely consist of ASCII characters
bool is_ascii_string(string& word);

//judge if a string starts with some other string
bool is_startwith(const string& word, const string& prefix);

#endif

