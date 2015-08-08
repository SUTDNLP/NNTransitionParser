#include "Parser.h"
#include <iostream>
#include <string>
#include "Argument_helper.h"

using namespace std;

int main(int argc, char* argv[])
{
  std::string trainFile = "", devFile = "", testFile = "", modelFile="";
  std::string wordEmbFile = "", optionFile = "";
  std::string outputFile = "";
  bool bTrain = false;
	dsr::Argument_helper ah;

	ah.new_flag("l", "learn", "train or test", bTrain);
	ah.new_named_string("train", "trainCorpus", "named_string", "training corpus to train a model, must when training", trainFile);
	ah.new_named_string("dev", "devCorpus", "named_string", "development corpus to train a model, optional when training", devFile);
	ah.new_named_string("test", "testCorpus", "named_string", "testing corpus to train a model or input file to test a model, optional when training and must when testing", testFile);
	ah.new_named_string("model", "modelFile", "named_string", "model file, must when training and testing", modelFile);
	ah.new_named_string("word", "wordEmbFile", "named_string", "pretrained word embedding file to train a model, optional when training", wordEmbFile);
	ah.new_named_string("option", "optionFile", "named_string", "option file to train a model, optional when training", optionFile);
	ah.new_named_string("output", "outputFile", "named_string", "output file to test, must when testing", outputFile);

  ah.process(argc, argv);

  Parser parser;
	if(bTrain)
	{
	  parser.train(trainFile, devFile, testFile,  modelFile, optionFile, wordEmbFile);
	}
	else
	{
	  parser.test(testFile, outputFile, modelFile);
	}


  //test(argv);
  //ah.write_values(std::cout);



}
