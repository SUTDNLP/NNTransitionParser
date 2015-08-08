cc=g++
#cflags = -g -O0 -larmadillo
cflags = -g -O0 -I/opt/armadillo/include -lopenblas -llapack
#cflags = -O3  -I/opt/armadillo/include -lopenblas -llapack

all: NNParser



obj = LinearClassifier.o main.o Parser.o LinearNNClassifier.o LinearSPNNClassifier.o LinearSNNClassifier.o MyLib.o NNClassifier.o 

NNParser: $(obj)
	$(cc) -o NNParser $(obj) $(cflags)

Parser.o: Parser.cpp Parser.h Alphabet.h MyLib.h Hash_map.hpp Pipe.h Dependency.h Utf.h Options.h State.h LinearSNNClassifier.h Example.h Feature.h Metric.h LinearHidderLayer.h SparseLinearHidderLayer.h NRMat.h
	$(cc) -c Parser.cpp $(cflags)

LinearNNClassifier.o: LinearNNClassifier.cpp LinearNNClassifier.h Example.h Feature.h MyLib.h Hash_map.hpp Metric.h LinearHidderLayer.h SparseLinearHidderLayer.h NRMat.h
	$(cc) -c LinearNNClassifier.cpp $(cflags)

main.o: main.cpp Parser.h Alphabet.h MyLib.h Hash_map.hpp Pipe.h Dependency.h Utf.h Options.h State.h LinearSNNClassifier.h Example.h Feature.h Metric.h LinearHidderLayer.h SparseLinearHidderLayer.h NRMat.h Argument_helper.h
	$(cc) -c main.cpp $(cflags)

NNClassifier.o: NNClassifier.cpp NNClassifier.h Example.h Feature.h MyLib.h Hash_map.hpp Metric.h LinearHidderLayer.h NRMat.h
	$(cc) -c NNClassifier.cpp $(cflags)

LinearSNNClassifier.o: LinearSNNClassifier.cpp LinearSNNClassifier.h Example.h Feature.h MyLib.h Hash_map.hpp Metric.h LinearHidderLayer.h SparseLinearHidderLayer.h NRMat.h
	$(cc) -c LinearSNNClassifier.cpp $(cflags)

LinearSPNNClassifier.o: LinearSPNNClassifier.cpp LinearSPNNClassifier.h Example.h Feature.h MyLib.h Hash_map.hpp Metric.h LinearHidderLayer.h SparseLinearHidderLayer.h NRMat.h
	$(cc) -c LinearSPNNClassifier.cpp $(cflags)

LinearClassifier.o: LinearClassifier.cpp LinearClassifier.h Example.h Feature.h MyLib.h Hash_map.hpp Metric.h LinearHidderLayer.h SparseLinearHidderLayer.h NRMat.h
	$(cc) -c LinearClassifier.cpp $(cflags)

MyLib.o: MyLib.cpp MyLib.h Hash_map.hpp
	$(cc) -c MyLib.cpp $(cflags)


clean:
	rm -rf *.o
	rm -rf 
	rm -rf NNParser

