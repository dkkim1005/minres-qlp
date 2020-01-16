#!/bin/bash
#g++ -o main main.cpp -lopenblas -L$OPENBLAS_LIB -std=c++11
g++ -o main main.cpp $MKL -std=c++11
