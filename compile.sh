#!/bin/bash
g++ -o main main.cpp -lopenblas -L$OPENBLAS_LIB -std=c++11
