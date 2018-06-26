#!/bin/bash

# for installation in the macos system
TARGET=__minresqlp.so
OPENBLASDEV="libopenblas_haswellp-r0.3.0.dev.dylib"
OLDOPENBLASLOC="..//lib/libopenblas_haswellp-r0.3.0.dev.dylib"

g++ -fPIC -shared -o $TARGET py_module.cpp -I/usr/include/python2.7 \
    -lpython2.7 -lboost_python -lopenblas\
    -I/Users/dongkyukim/Library/Python/2.7/lib/python/site-packages/numpy/core/include/ \
    -I$BOOST_INC -L$BOOST_LIB -L$OPENBLAS_LIB -std=c++11

# BOOST_LIBRARY : location for the boost_python.dylib
echo "BOOST Python dylib location: $BOOST_LIB"

install_name_tool -change libboost_python.dylib $BOOST_LIB/libboost_python.dylib $TARGET
install_name_tool -change $OLDOPENBLASLOC $OPENBLAS_LIB/$OPENBLASDEV $TARGET

# Checking linked libraries
otool -L $TARGET
