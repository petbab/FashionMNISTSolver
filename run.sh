#!/bin/bash

echo "#################"
echo "    COMPILING    "
echo "#################"

g++ -std=c++23 -O3 src/main.cpp src/*.h -o network

echo "#################"
echo "     RUNNING     "
echo "#################"

./network
