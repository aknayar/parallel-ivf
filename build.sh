#!/bin/bash
set -e
mkdir -p build
cd build
cmake ..
make -j
cd ..
echo "Build complete."
