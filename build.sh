#!/bin/bash
set -e
cd build
cmake ..
make -j
cd ..
echo "Build complete."
