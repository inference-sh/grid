#!/bin/bash

# Get the expected extension suffix for the current Python installation
SUFFIX=$(python3-config --extension-suffix)
OUTPUT_FILE="mesh_processor${SUFFIX}"

# Check if the binary already exists and is newer than the source file
if [ -f "$OUTPUT_FILE" ] && [ "$OUTPUT_FILE" -nt "mesh_processor.cpp" ]; then
    echo "Binary $OUTPUT_FILE is up to date, skipping compilation"
    exit 0
fi

echo "Compiling mesh_processor..."
c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` mesh_processor.cpp -o "$OUTPUT_FILE"