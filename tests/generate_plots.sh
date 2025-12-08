#!/bin/bash

cd "$(dirname "$0")"

for mode in 0 1 2; do
    for result_type in ghc-results psc-results; do
        for dataset in easy medium hard extreme gist; do
            path="../$result_type/$dataset"
            
            if [ -d "$path" ]; then
                echo "Running visualize.py for $result_type/$dataset (mode $mode)"
                python visualize.py "$path" "$mode"
            else
                echo "Skipping $result_type/$dataset (directory not found)"
            fi
        done
    done
done
