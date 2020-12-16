#!/bin/bash

# Create necessary directories for inputs and outputs
for scenario in hybrid no-dyntopo sealevel; do
    if [[ ! -d "inputs/${scenario}/data" ]]; then
        mkdir -p "inputs/${scenario}/data"
    fi
    if [[ ! -d "results/${scenario}" ]]; then
        mkdir -p "results/${scenario}"
    fi
done

# Download data from Zenodo
[[ ! -d "tmp" ]] && mkdir -p "tmp"
zip_filename="https://zenodo.org/record/4321853/files/data-bundle.zip"
curl "${zip_filename}" | tar -xz --directory "tmp"
# Copy to inputs directory
for scenario in "hybrid" "sealevel" "no-dyntopo"; do
    cp -R "tmp/data-bundle/${scenario}/" "inputs/${scenario}/data/"
done
rm -r "tmp"
