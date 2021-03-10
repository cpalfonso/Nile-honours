#!/bin/bash

# Create necessary directories for inputs and outputs
echo -n "Creating necessary directories...    "
for scenario in "hybrid" "sealevel" "no-dyntopo"; do
    if [[ ! -d "inputs/${scenario}/data" ]]; then
        mkdir -p "inputs/${scenario}/data"
    fi
    if [[ ! -d "results/${scenario}" ]]; then
        mkdir -p "results/${scenario}"
    fi
done
echo "Done"

# Download data from Zenodo
[[ ! -d "tmp" ]] && mkdir -p "tmp"
zip_url="https://zenodo.org/record/4321853/files/data-bundle.zip"
zip_filename="data-bundle"
echo "Downloading files from ${zip_url}"
wget -O "${zip_filename}.zip" "${zip_url}"
unzip -oq "${zip_filename}" -d "tmp"
# Copy to inputs directory
for scenario in "hybrid" "sealevel" "no-dyntopo"; do
    cp -R "tmp/data-bundle/${scenario}/" "inputs/${scenario}/data/"
done
rm -r "tmp"
rm "${zip_filename}"
echo "Setup completed"
