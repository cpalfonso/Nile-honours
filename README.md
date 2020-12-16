# The influence of dynamic topography, climate, and tectonics on the Nile River source-to-sink system – Badlands model scripts and inputs

This repository contains Badlands `.xml` config files for three of the models from Christopher Alfonso's 2020 Honours thesis, as well as Python scripts to run the models.
Model input data can be downloaded from Zenodo ([https://zenodo.org/record/4321853/files/data-bundle.zip](https://zenodo.org/record/4321853/files/data-bundle.zip)) and placed in the appropriate directory (`inputs/{model_name}/data`).
Alternatively, running the shell script `setup.sh` will download all input data and place it into the appropriate directories.

In order to run the models, `cd` into the appropriate subdirectory within the `scripts` directory (i.e. `scripts/{model_name}`), then run `python run_models.py`.
Model outputs will be placed in the `results/{model_name}` directory.
Running the models requires an installation of *Badlands* ([https://github.com/badlands-model/badlands](https://github.com/badlands-model/badlands)) in your Python environment.
