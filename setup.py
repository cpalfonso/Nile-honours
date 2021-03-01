'''
This file will download the necessary input data for the models from
Zenodo, placing it into the correct directories.

Usage:
    python setup.py  OR  python setup.py all  (download all input data)
'''


import os
import shutil
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib import request
from zipfile import ZipFile


DATA_URL = 'https://zenodo.org/record/4321853/files/data-bundle.zip'
DIRNAME = os.path.dirname(os.path.abspath(__file__))
ALL_SCENARIOS = {
    'hybrid',
    'sealevel',
    'no-dyntopo',
}


def main(scenarios):
    scenarios = set(scenarios)
    if not scenarios.issubset(ALL_SCENARIOS):
        raise ValueError(
            'Arguments must be one of the following: {}'.format(ALL_SCENARIOS)
        )
    print('Creating necessary directories...\t', end='', flush=True)
    for scenario in scenarios:
        inputs_dir = os.path.join(DIRNAME, 'inputs', scenario, 'data')
        if os.path.isdir(inputs_dir):
            shutil.rmtree(inputs_dir)
        os.makedirs(inputs_dir, exist_ok=True)
        outputs_dir = os.path.join(DIRNAME, 'results', scenario)
        if os.path.isdir(outputs_dir):
            shutil.rmtree(outputs_dir)
        os.makedirs(outputs_dir, exist_ok=True)
    print('Done')

    os.makedirs(os.path.join(DIRNAME, 'tmp'), exist_ok=True)
    print('Downloading data from {}...\t'.format(DATA_URL),
          end='', flush=True)
    with request.urlopen(DATA_URL) as response, \
            NamedTemporaryFile() as outfile, \
            TemporaryDirectory() as tmp_dir:
        shutil.copyfileobj(response, outfile)
        print('Done')

        print('Extracting data...\t', end='', flush=True)
        zipped = ZipFile(outfile)
        zipped.extractall(tmp_dir)
        for scenario in scenarios:
            src = os.path.join(tmp_dir, 'data-bundle', scenario)
            dst = os.path.join(DIRNAME, 'inputs', scenario, 'data')
            for f in os.listdir(src):
                if os.path.isfile(os.path.join(src, f)):
                    shutil.copy2(os.path.join(src, f),
                                 os.path.join(dst, f))
                else:
                    shutil.copytree(os.path.join(src, f),
                                    os.path.join(dst, f))
        print('Done')

    print('Setup complete.')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        args = ALL_SCENARIOS
    else:
        args = set(sys.argv[1:])
    if 'all' in args:
        args = ALL_SCENARIOS
    main(args)
