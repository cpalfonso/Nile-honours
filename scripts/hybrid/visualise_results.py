import multiprocessing as mp
import platform
import os
import sys
import warnings

import numpy as np

SCENARIO = 'hybrid'
CHUNKSIZE = 50

DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(DIR, '..', '..')
sys.path.insert(0, TOOLS_DIR)
from tools.grid import Grid, remap_TIN
from tools.vis import plot_output

START = 40.0  # Ma
END = 0.0  # Ma
DT = 0.2  # Myr
INPUT_TOPO = os.path.join(DIR, '..', '..',
                          'inputs', SCENARIO,
                          'data', 'topo0Ma.csv')
NUM_STEPS = abs(int((END - START) / DT)) + 1

OUTPUT_DIR = os.path.join(DIR, '..', '..',
                          'results', SCENARIO,
                          'images')

warnings.simplefilter('ignore', RuntimeWarning)


def visualise(step):
    frame = str(step)
    while len(frame) < len(str(NUM_STEPS - 1)):
        frame = '0' + frame
    time = np.around(START - np.abs(DT * step), 2)

    # times = np.flip(np.arange(END, START + DT, DT))
    output_elevation = os.path.join(OUTPUT_DIR,
                                    'elevation_{}.png'.format(frame))
    output_erodep = os.path.join(OUTPUT_DIR,
                                 'erodep_{}.png'.format(frame))

    badlands_out = os.path.join(OUTPUT_DIR,
                                '..', 'forward_{}'.format(SCENARIO))
    d = remap_TIN(input_topo=INPUT_TOPO,
                  step=step,
                  badlands_out=badlands_out,
                  return_all=True,
                  verbose=False)
    g = Grid(
        INPUT_TOPO,
        tin_file=os.path.join(
            badlands_out,
            'h5',
            'tin.time{}.hdf5'.format(step)
        )
    )
    d['z'] -= g.sealevel
    del g

    # Plot elevation and erosion/deposition
    for which, filename in zip(['elev_discharge', 'erodep'],
                               [output_elevation, output_erodep]):
        plot_output(d,
                    filename=filename,
                    time=time,
                    which=which,
                    contour_interval=0,
                    hillshade=True,
                    resize_factor=2,
                    time_decimals=1,
                    dzmin=-5000,
                    dzmax=5000)


if __name__ == '__main__':
    if platform.system() == 'Darwin':
        mp.set_start_method('spawn')

    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    args = list(range(NUM_STEPS))
    num_cpus = min([int(NUM_STEPS / CHUNKSIZE), int(mp.cpu_count() * 0.75)])
    print('Number of CPUs = {}'.format(num_cpus))
    pool = mp.Pool(num_cpus)
    pool.map(visualise, args, chunksize=CHUNKSIZE)
