import os
import shutil
import sys

SCENARIO = 'no-dyntopo'
DIR = os.path.dirname(os.path.abspath(__file__))

tools_dir = os.path.join(DIR, '..', '..')
sys.path.insert(0, tools_dir)
from tools.io import hdf5, xml

START_TIME = 40.0
END_TIME = 0.0
DT = 0.2

OUTPUT_DIR = os.path.join(
    DIR,
    '..', '..',
    'results',
    SCENARIO,
    'forward_{}'.format(SCENARIO),
)


def main():
    output_prerift = OUTPUT_DIR + '_prerift'
    output_postrift = OUTPUT_DIR + '_postrift'

    for subdir in ['h5', 'xmf']:
        i = os.path.join(OUTPUT_DIR, subdir)
        if not os.path.isdir(i):
            os.makedirs(i)

    # Copy prerift
    for subdir in ['h5', 'xmf']:
        src_dir = os.path.join(output_prerift, subdir)
        dst_dir = os.path.join(OUTPUT_DIR, subdir)
        for f in os.listdir(src_dir):
            src = os.path.join(src_dir, f)
            if not (
                    src.endswith('.hdf5')
                    or src.endswith('.xmf')
            ):
                continue
            dst = os.path.join(dst_dir, f)
            shutil.copy2(src, dst)

    # Copy postrift
    hdf5.copy_outputs(OUTPUT_DIR, output_postrift)

    # Generate .xdmf files
    num_steps = int(abs(END_TIME - START_TIME) / DT) + 1
    for which in ['flow', 'tin']:
        xdmf_filename = os.path.join(
            OUTPUT_DIR,
            '{}.series.xdmf'.format(which),
        )
        xml.generate_xdmf(
            xdmf_filename, which, num_steps, overwrite=True,
        )


if __name__ == "__main__":
    main()
