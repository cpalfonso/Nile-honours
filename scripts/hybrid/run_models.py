import os
import sys

from badlands.model import Model as badlandsModel

SCENARIO = 'hybrid'
DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR)

import combine_outputs

RIFT_TIME = 20.0
END_TIME = 0.0

# Currently have to change Python working directory so that
# Badlands understands relative file paths in XML input files
os.chdir(DIR)


def backward():
    '''
    Run backward Badlands model.
    '''
    xml_filename = os.path.join(
        DIR,
        '..',
        '..',
        'inputs',
        SCENARIO,
        'input_back_{}.xml'.format(SCENARIO),
    )
    model = badlandsModel()
    model.load_xml(xml_filename)
    model.run_to_time(END_TIME * -1.e6)


def forward():
    '''
    Run forward Badlands models.
    '''
    xml_filename = os.path.join(
        DIR,
        '..',
        '..',
        'inputs',
        SCENARIO,
        'input_forward_{}_prerift.xml'.format(SCENARIO),
    )
    model = badlandsModel()
    model.load_xml(xml_filename)
    model.run_to_time(RIFT_TIME * -1.e6)

    xml_filename = os.path.join(
        DIR,
        '..',
        '..',
        'inputs',
        SCENARIO,
        'input_forward_{}_postrift.xml'.format(SCENARIO),
    )
    model = badlandsModel()
    model.load_xml(xml_filename)
    model.run_to_time(END_TIME * -1.e6)


def main():
    backward()
    forward()


if __name__ == "__main__":
    main()
    combine_outputs.main()
