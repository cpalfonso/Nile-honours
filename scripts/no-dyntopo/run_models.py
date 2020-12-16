import os

from badlands.model import Model as badlandsModel

SCENARIO = 'no-dyntopo'
DIR = os.path.dirname(os.path.abspath(__file__))

RIFT_TIME = 20.0
END_TIME = 0.0


def backward():
    '''
    Run backward Badlands model.
    '''
    xml_filename = os.path.join(
        DIR,
        '..',
        'inputs',
        SCENARIO,
        f'input_back_{SCENARIO}.xml',
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
        'inputs',
        SCENARIO,
        f'input_forward_{SCENARIO}_prerift.xml',
    )
    model = badlandsModel()
    model.load_xml(xml_filename)
    model.run_to_time(RIFT_TIME * -1.e6)

    xml_filename = os.path.join(
        DIR,
        '..',
        'inputs',
        SCENARIO,
        f'input_forward_{SCENARIO}_postrift.xml',
    )
    model = badlandsModel()
    model.load_xml(xml_filename)
    model.run_to_time(END_TIME * -1.e6)


def main():
    backward()
    forward()


if __name__ == "__main__":
    main()
