'''
Class and functions for manipulating Badlands .xml input files.
'''

import os
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom


def _write_xml_to_file(data, filename, file_format='.xml'):
    try:
        if not file_format.startswith('.'):
            file_format = '.' + file_format
    except AttributeError as e:
        raise TypeError('Invalid output format: {}'.format(file_format)) from e
    filename = _check_filename(filename,
                               file_format=file_format,
                               in_out='output')

    if file_format == '.xml':
        data.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    elif file_format in ['.xdmf', '.xmf']:
        data.set('xmlns:xi', "http://www.w3.org/2001/XInclude")

    data = ET.tostring(data, encoding='unicode', method='xml')
    data = minidom.parseString(data).toprettyxml(indent='    ')
    # Remove empty lines
    lines = data.split('\n')
    lines_new = []
    for line in lines:
        if line.strip() != '':
            lines_new.append(line + '\n')
    with open(filename, 'w') as f:
        f.writelines(lines_new)


def change_xmf_time(input_filename, output_filename, old_time, new_time):
    input_filename = _check_filename(
        input_filename, file_format='.xmf', in_out='input'
    )
    output_filename = _check_filename(
        output_filename, file_format='.xmf', in_out='output'
    )

    tree = ET.parse(input_filename)
    root = tree.getroot()
    for DataItem in root.iter('DataItem'):
        text = DataItem.text
        s = 'time{}'.format(int(old_time))
        s_new = 'time{}'.format(int(new_time))
        text = text.replace(s, s_new)
        DataItem.text = text

    _write_xml_to_file(root, output_filename, file_format='.xmf')


def generate_xdmf(filename, which, num_steps, overwrite=False):
    filename = _check_filename(filename, file_format='.xdmf', in_out='output')
    num_steps = int(num_steps)
    if not isinstance(which, str):
        raise TypeError(
            "Invalid 'which' parameter (must be 'tin', 'sed', or 'flow')"
        )
    if which not in ['tin', 'sed', 'flow']:
        raise ValueError(
            "Invalid 'which' parameter (must be 'tin', 'sed', or 'flow')"
        )

    if os.path.isfile(filename):
        if overwrite:
            print('{} already exists; overwriting...'.format(filename))
        else:
            raise FileExistsError(
                '{} already exists; aborting...'.format(filename)
            )

    with open(filename, 'w') as f:
        l1 = '''<?xml version="1.0" encoding="UTF-8"?>'''
        l2 = '''<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd">'''
        l3 = (
            '''<Xdmf Version="2.0"'''
            + ''' xmlns:xi="http://www.w3.org/2001/XInclude">'''
        )
        l4 = ''' <Domain>'''
        l5 = '''    <Grid GridType="Collection" CollectionType="Temporal">'''
        for line in [l1, l2, l3, l4, l5]:
            f.writelines(line + '\n')
        line_format = (
            '''      <xi:include href="xmf/{}.'''.format(which)
            + '''time{}.xmf" xpointer="xpointer(//Xdmf/Domain/Grid)"/>'''
        )
        for i in range(num_steps):
            line = line_format.format(i)
            f.writelines(line + '\n')
        l6 = '''    </Grid>'''
        l7 = ''' </Domain>'''
        l8 = '''</Xdmf>'''
        for line in [l6, l7, l8]:
            f.writelines(line + '\n')


def _check_filename(filename, file_format='.xml', in_out=None):
    try:
        if not file_format.startswith('.'):
            file_format = '.' + file_format
    except AttributeError as e:
        raise TypeError('Invalid format: {}'.format(file_format)) from e

    try:
        if not filename.endswith(file_format):
            filename += file_format
    except AttributeError as e:
        if in_out is None:
            e_m = 'Invalid filename: {}'.format(filename)
        else:
            e_m = 'Invalid {} filename: {}'.format(in_out, filename)
        raise TypeError(e_m) from e

    return filename


def get_sealevel(filename):
    '''
    Get the sealevel from a given TIN .xmf file.
    '''
    try:
        filename = os.path.abspath(filename)
        if not os.path.isfile(filename):
            raise FileNotFoundError("'{}' not found.".format(filename))
        if not (
            os.path.basename(filename).startswith('tin.time')
            and os.path.basename(filename).endswith('.xmf')
        ):
            raise ValueError(
                "'{}' is not a valid TIN .xmf file.'".format(filename)
            )
    except (TypeError, AttributeError) as e:
        raise TypeError(
            "'{}' is not a valid filename.".format(filename)
        ) from e

    tree = ET.parse(filename)
    root = tree.getroot()
    domain = root.find('Domain')
    grid0 = domain.find('Grid')
    grid1 = grid0.find('Grid')
    for child in grid1.findall('Attribute'):
        if child.attrib['Name'] == 'Sealevel':
            di = child.find('DataItem')
            break
    else:
        raise AttributeError("Sea level not found in {}".format(filename))
    func = di.attrib['Function']
    sl = float(func.split()[-1])
    return sl
