'''
A few I/O functions for Badlands .hdf5 and .xmf files.
'''
import os
import shutil

from . import xml


def get_step(filename):
    '''
    Get the 'step' number of a .hdf5 or .xmf filename.

    --------------------------------------------------------------------------
    Parameters:
    --------------------------------------------------------------------------
        filename: the filename from which to extract the step number.

    Returns:
    --------------------------------------------------------------------------
        n: the step number corresponding to the filename.
    '''
    if not isinstance(filename, str):
        raise TypeError(
            f"{filename} is not of type 'str' "
            + f"(type = {type(filename)})"
        )
    if not (
        filename.endswith('.hdf5')
        or filename.endswith('.xmf')
    ):
        raise ValueError(f"Invalid .hdf5 or .xmf filename: {filename}")
    n = (
        filename.replace(
            'flow.time', ''
        ).replace(
            'tin.time', ''
        ).replace(
            'sed.time', ''
        ).replace(
            '.hdf5', ''
        ).replace(
            '.xmf', ''
        )
    )
    n = int(n)
    return n


def get_steps(filenames):
    '''
    Get the 'step' numbers of multiple .hdf5 or .xmf filenames.

    --------------------------------------------------------------------------
    Parameters:
    --------------------------------------------------------------------------
        filenames: the filenames from which to extract the step numbers.

    Returns:
    --------------------------------------------------------------------------
        steps: the step numbers corresponding to the filenames.
    '''
    if isinstance(filenames, str):
        filenames = [filenames]
    steps = []
    for filename in filenames:
        try:
            n = get_step(filename)
            steps.append(n)
        except ValueError:
            continue

    return steps


def get_max_step(filenames):
    '''
    Get the maximum 'step' number of multiple .hdf5 or .xmf filenames

    --------------------------------------------------------------------------
    Parameters:
    --------------------------------------------------------------------------
        filenames: the filenames from which to extract step numbers.

    Returns:
    --------------------------------------------------------------------------
        n: the maximum step number found in the filenames.
    '''
    try:
        if os.path.isdir(filenames):
            filenames = os.listdir(filenames)
    except TypeError:
        pass
    steps = get_steps(filenames)
    n = max(steps)
    return n


def copy_outputs(dir1, dir2=None, restart_step=None, delete=False):
    if dir2 is None:
        dir2 = dir1 + '_0'
    if restart_step is None:
        h5_dir = os.path.join(dir1, 'h5')
        h5_filenames = [
            i for i in os.listdir(h5_dir) if i.endswith('.hdf5')
        ]
        restart_step = get_max_step(h5_filenames)

    for subdir in ['h5', 'xmf']:
        src_dir = os.path.join(dir2, subdir)
        dst_dir = os.path.join(dir1, subdir)
        for f in os.listdir(src_dir):
            src = os.path.join(src_dir, f)
            if not (
                src.endswith('.hdf5')
                or src.endswith('.xmf')
            ):
                continue
            src_step = get_step(f)
            dst_step = src_step + restart_step
            dst = f.replace(f'time{src_step}', f'time{dst_step}')
            dst = os.path.join(dst_dir, dst)

            if f.endswith('.hdf5'):
                shutil.copy2(src, dst)
            else:  # elif f.endswith('.xmf'):
                xml.change_xmf_time(
                    src, dst, src_step, dst_step,
                )
            if delete:
                os.remove(src)
