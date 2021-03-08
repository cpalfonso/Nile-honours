import numpy as np


def format_time(duration):
    if duration < 0:
        raise ValueError("Duration must be greater than zero.")
    minutes = duration / 60.0
    if minutes > 1:
        seconds = duration % 60.0
        minutes = int(minutes)
        seconds = np.around(seconds, 1)
        message = "{}m {}s".format(minutes, seconds)
    else:
        seconds = np.around(duration, 2)
        message = "{}s".format(seconds)
    return message


def array_average(arr, N=2, axis=0):
    '''
    https://stackoverflow.com/questions/30379311/
    '''
    cum = np.cumsum(arr, axis)
    result = cum[N - 1::N] / float(N)
    result[1:] = result[1:] - result[:-1]
    remainder = arr.shape[0] % N
    if remainder != 0:
        if remainder < arr.shape[0]:
            lastAvg = (
                (cum[-1] - cum[-1 - remainder])
                / float(remainder)
            )
        else:
            lastAvg = cum[-1] / float(remainder)
        result = np.vstack([result, lastAvg])
    return result
