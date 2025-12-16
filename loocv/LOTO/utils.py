import numpy as np

def sliding_window2(data, window_size, overlap):
    num_segments, segment_length, *other_dims = data.shape
    step = window_size - overlap
    num_windows = (segment_length - window_size) // step + 1

    windows = []
    for i in range(num_segments):
        for start in range(0, segment_length - window_size + 1, step):
            window = data[i, start:start + window_size]
            windows.append(window)
    windows = np.array(windows)

    # return windows.reshape(-1, window_size, *other_dims)
    return windows


def sliding_window(data, window_size, overlap):
    segment_length, *other_dims = data.shape
    step = window_size - overlap
    num_windows = (segment_length - window_size) // step + 1

    windows = []
    for start in range(0, segment_length - window_size + 1, step):
        window = data[start:start + window_size]
        windows.append(window)
    windows = np.array(windows)

    # return windows.reshape(-1, window_size, *other_dims)
    return windows