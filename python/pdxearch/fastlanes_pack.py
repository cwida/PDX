import numpy as np
import math


def packed_length(in_len, bw):
    # Each element in in_len is 8 bits
    # Each element in out_len will be 6 bits, so we calculate accordingly
    current = in_len * 8
    needed = in_len * bw
    out_container_size = math.ceil(in_len * (needed / current))
    return out_container_size


def pack_6bit(in_array, out_array):
    """
    Python equivalent of the C++ pack_6bit_8ow function.

    Parameters:
    - in_array: 1D numpy array of 1024 uint8 values (8 * 128), each in [0, 63]
    - out_array: 1D numpy array of 768 uint8 values (6 * 128), to be written to
    """
    # assert in_array.shape == (1024,), "in_array must be 1D with 1024 elements"
    # assert out_array.shape == (768,), "out_array must be 1D with 768 elements"
    # assert np.all(in_array < 64), "All input values must be 6-bit"
    mask = 0x3F
    for i in range(128):
        x0 = in_array[128 * 0 + i] & mask
        x1 = in_array[128 * 1 + i] & mask
        x2 = in_array[128 * 2 + i] & mask
        x3 = in_array[128 * 3 + i] & mask
        x4 = in_array[128 * 4 + i] & mask
        x5 = in_array[128 * 5 + i] & mask
        x6 = in_array[128 * 6 + i] & mask
        x7 = in_array[128 * 7 + i] & mask

        out_array[i + 128 * 0] = x0 | (x1 << 6)
        out_array[i + 128 * 1] = (x1 >> 2) | (x2 << 4)
        out_array[i + 128 * 2] = (x2 >> 4) | (x3 << 2)
        out_array[i + 128 * 3] = x4 | (x5 << 6)
        out_array[i + 128 * 4] = (x5 >> 2) | (x6 << 4)
        out_array[i + 128 * 5] = (x6 >> 4) | (x7 << 2)


def pack_4bit(in_array, out_array):
    """
    Packs 8 rows of 4-bit values (128 values each) into 4 rows of packed bytes.

    Parameters:
    - in_array: 1D numpy array of 1024 uint8 values (8 * 128), each in [0, 15]
    - out_array: 1D numpy array of 512 uint8 values (4 * 128), to be filled
    """
    # assert in_array.shape == (1024,), "in_array must be 1D with 1024 elements"
    # assert out_array.shape == (512,), "out_array must be 1D with 512 elements"
    # assert np.all(in_array < 16), "All input values must be 4-bit (in range 0-15)"

    for i in range(128):
        x0 = in_array[128 * 0 + i] & 0x0F
        x1 = in_array[128 * 1 + i] & 0x0F
        out_array[128 * 0 + i] = x0 | (x1 << 4)

        x2 = in_array[128 * 2 + i] & 0x0F
        x3 = in_array[128 * 3 + i] & 0x0F
        out_array[128 * 1 + i] = x2 | (x3 << 4)

        x4 = in_array[128 * 4 + i] & 0x0F
        x5 = in_array[128 * 5 + i] & 0x0F
        out_array[128 * 2 + i] = x4 | (x5 << 4)

        x6 = in_array[128 * 6 + i] & 0x0F
        x7 = in_array[128 * 7 + i] & 0x0F
        out_array[128 * 3 + i] = x6 | (x7 << 4)


def pack_4bit_symmetric(in_array, out_array, original_length=1024):
    j = 0
    for i in range(0, original_length, 2):
        x0 = in_array[i] & 0x0F
        x1 = in_array[i + 1] & 0x0F
        out_array[j] = (x0 << 4) | x1
        j += 1
