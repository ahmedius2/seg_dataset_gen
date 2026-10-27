"""NumPy operations shared by training and inference."""

import numpy as np


COVERAGE_REACH_M = 1.6


def coverage_kernel_size(cell_size, target_m=COVERAGE_REACH_M):
    """Odd kernel size (cells) whose metric width is closest to target_m.

    Ties go to the smaller kernel.
    """
    lower = max(1, 2 * int(np.floor((target_m / cell_size - 1) / 2)) + 1)
    upper = lower + 2
    return min((lower, upper), key=lambda k: (abs(k * cell_size - target_m), k))


def dilate_bev_mask(mask, kernel_size=11):
    if kernel_size is None or kernel_size <= 1:
        return mask

    source = np.asarray(mask)
    is_bool = source.dtype == np.bool_
    if not is_bool:
        source = source.astype(np.float32, copy=False)
    pad = kernel_size // 2
    pad_value = False if is_bool else -np.inf
    padded = np.pad(source, ((pad, pad), (pad, pad)),
                    mode="constant", constant_values=pad_value)
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, (kernel_size, kernel_size)
    )
    result = windows.max(axis=(-2, -1))
    if is_bool:
        return result
    return result.astype(np.float32, copy=False)