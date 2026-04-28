import numpy as np
from typing import List, Optional, Tuple
import tqdm 

def group_bp_fmn(
    bpstack: np.ndarray,
    *,
    mode: int,
    out_frames: Optional[int] = None,
    bp_per_frame: Optional[int] = None,
    dtype: np.dtype = np.uint16,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Group BPs when the input array is shaped (bp_count, m, n).

    Parameters
    ----------
    bpstack : (bp_count, m, n) array
        3D array with BP/time as the first axis.
    mode : {1,2,3,4}
        1: Sequential even grouping (no overlap)
        2: Random even grouping (no overlap)
        3: Random overlap grouping, unique (without replacement)
        4: Random overlap grouping, non-unique (with replacement)
    out_frames : int, optional
        Number of output groups (required for mode 3; inferred for modes 1 & 2).
    bp_per_frame : int
        Number of BPs per output frame.
    dtype : np.dtype
        Output dtype (default uint16).
    rng : np.random.Generator, optional
        Random generator for reproducibility.

    Returns
    -------
    output : (out_frames, m, n) array
        Sum of selected BPs per group.
    groups_idx : list[np.ndarray]
        0-based indices into the first axis of `bpstack` for each output frame.
    """
    print("oo Start grouping")

    if rng is None:
        rng = np.random.default_rng()

    if bpstack.ndim != 3:
        raise ValueError("bpstack must be 3D (bp_count, m, n).")

    bp_count, m, n = bpstack.shape
    output = np.zeros((out_frames, m, n), dtype=dtype)
    groups_idx: List[np.ndarray] = [None] * out_frames  # filled below

    def _sum_to_dtype(arr: np.ndarray, out_dtype: np.dtype) -> np.ndarray:
        # Sum in safe dtype to avoid overflow, then clip/cast to desired dtype
        summed = arr.sum(axis=0, dtype=np.int64)  # sum along BP axis
        if np.issubdtype(out_dtype, np.integer):
            info = np.iinfo(out_dtype)
            summed = np.clip(summed, info.min, info.max).astype(out_dtype, copy=False)
        else:
            summed = summed.astype(out_dtype, copy=False)
        return summed

    if mode == 1:
        print("|| mode 1 Sequential even grouping")
        for i in range(out_frames):
            start = i * bp_per_frame
            end = (i + 1) * bp_per_frame
            idx = np.arange(start, end, dtype=int)
            groups_idx[i] = idx
            output[i] = _sum_to_dtype(bpstack[idx, :, :], out_dtype=dtype)
        print("** Done")

    elif mode == 2:
        print("|| mode 2 Random even grouping")
        perm = rng.permutation(bp_count)
        for i in range(out_frames):
            start = i * bp_per_frame
            end = (i + 1) * bp_per_frame
            idx = perm[start:end]
            groups_idx[i] = idx
            output[i] = _sum_to_dtype(bpstack[idx, :, :], out_dtype=dtype)
        print("** Done")

    elif mode == 3:
        print("|| mode 3 Random overlap grouping, no over-sampling BP within frame, allowing over-sampling BP between frames")
        for i in range(out_frames):
            idx = rng.choice(bp_count, size=bp_per_frame, replace=False)
            groups_idx[i] = idx
            output[i] = _sum_to_dtype(bpstack[idx, :, :], out_dtype=dtype)
        print("** Done")

    else: # mode == 4
        print("here")
        print("|| mode 4 Random overlap grouping, allowing over-sampling BP within frame, allowing over-sampling BP between frames")
        for i in tqdm(range(out_frames)):
            idx = rng.choice(bp_count, size=bp_per_frame, replace=True)
            groups_idx[i] = idx
            output[i] = _sum_to_dtype(bpstack[idx, :, :], out_dtype=dtype)
        print("** Done")

    return output, groups_idx