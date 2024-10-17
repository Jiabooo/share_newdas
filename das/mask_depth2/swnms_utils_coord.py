import numpy as np
from scipy.spatial import cKDTree, distance


def _ensure_spacing(coord, spacing_array, p_norm, max_out):
    """Returns a subset of coord where a minimum spacing is guaranteed.

    Parameters
    ----------
    coord : ndarray
        The coordinates of the considered points.
    spacing : float
        the maximum allowed spacing between the points.
    p_norm : float
        Which Minkowski p-norm to use. Should be in the range [1, inf].
        A finite large p may cause a ValueError if overflow can occur.
        ``inf`` corresponds to the Chebyshev distance and 2 to the
        Euclidean distance.
    max_out: int
        If not None, at most the first ``max_out`` candidates are
        returned.

    Returns
    -------
    output : ndarray
        A subset of coord where a minimum spacing is guaranteed.

    """

    # Use KDtree to find the peaks that are too close to each other
    tree = cKDTree(coord)

    # Initialize sets and counters
    rejected_peaks_indices = set()
    naccepted = 0

    # print(spacing_array)

    # Query the KDTree for each point
    for idx, point in enumerate(coord):
        if idx not in rejected_peaks_indices:
            # Get the dynamic spacing value for this point
            spacing = spacing_array[int(point[0]), int(point[1])]
            # print(max_out)

            # Find all points within the dynamic spacing
            candidates = tree.query_ball_point(point, r=spacing, p=p_norm)

            # Remove the current point from candidates
            candidates.remove(idx)

            # Calculate distances to the candidates
            dist = distance.cdist([point], coord[candidates], distance.minkowski, p=p_norm).reshape(-1)

            # Filter candidates based on the dynamic spacing
            candidates = [c for c, d in zip(candidates, dist) if d < spacing]

            # Mark these candidates as rejected
            rejected_peaks_indices.update(candidates)
            naccepted += 1

            # Check if we have reached the max_out limit
            if max_out is not None and naccepted >= max_out:
                break

    # Remove the peaks that are too close to each other
    output = np.delete(coord, tuple(rejected_peaks_indices), axis=0)

    # Limit the output to max_out if specified
    if max_out is not None:
        output = output[:max_out]

    return output


def ensure_spacing(coords, distance_array, p_norm=np.inf, min_split_size=50,
                   max_out=None, *, max_split_size=2000):
    """Returns a subset of coord where a minimum spacing is guaranteed.

    Parameters
    ----------
    coords : array_like
        The coordinates of the considered points.
    spacing : float
        the maximum allowed spacing between the points.
    p_norm : float
        Which Minkowski p-norm to use. Should be in the range [1, inf].
        A finite large p may cause a ValueError if overflow can occur.
        ``inf`` corresponds to the Chebyshev distance and 2 to the
        Euclidean distance.
    min_split_size : int
        Minimum split size used to process ``coords`` by batch to save
        memory. If None, the memory saving strategy is not applied.
    max_out : int
        If not None, only the first ``max_out`` candidates are returned.
    max_split_size : int
        Maximum split size used to process ``coords`` by batch to save
        memory. This number was decided by profiling with a large number
        of points. Too small a number results in too much looping in
        Python instead of C, slowing down the process, while too large
        a number results in large memory allocations, slowdowns, and,
        potentially, in the process being killed -- see gh-6010. See
        benchmark results `here
        <https://github.com/scikit-image/scikit-image/pull/6035#discussion_r751518691>`_.

    Returns
    -------
    output : array_like
        A subset of coord where a minimum spacing is guaranteed.

    """

    output = coords
    if len(coords):

        coords = np.atleast_2d(coords)
        if min_split_size is None:
            batch_list = [coords]
        else:
            coord_count = len(coords)
            split_idx = [min_split_size]
            split_size = min_split_size
            while coord_count - split_idx[-1] > max_split_size:
                split_size *= 2
                split_idx.append(split_idx[-1] + min(split_size,
                                                     max_split_size))
            batch_list = np.array_split(coords, split_idx)

        output = np.zeros((0, coords.shape[1]), dtype=coords.dtype)
        for batch in batch_list:
            output = _ensure_spacing(np.vstack([output, batch]),
                                     distance_array, p_norm, max_out)
            if max_out is not None and len(output) >= max_out:
                break

    return output
