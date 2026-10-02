import numpy as np

from alpharaw.feature.hills import filter_hills


def _hills(*hills):
    int_data = np.concatenate(hills)
    hill_ptrs = np.r_[0, np.cumsum([len(h) for h in hills])].astype(np.int64)
    hill_data = np.arange(len(int_data), dtype=np.int64)
    return hill_data, hill_ptrs, int_data


def test_filter_hills_removes_the_flagged_large_hill():
    """Large hills that are not peak-shaped are removed, and only those."""
    peak = np.exp(-0.5 * ((np.arange(50) - 25) / 6.0) ** 2) * 1e6 + 1e3  # kept
    flat = np.full(50, 1e5)  # max/edge ratio 1 < 2: removed
    short = np.full(5, 1e4)  # below hill_check_large: not checked
    # large hills are hills 2 and 5; hill 5 is flagged (position 1 among them)
    hill_data, hill_ptrs, int_data = _hills(short, short, peak, short, short, flat)

    data, ptrs = filter_hills(hill_data, hill_ptrs, int_data, hill_check_large=40)

    assert list(np.diff(ptrs)) == [5, 5, 50, 5, 5]
    assert np.array_equal(data, np.arange(70))
