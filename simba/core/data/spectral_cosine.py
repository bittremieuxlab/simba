import numpy as np

DEFAULT_BIN_WIDTH = 0.01
DEFAULT_MAX_MZ = 1100.0


def pairwise_spectral_cosine(
    mz_a: np.ndarray,
    intensity_a: np.ndarray,
    mz_b: np.ndarray,
    intensity_b: np.ndarray,
    bin_width: float = DEFAULT_BIN_WIDTH,
    max_mz: float = DEFAULT_MAX_MZ,
) -> float:
    """Binned spectral cosine similarity between two peak lists:
    bin peaks onto a fixed m/z grid, sqrt-compress intensities, L2-normalize
    each side, dot product over the intersecting bins."""

    def _binned(mz: np.ndarray, intensity: np.ndarray):
        mask = (intensity > 0) & (mz > 0) & (mz <= max_mz)
        if not mask.any():
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
        bins = np.round(mz[mask] / bin_width).astype(np.int64)
        sqrt_intensity = np.sqrt(intensity[mask].astype(np.float64))
        uniq_bins, inverse = np.unique(bins, return_inverse=True)
        summed = np.zeros(len(uniq_bins), dtype=np.float64)
        np.add.at(summed, inverse, sqrt_intensity)
        norm = np.linalg.norm(summed)
        if norm == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
        return uniq_bins, summed / norm

    bins_a, vals_a = _binned(np.asarray(mz_a), np.asarray(intensity_a))
    bins_b, vals_b = _binned(np.asarray(mz_b), np.asarray(intensity_b))
    if bins_a.size == 0 or bins_b.size == 0:
        return 0.0

    _, idx_a, idx_b = np.intersect1d(
        bins_a, bins_b, assume_unique=True, return_indices=True
    )
    if idx_a.size == 0:
        return 0.0
    return float(np.dot(vals_a[idx_a], vals_b[idx_b]))
