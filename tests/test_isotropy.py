"""PCA isotropy — the eigenvalues against a direct covariance eigendecomposition (CPU)."""

import numpy as np

from src import isotropy


def test_eigenvalues_match_covariance_eigendecomposition():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((2000, 16)) * np.linspace(0.1, 3.0, 16)).astype(np.float32)
    lam = isotropy.eigenvalues(x)

    ref = np.sort(np.linalg.eigvalsh(np.cov(x.astype(np.float64), rowvar=False)))[::-1]
    assert lam.dtype == np.float64
    np.testing.assert_allclose(lam, ref, rtol=1e-8)
    assert np.all(np.diff(lam) <= 0)


def test_isotropic_data_is_flat():
    lam = isotropy.eigenvalues(np.random.default_rng(1).standard_normal((50_000, 8)))
    np.testing.assert_allclose(lam / lam.mean(), 1.0, atol=0.05)


def test_dino_vectors_pool_and_centre_each_drawn_patch_by_its_position():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((50, 4, 3)).astype(np.float32) + np.arange(4)[None, :, None] * 10
    pos = isotropy.random_patch_positions(50, 4, seed=7)
    v = isotropy.dino_vectors(x, pos)

    np.testing.assert_allclose(v["mean_pool"], x.astype(np.float64).mean(axis=1))
    i = 3
    expected = x[i, pos[i]].astype(np.float64) - x[:, pos[i]].astype(np.float64).mean(axis=0)
    np.testing.assert_allclose(v["random_patch"][i], expected)
    assert v["random_patch"].dtype == np.float64
    np.testing.assert_array_equal(pos, isotropy.random_patch_positions(50, 4, seed=7))
