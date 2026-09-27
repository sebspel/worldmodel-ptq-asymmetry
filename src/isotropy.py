"""PCA isotropy of the captured encoder embeddings — off-pod, read-only over `src.embeddings`.

For every encoder engine of a track, fits scikit-learn PCA to one vector per frame of that engine's
`dataset` embeddings (the 10,000 held-out frames) in float64 and plots two eigenvalue spectra,
k = 1 the largest, one line per engine config:

  * `<name>.png`            — normalised, `lambda_k / mean(lambda)`, every engine;
  * `<name>.fp32_ratio.png` — FP32-relative, `lambda_k / lambda_k(FP32)`, every non-FP32 engine.

    uv run python -m src.isotropy track=<lewm|dino> [out=<dir>]

The per-frame vector depends on what the engine outputs:

  * LeWM (`lewm.dataset`) — the engine's one `projector(CLS)` latent `(192,)`.
  * DINO — the engine outputs a `(196, 384)` patch grid with no CLS, so two vectors are formed:
      - `dino.dataset.mean_pool`    — the mean over the 196 patches;
      - `dino.dataset.random_patch` — one patch per frame, its position drawn uniformly with the
        fixed `RANDOM_PATCH_SEED` (the same positions for every engine, so row `i` stays one
        frame and one patch across engines). Each drawn patch is centred by the mean of ITS
        position over all frames rather than by one global mean, so the spectrum is of the
        within-position variation, not of the spread between the 196 positions' means.
    The seed and the drawn positions are written once to `dino.dataset.random_patch.json` beside
    the figures; every later run must redraw exactly the same positions (a mismatch fails loudly).
    Both DINO plots stop at k = 383, and the normalised plot's mean is over those 383: the final
    LayerNorm removes each token's channel mean, so FP32's k = 384 direction (~ the uniform vector)
    carries zero variance and the lower precisions only rounding noise there — not a precision
    effect.

The embedding files are only read. The figures land in `$STABLEWM_HOME/reports/phase5/isotropy/`
beside the other analysis plots.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.decomposition import PCA

matplotlib.use("Agg")  # headless: save figures, never open a window (pod + CI)
import matplotlib.pyplot as plt  # noqa: E402

from src import embeddings  # noqa: E402
from src.report import _INK, _TRACK_DISPLAY, _prec_label, _style  # noqa: E402

SET = "dataset"
TRACKS = ("lewm", "dino")
BASELINE = ("fp32", None)
RANDOM_PATCH_SEED = 0
# Validated categorical hues (dataviz reference palette), fixed slot per ENGINE_CONFIGS entry so a
# config keeps its colour across renders and across every figure.
_CONFIG_COLORS = dict(zip(
    embeddings.ENGINE_CONFIGS, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
))
_DINO_VARIANT_DISPLAY = {"mean_pool": "mean-pooled patches",
                         "random_patch": "random patch, per-position centred"}


def eigenvalues(x: np.ndarray) -> np.ndarray:
    """Eigenvalues of the sample covariance of `x` `(n, d)`, descending, in float64."""
    return PCA(svd_solver="full").fit(x.astype(np.float64)).explained_variance_


def random_patch_positions(n_frames: int, n_patches: int, seed: int = RANDOM_PATCH_SEED):
    """One patch position per frame, uniform over `n_patches`, deterministic in `seed`."""
    return np.random.default_rng(seed).integers(0, n_patches, size=n_frames)


def dino_vectors(x: np.ndarray, positions: np.ndarray) -> dict[str, np.ndarray]:
    """Per-frame float64 vectors from a `(n, patches, d)` patch grid: the patch mean, and the patch
    at `positions[i]` minus that position's mean over all frames."""
    pos_mean = x.mean(axis=0, dtype=np.float64)  # (patches, d)
    drawn = x[np.arange(len(x)), positions].astype(np.float64)
    return {
        "mean_pool": x.mean(axis=1, dtype=np.float64),
        "random_patch": drawn - pos_mean[positions],
    }


def embedding_path(root: Path, track: str, precision: str, method: str | None) -> Path:
    name = f"encoder.{precision}" + (f".{method}" if method else "") + ".npy"
    return root / track / SET / name


def render(
    spectra: dict[tuple[str, str | None], np.ndarray], path: Path, *, ylabel: str, title: str,
    log: bool,
) -> None:
    """One line per engine config; the dashed line at 1 is the reference (isotropy, or FP32)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (prec, method), s in spectra.items():
        ax.plot(np.arange(1, len(s) + 1), s, color=_CONFIG_COLORS[(prec, method)], lw=1.5,
                label=_prec_label(prec, method or ""))
    ax.axhline(1.0, color="#8a8a8a", lw=1, ls="--", zorder=0)
    if log:
        ax.set_yscale("log")
    ax.set_xlim(1, len(s))
    ax.set_xlabel("Principal component $k$ (descending variance)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=_INK)
    _style(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_pair(lam: dict[tuple[str, str | None], np.ndarray], out_dir: Path, name: str,
                title: str) -> list[Path]:
    """The normalised and the FP32-relative spectrum of one set of per-engine eigenvalues."""
    path, ratio_path = out_dir / f"{name}.png", out_dir / f"{name}.fp32_ratio.png"
    render({c: v / v.mean() for c, v in lam.items()}, path, log=True,
           ylabel=r"$\lambda_k \,/\, \bar{\lambda}$", title=title.format(what="eigenvalue spectrum"))
    render({c: v / lam[BASELINE] for c, v in lam.items() if c != BASELINE},
           ratio_path, log=False, ylabel=r"$\lambda_k \,/\, \lambda_k^{\mathrm{FP32}}$",
           title=title.format(what="eigenvalue spectrum relative to FP32"))
    return [path, ratio_path]


def main() -> None:
    track, out_dir = None, None
    for a in sys.argv[1:]:
        if a.startswith("track="):
            track = a.split("=", 1)[1]
        elif a.startswith("out="):
            out_dir = Path(a.split("=", 1)[1])
        else:
            raise SystemExit(f"unknown argument {a!r}")
    if track not in TRACKS:
        raise SystemExit(f"src.isotropy requires track={'|'.join(TRACKS)} (got {track!r})")
    if out_dir is None:
        from src.study import default_out_dir

        out_dir = default_out_dir() / "isotropy"

    root = embeddings.default_out_dir()
    paths = {c: embedding_path(root, track, *c) for c in embeddings.ENGINE_CONFIGS}
    if track == "lewm":
        lam = {c: eigenvalues(np.load(p)) for c, p in paths.items()}
        written = render_pair(lam, out_dir, f"{track}.{SET}",
                              f"{_TRACK_DISPLAY[track]} {{what}} ({SET} frames)")
    else:
        n_frames, n_patches, dim = np.load(paths[BASELINE], mmap_mode="r").shape
        positions = random_patch_positions(n_frames, n_patches)
        embeddings.ensure_manifest(out_dir / f"{track}.{SET}.random_patch.json", {
            "seed": RANDOM_PATCH_SEED,
            "rule": f"one patch position per frame, uniform over {n_patches}, "
                    "numpy default_rng(seed).integers",
            "positions": positions.tolist(),
        })
        lam = {v: {} for v in _DINO_VARIANT_DISPLAY}
        for c, p in paths.items():
            for variant, vecs in dino_vectors(np.load(p), positions).items():
                lam[variant][c] = eigenvalues(vecs)[:dim - 1]  # drop the LayerNorm null direction
        written = [
            f for variant, display in _DINO_VARIANT_DISPLAY.items()
            for f in render_pair(lam[variant], out_dir, f"{track}.{SET}.{variant}",
                                 f"{_TRACK_DISPLAY[track]} {{what}}\n{display} ({SET} frames)")
        ]
    print(f"[isotropy] {track} {SET} spectra ({len(paths)} engines) -> "
          + ", ".join(str(f) for f in written))


if __name__ == "__main__":
    main()
