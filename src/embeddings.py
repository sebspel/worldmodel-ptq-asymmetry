"""Encoder-embedding capture — every encoder engine, over fixed frame sets.

Runs ONLY the encoder engines (no predictor, no planner) of one track over three frame sets and
persists each engine's embeddings, so the encoder representations can be compared across
precisions and calibration methods off-pod:

  * `dataset`   — `N_DATASET_FRAMES` frames evenly strided over `pusht_expert_train.lance`, drawn
                  ONLY from episodes that contribute no INT8/FP8 calibration clip (held-out from
                  the quantizer).
  * `eval_init` — the 50 eval episodes' initial frames, and
  * `eval_goal` — their goal frames (`start + goal_offset_steps`), both exactly as the Phase-3 /
                  SR eval feeds them to the encoder.

    uv run python -m src.embeddings track=<lewm|dino> [sets=dataset,eval_init,eval_goal]

**Which tokens are saved is whatever the engine outputs — nothing more is reconstructed.** Each
encoder engine has ONE output: DINO's is the CLS+register-sliced `(196, 384)` patch grid
(`DINOWMAdapter.encode`), LeWM's is `projector(CLS)` `(192,)` (`LeWMAdapter.encode`). So the DINO
CLS token and the LeWM patch tokens are not available from the engines and are skipped; the
per-engine sidecar JSON records what was saved and what was skipped. An engine with more than one
output fails loudly rather than guessing which output is which.

**Frame lists are fixed across runs.** Each set's frame list is written once to
`frames.<set>.json` under the output root; every later run (the other track, another pod session)
recomputes it and fails loudly if it differs, so row `i` of every embedding file is the same frame.

**The eval frames come from the vendored eval itself**, not a re-implementation: `eval_wm.run`
runs byte-unmodified with `swm.World` swapped for a recorder that captures the
`World.evaluate(dataset=, episodes_idx=, start_steps=, goal_offset=)` call and stops the run there
(no env, no model, no policy — `policy=random`). The pixels are then pulled by the platform's own
`_extract_init_goal`, the function `World.evaluate` uses to overwrite the env's first observation
and goal — so the frames are the ones the encoder saw.

Engines are driven exactly as the SR eval drives them: one frame per call through
`sr_shim._hist_adapt` (repeat-pad to the engine's static hist, slice the real frame back). Outputs
are written as float32 `.npy` (the engine's native output dtype — a narrower store would round away
part of the precision difference being measured) under
`$STABLEWM_HOME/reports/embeddings/<track>/<set>/encoder.<precision>[.<method>].npy`. Existing files
are never overwritten (CLAUDE §8): a re-run fills only the missing engines, and a file lands under
its final name only once complete.

Pod-only end to end (engines + dataset); the frame-selection and manifest helpers are pure and
unit-tested off-pod.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from src.calibrate import CALIB_DATASET, CALIB_FRAMESKIP, DEFAULT_N_CLIPS
from src.env import stablewm_home
from src.export import engine_filename, engine_root
from src.interfaces import CALIBRATION_METHODS, HISTORY_SIZE, QUANTIZED_PRECISIONS

N_DATASET_FRAMES = 10_000
SETS = ("dataset", "eval_init", "eval_goal")
# Every encoder engine the study builds per track: fp32/fp16 (method-invariant) + int8/fp8 at
# each calibration method (method-tagged plans, `export.engine_filename`).
ENGINE_CONFIGS: tuple[tuple[str, str], ...] = tuple(
    (p, m)
    for p in ("fp32", "fp16", "int8", "fp8")
    for m in (CALIBRATION_METHODS if p in QUANTIZED_PRECISIONS else (None,))
)
# What each engine output rank is, and what that engine therefore cannot provide. Keyed by the
# per-frame output rank (after the batch/hist axes are dropped).
_TOKENS_BY_RANK = {
    2: (
        "patch",
        {"cls": "not an engine output — the encoder graph slices off CLS + register tokens "
                "(DINOWMAdapter.encode)"},
    ),
    1: (
        "cls",
        {"patch": "not an engine output — the encoder graph returns projector(CLS) only "
                  "(LeWMAdapter.encode)"},
    ),
}


def default_out_dir() -> Path:
    return stablewm_home() / "reports" / "embeddings"


# --- frame selection (pure) -------------------------------------------------------------------


def calibration_episodes(calib_clip_indices: list[tuple[int, int]], n_clips: int) -> set[int]:
    """Episodes contributing at least one calibration clip. Mirrors the strided draw in
    `calibrate.draw_calibration_clips` (pinned against it by `tests/test_embeddings.py`) over the
    SAME windowed dataset's `clip_indices` (`num_steps=HISTORY_SIZE`, `frameskip=CALIB_FRAMESKIP`)."""
    total = len(calib_clip_indices)
    idxs = np.unique(np.linspace(0, total - 1, min(n_clips, total)).round().astype(int))
    return {int(calib_clip_indices[i][0]) for i in idxs}


def strided_frames(
    frame_indices: list[tuple[int, int]], excluded_episodes: set[int], n: int
) -> list[tuple[int, int]]:
    """`n` (episode, step) frames evenly strided over every frame of the non-excluded episodes.
    Deterministic (no RNG). Fails loudly if exclusion leaves fewer than `n` frames."""
    eligible = [f for f in frame_indices if f[0] not in excluded_episodes]
    if len(eligible) < n:
        raise RuntimeError(
            f"only {len(eligible)} frames remain after excluding {len(excluded_episodes)} "
            f"calibration episodes; need {n}"
        )
    idxs = np.linspace(0, len(eligible) - 1, n).round().astype(int)
    if len(np.unique(idxs)) != n:
        raise RuntimeError("strided draw produced duplicate frames")
    return [(int(eligible[i][0]), int(eligible[i][1])) for i in idxs]


def ensure_manifest(path: Path, manifest: dict) -> dict:
    """Write the frame manifest on first use; on every later run require the recomputed one to be
    identical, so the rows of every embedding file line up across runs and tracks."""
    if path.exists():
        stored = json.loads(path.read_text())
        if stored != manifest:
            raise RuntimeError(
                f"frame list {path} differs from the one recomputed now — the embeddings would "
                "no longer be row-aligned across runs. Investigate before re-running."
            )
        return stored
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1))
    return manifest


# --- frame sources (pod: need the dataset) -----------------------------------------------------


class _EvaluateCaptured(Exception):
    pass


class _RecordingWorld:
    """Stands in for `swm.World` inside the vendored eval: records the `evaluate` call and stops
    the run there, so the vendored episode selection runs byte-unmodified without an env."""

    captured: dict = {}

    def __init__(self, **kwargs):
        self.num_envs = kwargs["num_envs"]

    def set_policy(self, policy):
        pass

    def evaluate(self, **kwargs):
        type(self).captured = kwargs
        raise _EvaluateCaptured


def capture_eval_selection(hydra_argv: list[str]) -> dict:
    """Run the vendored `eval_wm.run` up to `World.evaluate` and return its call kwargs
    (`dataset`, `episodes_idx`, `start_steps`, `goal_offset`, ...). `policy=random` so no
    checkpoint is loaded — the selection does not depend on the policy."""
    from unittest.mock import patch

    import tempfile

    from scripts.plan import eval_wm
    from src.eval import _compose_eval_cfg

    out_file = Path(tempfile.mkdtemp(prefix="swm_embsel_")) / "results.txt"
    cfg = _compose_eval_cfg([*hydra_argv, "policy=random"], out_file)
    if cfg.get("compile", False):
        raise RuntimeError("compile=true runs a warm-up evaluate first; capture expects compile=false")
    with patch.object(eval_wm.swm, "World", _RecordingWorld):
        try:
            eval_wm.run(cfg)
        except _EvaluateCaptured:
            return _RecordingWorld.captured
    raise RuntimeError("eval_wm.run returned without calling World.evaluate")


def eval_frame_sets(hydra_argv: list[str]) -> dict[str, tuple[dict, np.ndarray]]:
    """`{eval_init|eval_goal: (manifest, pixels (N, H, W, C) uint8)}` for the eval episodes, via
    the platform's `_extract_init_goal` — the same arrays `World.evaluate` puts in `infos`."""
    from stable_worldmodel.world.world import _extract_init_goal

    call = capture_eval_selection(hydra_argv)
    episodes = [int(e) for e in call["episodes_idx"]]
    starts = [int(s) for s in call["start_steps"]]
    offset = int(call["goal_offset"])
    init_state, goal_state, _ = _extract_init_goal(call["dataset"], episodes, starts, offset)
    common = {"source": "scripts.plan.eval_wm selection (World.evaluate call)",
              "hydra_argv": hydra_argv, "goal_offset": offset}
    return {
        "eval_init": (
            {**common, "frames": [[e, s] for e, s in zip(episodes, starts)]},
            init_state["pixels"],
        ),
        "eval_goal": (
            {**common, "frames": [[e, s + offset] for e, s in zip(episodes, starts)]},
            goal_state["goal"],
        ),
    }


def dataset_frame_set(dataset_name: str = CALIB_DATASET, n: int = N_DATASET_FRAMES):
    """`(manifest, frame_dataset)` for the strided held-out set. The frame dataset (one frame per
    `clip_indices` entry) serves the pixels by `(episode, step)` via `load_chunk`."""
    import stable_worldmodel as swm

    calib_ds = swm.data.load_dataset(
        dataset_name, num_steps=HISTORY_SIZE, frameskip=CALIB_FRAMESKIP,
        keys_to_load=["pixels", "proprio", "action"],
    )
    excluded = calibration_episodes(calib_ds.clip_indices, DEFAULT_N_CLIPS)
    frame_ds = swm.data.load_dataset(dataset_name, keys_to_load=["pixels"])
    n_episodes = len(frame_ds.lengths)
    print(f"[embeddings] excluding {len(excluded)}/{n_episodes} episodes (calibration clips)")
    frames = strided_frames(frame_ds.clip_indices, excluded, n)
    manifest = {
        "n_episodes_total": n_episodes,
        "source": dataset_name,
        "rule": f"{n} frames evenly strided over all frames of episodes with no calibration clip",
        "excluded_calibration_episodes": sorted(excluded),
        "frames": [list(f) for f in frames],
    }
    return manifest, frame_ds


# --- engine capture (pod) ----------------------------------------------------------------------


def _open_encoders(track: str, root: Path) -> dict[str, tuple[Path, object]]:
    """`{file stem: (plan path, T-agnostic encode fn)}` for every built encoder engine of `track`.
    Engines not on the volume are reported absent and skipped."""
    from src.sr_shim import _hist_adapt
    from src.trt_runtime import EngineRunner

    out = {}
    for precision, method in ENGINE_CONFIGS:
        plan = root / track / engine_filename("encoder", precision, method or "max")
        if not plan.exists():
            print(f"[embeddings:{track}] absent, skipped: {plan}")
            continue
        runner = EngineRunner(plan)
        if len(runner.output_names) != 1:
            raise RuntimeError(
                f"{plan} has outputs {runner.output_names}; identify each before saving"
            )
        enc_hist = int(runner.engine.get_tensor_shape(runner.input_names[0])[1])
        fn = _hist_adapt(lambda px, r=runner: r.run((px,)), enc_hist)
        out[plan.name.removesuffix(".plan")] = (plan, fn)
    return out


def capture_set(
    track: str, set_name: str, n: int, frame_at, out_dir: Path, engine_dir: Path
) -> None:
    """Encode the `n` frames (`frame_at(i)` -> normalized `(3, 224, 224)`) through every built
    encoder engine of `track` and write one `.npy` per engine (+ a sidecar JSON)."""
    set_dir = out_dir / track / set_name
    set_dir.mkdir(parents=True, exist_ok=True)
    encoders = {
        stem: v for stem, v in _open_encoders(track, engine_dir).items()
        if not (set_dir / f"{stem}.npy").exists()
    }
    if not encoders:
        print(f"[embeddings:{track}/{set_name}] nothing to do (all present or absent)")
        return

    stores: dict[str, np.memmap] = {}
    with torch.no_grad():
        for i in range(n):
            obs = frame_at(i)[None, None].to("cuda")  # (1, 1, 3, 224, 224)
            for stem, (_, fn) in encoders.items():
                emb = fn(obs)[0, 0]
                if stem not in stores:
                    stores[stem] = np.lib.format.open_memmap(
                        set_dir / f"{stem}.npy.partial", mode="w+",
                        dtype=np.float32, shape=(n, *emb.shape),
                    )
                stores[stem][i] = emb.float().cpu().numpy()
            if (i + 1) % 1000 == 0:
                print(f"[embeddings:{track}/{set_name}] {i + 1}/{n}")

    for stem, store in stores.items():
        shape = store.shape[1:]
        store.flush()
        (set_dir / f"{stem}.npy.partial").rename(set_dir / f"{stem}.npy")
        saved, skipped = _TOKENS_BY_RANK[len(shape)]
        (set_dir / f"{stem}.json").write_text(json.dumps({
            "engine": str(encoders[stem][0]),
            "frames": f"frames.{set_name}.json",
            "shape": [n, *shape],
            "dtype": "float32",
            "saved": saved,
            "skipped": skipped,
        }, indent=1))
        print(f"[embeddings:{track}/{set_name}] {stem}: {saved} {[n, *shape]}")


def main() -> None:
    track, sets, out_dir = None, SETS, None
    for a in sys.argv[1:]:
        if a.startswith("track="):
            track = a.split("=", 1)[1]
        elif a.startswith("sets="):
            sets = tuple(a.split("=", 1)[1].split(","))
        elif a.startswith("out="):
            out_dir = Path(a.split("=", 1)[1])
        else:
            raise SystemExit(f"unknown argument {a!r}")
    if track is None:
        raise SystemExit("src.embeddings requires track=<lewm|dino>")
    if unknown := set(sets) - set(SETS):
        raise SystemExit(f"unknown sets {sorted(unknown)}; expected a subset of {list(SETS)}")
    out_dir = out_dir or default_out_dir()

    from torchvision import tv_tensors

    from src.calibrate import _pixel_transform

    transform = _pixel_transform()

    def to_input(chw_uint8):
        # As `WorldModelPolicy` feeds the encoder: channel-first frame -> tv Image -> the vendored
        # eval `img_transform` (ImageNet normalize + resize 224).
        return transform(tv_tensors.Image(chw_uint8))

    if "dataset" in sets:
        manifest, frame_ds = dataset_frame_set()
        frames = ensure_manifest(out_dir / "frames.dataset.json", manifest)["frames"]

        def frame_at(i):
            ep, step = frames[i]
            return to_input(frame_ds.load_chunk([ep], [step], [step + 1])[0]["pixels"][0])

        capture_set(track, "dataset", len(frames), frame_at, out_dir, engine_root())

    eval_sets = [s for s in sets if s != "dataset"]
    if eval_sets:
        # The selection is track-independent (seed / num_eval / goal_offset / dataset are shared);
        # the LeWM overlay only supplies the dataset name, identical in both eval overlays.
        by_set = eval_frame_sets(["--config-dir", "conf", "+experiment=eval_lewm"])
        for s in eval_sets:
            manifest, pixels = by_set[s]
            ensure_manifest(out_dir / f"frames.{s}.json", manifest)
            capture_set(
                track, s, len(pixels),
                lambda i, px=pixels: to_input(torch.from_numpy(px[i]).permute(2, 0, 1)),
                out_dir, engine_root(),
            )


if __name__ == "__main__":
    main()
