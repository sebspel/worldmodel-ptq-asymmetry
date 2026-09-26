"""Encoder-embedding capture — the pure frame-selection + manifest helpers and the vendored-eval
selection capture (CPU, fake dataset; the engine leg is pod-only)."""

import numpy as np
import pytest
import torch

from src import calibrate, embeddings
from src.export import engine_filename


class _FakeWindowedDataset:
    """Minimal `swm` Dataset: episode-grouped `clip_indices`, records which clips are drawn."""

    def __init__(self, lengths, span):
        self.clip_indices = [
            (ep, s) for ep, n in enumerate(lengths) for s in range(n - span + 1)
        ]
        self.drawn: list[int] = []

    def __len__(self):
        return len(self.clip_indices)

    def __getitem__(self, i):
        self.drawn.append(i)
        return {"pixels": torch.zeros(1), "proprio": torch.zeros(1), "action": torch.zeros(1)}


def test_calibration_episodes_matches_the_real_calibration_draw(monkeypatch):
    """The exclusion mirror must pick EXACTLY the episodes `draw_calibration_clips` draws from."""
    import stable_worldmodel as swm

    rng = np.random.default_rng(0)
    ds = _FakeWindowedDataset(rng.integers(20, 200, size=300), span=15)
    monkeypatch.setattr(swm.data, "load_dataset", lambda *a, **k: ds)
    monkeypatch.setattr(calibrate, "_pixel_transform", lambda: (lambda x: x))
    calibrate.draw_calibration_clips(512, 3, 5, "fake")

    drawn = {ds.clip_indices[i][0] for i in ds.drawn}
    assert embeddings.calibration_episodes(ds.clip_indices, 512) == drawn


def test_strided_frames_excludes_episodes_and_is_deterministic():
    frames = [(ep, s) for ep in range(10) for s in range(50)]
    out = embeddings.strided_frames(frames, {0, 3, 7}, 100)
    assert len(out) == len(set(out)) == 100
    assert not {ep for ep, _ in out} & {0, 3, 7}
    assert out == embeddings.strided_frames(frames, {0, 3, 7}, 100)
    assert out[0] == (1, 0) and out[-1] == (9, 49)  # spans the whole eligible range


def test_strided_frames_fails_loud_when_exclusion_leaves_too_few():
    frames = [(ep, s) for ep in range(3) for s in range(10)]
    with pytest.raises(RuntimeError, match="only 10 frames remain"):
        embeddings.strided_frames(frames, {0, 1}, 11)


def test_ensure_manifest_writes_once_then_guards(tmp_path):
    path = tmp_path / "frames.dataset.json"
    m = {"frames": [[1, 2], [3, 4]]}
    assert embeddings.ensure_manifest(path, m) == m
    assert embeddings.ensure_manifest(path, m) == m  # identical recompute passes
    with pytest.raises(RuntimeError, match="differs"):
        embeddings.ensure_manifest(path, {"frames": [[1, 2], [3, 5]]})
    assert embeddings.ensure_manifest(path, m) == m  # the stored list was not overwritten


def test_engine_configs_cover_every_encoder_plan():
    stems = [
        engine_filename("encoder", p, m or "max").removesuffix(".plan")
        for p, m in embeddings.ENGINE_CONFIGS
    ]
    assert stems == [
        "encoder.fp32", "encoder.fp16",
        "encoder.int8.max", "encoder.int8.entropy",
        "encoder.fp8.max", "encoder.fp8.entropy",
    ]


class _FakeEvalDataset:
    column_names = ["episode_idx", "step_idx", "action", "proprio", "state", "pixels"]

    def __init__(self, lengths):
        self.ep = np.concatenate([np.full(n, e) for e, n in enumerate(lengths)])
        self.step = np.concatenate([np.arange(n) for n in lengths])

    def get_col_data(self, col):
        if col == "episode_idx":
            return self.ep
        if col == "step_idx":
            return self.step
        return np.random.default_rng(0).normal(size=(len(self.ep), 2))


def test_capture_eval_selection_runs_the_vendored_selection(monkeypatch):
    """The vendored `eval_wm.run` executes up to `World.evaluate` with no env and no checkpoint,
    and hands back the 50-episode selection it would evaluate."""
    from scripts.plan import eval_wm

    ds = _FakeEvalDataset([80] * 120)
    monkeypatch.setattr(eval_wm, "get_dataset", lambda cfg, name: ds)
    call = embeddings.capture_eval_selection(["--config-dir", "conf", "+experiment=eval_lewm"])

    assert call["dataset"] is ds
    assert call["goal_offset"] == 25
    assert len(call["episodes_idx"]) == len(call["start_steps"]) == 50
    assert all(s <= 80 - 25 - 1 for s in call["start_steps"])  # goal stays in-episode
    again = embeddings.capture_eval_selection(["--config-dir", "conf", "+experiment=eval_lewm"])
    assert list(again["start_steps"]) == list(call["start_steps"])  # seeded -> fixed
