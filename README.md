# Post-training quantisation of world models: LeWM vs DINOv3-WM

A research repository comparing the **planning-cycle latency** and **Push-T task performance** of two
self-supervised world models under reduced-precision inference, on a single NVIDIA L40S.

- **LeWM** — the reference implementation from [`stable-worldmodel`](https://github.com/galilai-group/stable-worldmodel),
  used unmodified: encoder co-trained with the predictor, a single CLS-token latent.
- **DINOv3-WM** — this project's variant of DINO-WM: the platform's `prejepa` predictor with the
  reference DINOv2 backbone swapped for a frozen **DINOv3**, predicting over the full 196-patch grid.

Both are exported PyTorch → ONNX → TensorRT and evaluated closed-loop against a CEM planner at
**FP32, FP16, FP8 (E4M3) and INT8**, with the two 8-bit precisions calibrated by both **`max`** and
**`entropy`**. Every speed number is reported with the success rate measured on the same solves.

Training, the Push-T environment, the CEM solver and the evaluation loop come from
`stable-worldmodel` and are used as they are; their entrypoints are vendored here, so both
trainings reproduce from scratch. The code written in this repository starts at a trained
checkpoint: export, calibration, quantisation, benchmarking and the statistical analysis.

## Repository map

| Path | What it is |
| --- | --- |
| `SPEC.md` | The contract: requirements, invariants, scope, ownership boundaries |
| `docs/architecture.md` | Non-obvious design rationale — silent-failure traps, platform quirks, library deviations |
| `docs/platform_api.md` | The `stable-worldmodel` API as read from the pinned installed source |
| `src/` | The owned code: adapter, export, calibration, engine runtime, shims, benchmark, statistics, report |
| `scripts/` | Vendored platform entrypoints (`train/`, `plan/`) — provenance and the recorded divergences in each one's `VENDORED.md` |
| `conf/experiment/` | Hydra overlays layered on the vendored configs |
| `tests/` | The owned code's contract tests (CPU-only; the GPU legs are exercised on the pod) |
| `reports/figs/` | The one committed display figure, re-copied from a render and never hand-edited |

## Setup

```bash
cp .env.example .env    # then fill in STABLEWM_HOME, WANDB_API_KEY, HF_TOKEN
```

`.env` is the single source for runtime configuration — never `export` these into a shell. It is
gitignored and read once on `import src`. `STABLEWM_HOME` must point at persistent storage (on
RunPod, the mounted network volume): datasets, checkpoints, TensorRT engines and every report land
under it, and it is **required** — leaving it unset raises rather than silently writing to the
ephemeral container filesystem.

```bash
./setup.sh
```

Installs uv, syncs the locked dependencies (torch cu124), then installs TensorRT, the NVIDIA
TensorRT Model Optimizer and a CUDA-12 `onnxruntime-gpu` **outside** the lock — see
`docs/architecture.md` for why that split is load-bearing. It also fetches the Push-T expert
dataset. A later bare `uv sync` prunes the out-of-lock installs; re-run `setup.sh` to restore them.

## Reproducing the study

Two stages, run in order. The first is the platform's own training entrypoints and produces the
checkpoints; the second is this repository's code.

| Stage | Runs on | Produces |
| --- | --- | --- |
| 1. Training | L40S (LeWM) / H200 SXM (DINOv3-WM) | The two checkpoints, under `$STABLEWM_HOME/checkpoints/` |
| 2. Optimisation pipeline | L40S, then anywhere | The engines, the recorded samples, and the reported artifacts |

### 1. Training the two world models

Both trainings are the platform's own entrypoints, vendored under `scripts/train/` — the source
tag, the commit, and the deliberate divergences from it are recorded in `scripts/train/VENDORED.md`.
Each track is set up by a Hydra overlay in `conf/experiment/`; the only code this repository
contributes to a training run is the one encode-path override (`src/dino_patch.py`), which the
DINOv3-WM overlay reaches through `model._target_`.

```bash
uv run python -m scripts.train.lewm --config-dir conf +experiment=lewm
```

```bash
uv run python -m scripts.train.prejepa --config-dir conf +experiment=dinov3
```

`--config-dir conf` puts this repository's overlays on Hydra's search path; `+experiment=` selects
one. Invoke them as `python -m scripts.…` and never by file path: importing the `scripts` package
is what reads `.env`, and the vendored entrypoints never import `src`. Without that hook the
platform falls back to its `~/.stable_worldmodel` default — on the pod, the ephemeral container
filesystem where a multi-hour run's checkpoints are lost on restart — and `WandbLogger` stalls on an
interactive login prompt.

| | LeWM | DINOv3-WM |
| --- | --- | --- |
| Entrypoint | `scripts/train/lewm.py` | `scripts/train/prejepa.py` |
| Vendored base config | `scripts/train/config/lewm.yaml` | `scripts/train/config/prejepa.yaml` |
| Overlay | `conf/experiment/lewm.yaml` | `conf/experiment/dinov3.yaml` |
| Encoder | scratch ViT-Tiny, co-trained under SIGReg | DINOv3 ViT-S/16 (`dinov3_small`), frozen |
| What trains | the whole model — encoder, predictor, action encoder, projectors | the predictor and the proprio/action embedders; the backbone alone is frozen |
| Encoder latent | one CLS token, `(B, 192)` | the full patch grid, `(B, 196, 384)` |
| Predictor input width | 192 — the action enters as a separate conditioning argument, not concatenated | 404 per patch = 384 visual + 10 proprio + 10 action, the extras tiled onto every patch |
| Epochs / batch size | 10 / 128 | 10 / 128 |
| Training GPU | L40S | H200 SXM |
| Checkpoints written | `checkpoints/lewm/weights_epoch_{1..10}.pt` | `checkpoints/dino/weights_epoch_{5,10}.pt` |

Training is **epoch-capped, not wall-clock-capped** (`SPEC.md` §Execution Rules). Batch size is held
equal across the tracks; training *hardware* is not, and neither carries into inference — the engine
is built and benchmarked on the same L40S from the checkpoint weights alone.

Checkpoints land under `$STABLEWM_HOME/checkpoints/<output_model_name>/`: one `weights_epoch_N.pt`
per save, plus the `config.json` the model is later rebuilt from. Both entrypoints save on an
interval — every epoch for LeWM, every fifth for DINOv3-WM — so the directory holds several `.pt`
files and `load_pretrained` cannot resolve a bare folder. That is why the eval overlays name the
file (`policy: lewm/weights_epoch_10.pt`). Both entrypoints also hand `spt.Manager` a resume path —
`$STABLEWM_HOME/checkpoints/<subdir>/<output_model_name>_weights.ckpt`, where `subdir` is the Hydra
job id for LeWM and unset for DINOv3-WM — so if that file exists a re-run **resumes** from it
instead of training from scratch.

**The DINOv3-WM track is three config deltas on the platform's DINO-WM (`prejepa`) training**, and
nothing else — same predictor, same loss, same framework:

1. `backbone.name` / `backbone.type` → `dinov3_small`. There is no `backbone=` config *group*, so
   `backbone=dinov3` would select nothing; both keys are set directly.
2. `patch_size` 14 → 16. DINOv3 is a /16 model, and this key also drives the config-derived
   `num_patches = (224 // 16)² = 196`. Left at the DINOv2 default of 14 it yields 256 — silently
   wrong, not an error.
3. `model._target_` → `src.dino_patch.DINOv3PreJEPA`, the register-aware encode override that drops
   CLS **and** the four register tokens so the grid is the true 196 patches rather than 200. It is
   baked into the saved `config.json` at train time, which is how it reaches eval and export too.

The overlay also pins the run: `dataset_name` to the `pusht_expert_train.lance` that `setup.sh`
fetches (the vendored default names a *video* dataset that is never downloaded), the batch size up
from the vendored 32 to 128 for parity with LeWM, `trainer.precision` to `bf16-mixed`,
`output_model_name` to `dino`, and `num_workers` down to 6 — under DDP each rank spawns its own
worker set and each worker fans out OpenBLAS threads, so the vendored 16 exhausts the pod's thread
cap and surfaces as a spurious `KeyboardInterrupt` during worker spawn rather than as an
out-of-resources error. The LeWM overlay is smaller: the paper's 10 epochs, where the vendored
config defaults to 100, and batch size 128. Both enable W&B into the shared project, which the
vendored `launcher/local.yaml` leaves off, so training, eval and benchmark runs land together.

DINOv3's weights are gated on Hugging Face — set `HF_TOKEN` before the first run or the download
401s. The Push-T dataset itself is public.

Before trusting a checkpoint, confirm both encode paths still produce the latents the rest of the
study is sized for:

```bash
uv run python -m scripts.verify_encode
```

It builds the two real backbones and asserts LeWM's single-token latent and DINOv3-WM's 196-patch
grid, differentially against the un-overridden 200-token path. An inactive override is silent
everywhere else.

What the epoch cap costs is measurable rather than assumed. The epoch-5 DINOv3-WM snapshot has its
own overlay and its own `dino_ep5` track, evaluated under the identical pipeline — same dataset, CEM
config, seeds and callback — without touching the headline `dino` artifacts: its engines live in
`engines/dino_ep5/`, its SR rows key under `dino_ep5`, and it is deliberately absent from the
two-track headline. It builds its own engines and depends on nothing else in the study, so it needs
only the L40S and the epoch-5 checkpoint:

```bash
uv run python -m src.export model=dino_ep5 precision=fp32
```

```bash
uv run python -m src.sr_eval --config-dir conf +experiment=eval_dino_ep5 precision=fp32
```

### 2. The optimisation pipeline

Training is not part of `src.pipeline` — nothing in it runs through an engine. Everything from the
trained checkpoint onward is one orchestrated run:

```bash
uv run python -m src.pipeline
```

It executes the stages below in order, appending to `pipeline_manifest.json`. Add
`stages=<a,b,c>` to resume a subset (always run in canonical order), `tracks=lewm` for one model,
`diagnostics=true` to include the gates and smoke checks, or `dry_run=true` to print the plan.

| Stage | Command it runs | Produces |
| --- | --- | --- |
| `archive` | in-process | Snapshots the previous run's artifacts before they are superseded |
| `export` | `src.export model= precision= [calibration_method=]` | TensorRT engines under `$STABLEWM_HOME/engines/` |
| `sr_eval` | `src.sr_eval precision= calibration_method=` | `sr.json` — success rate + the raw per-cycle latency vectors |
| `isolation` | `src.sr_eval encoder_precision= predictor_precision=` | Component-isolation SR, under composite `enc-<A>+pred-<B>` keys |
| `benchmark` | `src.study track= precision= calibration_method=` | `results.<track>.json` + `latencies.<track>.json` (raw engine-step samples) |
| `stats` | `src.stats from= out=` | `stats.json` — the confidence intervals and independence tests |
| `report` | `src.report from= calibration_method=` | The reported tables and figure |
| `gpu_telemetry` | `src.gpu_telemetry from=` | Throttle diagnostics from the logged `nvidia-smi dmon` telemetry |
| `figs` | in-process | Refreshes the committed `reports/figs/` copy |

Diagnostic stages (`pytest`, `verify_encode`, `smoke`, `fidelity`, `sr_shim`, `precision_match`,
`probe_ranges`) are off by default. `fidelity`, `sr_shim` and `precision_match` are **owner
sign-off surfaces**: they print a drift table rather than passing or failing, because the judgement
they support is not a tolerance (`SPEC.md` §Implementation Boundaries).

Only the export, evaluation and benchmark stages need the L40S. Everything downstream reads the
stored samples, so a render or a re-analysis runs anywhere:

```bash
uv run python -m src.report
```

It resolves `$STABLEWM_HOME/reports/phase5` itself, from `.env`. Do not write that path into the
command — your shell expands `$STABLEWM_HOME` before Python reads `.env`, so it would collapse to
`/reports/phase5`. Pass `from=` only to read a *different* directory, as a literal path:

```bash
uv run python -m src.report from=/mnt/archive/2026-08-07
```

### 3. Encoder-embedding capture

Runs only the encoder engines of one track over three fixed frame sets and saves every engine's
embeddings: `dataset` (10,000 frames strided over the episodes the calibration set never touched),
`eval_init` and `eval_goal` (the 50 eval episodes' initial and goal frames). Needs the built encoder
engines; run it once per track:

```bash
uv run python -m src.embeddings track=lewm
```

```bash
uv run python -m src.embeddings track=dino
```

Add `sets=eval_init,eval_goal` to capture a subset. Output lands under
`$STABLEWM_HOME/reports/embeddings/`: `frames.<set>.json` (the frame list, written once and checked on
every later run) and `<track>/<set>/encoder.<precision>[.<method>].npy` (float32, row `i` = frame
`i`) with a sidecar `.json` recording which tokens were saved and which the engine cannot provide.
A re-run fills only the missing engines.

### 4. Embedding isotropy

Off-pod, over the captured `dataset` embeddings: a float64 PCA per encoder engine, plotted as the
normalised eigenvalue spectrum `λ_k / mean(λ)` and as each non-FP32 engine's `λ_k / λ_k(FP32)`. Run
it once per track:

```bash
uv run python -m src.isotropy track=lewm
```

```bash
uv run python -m src.isotropy track=dino
```

The figures land in `$STABLEWM_HOME/reports/phase5/isotropy/` as `<name>.png` and
`<name>.fp32_ratio.png`. LeWM has one `<name>`, `lewm.dataset`. DINO has two: `dino.dataset.mean_pool`
(the patch mean) and `dino.dataset.random_patch` (one seeded random patch per frame, centred by its
position's mean); the seed and drawn positions are kept in `dino.dataset.random_patch.json`.

## Artifacts

Everything durable lives under `$STABLEWM_HOME/reports/phase5/`, never in git. The **canonical**
artifacts are the raw ones — `results.<track>.json`, `latencies.<track>.json` and `sr.json`, which
carry the samples every statistic is re-derived from. `stats.json` and the rendered outputs are
regenerable views of them, and a render never rewrites its own inputs.

| Artifact | Contents |
| --- | --- |
| `speed_table.<method>.txt` | Per-cycle p50/p95, sample size, and success rate, each absolute value with its 95% interval |
| `latency_means_table.txt` | Mean encode-step and predictor-step latency per engine call, and the per-cycle component total, cycle and residual overhead, with bootstrap intervals |
| `isolation_table.<method>.txt` | Success rate per component-isolation configuration |
| `speed_vs_sr.png` | Success rate against median per-cycle latency, faceted per model |
| `gpu_logs/` | Per-run `nvidia-smi dmon` telemetry and its throttle diagnostics |

Tables carrying an 8-bit row are **method-scoped by filename** and name their calibration method in
the body: an INT8 or FP8 engine is a per-method build, so one method's numbers are never printed
under the other's label. `latency_means_table.txt` spans both, because its configuration column
names the method on every row.

## Running the tests

```bash
uv run python -m pytest -q
```

CPU-only and hardware-free — they pin the contracts of the owned code (call-count weighting,
equal-n truncation, method scoping, artefact immutability, the statistical constructions) against
synthetic fixtures. The engine legs are exercised on the pod through the diagnostic stages.

## Licence

MIT — see `LICENSE`. That covers the code written here; the vendored platform files under
`scripts/` remain under their own upstream MIT licence, reproduced in `licenses/` and pointed to
from each directory's `VENDORED.md`.
