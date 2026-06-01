# KV-Cache Timing Side-Channel Attack on SGLang (CS281)

Measures a prefix-cache timing side channel in LLM serving: SGLang reuses
KV-cache entries across requests sharing a prefix, so an attacker probing with a
recently-used prefix sees a lower time-to-first-token (TTFT) on a cache hit. This
repo implements **Experiment 1 (baseline attack)**: serve Llama-3.1-8B on an
NVIDIA L4, measure the cached-vs-uncached TTFT gap over MedQA prefixes, calibrate
a detection threshold, and report **AUC, attacker precision@80% recall, and
bits-leaked-per-probe**.

See `CS281_Milestone_Report.md` for the full project framing.

## Layout

```
src/kvleak/
  config.py          ExperimentConfig + YAML loader
  server.py          SGLangServer: launch/health/stop, flush_cache, server_info
  client.py          ProbeClient: streaming /generate TTFT measurement
  data/              tokenize_utils, medqa (probes), lmsys (background prep)
  experiments/       baseline.py  (Experiment 1 runner)
  analysis/          metrics.py, plots.py
  background.py      Phase-4 LMSYS replay skeleton (not used by the baseline)
configs/baseline.yaml   all experiment knobs
scripts/                setup_vm.sh, launch_server.sh
tests/                  Mac-runnable (no GPU) sanity tests
```

## Two environments

The SGLang/CUDA stack only installs on the GPU VM. Dependencies are split so the
analysis + data + tests run on a laptop:

- **Base** (`uv sync`): numpy/scipy/sklearn/pandas/matplotlib/datasets/transformers/httpx.
- **`gpu` extra** (`uv sync --extra gpu`, VM only): `sglang[all]` (torch + flashinfer).

A HuggingFace token (`HF_TOKEN`) is required for the gated Llama-3.1 tokenizer
and gated LMSYS-Chat-1M. Copy `.env.example` to `.env` and fill it in, or export
it in the shell.

## Laptop workflow (no GPU)

```bash
uv sync
uv run pytest                                  # metric + data-pipeline sanity
export HF_TOKEN=...                            # needed for the gated tokenizer
uv run python -m kvleak.data.medqa --config configs/baseline.yaml
```

## VM workflow (L4)

```bash
bash scripts/setup_vm.sh                        # install uv, sync gpu extra, hf login
# terminal 1: launch the server
bash scripts/launch_server.sh
# terminal 2: prep data, run the attack, analyze
uv run python -m kvleak.data.medqa   --config configs/baseline.yaml
uv run python -m kvleak.data.lmsys   --config configs/baseline.yaml   # background prep
uv run python -m kvleak.experiments.baseline --config configs/baseline.yaml
uv run python -m kvleak.analysis.metrics     --config configs/baseline.yaml
uv run python -m kvleak.analysis.plots       --config configs/baseline.yaml
```

Outputs land in `results/`: `baseline_raw.jsonl` (raw measurements),
`baseline_metrics.json` (AUC / precision@recall / bits-leaked, overall +
per-bucket), `ttft_hist.png`, `roc.png`.

### Sanity check (report Phase-1 gate)

Before the full run, confirm the signal exists: pick one long prefix, flush the
cache, measure cold TTFT, then warm TTFT — warm should be tens of ms faster. The
baseline runner's per-probe miss/hit pair is exactly this measurement at scale.

### Hard-uncached control

Set `server.disable_radix_cache: true` in the config (or pass `--disable-radix-cache`
to the launch script) to capture a reference run with prefix caching off.
