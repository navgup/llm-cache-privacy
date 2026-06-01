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

## Architecture: SGLang in Docker, client in Python

**SGLang runs in the official `lmsysorg/sglang` Docker image on the VM.** The
attack client / data pipeline / analysis are plain Python (`kvleak`) and talk to
the server over HTTP at `127.0.0.1:30000` — they have **no GPU/torch dependency**,
so the exact same `uv sync` works on the laptop and the VM. (Docker avoids the
prebuilt-kernel mismatch on the L4's Ada/sm89 architecture and the `pip --user`
conflicts that a from-source install hits.)

A HuggingFace token (`HF_TOKEN`) is required for the gated Llama-3.1 weights +
tokenizer and gated LMSYS-Chat-1M. Copy `.env.example` to `.env` and fill it in.

## Laptop workflow (no GPU)

```bash
uv sync
uv run pytest                                  # metric + data-pipeline sanity
export HF_TOKEN=...                            # needed for the gated tokenizer
uv run python -m kvleak.data.medqa --config configs/baseline.yaml
```

## VM workflow (L4)

```bash
bash scripts/setup_vm.sh        # install Docker + nvidia runtime, pull image, uv sync
# (log out/in once so `docker` works without sudo)

# terminal 1: launch the SGLang container (binds the HF cache so weights persist)
bash scripts/launch_server.sh configs/baseline.yaml

# terminal 2: prep data, run the attack, analyze
uv run python -m kvleak.data.medqa   --config configs/baseline.yaml
uv run python -m kvleak.data.lmsys   --config configs/baseline.yaml   # background prep (Phase 4)
uv run python -m kvleak.experiments.baseline --config configs/baseline.yaml
uv run python -m kvleak.analysis.metrics     --config configs/baseline.yaml
uv run python -m kvleak.analysis.plots       --config configs/baseline.yaml
```

### Prefix-length crossover sweep

Builds long medical contexts (concatenated MedQA stems), truncates to a length
ladder, and measures cold-vs-warm TTFT at each length to find where ΔTTFT flips
sign. Needs no running server arg beyond the default container:

```bash
uv run python -m kvleak.data.long_prefixes    --config configs/baseline.yaml
uv run python -m kvleak.experiments.length_sweep --config configs/baseline.yaml
uv run python -m kvleak.analysis.crossover     --config configs/baseline.yaml
```

### Phase 4: background traffic

Reruns the probe protocol under a concurrent LMSYS replayer at 0.5x/1x/2x to test
whether load masks the signal (server running, `lmsys_background.jsonl` prepared):

```bash
uv run python -m kvleak.experiments.background_baseline      --config configs/baseline.yaml
uv run python -m kvleak.analysis.background_robustness       --config configs/baseline.yaml
```

### Experiment 2: eviction-policy + cache-size sweep

Measures the attack window (background volume to evict a cached victim prefix)
across `--radix-eviction-policy` and `--max-total-tokens`. **This experiment
restarts the container itself** (one per config), so stop any manually-launched
server first (`docker stop sglang-kvleak`):

```bash
uv run python -m kvleak.experiments.eviction_sweep --config configs/baseline.yaml
uv run python -m kvleak.analysis.eviction          --config configs/baseline.yaml
```

The image tag, container name, shm size, and HF cache mount are set under
`server:` in `configs/baseline.yaml`. Setting `docker_image: null` falls back to
a local `python -m sglang.launch_server` process (requires sglang installed in
the env — not the supported path here).

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
