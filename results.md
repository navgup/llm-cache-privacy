# KV-Cache Timing Side-Channel — Results

All experiments measure a prefix-cache timing side channel in SGLang: a cache
**hit** (the probed prefix was recently processed and is still cached) yields a
different time-to-first-token (TTFT) than a **miss**, so an attacker who probes a
suspected prefix can tell whether it was recently used.

## Setup (common to all experiments)

- **Model / hardware:** Llama-3.1-8B-Instruct (bf16) on a single NVIDIA L4 (24 GB),
  served by **SGLang v0.5.3** in Docker (`lmsysorg/sglang:v0.5.3-cu129`),
  `--page-size 1`, flashinfer backend.
- **TTFT measurement:** client-side time to first streamed token, `max_new_tokens=1`,
  strictly **sequential** probing (concurrent probes batch and corrupt timing).
- **Cache-hit/miss labels:** a matched pair per prefix — flush → cold probe (miss)
  → warm probe (hit) — except under concurrent load, where flushing is impossible
  (see Phase 4).
- **Analysis is direction-agnostic:** the polarity of the signal (hit faster *or*
  slower) is learned from a calibration split; AUC measures *separability*, not a
  fixed direction. This turned out to matter (see Experiment 1).
- **Background corpus:** Open-Orca/OpenOrca (diverse real user prompts). LMSYS-Chat-1M
  was the intended source but its gated parquet trips a `datasets` load bug; for
  the cache-pressure/jitter measurements the corpus choice is immaterial.
- Code: `src/kvleak/`; raw data + figures in `results/`. Reproduced via
  `scripts/run_phase4_and_sweep.sh` and the per-experiment modules.

---

## Experiment 1 — Baseline attack AUC (MedQA)

**What & why.** The MVP deliverable: can TTFT alone distinguish a cached from an
uncached medical prefix? We extract 500 clinically-sensitive MedQA-USMLE prefixes
(bucketed short/medium/long, 32–256 tokens), measure a matched miss/hit TTFT pair
per prefix on an idle server, calibrate a threshold on 20%, and report AUC /
precision@80%-recall / bits-leaked on the held-out 80%. Hypothesis: AUC > 0.85.

**Data (held-out eval).**

| metric | value |
|---|---|
| AUC | **1.0000** |
| precision @ 80% recall | 1.0000 |
| bits leaked / probe | 0.936 |
| median TTFT — hit | 122.7 ms |
| median TTFT — miss | 77.9 ms |
| signal direction | **hit_slower** (hit is *slower*) |

Per-bucket separability AUC = 1.000 for all; median Δ(miss−hit) = −47.3 ms (short),
−44.7 ms (medium), −35.8 ms (long). Figures: `results/ttft_hist.png`, `results/roc.png`.

**Commentary.** The channel is **perfectly separable** (AUC 1.0) — hypothesis far
exceeded. The surprise: at these prefix lengths a cache **hit is ~45 ms *slower***
than a miss, the opposite of the literature's assumption. Mechanism: a miss runs
one efficient batched prefill of the whole prefix (~78 ms), while a hit reuses the
cached KV and only needs a 1-token "extend" through the batch-size-1 decode CUDA
graph (~123 ms), whose fixed overhead exceeds a cheap short prefill on the L4. The
attack works regardless of sign — which is exactly why the analysis is
direction-agnostic. The bucket trend (short most inverted, long least) pointed
straight at a length-dependent crossover, motivating Experiment 1b.

---

## Experiment 1b — Prefix-length crossover sweep

**Note** This is not really related to any privacy thing, just something to note b/c in milestone we assumed TTFT is always faster for cached 

**What & why.** If a hit is slower for short prefixes, where does the sign flip?
We build long medical contexts (concatenated MedQA stems), truncate each to a
length ladder (32–4096 tokens), and measure cold vs warm TTFT at each length
(18 contexts × 11 lengths, paired, bootstrap CIs).

**Data (median TTFT).**

| length (tok) | miss/cold (ms) | hit/warm (ms) | Δ = miss−hit (ms) |
|---|---|---|---|
| 32 | 74.6 | 122.6 | −47.9 |
| 128 | 86.7 | 123.9 | −37.2 |
| **256** | 123.3 | 124.1 | **−0.7 (≈ crossover)** |
| 512 | 187.1 | 124.4 | +62.7 |
| 1024 | 311.7 | 125.2 | +186.4 |
| 2048 | 640.6 | 125.5 | +515.1 |
| 4096 | 1262.5 | 127.6 | **+1135.2** |

**Crossover ≈ 259 tokens.** Figures: `results/crossover_ttft.png`, `results/crossover_delta.png`.

**Commentary.** The warm/hit TTFT is **flat (~123–128 ms) at every length** — a hit
is always a 1-token extend, independent of prefix length — while the cold/miss
prefill **scales with length**. They cross at ~259 tokens: below it the channel is
inverted (hit slower), above it the classic side channel appears and grows enormous
(a hit is **>1 second faster** at 4096 tokens). This both explains Experiment 1's
inversion and shows the leak is devastating in the long-context RAG / system-prompt
regime that real attacks target. It also corrects the milestone report's implicit
"hit is always faster" framing to "the sign is length-dependent; the channel is
detectable in both regimes."

---

## Experiment 2 (Phase 4) — Background traffic / does load mask the signal?

**What & why.** Real servers aren't idle. We rerun the probe protocol over 150
MedQA prefixes while an OpenOrca background replayer fires requests concurrently at
0.5× / 1× / 2× a base rate (~1.8 s mean inter-arrival), and ask whether the added
queueing jitter masks the timing signal. (Under load `/flush_cache` returns 400 —
the server is never idle — so we flush once per rate while idle, then use
first-send=cold-miss / second-send=warm-hit.)

**Data.**

| background rate | achieved bg req/s | attack AUC | probe p50 | probe p99 |
|---|---|---|---|---|
| 0.5× | 0.23 | **0.974** | 130.9 ms | 215.8 ms |
| 1× | 0.60 | **0.904** | 131.4 ms | 246.1 ms |
| 2× | 1.21 | **0.837** | 132.0 ms | 276.2 ms |

Median Δ(miss−hit) stayed ≈ +8.6 ms across all rates. Figures:
`results/background_auc.png`, `results/background_latency.png`.

**Commentary.** Load **partially masks but does not defeat** the channel: AUC falls
monotonically 0.974 → 0.837 as load rises, staying well above chance even at 2×.
The mechanism is **variance, not bias** — the median hit/miss gap is unchanged, but
the probe-TTFT tail inflates (p99 216 → 276 ms), spreading the distributions and
eroding separability. (Note: under concurrent load the *cold* TTFT inflates more
than the 1-token warm extend, so the observed sign flips back to "hit faster"
relative to the idle short-prefix case; separability is the quantity that
degrades, independent of sign.) Background rates are genuinely light at the
report's ~1.8 s inter-arrival, which is why even 2× barely dents the attack.

---

## Experiment 2 — Cache-size + eviction-policy sweep (attack window)

**What & why.** Reframed from "defense" to **attack-surface characterization**: at
the operating points operators actually run, how long does a sensitive prefix stay
detectable? We fix the prefix length at **512 tokens** (past the crossover, clean
"classic" signal) and measure the **attack window as background *volume*** — how
many distinct background prompts must be injected after a victim is cached before
an attacker probe no longer sees a hit. We sweep `--max-total-tokens` (cache size)
and `--radix-eviction-policy`.

**Data.**

| sweep | config | attack window (prompts) |
|---|---|---|
| cache size | 2048 tok | **7** |
| cache size | 4096 tok | **13** |
| cache size | 8192 tok | **32** |
| eviction policy | lru @ 4096 | 13 |
| eviction policy | lfu @ 4096 | 13 |

Figures: `results/eviction_cache_size.png`, `results/eviction_eviction_policy.png`.

**Commentary.** **Cache size cleanly controls the attack window** — bigger KV pool →
the sensitive prefix stays detectable through more background traffic (monotonic
7 → 13 → 32). This quantifies the "cache pressure shapes the attack surface" thesis.
The **eviction-policy sweep was a (informative) null**: `lru` and `lfu` gave
identical windows. Two reasons, both from the all-distinct background: every cache
entry has frequency 1, so LFU's frequency key is constant and degenerates to LRU's
recency tiebreak; and single-shot probing never re-accesses an entry. (Also: SGLang
v0.5.3 only ships `lru` and `lfu` — the report's `fifo/mru/filo/priority` don't
exist in this version.) The policies can only diverge when the background has a
**popularity structure** — which motivates Experiment 2b.

> Note on the report's "LRU self-sustaining" prediction: repeatedly probing the
> *same* prefix does **not** help an attacker, because the attacker's own first
> probe caches it — every later hit is self-induced and uninformative about the
> victim. Detection is inherently single-shot. The real LRU/LFU difference is about
> how background *popularity* (not attacker probing) shapes retention.

---

## Experiment 2b — Popularity structure vs eviction policy

**What & why.** The one configuration where LRU and LFU should diverge: give the
background a popularity skew (a small hot set recurs, building frequency) and ask
two questions at a fixed 512-token length, cache = 4096 tokens, per policy:
- **(a) Rare-victim window** — does a one-off sensitive prefix get evicted *faster*
  under LFU (because popular content is sticky and crowds it out)?
- **(b) Popularity oracle** — after a recency-*equalized* access schedule (all
  candidates share the same last-access; only frequency differs) plus cold filler,
  does P(cached) rise with a candidate's frequency under LFU but stay flat under LRU?
  (i.e., is an LFU cache a cross-user *popularity* oracle?)

**Data.** _(filled in after the VM run — see below.)_

**Commentary.** _(filled in after the VM run.)_
