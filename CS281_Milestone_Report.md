# CS 281 Milestone Report

## Introduction and Ethical Concern

LLM serving systems like SGLang and vLLM raise throughput by reusing KV-cache entries across requests that share a prefix. That reuse creates a timing side channel: an attacker who sends a probe with the same prefix as a recent user request sees a lower time-to-first-token (TTFT) when the prefix is already cached. This leaks whether specific content was recently processed — most damaging for sensitive medical, legal, or proprietary RAG / system-prompt prefixes. We measure this leakage by comparing TTFT for cached vs. uncached MedQA prefixes under realistic LMSYS background traffic, then study how the serving system's *own performance configuration* (cache pressure, eviction policy) shapes attacker success, and what dedicated defenses actually cost.

## Experimental Setup

**Hardware / model.** SGLang serving Llama-3.1-8B-Instruct in bf16 on a single NVIDIA L4 (24 GB, Ada). The 8B choice is deliberate: the cache-hit-vs-miss TTFT gap is the prefill time of the cached prefix, which scales roughly linearly with model size and prefix length. At 8B that gap is tens of ms for 128–256 token prefixes — well above local TTFT measurement noise (single-digit ms). The L4 is both the cheap option and, counterintuitively, the *stronger* one for this experiment: its lower prefill throughput widens the absolute ΔTTFT, so the side channel is easier to detect than on an A100/H100. bf16 is both the realistic deployment default and the precision that maximizes the gap (fp8 prefills faster and shrinks it). Llama-3.1-8B in bf16 (~16 GB of weights) fits on the L4 with ~6 GB left for the KV pool, and Ada gets the flashinfer backend with no Triton fallback.

**Measurement protocol.** TTFT is measured client-side via streaming (time to first chunk) with `max_new_tokens=1`, since the first token falls out of the prefill forward pass and we don't need generation. Probes are sent strictly *sequentially*: SGLang batches concurrent requests, which mixes prefills and destroys per-request timing, so concurrency would contaminate the signal. For the hard uncached control we either flush the radix tree between trials (`/flush_cache`) or run a `--disable-radix-cache` baseline.

**Page alignment.** SGLang only caches a prefix once it fills at least one full page, so a 32-token prompt under a 64-token page never produces a hit. Because prefix length (down to 32 tokens) is one of our variables, we run `--page-size 1` (token-level matching) so the short-prefix buckets can actually register cache hits and the length sweep stays clean.

## Experiments

### 1. Baseline Attack AUC (MVP)

Hypothesis: TTFT is a reliable binary cache-hit/miss signal under realistic load, yielding AUC > 0.85. We send 500 MedQA medical probes with background traffic at the normal rate, measure TTFT per probe, calibrate a threshold, and report AUC and attacker precision at 80% recall. This is the minimum viable result.

### 2. Attack-Window Characterization (cache pressure + eviction policy)

We originally framed cache size and eviction policy as *defenses*. They aren't. An operator already sizes the cache at the Pareto-efficient point for their memory and workload, so deliberately shrinking it to reduce leakage just trades latency for every legitimate user against an incidental, non-guaranteed privacy gain — strictly dominated by partitioning and noise. It also fails on its own terms: shrinking the cache only shortens the retention window, and a fast attacker who probes immediately after the victim still gets the hit regardless of size.

So we reframe this sweep as **attack-surface characterization**, which is the genuinely useful question: at the operating points operators *actually run*, how long does a sensitive prefix stay detectable, and how does the system's existing performance configuration change that?

- **Cache pressure (size as proxy).** We vary effective cache capacity via `--mem-fraction-static` / `--max-total-tokens` and measure the *attack window* — the interval after a victim request during which the attacker still observes a hit — as a function of background load. This maps the attacker's required probing rate across the natural operating region, rather than asking anyone to de-tune.
- **Eviction policy.** SGLang ships `lru`, `lfu`, `fifo`, `mru`, `filo`, `priority` (`--radix-eviction-policy`); there is no "random," so we compare the policies operators would really deploy. The point is that a policy chosen for hit-rate reasons silently changes leakage. Predictions: under LRU a rare sensitive prefix is kept alive by the attacker's own probes (LRU resets on access), making the attack self-sustaining; under LFU a one-off sensitive prefix has low frequency and evicts fast, shrinking the window for exactly the rare prompts that matter most. We plot AUC and attack-window length against p50/p99 TTFT for each policy.

### 3. Defense Evaluation

We evaluate dedicated defenses, each measured on the privacy–utility frontier (AUC reduction vs. latency cost):

1. **Per-tenant cache partitioning** — each tenant gets a separate cache, so attacker and victim never share cached prefixes. Eliminates cross-tenant leakage; the cost is lost cross-tenant reuse.
2. **Timing noise / randomized eviction** — add Laplace noise to TTFT and randomize eviction so a fast response no longer cleanly implies a hit, giving a tunable DP-style knob.
3. **TTL / max-age eviction** — evict entries after a fixed wall-clock age regardless of access, bounding the attack window *by design* (max leakage window = TTL) rather than incidentally. This is the principled version of the cache-size idea and requires a small modification to the eviction logic. We include it precisely because it is the cache-related knob that is an actual defense, unlike shrinking total size.

## Response to Proposal Feedback

### How MedQA examples will be selected/filtered
We use the MedQA-USMLE test split (1273 questions) and select probes that maximize prefix sensitivity: questions whose first 128 tokens (the "victim prefix") contain clinically sensitive terms such as diagnosis or medications. We expect ~400–500 sensitive prompts after filtering, bucketed by length — short (32–64), medium (64–128), long (128–256) tokens — to test how prefix length affects signal strength. With `--page-size 1` the short bucket still produces hits.

### Additional sensitive datasets
We sample 200 legal-QA prompts from MultiLegalPile as a secondary sensitive domain. Scope and time constraints keep us to these two datasets.

### Exact preprocessing
1. Tokenize with Llama's tokenizer to get exact token counts.
2. Strip trailing whitespace and NFC-normalize unicode so formatting doesn't perturb token count or cache behavior.
3. Pad/truncate into the length buckets, record the exact token count, and (given `page-size 1`) confirm each prefix clears the one-page minimum.
4. For LMSYS-Chat-1M background traffic, take the first user message per conversation, tokenize, and drop messages under 32 tokens, leaving ~600k usable background prompts. We interleave these between victim and attacker requests to create realistic cache pressure.

### Train/test methodology
There is no model training — we measure whether timing reveals cache hits — so instead of a conventional ML split we calibrate a threshold on one set and evaluate it on a held-out set:
- **Calibration (20%, 100 probes):** fit the TTFT threshold separating cache-hit from cache-miss via Youden's J on the ROC curve.
- **Evaluation (80%, 400 probes):** report AUC, attacker precision at 80% recall, and bits-leaked-per-probe, blind to the calibration set.
- **Background traffic** is not split; it is drawn fresh from LMSYS-Chat-1M at each run.

Note: because the attacker's own probe caches the prefix, each prefix yields one informative measurement (the first probe); repeated probing self-pollutes the cache. Our protocol treats each probe as a single-shot measurement accordingly.

### LMSYS-Chat-1M: sampling vs. synthetic replay
We replay real LMSYS-Chat-1M traffic (no prompt reused twice) to our local SGLang server, with inter-request gaps drawn to match the observed LMSYS timing (~1.8 s average at target load). No traffic is synthesized. We test at 0.5×, 1×, and 2× the normal rate; the L4 saturates under 2×, so we report whether load masks the signal rather than treating queueing jitter as a bug.

## Timeline to Completion

- **Phase 1 (May 20–21):** Finalize MedQA prefix extraction; verify SGLang + Llama-3.1-8B on the L4 (flashinfer backend, `page-size 1`); run a sequential TTFT sanity check confirming the cached/uncached gap.
- **Phase 2 (May 22–24):** Build the data pipeline — extract MedQA sensitive prefixes, select legal-QA prompts, preprocess/tokenize, prepare LMSYS background replay.
- **Phase 3 (May 24–26):** Baseline attack — measure cached vs. uncached TTFT distributions, calibrate the threshold, compute AUC over 500 MedQA + 200 legal probes.
- **Phase 4 (May 26–28):** Integrate background traffic at 0.5×/1×/2×; rerun the baseline under load and compare signal robustness.
- **Phase 5 (May 28–30):** Attack-window characterization — vary cache pressure and eviction policy; generate AUC and attack-window vs. p50/p99 TTFT plots.
- **Phase 6 (May 30–31):** Evaluate defenses — partitioning, timing noise, TTL eviction; estimate the privacy-utility tradeoff.
- **Phase 7 (June 1–2):** Prepare poster and present.
- **Phase 8 (June 3–7):** Clean up experiments, rerun unstable trials, improve plots, finish the final report and expand the sociotechnical analysis.

## Changes in Project Direction

The attack itself is unchanged; the contribution sharpened in two ways, from grader feedback and our own systems analysis.

First, MVP scoping: Experiment 1 is the deliverable, and Experiments 2 and 3 expand it time-permitting.

Second — the main conceptual change — we reexamined the cache-size/eviction "defense" and concluded it isn't one. Cache size is already set at the performance-Pareto point, so shrinking it for privacy is dominated by partitioning and noise and doesn't even stop a fast attacker. We therefore repositioned that sweep from a defense to **attack-surface characterization** (how realistic cache pressure and the operator's existing eviction-policy choice shape the attack window), switched the eviction comparison to the policies SGLang actually ships (dropping the unsupported "random"), and added TTL/max-age eviction as the one cache-related mechanism that is a genuine, by-design defense.

The sociotechnical analysis stays scoped to three written scenarios — hospital clinical decision support, enterprise RAG agent, consumer chatbot — analyzing adversary realism, harm distribution, and gaps in provider disclosures.

## References

[1] Luo, S. et al. (2024). InputSnatch: Stealing input in LLM services via timing side-channel attacks. arXiv:2411.18191.

[2] Jin, D. et al. (2021). What disease does this patient have? A large-scale open domain question answering dataset from medical exams. Applied Sciences.

[3] Zheng, L. et al. (2023). LMSYS-Chat-1M: A large-scale real-world LLM conversation dataset. arXiv:2309.11998.

[4] Gruss, D. et al. (2016). Flush+Flush: A fast and stealthy cache attack. DIMVA 2016.

[5] Dwork, C. & Roth, A. (2014). The algorithmic foundations of differential privacy. Foundations and Trends in Theoretical Computer Science.

[6] Kwon, W. et al. (2023). Efficient memory management for large language model serving with PagedAttention. SOSP 2023.

[7] Zheng, L. et al. (2023). Efficiently programming large language model inference using SGLang. arXiv:2312.07104.
