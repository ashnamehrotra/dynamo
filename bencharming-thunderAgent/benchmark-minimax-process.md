# ThunderAgent vs KvRouter — MiniMax-M2 Benchmark Report

## 1. Goal

Test whether ThunderAgent's program-aware scheduling delivers measurable TTFT
improvements over standard KvRouter for multi-turn agent workloads on a
2-worker Kubernetes deployment.

**ThunderAgent's claims:**
1. Sticky worker pinning → warm KV cache on follow-up turns → lower TTFT
2. Pause/resume lifecycle → better memory management under contention
3. No overhead for stateless (no-session) requests

## 2. Setup

### Cluster

- AKS `h100` cluster: 4× ND H100 nodes (8× NVIDIA H100 80GB each = 32 GPUs)
- 2 nodes available for benchmark (other 2 occupied by kimi-0905)
- 16Ti Azure Lustre PVC, Dynamo operator + NATS in `dynamo-system`

### Deployment

| DGD | Architecture | Node |
|-----|-------------|------|
| `bench-minimax-ta` | Frontend (round-robin) → ThunderAgentRouter → 2× VllmWorker (TP4) | vmss000001 |
| `bench-minimax-kv` | Frontend (kv mode) → 2× VllmWorker (TP4) | vmss000003 |

Each variant gets a full 8-GPU node. No resource contention between variants.

### Model

MiniMaxAI/MiniMax-M2.7 (MoE, ~56B total / ~20B active), TP4, fp8 KV cache,
prefix caching enabled, block size 16.

### Image

`docker.io/ashnam/dynamo-vllm-runtime:thunderagent` — built from `thunderagent-demo`
branch (includes DYN_NAMESPACE fix for Kubernetes namespace mismatch).

### Session Protocol

ThunderAgent identifies sessions via `x-dynamo-session-id` HTTP header (not request body).
Also recognizes `x-claude-code-session-id`, `session-id` (Codex), `x-session-id` (OpenCode).

## 3. Tests Conducted

### Test A: Single-turn (no session headers)

60-90 independent requests, random prefixes, concurrency 8, max 256 tokens.
Verifies ThunderAgent adds no overhead for stateless traffic.

### Test B: Multi-turn (3 turns per session)

20-30 sessions, each with 3 sequential turns using `x-dynamo-session-id`.
Measures per-turn TTFT to test sticky pinning benefit.

### Test C: High-contention

30-45 sessions, concurrency 16, max 256 tokens.
Attempts to stress KV cache and trigger pause/resume.

### Test D: Shared-prefix

30 sessions with identical system prompts (to confuse KvRouter's prefix matching).

### Test E: High-pressure

50 sessions, concurrency 24, max 512 tokens, long prompts designed to exhaust GPU memory.

## 4. Results

### Run History

| Run | Date | Config | Valid? |
|-----|------|--------|--------|
| Run 1 | Jul 23 15:49 | 60 prompts, 8 conc | Partial — multi-turn failed (wrong API format) |
| Run 2 | Jul 23 15:52 | 60 prompts, 8 conc | Invalid — session-final bug killed TA Turn 2 |
| Run 3 | Jul 23 15:54 | 60 prompts, 8 conc | **Valid** — all 60/60 both sides |
| Run 4 | Jul 23 17:04 | 60 prompts + 30 session high-contention | **Valid** — all succeeded |
| Run 5 | Jul 24 19:42 | 90 prompts + 45 session high-contention | **Valid** — all succeeded |
| Run 6 | Jul 24 | Shared-prefix, 30 sessions | **Valid** — all 90/90 |
| Run 7 | Jul 24 | High-pressure, 50 sessions, 24 conc, 512 tokens | **Valid** — all 150/150 |

### Multi-turn Turn 1 TTFT (the key metric)

| Run | ThunderAgent | KvRouter | Delta | Winner |
|-----|-------------|----------|-------|--------|
| Run 3 | 296.8ms | 347.8ms | -51ms (14.7%) | **TA** |
| Run 4 | 307.8ms | 389.9ms | -82ms (21.1%) | **TA** |
| **Run 5** | **348.4ms** | **312.5ms** | **+36ms (11.5%)** | **KV** |
| Run 7 (high-pressure) | 341ms | 331ms | +10ms (3.0%) | ~same |

**Result: Inconsistent.** ThunderAgent won Turn 1 in 2 of 4 valid runs, KvRouter won
in 1, and 1 was a tie. The results are within system variance and not reproducible.

### Single-turn (all runs consistent)

Both variants perform equivalently for stateless requests across all runs.
No overhead from ThunderAgent. This claim is **confirmed**.

### High-contention / High-pressure

| Metric | Run 4 (30 sessions) | Run 5 (45 sessions) | Run 7 (50 sessions) |
|--------|---------------------|---------------------|---------------------|
| Success rate | Both 100% | Both 100% | Both 100% |
| Wall time | ~same | ~same | ~same |
| Throughput | KV slightly higher | KV slightly higher | ~same |

Neither variant failed under any load we tested. The system never reached
memory pressure with MiniMax-M2.7 fp8 on 4×H100 80GB (ample headroom).

## 5. Analysis: Why No Consistent Benefit

### Root cause: 2 workers + ample GPU memory

ThunderAgent's sticky pinning only helps when KvRouter would route to the
**wrong** worker. With only 2 workers:

1. **KvRouter's prefix matching is already near-optimal** — with unique
   conversation prefixes and only 2 choices, KvRouter rarely picks wrong.
2. **No memory pressure** — MiniMax-M2.7 with fp8 KV cache on 4×H100 (320GB)
   has enormous headroom. Pause/resume never triggers because the cache never
   fills up.
3. **Sticky pinning creates hot spots** — at high concurrency, pinning many
   sessions to the same worker creates queueing, while KvRouter can
   dynamically rebalance.

### When ThunderAgent benefits would appear (not testable with 2 nodes)

| Condition | Why it helps | Our setup |
|-----------|-------------|-----------|
| 8+ workers | More routing choices → prefix matching gets wrong more often | Only 2 workers |
| Memory exhaustion | Pause/resume frees blocks for active sessions | Never reached |
| Tool-calling gaps | Idle time between turns wastes GPU memory | No tool calls |
| Long sessions (50+ turns) | Prefix matching degrades, sticky pinning stays O(1) | Only 3 turns |

### Published results (reference)

The ThunderAgent paper reports 1.5-3.6× throughput improvement over vLLM/SGLang
on SWE-Agent, OpenHands, and ToolOrchestra benchmarks. These use:
- Real tool-calling workflows with idle gaps between LLM turns
- Multiple workers (not just 2)
- Workloads that stress GPU memory

## 6. Files

| File | Purpose |
|------|---------|
| `benchmark-minimax-ta-dgd.yaml` | DGD: Frontend + ThunderAgentRouter + 2× VllmWorker (TP4) |
| `benchmark-minimax-kv-dgd.yaml` | DGD: Frontend + 2× VllmWorker (TP4) with KV routing |
| `benchmark-minimax-download.yaml` | K8s Job to download MiniMax-M2.7 to PVC |
| `bench_minimax_ab.py` | Benchmark client (asyncio + aiohttp, measures TTFT) |
| `run_minimax_benchmark_k8s.sh` | Orchestrator: deploy → bench → cleanup |
| `benchmark-results-*.json` | Raw results per run |
| `components/src/dynamo/thunderagent_router/run_minimax_8xh100.sh` | Bare-metal reference script |

## 7. How to Reproduce

```bash
# 1. Verify cluster
kubectl config current-context  # "h100"
kubectl get dgd -n default      # bench-minimax-ta, bench-minimax-kv

# 2. Port-forward
kubectl port-forward pod/<ta-frontend> 8200:8000 &
kubectl port-forward pod/<kv-frontend> 8201:8000 &

# 3. Install deps
python3 -m venv /tmp/bench-venv && source /tmp/bench-venv/bin/activate
pip install aiohttp

# 4. Run
python3 bench_minimax_ab.py                    # default: 60 prompts, 8 conc
NUM_PROMPTS=90 CONCURRENCY=16 python3 bench_minimax_ab.py  # higher load

# 5. Deploy from scratch (if DGDs don't exist)
kubectl apply -f benchmark-minimax-download.yaml   # download model
kubectl apply -f benchmark-minimax-ta-dgd.yaml     # ThunderAgent variant
kubectl apply -f benchmark-minimax-kv-dgd.yaml     # KvRouter variant
```

## 8. Conclusions

1. **"Do no harm" confirmed** — ThunderAgent adds no overhead for stateless requests.

2. **Sticky pinning TTFT benefit: not reproducible at 2-worker scale.** Some runs
   show 15-21% Turn 1 improvement, others show KvRouter equal or better. The
   variance across runs exceeds the signal.

3. **No failures under any load tested** — neither variant degraded to the point
   of rejecting requests, even at 50 sessions / 24 concurrency / 512 tokens.

4. **The published 1.5-3.6× improvements require conditions we cannot create with
   2 workers**: real tool-calling workloads, memory exhaustion, and more routing
   choices. A proper benchmark needs 8+ workers and agent frameworks like
   SWE-Agent or OpenHands.

5. **ThunderAgent is not yet first-class in Dynamo** — requires manual DGD editing
   (add router service, change frontend mode, rename worker model). No operator
   integration, recipes, or Helm chart support.

## 9. Recommendation

To demonstrate ThunderAgent's value convincingly:
- Use 8+ workers (need 2+ ND H100 nodes per variant, or use smaller model with TP1)
- Use real agent workloads with tool calls (SWE-Agent, Claude Code, OpenHands)
- Or run the bare-metal `run_minimax_8xh100.sh` reference script directly on a
  single node with higher worker count and the AIPerf load generator
