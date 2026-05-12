# DGDR Recipe Lifecycle Testing — Comprehensive Coverage

End-to-end validation of the v1beta1 `DynamoGraphDeploymentRequest` (DGDR) lifecycle
across the recipe matrix from issue [#8469](https://github.com/ai-dynamo/dynamo/issues/8469).

This doc describes the test harness, the focus/skip filters used to keep runs
recipe-relevant, the recipe matrix being exercised, and where results land.

---

## 1. Test setup

### Cluster

| Item | Value |
|---|---|
| Cluster | AKS, context `demo` |
| GPUs | H100 SXM5 80GB, ~32 across 4 nodes (`aks-ndh100pool-…`) |
| Namespace | `dynamo-test` |
| Operator | `nvcr.io/nvidia/ai-dynamo/kubernetes-operator:1.0.1` in `dynamo-system` |
| Profiler image | `nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.1.0` ⚠️ |
| Model cache PVC | `dynamo-test/model-cache` (Azure Lustre, RWX, 4Ti) |
| HF token secret | `dynamo-test/hf-token-secret` (key `HF_TOKEN`) |

> ⚠️ **Image gotcha**: `dynamo-vllm-runtime:1.1.0` is missing the
> `kubernetes_asyncio` Python module and crashes the profiler. Use
> `dynamo-planner:1.1.0` for the DGDR `image:` field.

### Test framework

The Go suite lives at [`deploy/operator/test/e2e/dgdr/`](../operator/test/e2e/dgdr/) (Ginkgo v2). It contains four `Describe` blocks:

| Describe | File | Specs | Recipe-relevant? |
|---|---|---:|:---:|
| `DGDR Lifecycle` | `lifecycle_test.go` | 2 | ✅ uses `--dgdr-backend` |
| `DGDR Lifecycle Scenarios` | `dgdr_test.go` | ~10 | ❌ hardcoded backends |
| `DGDR Profiling` | `profiling_test.go` | 3 | ✅ uses `--dgdr-backend` |
| `DGDR Validation` | `validation_test.go` | ~12 | ✅ backend-agnostic |

### Why a focus filter

`Lifecycle Scenarios` contains specs like
`should complete full lifecycle with vllm backend` /
`with sglang backend` / `with trtllm backend`. They **hardcode** the backend
and ignore the `--dgdr-backend` CLI flag. Running them against e.g. the
`q235b-tllm-agg` recipe would profile/deploy Qwen3-235B-FP8 three times
(once per backend), 2 of which aren't supported model formats and would
generate noise failures.

### Focus filter applied

```text
-ginkgo.focus="DGDR Lifecycle$|DGDR Profiling|DGDR Validation"
-ginkgo.skip="DGDR Lifecycle Scenarios"
```

→ ~17 specs per recipe, each one actually testing the recipe's
  `(model, backend, mode, gpus)` combo.

### Suite extensions added for recipe testing

Six new CLI flags wired through `suite_test.go` + `helpers_test.go`:

| Flag | Purpose |
|---|---|
| `-dgdr-pvc-name` | PVC holding pre-cached model weights |
| `-dgdr-pvc-model-path` | Path within PVC to the snapshot dir |
| `-dgdr-pvc-mount-path` | Mount path inside the worker container |
| `-dgdr-total-gpus` | Override `hardware.totalGpus` |
| `-dgdr-hf-token-secret` | Inject `HF_TOKEN` env on profiler |
| `-dgdr-name-prefix` | Stable short name prefix; required because the profiler enforces `<DGD-name> + <service-name> ≤ 45 chars` (worst svc = `TRTLLMPrefillWorker`, 19 chars → DGDR name ≤ 22 chars) |

---

## 2. Recipe matrix

Source: [`deploy/_recipe-tests/recipes.tsv`](recipes.tsv).
Format: `name|model|backend|mode|gpus|arch|pvcModelPath|skipReason` (pipe-separated; bash IFS=tab silently collapses empty fields).

### Runnable on this cluster (H100, model on PVC)

| Recipe | Model | Backend | Mode | GPUs |
|---|---|---|---|---:|
| `l3-70b-vllm-agg`   | nvidia/Llama-3.1-70B-Instruct-FP8 | vllm   | agg                | 4 |
| `l3-70b-vllm-dis`   | nvidia/Llama-3.1-70B-Instruct-FP8 | vllm   | disagg-single-node | 8 |
| `q235b-tllm-agg`    | Qwen/Qwen3-235B-A22B-FP8          | trtllm | agg-hopper         | 16 |
| `q235b-tllm-dis`    | Qwen/Qwen3-235B-A22B-FP8          | trtllm | disagg-hopper      | 16 |
| `n3s-vllm-agg`      | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 | vllm   | agg    | 4 |
| `n3s-sglang-agg`    | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 | sglang | agg    | 4 |
| `n3s-tllm-dis`      | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 | trtllm | disagg | 4 |

### Skipped (and why)

| Recipe | Reason |
|---|---|
| `dsr1-sgl-d8`        | known to fail (#8571) |
| `dsr1-sgl-d16`       | uses entire cluster (32 GPUs, no headroom) |
| `dsr1-vllm-dep16`    | uses entire cluster (32 GPUs) |
| `q32b-tllm-agg/dis/router` | model not on PVC; needs HF download |
| `l33-rh-vllm-agg`    | AIC: `compressed-tensors` quant unsupported (already in #8469) |
| `gpt-oss-120b`, `deepseek-v3.2-nvfp4`, `kimi-k2.5-nvfp4`, `glm-5-nvfp4`, `qwen3-vl-30b`, `nemotron-3-nano-omni`, `deepseek-v4-flash`, `deepseek-v4-pro`, `qwen3-235b-blackwell` | Blackwell-only, not testable on H100 |
| `q32b-vllm-a100`     | A100-only |

### PVC snapshot SHAs (verified against `dynamo-test/model-cache`)

| Model | Snapshot |
|---|---|
| `nvidia/Llama-3.1-70B-Instruct-FP8` | `07a08be3d8a8f5254c2aba375b79743bca8fd491` |
| `Qwen/Qwen3-235B-A22B-FP8` | `39eb2b067ea6b8e3e1dd97d3cd0c7ffeaf3e1a35` |
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8` | `afc6e8243c7b517501c2d5b9dd1b07cf89e1b81d` |
| `deepseek-ai/DeepSeek-R1` | `56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad` |
| `RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic` | `ddb4128556dfcff99e0c41aee159ea6c3e655dcd` |
| `openai/gpt-oss-120b` | `b5c939de8f754692c1647ca79fbf85e8c1e70f8a` |
| `deepseek-ai/DeepSeek-V3.2` | `a7e62ac04ecb2c0a54d736dc46601c5606cf10a6` |
| `nvidia/DeepSeek-V3.2-NVFP4` | `7c0f62c6da1da0c81c6e097010cc55854d206812` |
| `nvidia/Kimi-K2.5-NVFP4` | `c0285e649c34d4386b01e38abca642c06cbe014e` |

---

## 3. Running the matrix

```bash
cd deploy/_recipe-tests

# Dry-run (print plan only)
./run-matrix.sh --dry-run

# Full run, sequential (wrap with caffeinate to prevent macOS sleep)
caffeinate -dimsu ./run-matrix.sh 2>&1 | tee results/run-$(date +%Y%m%d-%H%M).log

# Single recipe (substring match)
caffeinate -dimsu ./run-matrix.sh l3-70b-vllm-agg

# Resume — skip recipes that already produced a result file
caffeinate -dimsu ./run-matrix.sh --resume
```

**Wall-clock estimate**: 7 runnable recipes × ~1.5 h (1 h profiling + 30 m deploy) ≈ **10 h worst case**. Validation specs (~12) add negligible time; profiling specs add ~5 min.

---

## 4. Outputs

Per recipe, under `deploy/_recipe-tests/results/`:

| File | What it contains |
|---|---|
| `<name>.log`           | full `go test` stdout/stderr |
| `<name>.json`          | Ginkgo JSON report (machine-readable) |
| `<name>.specs.txt`     | one line per spec: `[passed/failed/skipped] <hierarchy> > <name>` + indented error message — **copy/paste source for issue bodies** |
| `<name>.profiler.log`  | last 300 lines of the failed profiler pod (when applicable) |
| `<name>.reason`        | one-line summary of the first failure (suitable for issue title) |
| `chart.md`             | aggregate table: pass / fail / skip / total per recipe + first failure |

### Chart format

```text
| Recipe | Model | Backend | Mode | GPUs | Pass | Fail | Skip | Total | First failure |
```

### Filing GitHub issues from results

```bash
# All recipes with at least one failure
grep -L '^OK' results/*.reason

# Spec-level breakdown for a recipe
cat results/q235b-tllm-agg.specs.txt
```

---

## 5. Known issues already documented

| Issue | Where | Notes |
|---|---|---|
| `kubernetes_asyncio` missing in `vllm-runtime:1.1.0` | profiler pod | workaround = use `dynamo-planner:1.1.0` |
| `compressed-tensors` quant unsupported by AIC | `l33-rh-vllm-agg` | tracked in #8469 |
| 45-char pod-name limit | profiler `_validate_dgd_service_name_lengths` | workaround = `-dgdr-name-prefix` keeps recipe names ≤ 16 chars |
| `totalGpus` budget not respected in generated DGD | `profiling_test.go:should respect totalGpus budget` | tracked in #8583 (xfail label) |
| sglang-d8 disagg DeepSeek-R1 fails | `dsr1-sgl-d8` recipe | tracked in #8571 |

---

## 6. Results

**Run**: 2026-05-11 10:19 → 2026-05-12 (overnight, ~16h wall-clock)
**Image**: `nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.1.1`
**Specs per recipe**: 17 attempted (2 lifecycle + 3 profiling + 12 validation), 12 skipped (Lifecycle Scenarios block)

### Per-recipe summary

| # | Recipe | Backend | Mode | GPUs | Pass | Fail | Skip | Outcome |
|---|---|---|---|---:|---:|---:|---:|---|
| 1 | l3-70b-vllm-agg | vllm | agg | 4 | **19** | 0 | 12 | ✅ |
| 2 | l3-70b-vllm-dis | vllm | disagg-single-node | 8 | 18 | 1 | 12 | ❌ Bug A |
| 3 | q235b-tllm-agg | trtllm | agg-hopper | 16 | 18 | 1 | 12 | ❌ Bug B |
| 4 | q235b-tllm-dis | trtllm | disagg-hopper | 16 | 18 | 1 | 12 | ❌ Bug B |
| 5 | n3s-vllm-agg | vllm | agg | 4 | 18 | 1 | 12 | ❌ Bug C |
| 6 | n3s-sglang-agg | sglang | agg | 4 | **19** | 0 | 12 | ✅ |
| 7 | n3s-tllm-dis | trtllm | disagg | 4 | 18 | 1 | 12 | ❌ Bug B |

**Score**: 2/7 ✅, 5/7 ❌. Every failure is the same Ginkgo spec — `DGDR Lifecycle > Rapid profiling > should reach Deployed with autoApply=true` — but root causes differ.

### Distinct bugs surfaced

| ID | Affected recipes | Symptom | Root cause |
|---|---|---|---|
| **A** | l3-70b-vllm-dis | vllm prefill workers `CrashLoopBackOff` | `ValueError: To serve at least one request with the model's max seq len (131072), 20.0 GiB KV cache is needed, which is larger than the available KV cache memory (1.15 GiB).` AIC-generated DGD sets `max_model_len=131072` without scaling KV memory for disagg. |
| **B** | q235b-tllm-agg, q235b-tllm-dis, n3s-tllm-dis | DGDR stuck in `Deploying`; generated `DGD` object never visible in cluster (`kubectl get dgd <name>` → NotFound) for full 90-min timeout | Operator's autoApply path silently fails to persist the generated DGD for trtllm. Note: `trtllm-agg-89201d47` *eventually* became Ready=True ~15h later, suggesting an extremely slow or retry-driven reconcile rather than a hard failure. |
| **C** | n3s-vllm-agg | vllm decode worker `CrashLoopBackOff` | `pydantic ValidationError: The repository …Nemotron-3-Super-120B-A12B-FP8 contains custom code which must be executed to correctly load the model.` `trust_remote_code=True` not set in vllm worker config (sglang variant of same model passes — see recipe 6). |

### What worked

- All **15 webhook/CRD/conversion validation specs** passed on every recipe (DGDR API surface is stable).
- All **7 profiling jobs** completed successfully (AIC profiler runs fine).
- Both **simple aggregated** topologies (`l3-70b-vllm-agg`, `n3s-sglang-agg`) reached `Deployed` and the DGD became Ready.

### Files for issue filing

For each ❌ recipe, see:
- `results/<recipe>.reason`        — one-line failure summary (issue title)
- `results/<recipe>.specs.txt`     — per-spec breakdown
- `results/<recipe>.profiler.log`  — last 300 lines of failed pod (when applicable)
- `results/<recipe>.log`           — full `go test` output

Suggested issue groupings:
- **Bug A** → 1 issue ("vllm disagg: AIC-generated DGD requests max_model_len=131072 but doesn't reserve enough KV cache")
- **Bug B** → 1 issue ("operator: trtllm DGD object not applied after autoApply=true; DGDR stuck in Deploying for hours")
- **Bug C** → 1 issue ("vllm worker missing trust_remote_code=True for Nemotron-3-Super custom code models")

---

## 7. Draft GitHub issues

Three drafts ready to copy into `gh issue create` or the web UI.

### Issue A — vllm disagg KV-cache OOM in AIC-generated DGD

**Title**: `[bug] vllm disagg: AIC-generated DGD requests max_model_len=131072 but only ~1 GiB KV cache available, prefill workers crashloop`

**Labels**: `bug`, `area/profiler`, `area/vllm`, `priority/p1`

**Body**:

> ### Summary
> When `DGDR` runs the AIC profiler for a vllm disaggregated topology, the generated `DynamoGraphDeployment` configures `max_model_len=131072` for prefill workers but does not allocate enough GPU memory for the corresponding KV cache. All prefill workers enter `CrashLoopBackOff` immediately on startup.
>
> ### Reproduction
> Recipe: `l3-70b-vllm-dis` (in `deploy/_recipe-tests/recipes.tsv`)
>
> ```yaml
> apiVersion: nvidia.com/v1beta1
> kind: DynamoGraphDeploymentRequest
> spec:
>   model: nvidia/Llama-3.1-70B-Instruct-FP8
>   backend: vllm
>   image: nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.1.1
>   autoApply: true
>   hardware: { totalGpus: 8 }
>   modelCache: { pvcName: model-cache, pvcModelPath: hub/models--nvidia--Llama-3.1-70B-Instruct-FP8/snapshots/07a08be3..., pvcMountPath: /home/dynamo/.cache/huggingface }
> ```
>
> Or via the e2e suite:
> ```bash
> cd deploy/_recipe-tests && ./run-matrix.sh l3-70b-vllm-dis
> ```
>
> ### Observed
> - Profiling job completes successfully (40s).
> - DGDR transitions to `Deploying`, generates DGD `vllm-disagg-<hash>`.
> - All four `*-vllmprefillworker-*` pods enter `CrashLoopBackOff` and never become Ready.
> - DGDR remains `Deploying` until the test's 90-min timeout.
>
> ### Error
> ```
> ValueError: To serve at least one request with the model's max seq len (131072),
> 20.0 GiB KV cache is needed, which is larger than the available KV cache memory (1.15 GiB).
> Based on the available memory, the estimated maximum model length is 7536.
> Try increasing `gpu_memory_utilization` or decreasing `max_model_len` when initializing the engine.
> ```
> Source: `vllm/v1/core/kv_cache_utils.py:644 _check_enough_kv_cache_memory`
>
> ### Expected
> AIC profiler should either:
> - Set `max_model_len` to a value that fits within the available KV cache budget, OR
> - Increase `gpu_memory_utilization` / set explicit `kv_cache_memory_bytes`, OR
> - Refuse to emit a config it knows will OOM
>
> ### Environment
> Image: `nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.1.1`. AKS cluster, H100 SXM5 80GB ×8.

---

### Issue B — Operator does not apply trtllm DGD when DGDR autoApply=true

**Title**: `[bug] operator: trtllm DGD object not applied after autoApply=true; DGDR stuck in Deploying for hours`

**Labels**: `bug`, `area/operator`, `area/trtllm`, `priority/p1`

**Body**:

> ### Summary
> When a `DynamoGraphDeploymentRequest` with `autoApply: true` and `backend: trtllm` finishes profiling, the operator records the generated `DynamoGraphDeployment` name in `status.dgdName` and the spec is visible in `status.profilingResults.selectedConfig` — but the actual `DGD` object is never created in the cluster, and the DGDR remains in `Deploying` phase indefinitely. (One DGD eventually appeared ~15h after the test gave up, suggesting an extremely slow or retry-driven reconcile.)
>
> ### Reproduction
> Affected recipes: `q235b-tllm-agg`, `q235b-tllm-dis`, `n3s-tllm-dis` (3/3 trtllm recipes; vllm/sglang DGDs are applied promptly).
>
> ```yaml
> apiVersion: nvidia.com/v1beta1
> kind: DynamoGraphDeploymentRequest
> spec:
>   model: Qwen/Qwen3-235B-A22B-FP8
>   backend: trtllm
>   image: nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.1.1
>   autoApply: true
>   hardware: { totalGpus: 16 }
>   modelCache: { pvcName: model-cache, pvcModelPath: ..., pvcMountPath: /home/dynamo/.cache/huggingface }
> ```
>
> ### Observed
> ```bash
> $ kubectl get dgdr q235b-tllm-agg-depl -o jsonpath='{.status}'
> phase: Deploying
> profilingJobName: profile-q235b-tllm-agg-depl  # Complete
> dgdName: trtllm-disagg-4fa0be16
> profilingResults.selectedConfig: { ...full DGD spec present... }
>
> $ kubectl get dgd trtllm-disagg-4fa0be16
> Error from server (NotFound)
> ```
> DGDR remained in `Deploying` for the full 90-min test timeout. ~15h later, the DGD appeared and reached `Ready=True`.
>
> ### Expected
> Once profiling completes and `selectedConfig` is set, the operator should immediately apply the DGD. Reconcile loop should not take hours.
>
> ### Hypothesis
> trtllm-specific path in the operator's autoApply reconciler. vllm and sglang DGDs apply in seconds (verified via `l3-70b-vllm-agg` and `n3s-sglang-agg` in the same matrix run).
>
> ### Environment
> Image: `nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.1.1`, operator `nvcr.io/nvidia/ai-dynamo/kubernetes-operator:1.0.1`.

---

### Issue C — vllm worker missing `trust_remote_code=True` for custom-code models

**Title**: `[bug] vllm worker fails to load Nemotron-3-Super (custom-code model) — missing trust_remote_code=True`

**Labels**: `bug`, `area/vllm`, `area/profiler`

**Body**:

> ### Summary
> When the AIC profiler emits a vllm `DynamoGraphDeployment` for `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8`, the worker container fails to start because vllm refuses to load a model that contains custom code without `trust_remote_code=True`. The sglang variant of the same model in the same matrix passed (recipe `n3s-sglang-agg` reached `Deployed`), confirming the bug is vllm-specific.
>
> ### Reproduction
> Recipe: `n3s-vllm-agg`
>
> ```yaml
> apiVersion: nvidia.com/v1beta1
> kind: DynamoGraphDeploymentRequest
> spec:
>   model: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8
>   backend: vllm
>   image: nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.1.1
>   autoApply: true
>   hardware: { totalGpus: 4 }
> ```
>
> ### Error
> ```
> pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig
>   Value error, The repository /home/dynamo/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-FP8/snapshots/...
>   contains custom code which must be executed to correctly load the model.
> ```
>
> ### Expected
> Either:
> - AIC's vllm DGD template should set `--trust-remote-code` on the worker command line by default, OR
> - The DGDR API should expose a `trustRemoteCode` field that the user can opt into (and AIC should set it for known custom-code models).
>
> ### Workaround
> Use sglang backend for this model — the matrix run shows `n3s-sglang-agg` reaches Deployed without changes.
>
> ### Environment
> Image: `nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.1.1`, vllm worker image (auto-selected by planner).
