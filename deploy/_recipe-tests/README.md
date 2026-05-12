# DGDR Recipe Lifecycle Test Harness

Runs the Go DGDR lifecycle suite (`deploy/operator/test/e2e/dgdr/lifecycle_test.go`)
against the cluster for each recipe combination from issue #8469, capturing
per-recipe pass/fail + the exact failure reason so they can be filed as issues.

## Files

- `recipes.tsv` — pipe-separated matrix: `name|model|backend|mode|gpus|arch|pvcModelPath|skipReason`
- `run-matrix.sh` — runner: loops the TSV, invokes `go test`, parses ginkgo JSON, dumps profiler logs on FAIL
- `results/` — per-recipe `.log` / `.json` / `.profiler.log` / `.reason` plus `chart.md`

## Tests run per recipe

From `lifecycle_test.go`:

| Label | Spec |
|---|---|
| `ready`  | `should reach Ready with autoApply=false`  (profiling job completes) |
| `deploy` | `should reach Deployed with autoApply=true` (DGD reaches Successful) |

## Cluster prerequisites

- Namespace `dynamo-test` exists
- PVC `model-cache` mounted in `dynamo-test` with models pre-cached (sibling of `default/model-cache`)
- Secret `hf-token-secret` with key `HF_TOKEN`
- Operator `nvcr.io/nvidia/ai-dynamo/kubernetes-operator:1.0.1` running in `dynamo-system`
- Profiler image: **must be** `nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.1.0` (vllm-runtime is missing `kubernetes_asyncio`)

## Usage

```bash
# Dry-run (just print plan)
./run-matrix.sh --dry-run

# Run all non-skipped recipes (wrap with caffeinate on macOS)
caffeinate -dimsu ./run-matrix.sh

# Run only one recipe (substring match)
./run-matrix.sh l3-70b

# Resume — skip recipes that already have a results file
caffeinate -dimsu ./run-matrix.sh --resume
```

## Filing issues from results

After a run, each FAIL has:
- `results/<name>.reason`        — one-line summary suitable for issue title/body
- `results/<name>.log`           — full `go test` output
- `results/<name>.profiler.log`  — last 300 lines of the failed profiler pod (when applicable)
- `results/chart.md`             — summary table

To list all failures:

```bash
grep -L '^OK' results/*.reason
```

## Known gotchas

- DGDR name ≤ 22 chars (profiler enforces `<dgd>+<svc> ≤ 45`; worst svc = `TRTLLMPrefillWorker` = 19 chars). Keep `name` column ≤ 16 chars.
- Each recipe's two lifecycle tests share a `--dgdr-name-prefix` so they get distinct DGDR names (`<prefix>-rea`, `<prefix>-depl`).
- Tests run sequentially. Make sure cluster has enough free GPUs for the recipe's `gpus` value before running.
