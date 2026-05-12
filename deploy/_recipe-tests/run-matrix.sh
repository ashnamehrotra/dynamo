#!/usr/bin/env bash
# Recipe lifecycle test matrix runner.
#
# For each recipe in recipes.tsv, runs the DGDR lifecycle test suite
# (deploy/operator/test/e2e/dgdr/lifecycle_test.go) against the cluster
# and captures pass/fail + the exact error message into per-recipe files.
#
# Outputs:
#   results/<name>.log           - full go test output
#   results/<name>.json          - ginkgo json report
#   results/<name>.profiler.log  - profiler pod tail (if profiler crashed)
#   results/<name>.reason        - one-line failure reason for issue filing
#   results/chart.md             - summary table
#
# Usage:
#   ./run-matrix.sh                      # run everything not skipped
#   ./run-matrix.sh --dry-run            # print plan only
#   ./run-matrix.sh --resume             # skip recipes that already have a .log
#   ./run-matrix.sh <substring>          # only recipes whose name contains substr
#
# Tip: wrap with `caffeinate -dimsu` on macOS to prevent sleep killing the run.

set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
TSV="$ROOT/recipes.tsv"
RESULTS="$ROOT/results"
mkdir -p "$RESULTS"

# --- Cluster + suite settings (override via env) ---
NS="${NS:-dynamo-test}"
IMAGE="${IMAGE:-nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.1.1}"
PVC_NAME="${PVC_NAME:-model-cache}"
PVC_MOUNT="${PVC_MOUNT:-/home/dynamo/.cache/huggingface}"
HF_SECRET="${HF_SECRET:-hf-token-secret}"
PROFILING_TIMEOUT="${PROFILING_TIMEOUT:-3600}"   # 1h per profiling job
DEPLOY_TIMEOUT="${DEPLOY_TIMEOUT:-1800}"         # 30m for DGD to come up

DRY=0
RESUME=0
FILTER=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    --resume)  RESUME=1 ;;
    --help|-h) sed -n '2,22p' "$0"; exit 0 ;;
    *)         FILTER="$arg" ;;
  esac
done

CHART="$RESULTS/chart.md"
{
  echo "# DGDR Recipe Lifecycle Test Matrix"
  echo
  echo "Generated: $(date)"
  echo "Namespace: \`$NS\` | Image: \`$IMAGE\` | PVC: \`$PVC_NAME\`"
  echo
  echo "Runs the **entire** \`./test/e2e/dgdr/\` suite per recipe (Lifecycle + Lifecycle Scenarios + Profiling + Validation)."
  echo "Per-recipe columns: passed / failed / skipped / total. Failure details are in \`results/<name>.reason\`."
  echo
  echo "| Recipe | Model | Backend | Mode | GPUs | Pass | Fail | Skip | Total | First failure |"
  echo "|---|---|---|---|---:|---:|---:|---:|---:|---|"
} > "$CHART"

run_recipe() {
  local name="$1" model="$2" backend="$3" mode="$4" gpus="$5" arch="$6" path="$7" skip="$8"

  if [[ -n "$FILTER" && "$name" != *"$FILTER"* ]]; then return; fi
  if [[ -n "$skip" ]]; then
    echo ">>> SKIP $name : $skip"
    printf "| %s | %s | %s | %s | %s | - | - | - | - | %s |\n" "$name" "$model" "$backend" "$mode" "$gpus" "$skip" >> "$CHART"
    return
  fi

  local log="$RESULTS/$name.log"
  local jrep="$RESULTS/$name.json"
  local plog="$RESULTS/$name.profiler.log"
  local reason_f="$RESULTS/$name.reason"

  if [[ "$RESUME" == 1 && -f "$log" && -f "$reason_f" ]]; then
    echo ">>> SKIP $name (resume: existing result)"
    local r; r=$(cat "$reason_f")
    local p=- f=- s=- t=-
    if [[ -f "$jrep" ]]; then
      p=$(jq '[.[].SpecReports[]? | select(.State=="passed")] | length' "$jrep")
      f=$(jq '[.[].SpecReports[]? | select(.State=="failed" or .State=="panicked" or .State=="interrupted" or .State=="timedout")] | length' "$jrep")
      s=$(jq '[.[].SpecReports[]? | select(.State=="skipped" or .State=="pending")] | length' "$jrep")
      t=$(jq '[.[].SpecReports[]?] | length' "$jrep")
    fi
    printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" "$name" "$model" "$backend" "$mode" "$gpus" "$p" "$f" "$s" "$t" "${r//|/\\|}" >> "$CHART"
    return
  fi

  echo ">>> RUN  $name  model=$model backend=$backend mode=$mode gpus=$gpus"
  if [[ "$DRY" == 1 ]]; then return; fi

  pushd "$REPO_ROOT/deploy/operator" > /dev/null

  # Focus = Lifecycle + Profiling + Validation (recipe-relevant).
  # Skip = Lifecycle Scenarios (hardcoded backends; would re-run same model on
  # vllm/sglang/trtllm regardless of the recipe).
  # NOTE: Ginkgo focus matches against the full hierarchical spec text, so we
  # cannot anchor with `DGDR Lifecycle$`. Use the skip pattern to exclude the
  # "Scenarios" Describe block.
  go test -v -count=1 -timeout=8h ./test/e2e/dgdr/ \
    -ginkgo.v \
    -ginkgo.timeout=7h30m \
    -ginkgo.focus="DGDR Lifecycle|DGDR Profiling|DGDR Validation" \
    -ginkgo.skip="DGDR Lifecycle Scenarios" \
    -ginkgo.json-report="$jrep" \
    -dgdr-namespace="$NS" \
    -dgdr-image="$IMAGE" \
    -dgdr-model="$model" \
    -dgdr-backend="$backend" \
    -dgdr-no-mocker \
    -dgdr-profiling-timeout="$PROFILING_TIMEOUT" \
    -dgdr-deploy-timeout="$DEPLOY_TIMEOUT" \
    -dgdr-pvc-name="$PVC_NAME" \
    -dgdr-pvc-model-path="$path" \
    -dgdr-pvc-mount-path="$PVC_MOUNT" \
    -dgdr-total-gpus="$gpus" \
    -dgdr-hf-token-secret="$HF_SECRET" \
    -dgdr-name-prefix="$name" \
    > "$log" 2>&1
  local rc=$?

  popd > /dev/null

  # Parse json report for per-test status
  local p=- f=- s=- t=-
  if [[ -f "$jrep" ]]; then
    p=$(jq '[.[].SpecReports[]? | select(.State=="passed")] | length' "$jrep")
    f=$(jq '[.[].SpecReports[]? | select(.State=="failed" or .State=="panicked" or .State=="interrupted" or .State=="timedout")] | length' "$jrep")
    s=$(jq '[.[].SpecReports[]? | select(.State=="skipped" or .State=="pending")] | length' "$jrep")
    t=$(jq '[.[].SpecReports[]?] | length' "$jrep")
    # Per-spec breakdown saved separately for issue filing
    jq -r '[.[].SpecReports[]?
      | {state: .State,
         text: ((.ContainerHierarchyTexts // []) + [.LeafNodeText // "?"] | join(" > ")),
         msg: ((.Failure.Message // "") | gsub("\n"; " ") | .[0:500])}
      ] | .[] | "[\(.state)] \(.text)" + (if .msg == "" then "" else "\n    \(.msg)" end)
    ' "$jrep" > "$RESULTS/$name.specs.txt"
  fi

  # Capture profiler pod logs on failure for issue filing
  if [[ "$rc" != 0 ]]; then
    local pp
    pp=$(kubectl -n "$NS" get pods -l "nvidia.com/dynamo-graph-deployment-request=$name" --no-headers 2>/dev/null \
         | awk '$3 ~ /Error|Failed|CrashLoopBackOff|Init/ {print $1}' | head -1)
    if [[ -z "$pp" ]]; then
      pp=$(kubectl -n "$NS" get pods --no-headers 2>/dev/null | awk -v n="$name" '$1 ~ n && $3 ~ /Error|Failed|CrashLoopBackOff/ {print $1}' | head -1)
    fi
    if [[ -n "$pp" ]]; then
      kubectl -n "$NS" logs --tail=300 "$pp" --all-containers=true > "$plog" 2>&1 || true
    fi
  fi

  # Distill a failure reason
  local reason="OK"
  if [[ "$rc" != 0 ]]; then
    if [[ -f "$jrep" ]]; then
      reason=$(jq -r '
        [.[].SpecReports[]?
          | select(.State == "failed" or .State == "panicked" or .State == "timedout")
          | ((.ContainerHierarchyTexts // []) + [.LeafNodeText // "?"] | join(" > ")) + ": " + ((.Failure.Message // .Failure.ProgressReport.Message // "no message") | gsub("\n"; " ") | .[0:300])
        ][0] // "no failed spec found"
      ' "$jrep")
    else
      reason=$(grep -m1 -E "FAIL|panic:|Error:|exit status" "$log" | head -c 300)
      reason=${reason:-"no json report; see $log"}
    fi
    if [[ -f "$plog" ]]; then
      local prof
      prof=$(grep -m1 -E "ValueError|RuntimeError|exceeds|Failed|Traceback" "$plog" | head -c 200)
      [[ -n "$prof" ]] && reason="$reason | profiler: $prof"
    fi
  fi
  echo "$reason" > "$reason_f"

  printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" \
    "$name" "$model" "$backend" "$mode" "$gpus" "$p" "$f" "$s" "$t" "${reason//|/\\|}" >> "$CHART"

  echo "<<< DONE $name : passed=$p failed=$f skipped=$s total=$t : $reason"
}

# Read the TSV (pipe-delimited, '#' = comment)
while IFS='|' read -r name model backend mode gpus arch path skip _rest; do
  [[ -z "${name:-}" || "${name:0:1}" == "#" ]] && continue
  run_recipe "$name" "$model" "$backend" "$mode" "$gpus" "$arch" "$path" "${skip:-}"
done < "$TSV"

echo
echo "=== Done. Chart: $CHART ==="
