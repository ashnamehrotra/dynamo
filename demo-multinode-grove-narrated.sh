#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# ============================================================================
# Narrated Multinode Grove Demo for NVIDIA Dynamo
# ============================================================================
#
# This demo answers a single question:
#
#     "How do I deploy a model that doesn't fit on one node?"
#
# Act 1 — The Wall
#     Show the cluster, do the GPU math, prove the model can't fit on one node,
#     and explain why a vanilla Kubernetes Deployment can't help.
#
# Act 2 — LWS vs. Grove
#     Show what LeaderWorkerSet (LWS) gives you, then walk through what
#     Grove adds on top: PodCliqueSet / PodClique / PodCliqueScalingGroup,
#     hierarchical gang scheduling, and startup ordering.
#
# Act 3 — Watch It Work
#     Apply a multinode DynamoGraphDeployment with `multinode.nodeCount`,
#     then watch Grove primitives come up live: the gang stays Pending until
#     all pods can schedule together, then they all transition together.
#     Finally, port-forward to the frontend and prove it serves a request.
#
# Requirements:
#   - Kubernetes cluster with the Dynamo operator + Grove + KAI Scheduler
#   - At least 2 GPU nodes free (16 H100s total)
#   - kubectl, python3, curl available locally
#   - Model pre-cached on a PVC (default: `model-cache` in `dynamo-test`)
#
# Usage:
#   ./demo-multinode-grove-narrated.sh
#   ./demo-multinode-grove-narrated.sh --namespace dynamo-test --model nvidia/DeepSeek-V3.2-NVFP4
#   ./demo-multinode-grove-narrated.sh --no-cleanup        # keep DGD after demo
#   ./demo-multinode-grove-narrated.sh --skip-scale        # skip scale-out demo
#
set -e

# =============================================================================
# Configuration
# =============================================================================
NAMESPACE="${NAMESPACE:-dynamo-test}"
MODEL="${MODEL:-nvidia/Llama-3.1-70B-Instruct-FP8}"
BACKEND="${BACKEND:-sglang}"
DGD_NAME="${DGD_NAME:-multinode-grove-demo}"
NODES_PER_REPLICA="${NODES_PER_REPLICA:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
DO_CLEANUP=true
SKIP_SCALE=false
PORT_FORWARD_PORT=8000

# Model cache PVC
PVC_NAME="${PVC_NAME:-model-cache}"
PVC_MOUNT_PATH="${PVC_MOUNT_PATH:-/model-store}"
# Direct snapshot path inside the PVC. We pass this as --model-path to bypass
# HuggingFace name resolution entirely (the cache on this PVC is missing
# refs/main, which transformers needs in offline mode). The HF id above is
# still used as --served-model-name so the OpenAI API answers to it.
MODEL_PATH="${MODEL_PATH:-${PVC_MOUNT_PATH}/hub/models--nvidia--Llama-3.1-70B-Instruct-FP8/snapshots/07a08be3d8a8f5254c2aba375b79743bca8fd491}"

# Container image (override per cluster)
# NOTE: NGC does NOT publish a `:latest` tag — pin to a versioned release.
WORKER_IMAGE="${WORKER_IMAGE:-nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.0.1}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
WHITE='\033[1;37m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'
NAR='\033[0;36m'

# =============================================================================
# Args
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --namespace)         NAMESPACE="$2"; shift 2 ;;
        --model)             MODEL="$2"; shift 2 ;;
        --backend)           BACKEND="$2"; shift 2 ;;
        --dgd-name)          DGD_NAME="$2"; shift 2 ;;
        --nodes-per-replica) NODES_PER_REPLICA="$2"; shift 2 ;;
        --gpus-per-node)     GPUS_PER_NODE="$2"; shift 2 ;;
        --pvc-name)          PVC_NAME="$2"; shift 2 ;;
        --pvc-mount-path)    PVC_MOUNT_PATH="$2"; shift 2 ;;
        --image)             WORKER_IMAGE="$2"; shift 2 ;;
        --port)              PORT_FORWARD_PORT="$2"; shift 2 ;;
        --no-cleanup)        DO_CLEANUP=false; shift ;;
        --skip-scale)        SKIP_SCALE=true; shift ;;
        --help|-h)
            sed -n '3,40p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# =============================================================================
# Helpers
# =============================================================================
# SPEED multiplies all pause durations and the type-out delay below.
# Bump above 1.0 for slower / easier-to-follow pacing (good for recordings),
# drop below 1.0 to fly through. Default 1.4 was tuned from asciinema review
# where viewers said a few sections were hard to follow.
SPEED="${SPEED:-1.4}"

narrate()      { echo -e "\n${NAR}# $1${NC}"; }
show_command() { echo -e "${GREEN}❯${NC} $1"; }
pause()        {
    local d="${1:-2}"
    # awk handles float multiplication portably on macOS bash
    sleep "$(awk -v d="$d" -v s="$SPEED" 'BEGIN { printf "%.2f", d*s }')"
}

step_header() {
    local emoji="$1" title="$2"
    echo ""; echo ""
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "  ${MAGENTA}${BOLD}${emoji} ${title}${NC}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

type_yaml() {
    local delay
    delay=$(awk -v s="$SPEED" 'BEGIN { printf "%.3f", 0.05*s }')
    while IFS= read -r line; do echo "$line"; sleep "$delay"; done <<< "$1"
}

# =============================================================================
# Cleanup
# =============================================================================
PF_PID=""
cleanup() {
    if [[ -n "$PF_PID" ]]; then
        kill "$PF_PID" 2>/dev/null || true
        wait "$PF_PID" 2>/dev/null || true
    fi
    if [[ "$DO_CLEANUP" == true ]]; then
        echo ""
        echo -e "${DIM}# Cleaning up DGD (${DGD_NAME})...${NC}"
        kubectl delete dgd "$DGD_NAME" -n "$NAMESPACE" --ignore-not-found --wait=false 2>/dev/null || true
    else
        echo ""
        echo -e "${DIM}# DGD '${DGD_NAME}' left in place (--no-cleanup).${NC}"
    fi
}
trap cleanup EXIT

# =============================================================================
# Preflight
# =============================================================================
for tool in kubectl python3 curl; do
    if ! command -v "$tool" &>/dev/null; then
        echo -e "${RED}Error: $tool is required.${NC}"; exit 1
    fi
done

if ! kubectl get crd dynamographdeployments.nvidia.com &>/dev/null; then
    echo -e "${RED}Error: Dynamo operator CRDs not found.${NC}"; exit 1
fi

if ! kubectl get crd podcliquesets.grove.io &>/dev/null; then
    echo -e "${RED}Error: Grove CRDs not found. This demo requires Grove + KAI Scheduler.${NC}"
    echo -e "${DIM}See: https://github.com/NVIDIA/grove${NC}"
    exit 1
fi

if ! kubectl get pvc "$PVC_NAME" -n "$NAMESPACE" &>/dev/null; then
    echo -e "${YELLOW}Warning: PVC '${PVC_NAME}' not found in '${NAMESPACE}'. Workers will download from HuggingFace.${NC}"
fi

# =============================================================================
# DGD YAML — multinode disagg via Grove
# =============================================================================
TOTAL_GPUS=$((NODES_PER_REPLICA * GPUS_PER_NODE))
DGD_YAML="apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: ${DGD_NAME}
  namespace: ${NAMESPACE}
  labels:
    demo: multinode-grove
spec:
  backendFramework: ${BACKEND}
  envs:
    # Point HuggingFace cache at the PVC mount; weights are pre-downloaded
    # so workers don't have to hit the HuggingFace API.
    - name: HF_HOME
      value: ${PVC_MOUNT_PATH}
    # The Azure ND_H100_v5 pool exposes NDR InfiniBand to pods, so we let
    # NCCL auto-detect the IB fabric for cross-node tensor parallel — that's
    # what makes TP across nodes performant. Keep NCCL_DEBUG=INFO so the
    # transport (IB vs. socket) is visible in the worker logs.
    - name: NCCL_DEBUG
      value: INFO

  services:
    Frontend:
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: ${WORKER_IMAGE}
          command: [\"python3\"]
          args: [\"-m\", \"dynamo.frontend\"]
          volumeMounts:
            - name: model-cache
              mountPath: ${PVC_MOUNT_PATH}
        volumes:
          - name: model-cache
            persistentVolumeClaim:
              claimName: ${PVC_NAME}

    decode:
      componentType: worker
      subComponentType: decode
      replicas: 1
      # ── This is the Grove multinode primitive ──
      # Tells the operator to create a PodCliqueScalingGroup with
      # ${NODES_PER_REPLICA} pods that scale and gang-schedule together.
      multinode:
        nodeCount: ${NODES_PER_REPLICA}
      resources:
        limits:
          gpu: \"${GPUS_PER_NODE}\"
      extraPodSpec:
        mainContainer:
          image: ${WORKER_IMAGE}
          command: [\"python3\"]
          args:
            - \"-m\"
            - \"dynamo.${BACKEND}\"
            - \"--model-path\"
            - \"${MODEL_PATH}\"
            - \"--served-model-name\"
            - \"${MODEL}\"
            - \"--tensor-parallel-size\"
            - \"${TOTAL_GPUS}\"
            - \"--trust-remote-code\"
          volumeMounts:
            - name: model-cache
              mountPath: ${PVC_MOUNT_PATH}
        volumes:
          - name: model-cache
            persistentVolumeClaim:
              claimName: ${PVC_NAME}"

# =============================================================================
# BANNER
# =============================================================================
clear
echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}  ${BOLD}🌳  Dynamo + Grove — Multinode Inference Made Simple  🌳${NC}      ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}                                                                ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}     ${MAGENTA}One DGD. Multiple nodes. Hierarchical gang scheduling.${NC}     ${CYAN}║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
pause 3

narrate "Today's question:  what do you do when the model doesn't fit on one node?"
echo ""
echo "   We'll cover three things:"
echo ""
echo "   1. 🧱  Why a normal Kubernetes Deployment can't help."
echo "   2. 🤔  LeaderWorkerSet (LWS) — the obvious fallback — and where it stops."
echo "   3. 🌳  Grove — Dynamo's preferred solution for k8s orchestration."
echo ""
pause 5


# =============================================================================
# STEP 1: The Wall — multinode is required, here's the math
# =============================================================================
step_header "🧱" "Step 1: The Multinode Wall"

narrate "Let's look at the cluster we have to work with."
echo ""
pause 1

show_command "kubectl get nodes -o wide"
kubectl get nodes -o wide 2>&1 | head -10
echo ""
pause 3

show_command "kubectl get nodes -o json | jq '.items[] | {node: .metadata.name, gpus: .status.allocatable[\"nvidia.com/gpu\"]}'"
kubectl get nodes -o json 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
total = 0
nodes_with_gpu = 0
for n in d['items']:
    g = int(n['status'].get('allocatable',{}).get('nvidia.com/gpu','0') or 0)
    if g:
        nodes_with_gpu += 1
        total += g
        print(f'   {n[\"metadata\"][\"name\"]:<45s}  {g} × H100')
print()
print(f'   Total: {nodes_with_gpu} GPU nodes, {total} GPUs, ~{total*80} GB VRAM')
" 2>/dev/null
pause 4

echo ""
narrate "Now consider the model we want to deploy: ${MODEL}"
echo ""
echo "   Models like this:"
echo "   - DeepSeek-R1 (671B FP8)          → ~640 GB of weights"
echo "   - DeepSeek-R1 (671B BF16)         → ~1.3 TB of weights"
echo "   - Llama-4-Maverick                → multi-node by design"
echo ""
echo -e "   A single H100 node = 8 × 80 GB = ${BOLD}640 GB${NC} of VRAM."
echo ""
echo -e "   ${RED}❌  Even FP8 671B fills a node — with no room for KV cache.${NC}"
echo "      You need a second node just to serve a single user, let alone scale."
echo ""
echo -e "   ${DIM}(For this live demo we'll deploy ${MODEL} across ${NODES_PER_REPLICA} nodes${NC}"
echo -e "   ${DIM} so we don't sit through 30 minutes of CUDA-graph capture.${NC}"
echo -e "   ${DIM} The Grove story below is identical for R1 — just change the${NC}"
echo -e "   ${DIM} model path and bump nodeCount.)${NC}"
pause 5

echo ""
narrate "The naive fix is a Kubernetes Deployment with replicas: 8. That doesn't work."
echo ""
echo -e "   ${BOLD}Why a vanilla Deployment fails for multinode inference:${NC}"
echo ""
echo -e "   - ${RED}A Pod runs on ONE node.${NC} Tensor parallelism across nodes needs"
echo "     N pods that boot together, find each other, and form one logical worker."
echo -e "   - ${RED}No gang scheduling.${NC} If 7 pods schedule and the 8th pends on GPUs,"
echo "     you've burned 7 nodes' worth of GPUs holding nothing."
echo -e "   - ${RED}No startup ordering.${NC} Workers must reach the leader on a known address"
echo "     before the leader can call \`init_process_group\`."
echo -e "   - ${RED}No topology awareness.${NC} Pods can land in different racks; NCCL"
echo "     all-reduce across the wrong fabric tanks throughput."
echo ""
echo -e "   ${BOLD}You need an orchestrator that treats N pods × N nodes as one unit.${NC}"
pause 6


# =============================================================================
# STEP 2: LWS vs. Grove
# =============================================================================
step_header "🤔" "Step 2: Option A — LeaderWorkerSet (LWS)"

narrate "LeaderWorkerSet is the SIG-Apps project for this. It gets you partway there."
echo ""

LWS_SNIPPET='apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: deepseek-decode
spec:
  replicas: 1                     # one logical worker
  leaderWorkerTemplate:
    size: 2                       # 1 leader + 1 worker pod
    leaderTemplate: { ... }
    workerTemplate: { ... }       # both pods together = one TP=16 worker'
type_yaml "$LWS_SNIPPET"
echo ""
pause 3

echo -e "   ${GREEN}What LWS gives you:${NC}"
echo "   ✅  Leader+workers as one logical unit, replicated as a group"
echo "   ✅  Stable hostnames so the leader can find workers"
echo "   ✅  Basic gang behavior — the group restarts together on failure"
echo ""
echo -e "   ${YELLOW}Where LWS stops:${NC}"
echo -e "   ⚠️   ${BOLD}Multinode units may not be leader-worker shaped.${NC} LWS uses"
echo "        leader-worker groups to describe a multinode scaling unit. That works"
echo "        for true leader-worker workloads, but gets awkward for disaggregated"
echo "        serving: if prefill and decode are forced into one LWS, they can be"
echo "        scheduled together, but cannot scale independently and the API gets clunky."
echo ""
echo -e "   ⚠️   ${BOLD}Separate LWS objects lose cross-component scheduling guarantees.${NC} Modeling"
echo "        prefill and decode separately avoids that awkward fit and preserves"
echo "        independent scaling. Each LWS may still be gang-scheduled internally,"
echo "        but there is no hierarchical gang scheduling or topology awareness"
echo "        between them, so in the worst case you can get prefill with no decode."
echo ""
echo -e "   ⚠️   ${BOLD}Startup ordering is tied to creation, not startup intent.${NC}"
echo "        LWS can order pod creation, but creation order is not startup or"
echo "        readiness order. You want to gang-schedule the unit, then separately"
echo "        say whether leaders or workers should start first."
pause 7

step_header "🌳" "Step 2 (cont): Option B — Grove"

narrate "Grove is an open-source, Kubernetes-native API for describing an AI"
narrate "inference service — a modular component of Dynamo that can also run"
narrate "standalone or integrate with other inference frameworks. Where standard"
narrate "Kubernetes resources describe individual pods and services, Grove lets"
narrate "you describe an entire inference serving system as one object: routing,"
narrate "prefill, decode, leader-worker groups, startup dependencies, and scaling"
narrate "boundaries — all in a single workload specification."
narrate ""
narrate "Three hierarchical Kubernetes resources do the modeling:"
echo ""
pause 2

cat <<'EOF'
   ┌─────────────────────────────────────────────────────────────┐
   │                      PodCliqueSet  (PCS)                    │
   │     The whole disaggregated system as ONE k8s object        │
   │                                                             │
   │   ┌──────────────┐   ┌────────────────────────────────────┐ │
   │   │  PodClique   │   │   PodCliqueScalingGroup (PCSG)     │ │
   │   │  (Frontend)  │   │    ┌──────────┐    ┌──────────┐    │ │
   │   │              │   │    │PodClique │    │PodClique │    │ │
   │   │  1 pod       │   │    │ (decode  │    │ (decode  │    │ │
   │   │              │   │    │  leader) │    │  worker) │    │ │
   │   └──────────────┘   │    └──────────┘    └──────────┘    │ │
   │                      │   N pods, scheduled & scaled       │ │
   │                      │   together as a multinode unit     │ │
   │                      └────────────────────────────────────┘ │
   └─────────────────────────────────────────────────────────────┘
EOF
pause 5

echo ""
echo -e "   ${GREEN}What Grove adds on top of LWS:${NC}"
echo ""
echo -e "   ✅  ${BOLD}Multi-component in one spec${NC} — frontend + prefill + decode in one PCS."
echo "       The whole disagg system is one k8s object, not three."
echo ""
echo -e "   ✅  ${BOLD}Flexible gang scheduling${NC} — gang the entire PCS (nothing runs"
echo "       until the whole stack can schedule), or gang inside a PCSG only"
echo "       (multinode workers boot together but frontend can come up early)."
echo ""
echo -e "   ✅  ${BOLD}Hierarchical gang scheduling${NC} — the workload can express"
echo "       service-level viability, e.g. require a complete prefill AND a complete"
echo "       decode before the deployment is considered ready to serve. The scheduler"
echo "       won't admit complete-but-unbalanced capacity that can't handle an end-to-end request."
echo ""
echo -e "   ✅  ${BOLD}Declarative startup dependencies${NC} — \"decode starts after prefill"
echo "       is ready\" is a field (cliqueStartupType / startsAfter), not a"
echo "       scripted readiness probe dance or an init container."
echo ""
echo -e "   ✅  ${BOLD}Independent multi-level autoscaling${NC} — scale prefill PCSGs and"
echo "       decode PCSGs on different metrics, in the same DGD. Planner-driven"
echo "       scale-out adds a complete, gang-admitted instance — not loose pods."
echo ""
echo -e "   ✅  ${BOLD}Generates PodGang resources${NC} for a gang-aware scheduler such as"
echo "       KAI — no half-scheduled deadlocks where 7 of 8 pods run and 1 pends forever."
pause 8


# =============================================================================
# STEP 3: The DGD — Multinode in one field
# =============================================================================
step_header "📝" "Step 3: The DGD — All Of That, In One YAML"

narrate "Here's the DynamoGraphDeployment we're about to apply."
narrate "Notice how little there is: ONE field unlocks multinode."
echo ""
pause 2

echo -e "${DIM}# ${DGD_NAME}.yaml${NC}"
type_yaml "$DGD_YAML"
echo ""
pause 3

echo -e "   ${BOLD}The single field that does the work:${NC}"
echo ""
echo -e "      ${CYAN}services.decode.multinode.nodeCount: ${NODES_PER_REPLICA}${NC}"
echo ""
echo "   That tells the Dynamo operator: \"the decode worker is ${NODES_PER_REPLICA} pods"
echo "   pinned to ${NODES_PER_REPLICA} nodes, gang-scheduled, with stable hostnames\"."
echo ""
echo "   The operator translates that into a Grove PodCliqueScalingGroup."
echo "   You don't write Grove YAML directly — the DGD is your interface."
pause 5


# =============================================================================
# STEP 4: Apply
# =============================================================================
step_header "🚀" "Step 4: Apply"

narrate "One kubectl apply. Then we watch Grove do its thing."
echo ""
pause 2

show_command "kubectl apply -f ${DGD_NAME}.yaml"
echo "$DGD_YAML" | kubectl apply -f - 2>&1
pause 3


# =============================================================================
# STEP 5: Watch Grove primitives come up
# =============================================================================
step_header "👀" "Step 5: Watch the Gang Form"

narrate "Watch closely: with Grove, pods stay Pending until the WHOLE gang"
narrate "can schedule. Then they all transition together. No half-scheduled"
narrate "deadlocks where 7 nodes hold GPUs waiting for an 8th that never comes."
echo ""
narrate "We'll use \`watch -n1\` to refresh the view every second. It exits"
narrate "automatically after the live-watch window; then we wait for the model"
narrate "weights to finish loading."
echo ""
pause 3

# Live watch window: ~2 min of 1-Hz refresh shows the gang go from
# Pending -> all-admitted -> ContainerCreating -> Running together.
WATCH_DURATION=120
show_command "watch -n 1 'kubectl get podcliqueset,podclique,podcliquescalinggroup,pods -n ${NAMESPACE} -l nvidia.com/dynamo-graph-deployment-name=${DGD_NAME}'"
timeout "${WATCH_DURATION}s" watch -n 1 -t \
    "kubectl get podcliqueset,podclique,podcliquescalinggroup,pods -n ${NAMESPACE} -l nvidia.com/dynamo-graph-deployment-name=${DGD_NAME} -o wide 2>/dev/null" \
    || true

# Now wait for the DGD to reach a ready state (model weights load).
echo ""
narrate "Gang is admitted. Waiting for the model weights to finish loading..."
show_command "kubectl wait dgd ${DGD_NAME} -n ${NAMESPACE} --for=jsonpath='{.status.state}'=successful --timeout=900s"
kubectl wait dgd "$DGD_NAME" -n "$NAMESPACE" \
    --for=jsonpath='{.status.state}'=successful --timeout=900s 2>/dev/null \
    || kubectl wait dgd "$DGD_NAME" -n "$NAMESPACE" \
        --for=jsonpath='{.status.state}'=ready --timeout=60s 2>/dev/null \
    || true

dgd_ready=$(kubectl get dgd "$DGD_NAME" -n "$NAMESPACE" -o jsonpath='{.status.state}' 2>/dev/null || echo "")
echo ""
echo -e "   ${GREEN}✅ DGD is ${dgd_ready:-running} — multinode worker is online${NC}"

pause 3


# =============================================================================
# STEP 6: Topology — where did the gang land?
# =============================================================================
step_header "🗺️ " "Step 6: Topology — Where the Gang Landed"

narrate "With the gang up, let's see how Grove placed it."
echo ""
pause 2

show_command "kubectl get pods -n ${NAMESPACE} -l nvidia.com/dynamo-graph-deployment-name=${DGD_NAME} -o wide"
kubectl get pods -n "$NAMESPACE" -l "nvidia.com/dynamo-graph-deployment-name=${DGD_NAME}" -o wide 2>/dev/null
echo ""
pause 3

echo -e "   ${BOLD}What to look for:${NC}"
echo "   - Decode worker pods land on ${NODES_PER_REPLICA} different nodes"
echo "     (one PCSG member per node — that's the multinode unit)."
echo "   - Frontend lands wherever there's room — independent placement."
pause 5


# =============================================================================
# STEP 7 (optional): Scale out — hierarchical gang scheduling
# =============================================================================
if [[ "$SKIP_SCALE" != true ]]; then
    step_header "📈" "Step 7: Scale Out — Hierarchical Gang Scheduling"

    narrate "Now the payoff. We have ONE multinode replica running (${NODES_PER_REPLICA} pods,"
    narrate "${NODES_PER_REPLICA} nodes, gang-scheduled together). Let's scale to TWO replicas."
    echo ""
    narrate "What we expect Grove to do:"
    echo ""
    echo "   - Create a SECOND PodCliqueScalingGroup with ${NODES_PER_REPLICA} new pods."
    echo "   - Hold those ${NODES_PER_REPLICA} new pods Pending until ALL of them can"
    echo "     schedule together (the gang within the new PCSG)."
    echo "   - Treat each PCSG as an independent gang — the existing replica keeps"
    echo "     serving traffic regardless of what happens to the new one."
    echo ""
    echo -e "   ${BOLD}This shows Grove's layered scaling boundaries:${NC} gang within a PCSG"
echo "   (the multinode worker), and each PCSG admitted as an independent unit."
echo "   Scale-out doesn't add loose pods — it adds a complete, gang-admitted"
echo "   inference instance that's schedulable and ready to serve."
    pause 6

    echo ""
    show_command "kubectl patch dgd ${DGD_NAME} -n ${NAMESPACE} --type merge -p '{\"spec\":{\"services\":{\"decode\":{\"replicas\":2}}}}'"
    kubectl patch dgd "$DGD_NAME" -n "$NAMESPACE" --type merge \
        -p '{"spec":{"services":{"decode":{"replicas":2}}}}' 2>&1
    echo ""
    pause 2

    narrate "Watching the second gang form..."
    echo ""

    SCALE_MAX_WAIT=300
    s_elapsed=0
    s_first=true
    target_pods=$((NODES_PER_REPLICA * 2))

    while [[ $s_elapsed -lt $SCALE_MAX_WAIT ]]; do
        if [[ "$s_first" != true ]]; then
            printf '\e[H\e[2J'
        fi
        s_first=false

        echo -e "${BOLD}  📈 Step 7: Scaling decode 1 → 2 replicas${NC}"
        echo -e "  ${DIM}Every 5s · ${s_elapsed}s elapsed${NC}"
        echo ""

        show_command "kubectl get podcliquescalinggroup -n ${NAMESPACE}"
        kubectl get podcliquescalinggroup -n "$NAMESPACE" 2>/dev/null \
            | grep -E "NAME|${DGD_NAME}" || true
        echo ""

        show_command "kubectl get pods -n ${NAMESPACE} -l nvidia.com/dynamo-graph-deployment-name=${DGD_NAME} -o wide"
        decode_lines=$(kubectl get pods -n "$NAMESPACE" \
            -l "nvidia.com/dynamo-graph-deployment-name=${DGD_NAME}" \
            -o wide --no-headers 2>/dev/null | grep -i decode || echo "")
        if [[ -n "$decode_lines" ]]; then
            kubectl get pods -n "$NAMESPACE" \
                -l "nvidia.com/dynamo-graph-deployment-name=${DGD_NAME}" -o wide 2>/dev/null \
                | grep -E "NAME|decode"
        fi
        echo ""

        if [[ -n "$decode_lines" ]]; then
            total=$(echo "$decode_lines" | wc -l | tr -d ' ')
            pending=$(echo "$decode_lines" | grep -c "Pending" || true)
            running=$(echo "$decode_lines" | grep -c "Running" || true)
            ready=$(echo "$decode_lines" | awk '{print $2}' | grep -c "1/1" || true)

            echo -e "   ${CYAN}Decode pods: ${BOLD}${ready}/${total} ready  (${running} running, ${pending} pending)${NC}"

            # Group pods by PCSG. Pod names look like:
            #   <dgd>-0-decode-<replicaIdx>-decode-<role>-<hash>
            # so the replica index is the field right after "-decode-".
            echo ""
            echo -e "   ${DIM}Grouped by PCSG (replica):${NC}"
            echo "$decode_lines" | awk '{
                name=$1; node=$7
                # extract the number that follows the FIRST "-decode-"
                n=name
                sub(/.*-decode-/, "", n)   # n now starts with "<idx>-decode-..."
                split(n, a, "-")
                printf "      replica %s  →  %-55s  on %s\n", a[1], $1, node
            }' | sort

            if [[ "$pending" -gt 0 && "$running" -eq "$NODES_PER_REPLICA" ]]; then
                echo ""
                echo -e "   ${MAGENTA}▸ New gang is Pending as a unit${NC} — KAI is reserving GPUs for ALL ${NODES_PER_REPLICA} new pods together"
                echo -e "   ${MAGENTA}▸ Existing replica keeps serving${NC} — independent gangs, independent fates"
            elif [[ "$ready" == "$target_pods" ]]; then
                echo ""
                echo -e "   ${GREEN}▸ Both gangs ready${NC} — ${target_pods} pods across $((NODES_PER_REPLICA * 2)) nodes, two independent multinode workers"
                break
            fi
        fi

        sleep 5
        s_elapsed=$((s_elapsed + 5))
    done

    echo ""
    echo -e "   ${BOLD}What just happened:${NC}"
    echo -e "   - You changed ${CYAN}replicas: 1 → 2${NC}. One field."
    echo "   - Grove created a second PCSG with its own ${NODES_PER_REPLICA}-pod gang."
    echo "   - Pods of the new gang stayed Pending TOGETHER until the scheduler"
    echo "     could admit them as a unit. No half-scheduled deadlock."
    echo "   - The original replica kept serving the whole time — gangs are"
    echo "     independent, so scaling can't break what's already running."
    echo ""
    echo -e "   ${DIM}If we had asked for replicas: 3 here (24 GPUs / 3 nodes), the third${NC}"
    echo -e "   ${DIM}gang would sit Pending as a whole until 8 GPUs free up — never${NC}"
    echo -e "   ${DIM}half-scheduled, never wasting GPUs. That's the gang guarantee.${NC}"
    pause 6
fi


# =============================================================================
# STEP 8: Serve a request
# =============================================================================
step_header "💬" "Step 8: Prove It Works"

narrate "Final check — let's actually talk to the multinode worker."
echo ""
pause 2

# Wait for the frontend + at least one decode replica to be Running 1/1.
# Note: after a scale-out, the Dynamo operator's DGD.status.state may stay
# "pending" even when Grove has admitted both gangs and all pods are 1/1 Running
# (operator status lags PCSG availability). Pod readiness is the real signal
# we care about for serving traffic.
echo "   Waiting up to 2 min for serving pods to be ready..."
for i in $(seq 1 24); do
    fe_ready=$(kubectl get pods -n "$NAMESPACE" \
        -l "nvidia.com/dynamo-graph-deployment-name=${DGD_NAME},nvidia.com/dynamo-component-type=frontend" \
        --no-headers 2>/dev/null | awk '$2=="1/1" && $3=="Running"' | wc -l | tr -d ' ')
    decode_ready=$(kubectl get pods -n "$NAMESPACE" \
        -l "nvidia.com/dynamo-graph-deployment-name=${DGD_NAME}" \
        --no-headers 2>/dev/null | grep -- '-decode-' | awk '$2=="1/1" && $3=="Running"' | wc -l | tr -d ' ')
    if [[ "${fe_ready:-0}" -ge 1 && "${decode_ready:-0}" -ge 1 ]]; then
        break
    fi
    sleep 5
done

# Frontend svc is named "<dgd>-frontend" by the Dynamo operator.
# (It is NOT labeled with nvidia.com/dynamo-graph-deployment-name on this version.)
frontend_svc="${DGD_NAME}-frontend"
if ! kubectl get svc "$frontend_svc" -n "$NAMESPACE" >/dev/null 2>&1; then
    frontend_svc=$(kubectl get svc -n "$NAMESPACE" --no-headers -o name 2>/dev/null \
        | grep -i "${DGD_NAME}.*frontend" | head -1 | sed 's|service/||')
fi

if [[ -z "$frontend_svc" ]]; then
    echo -e "   ${YELLOW}Frontend service not found — skipping live request.${NC}"
else
    show_command "kubectl port-forward svc/${frontend_svc} ${PORT_FORWARD_PORT}:8000 -n ${NAMESPACE} &"
    kubectl port-forward "svc/$frontend_svc" "${PORT_FORWARD_PORT}:8000" -n "$NAMESPACE" >/dev/null 2>&1 &
    PF_PID=$!

    echo -n "   Waiting for port-forward"
    pf_ready=false
    for _i in $(seq 1 30); do
        if curl -s -o /dev/null --connect-timeout 1 "http://localhost:${PORT_FORWARD_PORT}/v1/models" 2>/dev/null; then
            pf_ready=true; break
        fi
        echo -n "."; sleep 1
    done
    echo ""

    if [[ "$pf_ready" == true ]]; then
        narrate "Sending a streaming request to the multinode decode worker..."
        echo ""
        show_command "curl http://localhost:${PORT_FORWARD_PORT}/v1/chat/completions ..."
        echo -e -n "   ${BOLD}Response:${NC} "

        curl -s -N "http://localhost:${PORT_FORWARD_PORT}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"In one enthusiastic sentence, celebrate being deployed across ${NODES_PER_REPLICA} nodes via Grove gang scheduling.\"}],\"stream\":true,\"max_tokens\":200}" 2>/dev/null \
            | python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if line.startswith('data: ') and line != 'data: [DONE]':
        try:
            d = json.loads(line[6:])
            c = d.get('choices', [{}])[0].get('delta', {}).get('content', '')
            if c: print(c, end='', flush=True)
        except: pass
print()
" 2>/dev/null || echo -e "${YELLOW}(request failed — model may still be warming)${NC}"
        echo ""
    else
        echo -e "   ${YELLOW}Port-forward failed.${NC}"
    fi

    kill "$PF_PID" 2>/dev/null || true
    wait "$PF_PID" 2>/dev/null || true
    PF_PID=""
fi
pause 3


# =============================================================================
# SUMMARY
# =============================================================================
step_header "🎉" "Recap"

echo ""
echo -e "   The problem:  ${BOLD}models that don't fit on one node.${NC}"
echo ""
echo "   The options:"
echo "     • Vanilla Deployment   →  can't span nodes as one worker"
echo "     • LWS                  →  one role, one knob, no gang scheduler"
echo -e "     • ${BOLD}Grove (via Dynamo DGD)${NC}  →  full disagg system in one spec"
echo ""
echo "   Grove gives you, in one DGD:"
echo "     ✅  PodCliqueSet           — the whole inference system as one object"
echo "     ✅  PodCliqueScalingGroup  — multinode workers as one gang"
echo "     ✅  Hierarchical gang      — gang scheduling at multiple layers, both within PCSGs and between them"
echo "     ✅  No half-scheduled      — pods of a gang admit together or wait together"
echo "     ✅  Startup ordering       — declarative, not scripted"
echo "     ✅  Independent autoscale  — prefill and decode on their own metrics"
echo ""
echo -e "   ${BOLD}You wrote one field — multinode.nodeCount: ${NODES_PER_REPLICA}.${NC}"
echo "   Grove and Dynamo did the rest."
echo ""
if [[ "$DO_CLEANUP" == true ]]; then
    echo -e "   ${DIM}(DGD will be cleaned up on exit. Use --no-cleanup to keep it.)${NC}"
else
    echo -e "   ${DIM}(DGD '${DGD_NAME}' left in place.)${NC}"
fi
echo ""
echo "   Learn more:"
echo "     • Grove:  https://github.com/NVIDIA/grove"
echo "     • Dynamo: https://github.com/ai-dynamo/dynamo"
echo "     • Multinode docs: docs/kubernetes/deployment/multinode-deployment.md"
echo ""
