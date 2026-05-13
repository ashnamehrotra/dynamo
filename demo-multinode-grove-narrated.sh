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
#     gang scheduling, startup ordering, and topology-aware placement.
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
#   ./demo-multinode-grove-narrated.sh --skip-chaos        # skip pod-kill demo
#
set -e

# =============================================================================
# Configuration
# =============================================================================
NAMESPACE="${NAMESPACE:-dynamo-test}"
MODEL="${MODEL:-nvidia/DeepSeek-V3.2-NVFP4}"
BACKEND="${BACKEND:-sglang}"
DGD_NAME="${DGD_NAME:-multinode-grove-demo}"
NODES_PER_REPLICA="${NODES_PER_REPLICA:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
DO_CLEANUP=true
SKIP_CHAOS=false
PORT_FORWARD_PORT=8000

# Model cache PVC
PVC_NAME="${PVC_NAME:-model-cache}"
PVC_MOUNT_PATH="${PVC_MOUNT_PATH:-/model-store}"

# Container image (override per cluster)
WORKER_IMAGE="${WORKER_IMAGE:-nvcr.io/nvidia/ai-dynamo/sglang-runtime:latest}"

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
        --skip-chaos)        SKIP_CHAOS=true; shift ;;
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
narrate()      { echo -e "\n${NAR}# $1${NC}"; }
show_command() { echo -e "${GREEN}❯${NC} $1"; }
pause()        { sleep "${1:-2}"; }

step_header() {
    local emoji="$1" title="$2"
    echo ""; echo ""
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "  ${MAGENTA}${BOLD}${emoji} ${title}${NC}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

type_yaml() {
    while IFS= read -r line; do echo "$line"; sleep 0.04; done <<< "$1"
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
  pvcs:
    - name: model-cache
      create: false
      existingClaimName: ${PVC_NAME}

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
            - \"${MODEL}\"
            - \"--tensor-parallel-size\"
            - \"${TOTAL_GPUS}\"
          volumeMounts:
            - name: model-cache
              mountPath: ${PVC_MOUNT_PATH}"

# =============================================================================
# BANNER
# =============================================================================
clear
echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}  ${BOLD}🌳  Dynamo + Grove — Multinode Inference Made Simple  🌳${NC}      ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}                                                                ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}     ${MAGENTA}One DGD. Multiple nodes. Gang-scheduled. Topology-aware.${NC}   ${CYAN}║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
pause 3

narrate "Today's question:  what do you do when the model doesn't fit on one node?"
echo ""
echo "   We'll cover three things:"
echo ""
echo "   1. 🧱  Why a normal Kubernetes Deployment can't help."
echo "   2. 🤔  LeaderWorkerSet (LWS) — the obvious fallback — and where it stops."
echo "   3. 🌳  Grove — what Dynamo actually uses, and why."
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
echo "   - DeepSeek-R1 (671B BF16)         → ~1.3 TB of weights"
echo "   - DeepSeek-V3.2 (671B BF16)       → ~1.3 TB"
echo "   - Llama-4-Maverick                → multi-node by design"
echo ""
echo "   A single H100 node = 8 × 80 GB = ${BOLD}640 GB${NC} of VRAM."
echo ""
echo -e "   ${RED}❌  The weights alone don't fit on one node.${NC}"
echo "      And once you add KV cache + activations + expert routing for MoE,"
echo "      even quantized variants want to span multiple nodes for throughput."
pause 5

echo ""
narrate "The naive fix is a Kubernetes Deployment with replicas: 8. That doesn't work."
echo ""
echo "   ${BOLD}Why a vanilla Deployment fails for multinode inference:${NC}"
echo ""
echo "   - ${RED}A Pod runs on ONE node.${NC} Tensor parallelism across nodes needs"
echo "     N pods that boot together, find each other, and form one logical worker."
echo "   - ${RED}No gang scheduling.${NC} If 7 pods schedule and the 8th pends on GPUs,"
echo "     you've burned 7 nodes' worth of GPUs holding nothing."
echo "   - ${RED}No startup ordering.${NC} Workers must reach the leader on a known address"
echo "     before the leader can call \`init_process_group\`."
echo "   - ${RED}No topology awareness.${NC} Pods can land in different racks; NCCL"
echo "     all-reduce across the wrong fabric tanks throughput."
echo ""
echo "   ${BOLD}You need an orchestrator that treats N pods × N nodes as one unit.${NC}"
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
echo "   ⚠️   ${BOLD}Single role per group.${NC} Frontend, prefill, decode all need separate"
echo "        LWS objects, glued together by you with services and config."
echo "   ⚠️   ${BOLD}No cross-group startup ordering.${NC} Decode might come up before the"
echo "        prefill workers it depends on are ready."
echo "   ⚠️   ${BOLD}One scaling knob per group.${NC} You can't independently scale prefill"
echo "        nodes vs. decode nodes within a single declarative spec."
echo "   ⚠️   ${BOLD}No topology hints.${NC} Pods land wherever; cross-rack traffic is on you."
echo "   ⚠️   ${BOLD}Default pod-by-pod scheduling.${NC} Without a gang scheduler underneath,"
echo "        you can still deadlock when GPUs are tight."
pause 7

step_header "🌳" "Step 2 (cont): Option B — Grove"

narrate "Grove is built specifically for disaggregated multinode inference."
narrate "It's three Kubernetes resources, layered:"
echo ""
pause 2

cat <<'EOF'
   ┌─────────────────────────────────────────────────────────────┐
   │                      PodCliqueSet  (PCS)                    │
   │     The whole disaggregated system as ONE k8s object        │
   │                                                             │
   │   ┌──────────────┐   ┌────────────────────────────────────┐ │
   │   │  PodClique   │   │   PodCliqueScalingGroup (PCSG)     │ │
   │   │  (Frontend)  │   │   ┌──────────┐    ┌──────────┐    │ │
   │   │              │   │   │PodClique │    │PodClique │    │ │
   │   │  1 pod       │   │   │ (decode  │    │ (decode  │    │ │
   │   │              │   │   │  leader) │    │  worker) │    │ │
   │   └──────────────┘   │   └──────────┘    └──────────┘    │ │
   │                      │   N pods, scheduled & scaled       │ │
   │                      │   together as a multinode unit     │ │
   │                      └────────────────────────────────────┘ │
   └─────────────────────────────────────────────────────────────┘
EOF
pause 5

echo ""
echo -e "   ${GREEN}What Grove adds on top of LWS:${NC}"
echo ""
echo "   ✅  ${BOLD}Multi-role in one spec${NC} — frontend + prefill + decode in one PCS."
echo "       The whole disagg system is one k8s object, not three."
echo ""
echo "   ✅  ${BOLD}Flexible gang scheduling${NC} — gang the entire PCS (nothing runs"
echo "       until the whole stack can schedule), or gang inside a PCSG only"
echo "       (multinode workers boot together but frontend can come up early)."
echo ""
echo "   ✅  ${BOLD}Declarative startup dependencies${NC} — \"decode starts after prefill"
echo "       is ready\" is a field, not a scripted readiness probe dance."
echo ""
echo "   ✅  ${BOLD}Independent multi-level autoscaling${NC} — scale prefill PCSGs and"
echo "       decode PCSGs on different metrics, in the same DGD."
echo ""
echo "   ✅  ${BOLD}Network topology-aware placement${NC} — pack a PCSG inside one"
echo "       NVLink domain or rack; spread replicas across domains for HA."
echo "       Surfaced in Dynamo via the DGD \`topologyConstraint\` field."
echo ""
echo "   ✅  ${BOLD}Built on a gang scheduler${NC} (KAI / podgang) — no half-scheduled"
echo "       deadlocks where 7 of 8 pods are running and 1 is pending forever."
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
pause 3

WATCH_MAX_WAIT=900
elapsed=0
_first=true

while [[ $elapsed -lt $WATCH_MAX_WAIT ]]; do
    if [[ "$_first" != true ]]; then
        printf '\e[H\e[2J'
    fi
    _first=false

    echo -e "${BOLD}  👀 Step 5: Watching Grove Primitives${NC}"
    echo -e "  ${DIM}Every 5s · ${elapsed}s elapsed${NC}"
    echo ""

    show_command "kubectl get podcliqueset,podclique,podcliquescalinggroup -n ${NAMESPACE}"
    kubectl get podcliqueset,podclique,podcliquescalinggroup -n "$NAMESPACE" 2>/dev/null \
        | grep -E "NAME|${DGD_NAME}" || true
    echo ""

    show_command "kubectl get pods -n ${NAMESPACE} -l nvidia.com/dynamo-graph-deployment-name=${DGD_NAME} -o wide"
    pod_lines=$(kubectl get pods -n "$NAMESPACE" \
        -l "nvidia.com/dynamo-graph-deployment-name=${DGD_NAME}" \
        -o wide --no-headers 2>/dev/null || echo "")
    if [[ -n "$pod_lines" ]]; then
        kubectl get pods -n "$NAMESPACE" \
            -l "nvidia.com/dynamo-graph-deployment-name=${DGD_NAME}" -o wide 2>/dev/null
    else
        echo "   (no pods yet — operator is still creating Grove resources)"
    fi
    echo ""

    # Narrate the current phase
    if [[ -z "$pod_lines" ]]; then
        echo -e "   ${MAGENTA}▸ Operator is materializing Grove resources${NC}"
    else
        total=$(echo "$pod_lines" | wc -l | tr -d ' ')
        pending=$(echo "$pod_lines" | grep -c "Pending" || true)
        creating=$(echo "$pod_lines" | grep -c "ContainerCreating\|Init:" || true)
        running=$(echo "$pod_lines" | grep -c "Running" || true)
        ready=$(echo "$pod_lines" | awk '{print $2}' | grep -c "1/1" || true)

        echo -e "   ${CYAN}Pods: ${BOLD}${ready}/${total} ready  (${running} running, ${creating} creating, ${pending} pending)${NC}"

        if [[ "$pending" == "$total" && "$total" -gt 0 ]]; then
            echo -e "   ${MAGENTA}▸ Gang-scheduling${NC} — KAI is reserving GPUs for the whole gang at once"
        elif [[ "$creating" -gt 0 ]]; then
            echo -e "   ${MAGENTA}▸ All gang members admitted${NC} — pods transitioned together (this is the gang win)"
        elif [[ "$ready" == "$total" && "$total" -gt 0 ]]; then
            echo -e "   ${GREEN}▸ Gang fully ready${NC}"
        elif [[ "$running" -gt 0 ]]; then
            echo -e "   ${MAGENTA}▸ Loading model weights from PVC${NC}"
        fi

        # Show node distribution
        echo ""
        echo -e "   ${DIM}Node placement:${NC}"
        echo "$pod_lines" | awk '{print "      "$1" → "$7}' | head -10
    fi

    # Check DGD readiness
    dgd_ready=$(kubectl get dgd "$DGD_NAME" -n "$NAMESPACE" -o jsonpath='{.status.state}' 2>/dev/null || echo "")
    if [[ "$dgd_ready" == "successful" || "$dgd_ready" == "ready" ]]; then
        echo ""
        echo -e "   ${GREEN}✅ DGD is ${dgd_ready} — multinode worker is online${NC}"
        break
    fi

    sleep 5
    elapsed=$((elapsed + 5))
done

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
echo ""
echo "   In production you'd add ${CYAN}spec.topologyConstraint${NC} to pin the PCSG"
echo "   inside one NVLink rack, and spread replicas across racks for HA."
pause 5


# =============================================================================
# STEP 7 (optional): Chaos — kill a worker, watch the gang restart together
# =============================================================================
if [[ "$SKIP_CHAOS" != true ]]; then
    step_header "💥" "Step 7: Chaos — What Happens When a Worker Dies?"

    narrate "Multinode workers are ALL-OR-NOTHING. If one pod in the gang dies,"
    narrate "the rest are useless — they're a half-broken TP=${TOTAL_GPUS} worker"
    narrate "that can't run inference. Grove enforces this: the gang restarts together."
    echo ""
    pause 3

    victim=$(kubectl get pods -n "$NAMESPACE" \
        -l "nvidia.com/dynamo-graph-deployment-name=${DGD_NAME}" \
        --no-headers 2>/dev/null | grep -i "decode" | grep "Running" | head -1 | awk '{print $1}')

    if [[ -n "$victim" ]]; then
        show_command "kubectl delete pod ${victim} -n ${NAMESPACE}"
        kubectl delete pod "$victim" -n "$NAMESPACE" --wait=false 2>&1
        echo ""
        narrate "Watch what happens to the OTHER decode pods..."
        echo ""
        pause 2

        for i in 1 2 3; do
            echo -e "${DIM}--- snapshot ${i}/3 ---${NC}"
            kubectl get pods -n "$NAMESPACE" \
                -l "nvidia.com/dynamo-graph-deployment-name=${DGD_NAME}" 2>/dev/null \
                | grep -E "NAME|decode"
            echo ""
            sleep 4
        done

        echo "   ${BOLD}Compare to LWS or a vanilla Deployment:${NC} you'd lose 1 pod and the"
        echo "   other ${NODES_PER_REPLICA} would keep running, holding GPUs but unable to serve."
        echo "   Grove's gang-restart prevents that wasted state."
        pause 5
    else
        echo -e "   ${YELLOW}(no Running decode pod found to demonstrate chaos)${NC}"
    fi
fi


# =============================================================================
# STEP 8: Serve a request
# =============================================================================
step_header "💬" "Step 8: Prove It Works"

narrate "Final check — let's actually talk to the multinode worker."
echo ""
pause 2

# Wait for DGD to be ready again post-chaos
echo "   Waiting up to 5 min for DGD to be ready..."
for i in $(seq 1 60); do
    state=$(kubectl get dgd "$DGD_NAME" -n "$NAMESPACE" -o jsonpath='{.status.state}' 2>/dev/null || echo "")
    if [[ "$state" == "successful" || "$state" == "ready" ]]; then
        break
    fi
    sleep 5
done

frontend_svc=$(kubectl get svc -n "$NAMESPACE" \
    -l "nvidia.com/dynamo-graph-deployment-name=${DGD_NAME}" --no-headers -o name 2>/dev/null \
    | grep -i frontend | head -1 | sed 's|service/||')

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
echo "     • LWS                  →  one role, one knob, no gang scheduler, no topology"
echo "     • ${BOLD}Grove (via Dynamo DGD)${NC}  →  full disagg system in one spec"
echo ""
echo "   Grove gives you, in one DGD:"
echo "     ✅  PodCliqueSet           — the whole disagg system as one object"
echo "     ✅  PodCliqueScalingGroup  — multinode workers as one gang"
echo "     ✅  Gang scheduling        — no half-scheduled GPU deadlocks"
echo "     ✅  Startup ordering       — declarative, not scripted"
echo "     ✅  Topology hints         — pack on NVLink, spread for HA"
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
