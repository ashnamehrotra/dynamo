#!/usr/bin/env bash
# ThunderAgent Demo — asciinema recording script
#
# Usage:
#   asciinema rec thunderagent-demo.cast --command="bash thunderagent-demo.sh" \
#     --title="ThunderAgent: Program-Aware Agentic Scheduling in Dynamo" \
#     --idle-time-limit=3
#
# Prerequisites:
#   - kubectl configured with cluster access (GPU node + Dynamo Operator installed)
#   - thunderagent-dgd.yaml in current directory
#   - model-cache PVC and hf-token-secret already created in dynamo-system namespace
#   - Container image already pushed: docker.io/ashnam/dynamo-vllm-runtime:thunderagent
#
# After recording:
#   asciinema upload thunderagent-demo.cast

set -e

# --- Helpers ---
DELAY=${DEMO_DELAY:-2}
TYPE_DELAY=${DEMO_TYPE_DELAY:-0.03}

# Colors
CYAN='\033[1;36m'
GREEN='\033[1;32m'
YELLOW='\033[0;33m'
WHITE='\033[1;37m'
DIM='\033[0;90m'
RESET='\033[0m'

prompt() {
  echo ""
  echo -e "${CYAN}# $1${RESET}"
  sleep 1
}

note() {
  # Educational note — displayed in dim white
  echo -e "${DIM}$1${RESET}"
  sleep "${2:-2}"
}

run() {
  # Simulate typing the command
  echo -ne "${GREEN}❯ ${RESET}"
  echo "$1" | while IFS= read -r -n1 char; do
    echo -n "$char"
    sleep "$TYPE_DELAY"
  done
  echo ""
  sleep 0.5
  # Execute it
  eval "$1"
  sleep "$DELAY"
}

# ═══════════════════════════════════════════════════════════════════════════════
# INTRO: What is ThunderAgent?
# (Corresponds to doc Parts 0-1: The problem and the research)
# ═══════════════════════════════════════════════════════════════════════════════
clear
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ThunderAgent — Program-Aware Agentic Scheduling in Dynamo ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
sleep 2

echo -e "${WHITE}THE PROBLEM:${RESET}"
echo ""
echo "  Agentic LLM workloads (Claude Code, SWE-Agent, OpenHands) work like:"
echo ""
echo "    LLM call → tool call (pytest, docker, curl) → LLM call → tool call → ..."
echo ""
echo "  Between LLM turns, the agent runs tools (non-GPU work), but its"
echo "  KV cache stays resident on the GPU — wasting memory."
echo ""
echo "  Standard routers see each turn independently. They don't know that"
echo "  turns belong to the same agent, so they can't:"
echo "    • Pin an agent to one worker (for KV cache reuse)"
echo "    • Pause idle agents at tool boundaries to free GPU memory"
echo "    • Resume them when capacity is available"
echo ""
sleep 8

echo -e "${WHITE}THE SOLUTION — ThunderAgent:${RESET}"
echo ""
echo "  ThunderAgent groups all turns from one agent into a 'Program' and"
echo "  schedules at the program level. Each program has:"
echo ""
echo "    • Status:    REASONING (LLM active) or ACTING (running tools)"
echo "    • Lifecycle: ACTIVE (can submit) or PAUSED (held until memory frees)"
echo ""
echo "  A background scheduler ticks every 5s with three phases:"
echo "    1. Soft-demote: priority penalty in a warning band"
echo "    2. Greedy resume: BFD bin-pack smallest paused programs"
echo "    3. Pause until safe: park idle programs at tool boundaries"
echo ""
echo "  Paper: 1.5–3.6× throughput improvement (ICML 2026 Spotlight)"
echo ""
sleep 8

# ═══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE: Where ThunderAgent sits in Dynamo
# (Corresponds to doc Part 2: Architecture)
# ═══════════════════════════════════════════════════════════════════════════════
clear
echo ""
echo -e "${WHITE}ARCHITECTURE — Where ThunderAgent fits in Dynamo:${RESET}"
echo ""
echo "  ┌───────────────────────────────────────────────────┐"
echo "  │ dynamo.frontend  (HTTP + OpenAI-compatible API)   │"
echo "  │ Receives chat completions with session headers    │"
echo "  └───────────────────────┬───────────────────────────┘"
echo "                          │  x-dynamo-session-id"
echo "                          ▼"
echo "  ┌───────────────────────────────────────────────────┐"
echo "  │ dynamo.thunderagent_router  ← THIS SERVICE        │"
echo "  │                                                   │"
echo "  │  • ProgramTable: session_id → program state       │"
echo "  │  • Admission gate: pause/resume at tool boundary  │"
echo "  │  • Sticky worker pin per program                  │"
echo "  │  • Wraps KvRouter for actual routing decisions    │"
echo "  └───────────────────────┬───────────────────────────┘"
echo "                          │  dispatches to best worker"
echo "                          ▼"
echo "  ┌───────────────────────────────────────────────────┐"
echo "  │ dynamo.vllm  (GPU workers, publish KV events)     │"
echo "  └───────────────────────────────────────────────────┘"
echo ""
echo "  Key design: ThunderAgent owns a KvRouter in-process (no extra hop)."
echo "  Requests without session headers pass through unchanged."
echo ""
sleep 10

# ═══════════════════════════════════════════════════════════════════════════════
# HOW SESSIONS WORK
# (Corresponds to doc Part 0: How does Dynamo know it's agentic?)
# ═══════════════════════════════════════════════════════════════════════════════
clear
echo ""
echo -e "${WHITE}HOW SESSION TRACKING WORKS:${RESET}"
echo ""
echo "  Dynamo doesn't auto-detect agentic workloads. It's opt-in via headers:"
echo ""
echo "  ┌─────────────────────────┬──────────────────────────────────┐"
echo "  │ Agent Framework         │ Session Header                   │"
echo "  ├─────────────────────────┼──────────────────────────────────┤"
echo "  │ Generic / any agent     │ x-dynamo-session-id              │"
echo "  │ Claude Code             │ x-claude-code-session-id         │"
echo "  │ Codex                   │ session-id                       │"
echo "  │ OpenCode                │ x-session-id                     │"
echo "  └─────────────────────────┴──────────────────────────────────┘"
echo ""
echo "  The frontend normalizes these into agent_context.session_id."
echo "  ThunderAgent uses this to create/lookup a Program."
echo ""
echo "  To end a session: send x-dynamo-session-final: true"
echo "  This releases the program from the table and frees resources."
echo ""
sleep 8

# ═══════════════════════════════════════════════════════════════════════════════
# LIVE DEMO: Deploying and using ThunderAgent
# (Corresponds to doc Part 5: Demo steps)
# ═══════════════════════════════════════════════════════════════════════════════
clear
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  LIVE DEMO — ThunderAgent on Kubernetes                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Deployment: Frontend → ThunderAgentRouter → vLLM Worker"
echo "  Model: Qwen/Qwen3-0.6B on 1x A100 GPU"
echo "  Deployed via Dynamo Graph Deployment (DGD) on the Dynamo Operator"
echo ""
sleep 4

# --- Step 1: Deploy the DGD ---
prompt "Step 1: Deploy ThunderAgent via DGD (Dynamo Graph Deployment)"
note "  The DGD defines three services: Frontend, ThunderAgentRouter, VllmDecodeWorker."
note "  The Dynamo Operator creates pods, configures networking, and mounts the model PVC." 1
run "kubectl apply -f thunderagent-dgd.yaml"

prompt "Waiting for all pods to be Running..."
note "  The vLLM worker takes ~30-60s to load model weights into GPU memory."
echo ""
# Wait loop — show pod status every 10s until all are running
for i in $(seq 1 30); do
  READY=$(kubectl get pods -n dynamo-system -l "nvidia.com/dynamo-graph-deployment-name=thunderagent-demo" --no-headers 2>/dev/null | grep -v Completed | grep -c "1/1" || true)
  READY=${READY:-0}
  echo -ne "\r  Pods running: ${READY}/3"
  if [[ "$READY" -ge 3 ]]; then
    echo -e "  ${GREEN}✓ All pods ready${RESET}"
    break
  fi
  sleep 10
done
echo ""
sleep 2

run "kubectl get pods -n dynamo-system | grep thunderagent-demo"

# --- Step 2: Port-forward and verify ---
prompt "Step 2: Port-forward to the frontend and verify model is registered"
note "  The frontend exposes an OpenAI-compatible API on port 8000."
run "kubectl port-forward -n dynamo-system svc/thunderagent-demo-frontend 8000:8000 &"
echo ""
note "  Waiting for frontend to be ready..." 1
sleep 5

run "curl -s http://localhost:8000/v1/models | python3 -m json.tool"

# Resolve router pod name (used for log checks throughout)
ROUTER_POD=$(kubectl get pods -n dynamo-system -l "nvidia.com/dynamo-graph-deployment-name=thunderagent-demo,nvidia.com/dynamo-component=ThunderAgentRouter" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

# --- Step 3: Verify ThunderAgent scheduler started ---
prompt "Step 3: Verify ThunderAgent scheduler is running"
note "  Check router logs for scheduler initialization."
run "kubectl logs -n dynamo-system $ROUTER_POD --tail=30 2>&1 | grep -i 'scheduler\|started\|routing\|model\|KV' | tail -8"
# --- Step 4: Request without session ---
prompt "Step 4: Request WITHOUT session header — plain pass-through"
note "  Without x-dynamo-session-id, ThunderAgent has no program to track."
note "  It forwards directly via the inner KvRouter — no lifecycle, no pinning." 1
run "curl -s http://localhost:8000/v1/chat/completions \\
  -H 'Content-Type: application/json' \\
  -d '{\"model\": \"Qwen/Qwen3-0.6B\", \"messages\": [{\"role\": \"user\", \"content\": \"What is 2+2?\"}], \"max_tokens\": 50}' | python3 -m json.tool"

echo ""
echo -e "${YELLOW}→ No program created. This is how non-agentic requests work — fully backwards compatible.${RESET}"
sleep 3

# --- Step 5: Request with session (creates a program) ---
prompt "Step 5: Request WITH x-dynamo-session-id — creates a Program"
note "  Adding the session header tells ThunderAgent this is an agent turn."
note "  It creates a Program entry, sets status=REASONING, and pins to a worker." 1
run "curl -s http://localhost:8000/v1/chat/completions \\
  -H 'Content-Type: application/json' \\
  -H 'x-dynamo-session-id: demo-agent-1' \\
  -d '{\"model\": \"Qwen/Qwen3-0.6B\", \"messages\": [{\"role\": \"user\", \"content\": \"Write a Python sort function\"}], \"max_tokens\": 100}' | python3 -m json.tool"

echo ""
echo -e "${YELLOW}→ Program 'demo-agent-1' created. Status: REASONING → ACTING (after response).${RESET}"
echo -e "${YELLOW}  Worker assigned — all future turns in this session go to the same GPU.${RESET}"
sleep 3

prompt "Check ThunderAgent status — program lifecycle after first turn"
note "  The router logs show the program state machine in action."
run "kubectl logs -n dynamo-system $ROUTER_POD --tail=50 2>&1 | grep -i 'program\|session\|worker.*assign\|REASONING\|ACTING\|created\|demo-agent' | tail -8"

# --- Step 6: Second turn (same session → sticky worker) ---
prompt "Step 6: Second turn — same session → sticky worker pinning"
note "  In a real agent, this is the next LLM call after a tool ran (e.g., pytest)."
note "  ThunderAgent routes it to the SAME worker so the KV cache is already warm." 1
run "curl -s http://localhost:8000/v1/chat/completions \\
  -H 'Content-Type: application/json' \\
  -H 'x-dynamo-session-id: demo-agent-1' \\
  -d '{\"model\": \"Qwen/Qwen3-0.6B\", \"messages\": [{\"role\": \"user\", \"content\": \"Write a sort function\"}, {\"role\": \"assistant\", \"content\": \"def sort_list(lst): return sorted(lst)\"}, {\"role\": \"user\", \"content\": \"Now add type hints\"}], \"max_tokens\": 150}' | python3 -m json.tool"

echo ""
echo -e "${YELLOW}→ Same worker handled both turns. The KV cache from turn 1 is reused —${RESET}"
echo -e "${YELLOW}  no expensive re-prefill of the conversation history.${RESET}"
sleep 3

prompt "Check ThunderAgent status — program still ACTIVE, same worker"
note "  The program transitioned: ACTING → REASONING (turn 2) → ACTING again."
run "kubectl logs -n dynamo-system $ROUTER_POD --tail=50 2>&1 | grep -i 'program\|session\|worker.*assign\|REASONING\|ACTING\|sticky\|demo-agent' | tail -8"

# --- Step 7: End session ---
prompt "Step 7: End the session with x-dynamo-session-final: true"
note "  When the agent is done (task complete, user disconnects, etc.),"
note "  the harness sends session-final to release the program." 1
note "  This frees the worker assignment and removes the program from the table." 1
run "curl -s http://localhost:8000/v1/chat/completions \\
  -H 'Content-Type: application/json' \\
  -H 'x-dynamo-session-id: demo-agent-1' \\
  -H 'x-dynamo-session-final: true' \\
  -d '{\"model\": \"Qwen/Qwen3-0.6B\", \"messages\": [{\"role\": \"user\", \"content\": \"done\"}]}' | python3 -m json.tool"

echo ""
echo -e "${YELLOW}→ Program 'demo-agent-1' terminated and released.${RESET}"
echo -e "${YELLOW}  Worker slot freed for other agents.${RESET}"
sleep 3

prompt "Check ThunderAgent status — program TERMINATED and removed"
note "  The program is gone from the ProgramTable. Worker assignment released."
run "kubectl logs -n dynamo-system $ROUTER_POD --tail=50 2>&1 | grep -i 'program\|session\|release\|terminated\|final\|demo-agent' | tail -8"

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Demo Summary                                              ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                            ║"
echo "║  What we showed:                                           ║"
echo "║  ✓ ThunderAgent deploys as a Dynamo router service         ║"
echo "║  ✓ No session header → plain pass-through (compatible)     ║"
echo "║  ✓ With session header → program tracking + sticky pin     ║"
echo "║  ✓ Multi-turn → same GPU worker (KV cache reuse)          ║"
echo "║  ✓ Session final → clean program release                   ║"
echo "║                                                            ║"
echo "║  What happens under memory pressure (large model/many      ║"
echo "║  agents): ThunderAgent's scheduler pauses idle programs    ║"
echo "║  at tool boundaries and resumes them via BFD bin-packing   ║"
echo "║  when GPU capacity frees up. This achieves 1.5–3.6×        ║"
echo "║  throughput improvement on agentic benchmarks.             ║"
echo "║                                                            ║"
echo "║  Learn more:                                               ║"
echo "║  • Paper: arxiv.org/abs/2602.13692 (ICML 2026 Spotlight)  ║"
echo "║  • Code: components/src/dynamo/thunderagent_router/        ║"
echo "║  • Docs: docs/agents/thunderagent-router.md               ║"
echo "║                                                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
sleep 6

# --- Cleanup ---
prompt "Cleanup: Tear down the deployment"
run "kill %1 2>/dev/null || true"
run "kubectl delete dgd thunderagent-demo -n dynamo-system"
echo ""
echo -e "${DIM}Done. All resources released.${RESET}"
sleep 2
