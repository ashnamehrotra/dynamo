# ThunderAgent: Kubernetes Integration Bugs & Observability Gaps

## Summary

During end-to-end testing of ThunderAgent on Kubernetes via the Dynamo Operator (DGD deployment), several bugs prevent core functionality from working and observability gaps make it impossible to verify behavior. This tracking issue covers the fixes needed to make ThunderAgent demo-ready and production-viable on K8s.

## Context

- Deployment: DGD with Frontend → ThunderAgentRouter → VllmDecodeWorker (1x A100, Qwen3-0.6B)
- Image: built from main branch (post-PR #9448)
- Operator: Dynamo Operator in `dynamo-system` namespace

---

## Bugs

### 1. WorkerCapacityProvider namespace mismatch prevents pause/resume

**Severity: High — core scheduling feature is non-functional on K8s**

The ThunderAgent scheduler's pause/resume logic never fires in K8s because `WorkerCapacityProvider` can't discover the worker's MDC.

- Worker publishes MDC at: `dynamo-system-thunderagent-demo-<hash>.backend`
- Router subscribes at: `dynamo.backend` (from `--endpoint dynamo.backend.generate`)
- Result: `capacity.snapshot()` always returns empty, scheduler tick no-ops

The CLI reference script (`run_minimax_8xh100.sh`) works because all components share a flat namespace. The K8s operator uses namespaced discovery that doesn't match.

**Fix:** Either the operator needs to wire up the namespaced endpoint correctly, or ThunderAgentRouter needs to discover the worker's actual published namespace from the runtime context.

---

### 2. `session-final` still forwards to GPU engine

When `x-dynamo-session-final: true` is sent, ThunderAgent still calls `KvRouter.generate_from_request()` and runs full inference before releasing the program. This wastes a GPU forward pass on a throwaway "done" message.

**Expected:** Short-circuit after releasing the program — return immediately without an engine call.

**File:** `components/src/dynamo/thunderagent_router/__main__.py`

---

### 3. DGDR profiler doubles model path

The profiler generates `--model-path /opt/models/opt/models` by concatenating `modelCache.pvcMountPath` with `HF_HOME`. Requires manual fixup of the generated DGD.

**Steps to reproduce:**
1. Submit a DGDR with `modelCache.pvcMountPath: /opt/models` and env `HF_HOME=/opt/models`
2. Let profiler generate the DGD
3. Inspect `--model-path` in the Frontend and Worker args

**Note:** This is a Dynamo Operator/profiler bug, not ThunderAgent-specific, but blocks ThunderAgent deployment.

---

### 4. DGDR profiler service name undocumented

The profiler generates `VllmDecodeWorker` (not `VllmWorker`). The `overrides.dgd` section requires exact service name matches, but there's no documentation of what names the profiler generates. Users must first run with `autoApply: false` and inspect the output.

**Fix:** Document the generated service names, or make overrides match by component type rather than exact name.

---

## Observability Improvements

### 5. No program lifecycle logging

ThunderAgent doesn't log program creation, status transitions, worker assignment, or release at INFO level. `kubectl logs` grepping for `program|session|REASONING|ACTING` returns nothing meaningful.

**Expected logs:**
```
INFO Program 'abc-123' created, assigned_worker=0x1a2b
INFO Program 'abc-123' status: REASONING → ACTING, tokens=113
INFO Program 'abc-123' status: ACTING → REASONING (turn 2), worker=0x1a2b
INFO Program 'abc-123' released (session-final)
```

**File:** `components/src/dynamo/thunderagent_router/router.py`, `__main__.py`

---

### 6. No status/metrics endpoint

There's no way to query the current ProgramTable state (active programs, paused count, per-worker utilization, assigned workers). A `/debug/programs` HTTP endpoint or Prometheus metrics would make demos and debugging significantly easier.

**Suggested output:**
```json
{
  "active_programs": 3,
  "paused_programs": 1,
  "programs": [
    {"id": "demo-agent-1", "status": "acting", "lifecycle": "active", "worker": "0x1a2b", "tokens": 245, "turns": 3}
  ],
  "workers": [
    {"id": "0x1a2b", "utilization": 0.42, "programs_pinned": 2}
  ]
}
```

---

### 7. No routing proof in responses

Curl responses don't include any evidence that ThunderAgent processed the request (no `x-dynamo-worker-id`, `x-dynamo-session-status`, or similar headers). It's impossible to externally verify sticky pinning or program state without logs.

**Suggestion:** Add response headers like:
- `x-dynamo-worker-id: <worker_hash>`
- `x-dynamo-program-status: active`
- `x-dynamo-program-turn: 3`

---

## Task List

- [ ] Fix WorkerCapacityProvider namespace mismatch on K8s (#1)
- [ ] Short-circuit session-final without engine call (#2)
- [ ] Fix DGDR profiler model path doubling (#3)
- [ ] Document or fix DGDR profiler service names (#4)
- [ ] Add program lifecycle INFO logging (#5)
- [ ] Add /debug/programs status endpoint (#6)
- [ ] Add routing metadata to response headers (#7)

---

## Labels

`thunderagent`, `bug`, `enhancement`, `kubernetes`
