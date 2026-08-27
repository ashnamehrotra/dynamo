# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Sweeper-container side of one DGDRRun Job.

This is the piece #13092's Job design refers to as "a Kubernetes-neutral
Python execution wrapper" that "registers the versioned observer." It runs
inside the same Python process as aisimulate (both packages installed in one
Sweeper container image -- see the DGDR v1beta2 architecture discussion: a Go
publisher cannot reuse this module's templating logic, so materialization has
to happen here, before anything crosses the observer socket).

What this module does NOT do, deliberately:
  - Own the Unix socket transport itself. `emit` is an injected callback
    (dict -> None); wiring it to a real socket is Job/Pod plumbing, not
    orchestration logic, and untestable without a running Job.
  - Resolve backend_version -> container image. `image_resolver` is injected
    for the same reason materialize_dgd_from_candidate() takes a pre-resolved
    `image` -- confirmed nowhere in this codebase, out of scope here too.
  - Fix the on_round signal gap. Sweeper.run()'s on_round callback exposes
    only (round_no, cumulative_candidates) today -- no tally, no
    failure_reasons, no branch label (confirmed directly against
    aisimulate.sweeper.search.py). round.completed events below carry only
    what's actually available; richer progress needs the on_round widening
    tracked separately (upstream aisimulate work, not a Dynamo-side fix).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol

from dynamo.profiler.v2.materializer import (
    MaterializationError,
    materialize_dgd_from_candidate,
)


class _CandidateLike(Protocol):
    config: dict[str, Any]
    used_gpus: int
    score: float
    metrics: dict[str, float]
    objectives: dict[str, float] | None


ImageResolver = Callable[[str, str | None], str]
EventSink = Callable[[dict[str, Any]], None]


def _default_image_resolver(backend: str, backend_version: str | None) -> str:
    """Placeholder only -- confirmed no real backend+version -> image lookup
    exists anywhere in aisimulate or dynamo.profiler. A real deployment must
    inject its own resolver; this default exists so the wrapper is testable
    without one, and intentionally looks like a placeholder, not a rule."""
    tag = backend_version or "latest"
    return f"my-registry/{backend}-runtime:{tag}"


@dataclass
class _Sequencer:
    run_uid: str
    stream_id: str
    _n: int = 0

    def envelope(self, event_type: str, data: dict[str, Any]) -> dict[str, Any]:
        self._n += 1
        return {
            "apiVersion": "sweeper.dynamo.nvidia.com/v1",
            "runUID": self.run_uid,
            "streamID": self.stream_id,
            "sequence": self._n,
            "type": event_type,
            "data": data,
        }


def _materialize_one(candidate: _CandidateLike, image_resolver: ImageResolver) -> dict:
    """One candidate's outcome, shaped for a round.completed event.

    Never raises: a materialization failure is a data value in the returned
    dict (outcome="materialization_failed"), not an exception propagating out
    of the round loop -- one bad candidate must not abort the run, matching
    #13092's "isolated trials" behavior for evaluation itself.
    """
    backend = candidate.config.get("backend")
    backend_version = candidate.config.get("backend_version")
    try:
        image = image_resolver(backend, backend_version)
        result = materialize_dgd_from_candidate(
            candidate.config,
            image=image,
        )
    except MaterializationError as exc:
        return {
            "outcome": "materialization_failed",
            "reason": str(exc),
            "candidate_config": candidate.config,
        }
    return {
        "outcome": "materialized",
        "dgd": result.dgd,
        "experimental": result.experimental,
        "used_gpus": candidate.used_gpus,
        "score": candidate.score,
        "metrics": candidate.metrics,
        "objectives": candidate.objectives,
    }


def run_and_emit(
    sweeper: Any,
    config: Any,
    *,
    run_uid: str,
    stream_id: str,
    emit: EventSink,
    image_resolver: ImageResolver = _default_image_resolver,
) -> None:
    """Run `sweeper.run(config, on_round=...)`, materializing each candidate
    in every round and emitting observer-protocol events via `emit`.

    `sweeper` is an aisimulate.sweeper.Sweeper instance (typed as Any here,
    matching materializer.py's own choice not to hard-import aisimulate --
    see that module's docstring for why). Duck-typed: anything with a
    `.run(config, on_round=callback)` method matching Sweeper's signature
    works, which is also what keeps this function's own tests independent of
    an installed aisimulate wheel.
    """
    seq = _Sequencer(run_uid=run_uid, stream_id=stream_id)
    emit(seq.envelope("search.resolved", {"config": _safe_config_summary(config)}))

    def on_round(round_no: int, candidates: Iterable[_CandidateLike]) -> None:
        materialized = [_materialize_one(c, image_resolver) for c in candidates]
        emit(
            seq.envelope(
                "round.completed",
                {
                    "round": round_no,
                    # Cumulative, not delta -- this is what on_round actually
                    # gives us today; see module docstring.
                    "cumulative_candidate_count": len(materialized),
                    "candidates": materialized,
                },
            )
        )

    try:
        final_candidates = sweeper.run(config, on_round=on_round)
    except Exception as exc:  # noqa: BLE001 -- must always emit a terminal event
        emit(
            seq.envelope(
                "run.completed",
                {"outcome": "failed", "error": str(exc), "error_type": type(exc).__name__},
            )
        )
        raise

    final_materialized = [
        _materialize_one(c, image_resolver) for c in final_candidates
    ]
    emit(
        seq.envelope(
            "run.completed",
            {"outcome": "succeeded", "candidates": final_materialized},
        )
    )


def _safe_config_summary(config: Any) -> dict[str, Any]:
    """Best-effort JSON-safe summary for search.resolved; falls back to
    str() for anything not a pydantic BaseModel, so this never breaks the
    wrapper regardless of what `config`'s real type turns out to be."""
    model_dump_json = getattr(config, "model_dump_json", None)
    if callable(model_dump_json):
        import json

        return json.loads(model_dump_json())
    return {"repr": str(config)}
