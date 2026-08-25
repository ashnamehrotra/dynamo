# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test for the DGDR v1beta2 Job wrapper.

Unlike test_v2_materializer.py (which uses hand-copied Candidate.config
fixtures and needs no aisimulate install), this test runs the real
aisimulate.sweeper.Sweeper orchestration end to end -- Sweeper.run() with a
fake replay runner, feeding real Candidate objects through the real
materializer, producing real observer-protocol events. It requires aisimulate
to actually be installed (skips otherwise) since it is exercising the
integration seam between the two packages, not just this module's own logic.
"""

from __future__ import annotations

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]

try:
    from aisimulate.sweeper.config import SmartSearchConfig
    from aisimulate.sweeper.parallel_enum import ParallelShape, ReplicaParallelConfig
    from aisimulate.sweeper.replay import ReplayReport, RunnerCapabilities
    from aisimulate.sweeper.sampler import Suggestion
    from aisimulate.sweeper.search import Sweeper
    import aisimulate.sweeper.search as search_mod
    from aisimulate.sweeper.search_space import BranchSpace

    from dynamo.profiler.v2.job_wrapper import run_and_emit
except ImportError as exc:
    pytest.skip(f"Skip (missing dependency): {exc}", allow_module_level=True)


def _two_shape_branch() -> "BranchSpace":
    """A dense-TP shape and a TEP (MoE expert-parallel) shape -- genuinely
    different (tp, dp, moe_tp, moe_ep) tuples, since strategy is a derived
    property (ParallelShape.strategy) and cannot be injected via selection
    dicts. Confirmed directly: an earlier version of this test tried to force
    strategy via the selection dict and both candidates silently came back
    "tep" regardless, because the real code derives it from shape alone."""
    tp_shape = ReplicaParallelConfig(
        ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=1), replicas=2
    )
    tep_shape = ReplicaParallelConfig(
        ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2
    )
    return BranchSpace(
        deployment_mode="agg",
        parallel_configs=(tp_shape, tep_shape),
        supported_backends={
            tp_shape: frozenset({"trtllm"}),
            tep_shape: frozenset({"trtllm"}),
        },
        knob_choices={"backend": ["trtllm"]},
    )


class _FakeRunner:
    def run(self, spec):
        args = spec.backend_deployment.agg_engine_args
        seqs = float(args["max_num_seqs"])
        return ReplayReport(
            metrics={
                "output_throughput_tok_s": seqs * 12.5,
                "gpu_hours": float(spec.backend_deployment.num_workers),
            }
        )

    def close(self):
        pass


class _FakeRunnerFactory:
    def capabilities(self):
        return RunnerCapabilities(supported_backend_topologies=(("*", "*"),))

    def create(self, worker_id):
        return _FakeRunner()


class _TwoCandidateSampler:
    """Suggests one TP candidate (expect: materializes) and one TEP
    candidate on trtllm (expect: the real, confirmed set_config_tep_size
    NotImplementedError -> MaterializationError path)."""

    def __init__(self, branch, study_id, objectives=None):
        self.branch = branch

    def suggest(self, count):
        selections = [
            {
                "deployment_mode": "agg",
                "backend": "trtllm",
                "agg_max_num_batched_tokens": 8192,
                "agg_max_num_seqs": seqs,
            }
            for seqs in (512, 1024)[:count]
        ]
        return [
            Suggestion(
                selection=sel,
                parallel_config=self.branch.parallel_configs[i],
                handle=sel,
            )
            for i, sel in enumerate(selections)
        ]

    def observe(self, suggestion, metrics):
        pass

    def observe_infeasible(self, suggestion, reason):
        pass


def test_run_and_emit_materializes_one_candidate_and_fails_the_other(monkeypatch) -> None:
    branch = _two_shape_branch()
    monkeypatch.setattr(
        search_mod,
        "enumerate_branches",
        lambda config, *, max_seq_len=None, runner_capabilities=None: [branch],
    )
    monkeypatch.setattr(search_mod, "resolve_backend_version", lambda hw, be: "1.3.0rc10")

    config = SmartSearchConfig(
        search_space={
            "model_name": "deepseek-ai/DeepSeek-V3",
            "hardware_sku": "gb200",
            "backend": ["trtllm"],
            "deployment_mode": ["agg"],
            "gpu_budget": 32,
        },
        workload={
            "isl": 1024,
            "osl": 1024,
            "concurrency": 64,
            "num_request_ratio": 10,
        },
        sweep={"max_rounds": 1, "candidates_per_round": 2, "parallel_evals": 1},
        goal={"target": "throughput"},
    )

    sweeper = Sweeper(
        runner_factory=_FakeRunnerFactory(),
        sampler_factory=_TwoCandidateSampler,
        show_progress=False,
    )

    events: list[dict] = []
    run_and_emit(
        sweeper,
        config,
        run_uid="test-run-uid",
        stream_id="test-stream",
        emit=events.append,
    )

    assert events[0]["type"] == "search.resolved"
    assert events[-1]["type"] == "run.completed"
    assert events[-1]["data"]["outcome"] == "succeeded"

    round_events = [e for e in events if e["type"] == "round.completed"]
    assert len(round_events) == 1
    candidates = round_events[0]["data"]["candidates"]
    outcomes = {c["outcome"] for c in candidates}
    assert outcomes == {"materialized", "materialization_failed"}

    failed = next(c for c in candidates if c["outcome"] == "materialization_failed")
    assert "tep" in failed["reason"] and "not supported" in failed["reason"]

    succeeded = next(c for c in candidates if c["outcome"] == "materialized")
    worker = [
        comp for comp in succeeded["dgd"]["spec"]["components"] if comp.get("type") == "worker"
    ][0]
    args = worker["podTemplate"]["spec"]["containers"][0]["args"]
    assert "--trtllm.tensor_parallel_size" in args


def test_run_and_emit_always_emits_a_terminal_event_on_sweeper_failure(monkeypatch) -> None:
    """A run that raises must still emit run.completed(outcome=failed) --
    matching #13092's "EOF without a terminal record fails the Job" rule:
    the wrapper's job is to guarantee a terminal record exists either way."""

    class _BrokenSweeper:
        def run(self, config, on_round=None):
            raise RuntimeError("simulated Sweeper crash")

    events: list[dict] = []
    with pytest.raises(RuntimeError, match="simulated Sweeper crash"):
        run_and_emit(
            _BrokenSweeper(),
            config=object(),
            run_uid="test-run-uid",
            stream_id="test-stream",
            emit=events.append,
        )

    assert events[-1]["type"] == "run.completed"
    assert events[-1]["data"]["outcome"] == "failed"
    assert events[-1]["data"]["error_type"] == "RuntimeError"
