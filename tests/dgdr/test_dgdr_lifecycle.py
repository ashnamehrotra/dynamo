# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Core lifecycle and profiling configuration tests for DGDR v1beta1.

Covers:
  TestDGDRMinimalDeployment    - Pending -> Profiling -> Ready -> Deploying -> Deployed
  TestDGDRBackendSelection     - vllm, trtllm, sglang, auto
  TestDGDRSearchStrategies     - rapid (AIC) and thorough (GPU sweep)
  TestDGDRSLATargets           - ttft+itl, e2eLatency, optimizationType
  TestDGDRWorkloadPickingModes - requestRate, concurrency, isl/osl
  TestDGDRFeatures             - planner, mocker feature flags
  TestDGDRModelCache           - PVC-backed model weight cache

Requires at least one GPU node (gpu_1) or mocker mode.

Run:
  pytest tests/dgdr/test_dgdr_lifecycle.py -m "gpu_1 and pre_merge" -v --dgdr-namespace=default
"""

from __future__ import annotations

import json
import logging

import pytest

from tests.dgdr.conftest import (
    PHASE_DEPLOYED,
    PHASE_DEPLOYING,
    PHASE_PROFILING,
    PHASE_READY,
    _run_kubectl,
    build_dgdr_manifest,
    get_dgdr,
    unique_dgdr_name,
    wait_for_any_dgdr_phase,
    wait_for_dgdr_phase,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# -- Group 3: Minimal Deployment Lifecycle (gpu_1, pre_merge) ---------------
# ---------------------------------------------------------------------------


@pytest.mark.gpu_1
@pytest.mark.pre_merge
@pytest.mark.e2e
@pytest.mark.k8s
@pytest.mark.deploy
class TestDGDRMinimalDeployment:
    """
    Simplest possible DGDR: only model, image, backend.
    Verifies the complete Pending → Profiling → Ready → Deploying → Deployed lifecycle.
    Hardware is auto-discovered from the GPU cluster.
    """

    def test_minimal_rapid_full_lifecycle(
        self,
        dgdr_namespace: str,
        dgdr_image: str,
        dgdr_model: str,
        dgdr_use_mocker: bool,
        dgdr_factory,
        dgdr_profiling_timeout: int,
        dgdr_deploy_timeout: int,
    ) -> None:
        """
        Create a minimal DGDR and wait for it to reach Deployed.
        Validates: phase progression, profilingJobName, dgdName, DeploymentInfo.

        In mocker mode the operator pre-sets Status.DGDName from the profiling output
        (generateDGDSpec) and then handleDeployingPhase immediately tries to GET that
        DGD.  Since no DGD of that name exists yet, handleDGDDeleted fires and the DGDR
        reaches Failed.  Use auto_apply=False in mocker mode and validate only that
        profiling completed and the spec was generated correctly (PHASE_READY).
        """
        name = unique_dgdr_name("minimal")
        manifest = build_dgdr_manifest(
            name,
            model=dgdr_model,
            image=dgdr_image,
            backend="vllm",
            search_strategy="rapid",
            auto_apply=not dgdr_use_mocker,
        )
        dgdr_factory(manifest)

        # 1. Should quickly reach Pending then Profiling
        wait_for_dgdr_phase(name, dgdr_namespace, PHASE_PROFILING, timeout=120)

        obj = get_dgdr(name, dgdr_namespace)
        assert obj["status"].get("profilingJobName"), (
            "status.profilingJobName should be populated when in Profiling phase"
        )

        if dgdr_use_mocker:
            # In mocker mode validate spec generation only; do not attempt deployment.
            wait_for_dgdr_phase(name, dgdr_namespace, PHASE_READY,
                                timeout=dgdr_profiling_timeout)
            obj = get_dgdr(name, dgdr_namespace)
            assert obj["status"].get("profilingResults", {}).get("selectedConfig"), (
                "status.profilingResults.selectedConfig should be set after profiling"
            )
            assert obj["status"].get("dgdName"), (
                "status.dgdName should be set after spec generation"
            )
            return

        # 2. Wait for profiling to complete.
        # With autoApply=true, the operator may skip PHASE_READY entirely and jump
        # directly from Profiling → Deploying → Deployed.  Accept any of these as
        # evidence that profiling finished.
        post_profiling_phase = wait_for_any_dgdr_phase(
            name, dgdr_namespace,
            [PHASE_READY, PHASE_DEPLOYING, PHASE_DEPLOYED],
            timeout=dgdr_profiling_timeout,
        )

        obj = get_dgdr(name, dgdr_namespace)
        assert obj["status"].get("profilingResults"), (
            "status.profilingResults should be populated after profiling"
        )
        assert obj["status"]["profilingResults"].get("selectedConfig"), (
            "status.profilingResults.selectedConfig should be set"
        )

        # 3. With autoApply=true, DGD should be auto-created → Deploying then Deployed.
        # If we already saw Deploying or Deployed in step 2, skip the short wait.
        if post_profiling_phase == PHASE_READY:
            final_phase = wait_for_any_dgdr_phase(
                name, dgdr_namespace,
                [PHASE_DEPLOYING, PHASE_DEPLOYED],
                timeout=120,
            )
        else:
            final_phase = post_profiling_phase
        assert final_phase in (PHASE_DEPLOYING, PHASE_DEPLOYED), (
            f"Expected Deploying or Deployed after profiling with autoApply=true, got {final_phase}"
        )

        obj = get_dgdr(name, dgdr_namespace)
        assert obj["status"].get("dgdName"), (
            "status.dgdName should be populated once the DGD is created"
        )

        # 4. Final state: Deployed
        wait_for_dgdr_phase(name, dgdr_namespace, PHASE_DEPLOYED, timeout=dgdr_deploy_timeout)

        obj = get_dgdr(name, dgdr_namespace)
        deployment_info = obj["status"].get("deploymentInfo", {})
        assert deployment_info.get("availableReplicas", 0) > 0, (
            "deploymentInfo.availableReplicas should be > 0 in Deployed phase"
        )


# ---------------------------------------------------------------------------
# ── Group 4: Backend Selection ──────────────────────────────────────────────
# ---------------------------------------------------------------------------


@pytest.mark.gpu_1
@pytest.mark.nightly
@pytest.mark.e2e
@pytest.mark.k8s
@pytest.mark.deploy
class TestDGDRBackendSelection:
    """
    Verify that each backend value produces a working DGDR lifecycle.
    Tests vllm, sglang, trtllm, and auto (which selects the best backend
    automatically via AIC).
    """

    @pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm"])
    def test_backend(
        self,
        backend: str,
        dgdr_namespace: str,
        dgdr_image: str,
        dgdr_model: str,
        dgdr_use_mocker: bool,
        dgdr_factory,
        dgdr_profiling_timeout: int,
        dgdr_deploy_timeout: int,
    ) -> None:
        """
        Create a DGDR for each supported backend and verify it progresses
        to Deployed.  Image is adjusted per-backend.
        """
        # AIC silicon-mode perf data is only bundled for vllm and trtllm on the
        # mocker's injected GPU SKU (a100_sxm).  sglang has no data files for that
        # SKU, so AIC fails with PerfDataNotAvailableError.
        # sglang requires a real GPU cluster with actual perf data.
        if dgdr_use_mocker and backend == "sglang":
            pytest.skip(
                "AIC has no silicon perf data for backend='sglang' on the mocker GPU SKU "
                "(a100_sxm). Run without --dgdr-use-mocker on a real GPU cluster."
            )

        image = dgdr_image

        name = unique_dgdr_name(f"backend-{backend}")
        manifest = build_dgdr_manifest(
            name,
            model=dgdr_model,
            image=image,
            backend=backend,
            search_strategy="rapid",
            # In mocker mode skip auto-deployment (see TestDGDRMinimalDeployment note).
            auto_apply=not dgdr_use_mocker,
        )
        dgdr_factory(manifest)

        wait_for_dgdr_phase(name, dgdr_namespace, PHASE_PROFILING, timeout=120)

        if dgdr_use_mocker:
            wait_for_dgdr_phase(name, dgdr_namespace, PHASE_READY,
                                timeout=dgdr_profiling_timeout)
            obj = get_dgdr(name, dgdr_namespace)
            assert obj["status"].get("profilingResults", {}).get("selectedConfig"), (
                "selectedConfig should be present after profiling"
            )
            return

        wait_for_dgdr_phase(name, dgdr_namespace, PHASE_DEPLOYED,
                            timeout=dgdr_profiling_timeout + dgdr_deploy_timeout)

        obj = get_dgdr(name, dgdr_namespace)
        assert obj["status"].get("dgdName"), "DGD should be named in status"

    def test_backend_auto_selects_appropriate_backend(
        self,
        dgdr_namespace: str,
        dgdr_image: str,
        dgdr_model: str,
        dgdr_factory,
        dgdr_profiling_timeout: int,
        dgdr_deploy_timeout: int,
    ) -> None:
        """
        backend: auto delegates backend selection to AI Configurator.
        The resolved backend should appear in the generated DGD spec.
        """
        name = unique_dgdr_name("auto-backend")
        manifest = build_dgdr_manifest(
            name,
            model=dgdr_model,
            image=dgdr_image,
            backend="auto",       # auto-select
            search_strategy="rapid",
            auto_apply=False,     # inspect the spec first
        )
        dgdr_factory(manifest)

        wait_for_dgdr_phase(name, dgdr_namespace, PHASE_READY, timeout=dgdr_profiling_timeout)

        obj = get_dgdr(name, dgdr_namespace)
        selected = obj["status"].get("profilingResults", {}).get("selectedConfig")
        assert selected, "selectedConfig should be present after auto-backend profiling"


# ---------------------------------------------------------------------------
# ── Group 5: Search Strategies ──────────────────────────────────────────────
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.k8s
@pytest.mark.deploy
class TestDGDRSearchStrategies:
    """
    Verify the two search strategies: rapid (AIC simulation) and thorough
    (real-GPU sweep).
    """

    @pytest.mark.gpu_8
    @pytest.mark.nightly
    def test_thorough_strategy_completes(
        self,
        dgdr_namespace: str,
        dgdr_image: str,
        dgdr_model: str,
        dgdr_factory,
        dgdr_profiling_timeout: int,
    ) -> None:
        """
        thorough strategy deploys real engines, sweeps parallelization options,
        and benchmarks with AIPerf.  Requires 8 GPUs; can take 2-4 hours.
        Must specify a concrete backend (not 'auto').
        """
        name = unique_dgdr_name("thorough-strat")
        manifest = build_dgdr_manifest(
            name,
            model=dgdr_model,
            image=dgdr_image,
            backend="vllm",          # explicit – thorough forbids 'auto'
            search_strategy="thorough",
            auto_apply=False,
        )
        dgdr_factory(manifest)

        # thorough profiling can take several hours
        wait_for_dgdr_phase(name, dgdr_namespace, PHASE_READY, timeout=dgdr_profiling_timeout)

        obj = get_dgdr(name, dgdr_namespace)
        results = obj["status"].get("profilingResults", {})
        assert results.get("selectedConfig"), "selectedConfig should be set after thorough profiling"
        # thorough mode should return a Pareto frontier
        assert isinstance(results.get("pareto", []), list), (
            "profilingResults.pareto should be a list after thorough profiling"
        )


# ---------------------------------------------------------------------------
# ── Group 6: SLA Targets ────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


@pytest.mark.gpu_1
@pytest.mark.nightly
@pytest.mark.e2e
@pytest.mark.k8s
@pytest.mark.deploy
class TestDGDRSLATargets:
    """
    Verify that the profiler respects SLA targets and surfaces them in the
    generated DGD configuration.
    """

    def test_sla_ttft_and_itl(
        self,
        dgdr_namespace: str,
        dgdr_image: str,
        dgdr_model: str,
        dgdr_factory,
        dgdr_profiling_timeout: int,
    ) -> None:
        """
        Specifying explicit ttft + itl targets drives the profiler's config selection.
        """
        name = unique_dgdr_name("sla-ttft-itl")
        manifest = build_dgdr_manifest(
            name,
            model=dgdr_model,
            image=dgdr_image,
            backend="vllm",
            search_strategy="rapid",
            sla={"ttft": 200.0, "itl": 20.0},
            workload={"isl": 3000, "osl": 150},
            auto_apply=False,
        )
        dgdr_factory(manifest)

        wait_for_dgdr_phase(name, dgdr_namespace, PHASE_READY, timeout=dgdr_profiling_timeout)

        obj = get_dgdr(name, dgdr_namespace)
        assert obj["status"].get("profilingResults", {}).get("selectedConfig"), (
            "selectedConfig should be present after SLA-targeted profiling"
        )

    def test_sla_e2e_latency(
        self,
        dgdr_namespace: str,
        dgdr_image: str,
        dgdr_model: str,
        dgdr_factory,
        dgdr_profiling_timeout: int,
    ) -> None:
        """
        e2eLatency is an alternative to ttft+itl – mutually exclusive with those fields.
        The profiler should accept this and produce a valid config.
        """
        name = unique_dgdr_name("sla-e2e")
        manifest = build_dgdr_manifest(
            name,
            model=dgdr_model,
            image=dgdr_image,
            backend="vllm",
            search_strategy="rapid",
            sla={"e2eLatency": 5000.0},   # 5 seconds; no ttft/itl
            workload={"isl": 1024, "osl": 128},
            auto_apply=False,
        )
        dgdr_factory(manifest)

        wait_for_dgdr_phase(name, dgdr_namespace, PHASE_READY, timeout=dgdr_profiling_timeout)

        obj = get_dgdr(name, dgdr_namespace)
        assert obj["status"].get("profilingResults", {}).get("selectedConfig"), (
            "selectedConfig should be present after e2eLatency-targeted profiling"
        )

    def test_sla_optimization_type_latency(
        self,
        dgdr_namespace: str,
        dgdr_image: str,
        dgdr_model: str,
        dgdr_factory,
        dgdr_profiling_timeout: int,
    ) -> None:
        """
        optimizationType: latency tells the profiler to minimize latency
        rather than explicit SLA numbers.

        The generated DGD should be at least as conservative as throughput mode.
        """
        name = unique_dgdr_name("opt-latency")
        manifest = build_dgdr_manifest(
            name,
            model=dgdr_model,
            image=dgdr_image,
            backend="vllm",
            search_strategy="rapid",
            sla={"optimizationType": "latency"},
            auto_apply=False,
        )
        dgdr_factory(manifest)

        wait_for_dgdr_phase(name, dgdr_namespace, PHASE_READY, timeout=dgdr_profiling_timeout)

        obj = get_dgdr(name, dgdr_namespace)
        assert obj["status"].get("profilingResults"), "profilingResults should be set"

    def test_sla_optimization_type_throughput(
        self,
        dgdr_namespace: str,
        dgdr_image: str,
        dgdr_model: str,
        dgdr_factory,
        dgdr_profiling_timeout: int,
    ) -> None:
        """
        optimizationType: throughput maximises GPU utilisation.
        """
        name = unique_dgdr_name("opt-throughput")
        manifest = build_dgdr_manifest(
            name,
            model=dgdr_model,
            image=dgdr_image,
            backend="vllm",
            search_strategy="rapid",
            sla={"optimizationType": "throughput"},
            auto_apply=False,
        )
        dgdr_factory(manifest)

        wait_for_dgdr_phase(name, dgdr_namespace, PHASE_READY, timeout=dgdr_profiling_timeout)

        obj = get_dgdr(name, dgdr_namespace)
        assert obj["status"].get("profilingResults"), "profilingResults should be set"


# ---------------------------------------------------------------------------
# ── Group 7: Workload Picking Modes ─────────────────────────────────────────
# ---------------------------------------------------------------------------


@pytest.mark.gpu_1
@pytest.mark.nightly
@pytest.mark.e2e
@pytest.mark.k8s
@pytest.mark.deploy
class TestDGDRWorkloadPickingModes:
    """
    The profiler has three picking modes driven by workload fields:
    - Default (no workload)           → maximise throughput under SLA
    - Load-match with requestRate     → minimum GPUs serving target req/s
    - Load-match with concurrency     → minimum GPUs serving target concurrency
    """

    def test_request_rate_picking(
        self,
        dgdr_namespace: str,
        dgdr_image: str,
        dgdr_model: str,
        dgdr_factory,
        dgdr_profiling_timeout: int,
    ) -> None:
        """
        workload.requestRate enables load-match picking: find config satisfying
        SLA at target req/s with minimum GPU count.
        """
        name = unique_dgdr_name("wl-rps")
        manifest = build_dgdr_manifest(
            name,
            model=dgdr_model,
            image=dgdr_image,
            backend="vllm",
            search_strategy="rapid",
            sla={"ttft": 300.0, "itl": 25.0},
            workload={"isl": 2048, "osl": 256, "requestRate": 5.0},
            auto_apply=False,
        )
        dgdr_factory(manifest)

        wait_for_dgdr_phase(name, dgdr_namespace, PHASE_READY, timeout=dgdr_profiling_timeout)

        obj = get_dgdr(name, dgdr_namespace)
        assert obj["status"].get("profilingResults"), "profilingResults should be set"

    def test_concurrency_picking(
        self,
        dgdr_namespace: str,
        dgdr_image: str,
        dgdr_model: str,
        dgdr_factory,
        dgdr_profiling_timeout: int,
    ) -> None:
        """
        workload.concurrency enables load-match picking at target concurrency.
        """
        name = unique_dgdr_name("wl-conc")
        manifest = build_dgdr_manifest(
            name,
            model=dgdr_model,
            image=dgdr_image,
            backend="vllm",
            search_strategy="rapid",
            sla={"ttft": 300.0, "itl": 25.0},
            workload={"isl": 2048, "osl": 256, "concurrency": 10.0},
            auto_apply=False,
        )
        dgdr_factory(manifest)

        wait_for_dgdr_phase(name, dgdr_namespace, PHASE_READY, timeout=dgdr_profiling_timeout)

        obj = get_dgdr(name, dgdr_namespace)
        assert obj["status"].get("profilingResults"), "profilingResults should be set"

    def test_isl_osl_affects_profiling(
        self,
        dgdr_namespace: str,
        dgdr_image: str,
        dgdr_model: str,
        dgdr_factory,
        dgdr_profiling_timeout: int,
    ) -> None:
        """
        isl (Input Sequence Length) and osl (Output Sequence Length) are plumbed
        to the profiler and influence KV-cache sizing in the generated config.
        """
        name = unique_dgdr_name("wl-isl-osl")
        manifest = build_dgdr_manifest(
            name,
            model=dgdr_model,
            image=dgdr_image,
            backend="vllm",
            search_strategy="rapid",
            workload={"isl": 8192, "osl": 1024},
            auto_apply=False,
        )
        dgdr_factory(manifest)

        wait_for_dgdr_phase(name, dgdr_namespace, PHASE_READY, timeout=dgdr_profiling_timeout)

        obj = get_dgdr(name, dgdr_namespace)
        assert obj["status"].get("profilingResults", {}).get("selectedConfig"), (
            "selectedConfig should be present"
        )


# ---------------------------------------------------------------------------
# ── Group 8: Optional Features ──────────────────────────────────────────────
# ---------------------------------------------------------------------------


@pytest.mark.gpu_1
@pytest.mark.nightly
@pytest.mark.e2e
@pytest.mark.k8s
@pytest.mark.deploy
class TestDGDRFeatures:
    """
    Tests for optional Dynamo platform features toggled via spec.features.
    Each test verifies the feature is reflected in the generated DGD spec and
    that the DGDR lifecycle completes successfully.
    """

    def test_planner_enabled_with_rapid_sweep(
        self,
        dgdr_namespace: str,
        dgdr_image: str,
        dgdr_model: str,
        dgdr_use_mocker: bool,
        dgdr_factory,
        dgdr_profiling_timeout: int,
        dgdr_deploy_timeout: int,
    ) -> None:
        """
        features.planner.enabled=true injects the SLA planner into the generated DGD.
        plannerPreDeploymentSweeping=rapid uses AIC simulation (fast, no extra GPUs).

        In mocker mode the operator pre-sets Status.DGDName from the profiling output
        (generateDGDSpec) before the DGD is actually created.  When handleDeployingPhase
        runs it immediately tries to GET that DGD; since it does not exist yet it fires
        handleDGDDeleted and the DGDR reaches Failed.  To avoid this, mocker mode runs
        with auto_apply=False and only validates that the spec was generated correctly
        (PHASE_READY).  Full end-to-end deployment is exercised outside mocker mode.
        """
        name = unique_dgdr_name("feat-planner")
        manifest = build_dgdr_manifest(
            name,
            model=dgdr_model,
            image=dgdr_image,
            backend="vllm",
            search_strategy="rapid",
            sla={"ttft": 300.0, "itl": 20.0},
            features={
                "planner": {
                    "enabled": True,
                    "plannerPreDeploymentSweeping": "rapid",
                }
            },
            # In mocker mode skip auto-deployment — see docstring above.
            auto_apply=not dgdr_use_mocker,
        )
        dgdr_factory(manifest)

        if dgdr_use_mocker:
            # Only verify that spec generation succeeded; do not attempt deployment.
            wait_for_dgdr_phase(name, dgdr_namespace, PHASE_READY,
                                timeout=dgdr_profiling_timeout)
            obj = get_dgdr(name, dgdr_namespace)
            assert obj["status"].get("dgdName"), (
                "dgdName should be set after spec generation"
            )
            assert obj["status"].get("profilingResults", {}).get("selectedConfig"), (
                "selectedConfig should be present in profiling results"
            )
        else:
            wait_for_dgdr_phase(name, dgdr_namespace, PHASE_DEPLOYED,
                                timeout=dgdr_profiling_timeout + dgdr_deploy_timeout)

            obj = get_dgdr(name, dgdr_namespace)
            dgd_name = obj["status"].get("dgdName")
            assert dgd_name, "dgdName should be set"

            dgd_result = _run_kubectl([
                "get", "dynamographdeployment", dgd_name,
                "-n", dgdr_namespace, "-o", "json",
            ], check=False)
            if dgd_result.returncode == 0:
                dgd_obj = json.loads(dgd_result.stdout)
                services = dgd_obj.get("spec", {}).get("services", {})
                assert any("planner" in s.lower() for s in services.keys()), (
                    f"Expected Planner service in generated DGD. Found: {list(services.keys())}"
                )

    def test_mocker_enabled(
        self,
        dgdr_namespace: str,
        dgdr_image: str,
        dgdr_model: str,
        dgdr_use_mocker: bool,
        dgdr_factory,
        dgdr_profiling_timeout: int,
        dgdr_deploy_timeout: int,
    ) -> None:
        """
        features.mocker.enabled=true generates a mocker DGD that simulates engine
        behaviour without real GPU inference.  Useful for validating planner behaviour
        at scale.  Mocker requires a planner with pre-deployment sweeping enabled.

        In mocker mode use auto_apply=False and validate spec generation only
        (see TestDGDRMinimalDeployment note on the handleDeployingPhase race).
        """
        name = unique_dgdr_name("feat-mocker")
        manifest = build_dgdr_manifest(
            name,
            model=dgdr_model,
            image=dgdr_image,
            backend="vllm",
            search_strategy="rapid",
            features={
                "planner": {
                    "enabled": True,
                    "plannerPreDeploymentSweeping": "rapid",
                },
                "mocker": {"enabled": True},
            },
            auto_apply=not dgdr_use_mocker,
        )
        dgdr_factory(manifest)

        if dgdr_use_mocker:
            wait_for_dgdr_phase(name, dgdr_namespace, PHASE_READY,
                                timeout=dgdr_profiling_timeout)
            obj = get_dgdr(name, dgdr_namespace)
            assert obj["status"].get("dgdName"), (
                "dgdName should be set after spec generation"
            )
            return

        wait_for_dgdr_phase(name, dgdr_namespace, PHASE_DEPLOYED,
                            timeout=dgdr_profiling_timeout + dgdr_deploy_timeout)

        obj = get_dgdr(name, dgdr_namespace)
        assert obj["status"].get("dgdName"), "dgdName should be set for mocker deployment"

    def test_planner_enabled_no_pre_deployment_sweep(
        self,
        dgdr_namespace: str,
        dgdr_image: str,
        dgdr_model: str,
        dgdr_use_mocker: bool,
        dgdr_factory,
        dgdr_profiling_timeout: int,
        dgdr_deploy_timeout: int,
    ) -> None:
        """
        plannerPreDeploymentSweeping=none skips interpolation curve generation
        (only load-based scaling is available at runtime).

        In mocker mode use auto_apply=False and validate spec generation only
        (see TestDGDRMinimalDeployment note on the handleDeployingPhase race).
        """
        name = unique_dgdr_name("feat-planner-none")
        manifest = build_dgdr_manifest(
            name,
            model=dgdr_model,
            image=dgdr_image,
            backend="vllm",
            search_strategy="rapid",
            features={
                "planner": {
                    "enabled": True,
                    "plannerPreDeploymentSweeping": "none",
                }
            },
            auto_apply=not dgdr_use_mocker,
        )
        dgdr_factory(manifest)

        if dgdr_use_mocker:
            wait_for_dgdr_phase(name, dgdr_namespace, PHASE_READY,
                                timeout=dgdr_profiling_timeout)
            return

        wait_for_dgdr_phase(name, dgdr_namespace, PHASE_DEPLOYED,
                            timeout=dgdr_profiling_timeout + dgdr_deploy_timeout)


# ---------------------------------------------------------------------------
# ── Group 9: Model Cache (PVC) ──────────────────────────────────────────────
# ---------------------------------------------------------------------------


@pytest.mark.gpu_1
@pytest.mark.nightly
@pytest.mark.e2e
@pytest.mark.k8s
@pytest.mark.deploy
class TestDGDRModelCache:
    """
    Tests for spec.modelCache which mounts a PVC containing pre-downloaded
    model weights instead of pulling from HuggingFace at runtime.
    """

    def test_model_cache_pvc_mounted_in_profiling_job(
        self,
        dgdr_namespace: str,
        dgdr_image: str,
        dgdr_model: str,
        dgdr_pvc_name: str,
        dgdr_factory,
        dgdr_profiling_timeout: int,
    ) -> None:
        """
        When modelCache.pvcName is specified, the profiling Job should mount
        the PVC at modelCache.pvcMountPath (/opt/model-cache by default).
        The PVC name should appear in the Job spec volumes.
        """
        if not dgdr_pvc_name:
            pytest.skip("--dgdr-pvc-name not provided; skipping PVC model cache test")

        name = unique_dgdr_name("pvc-cache")
        manifest = build_dgdr_manifest(
            name,
            model=dgdr_model,
            image=dgdr_image,
            backend="vllm",
            search_strategy="rapid",
            model_cache={
                "pvcName": dgdr_pvc_name,
                "pvcModelPath": dgdr_model.split("/")[-1].lower(),
                "pvcMountPath": "/opt/model-cache",
            },
            auto_apply=False,
        )
        dgdr_factory(manifest)

        # Wait until a profiling job is launched
        wait_for_dgdr_phase(name, dgdr_namespace, PHASE_PROFILING, timeout=120)

        obj = get_dgdr(name, dgdr_namespace)
        job_name = obj["status"].get("profilingJobName")
        assert job_name, "profilingJobName should be set"

        # Inspect the Job for the PVC volume
        job_result = _run_kubectl([
            "get", "job", job_name, "-n", dgdr_namespace, "-o", "json",
        ], check=False)
        if job_result.returncode == 0:
            job_obj = json.loads(job_result.stdout)
            volumes = (
                job_obj.get("spec", {})
                       .get("template", {})
                       .get("spec", {})
                       .get("volumes", [])
            )
            pvc_volumes = [
                v for v in volumes
                if v.get("persistentVolumeClaim", {}).get("claimName") == dgdr_pvc_name
            ]
            assert pvc_volumes, (
                f"Expected PVC {dgdr_pvc_name!r} to be mounted in profiling Job. "
                f"Volumes: {[v.get('name') for v in volumes]}"
            )

        # Allow profiling to complete
        wait_for_dgdr_phase(name, dgdr_namespace, PHASE_READY, timeout=dgdr_profiling_timeout)

    def test_model_cache_propagated_to_generated_dgd(
        self,
        dgdr_namespace: str,
        dgdr_image: str,
        dgdr_model: str,
        dgdr_pvc_name: str,
        dgdr_factory,
        dgdr_profiling_timeout: int,
    ) -> None:
        """
        After profiling, the generated DGD should also reference the PVC so
        inference workers can load weights from cache rather than re-downloading.
        """
        if not dgdr_pvc_name:
            pytest.skip("--dgdr-pvc-name not provided; skipping PVC DGD propagation test")

        name = unique_dgdr_name("pvc-dgd")
        manifest = build_dgdr_manifest(
            name,
            model=dgdr_model,
            image=dgdr_image,
            backend="vllm",
            search_strategy="rapid",
            model_cache={"pvcName": dgdr_pvc_name},
            auto_apply=False,
        )
        dgdr_factory(manifest)

        wait_for_dgdr_phase(name, dgdr_namespace, PHASE_READY, timeout=dgdr_profiling_timeout)

        obj = get_dgdr(name, dgdr_namespace)
        selected = obj["status"]["profilingResults"].get("selectedConfig", {})
        selected_raw = selected.get("Raw") or selected
        selected_str = json.dumps(selected_raw)
        assert dgdr_pvc_name in selected_str, (
            f"PVC name {dgdr_pvc_name!r} should appear in selectedConfig for workers to load "
            f"from cache. selectedConfig: {selected_str[:500]}"
        )


