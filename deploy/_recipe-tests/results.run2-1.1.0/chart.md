# DGDR Recipe Lifecycle Test Matrix

Generated: Mon May 11 09:34:20 EDT 2026
Namespace: `dynamo-test` | Image: `nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.1.0` | PVC: `model-cache`

Runs the **entire** `./test/e2e/dgdr/` suite per recipe (Lifecycle + Lifecycle Scenarios + Profiling + Validation).
Per-recipe columns: passed / failed / skipped / total. Failure details are in `results/<name>.reason`.

| Recipe | Model | Backend | Mode | GPUs | Pass | Fail | Skip | Total | First failure |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| l3-70b-vllm-agg | nvidia/Llama-3.1-70B-Instruct-FP8 | vllm | agg | 4 | 19 | 0 | 12 | 31 | OK |
