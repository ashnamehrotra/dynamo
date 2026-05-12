# DGDR Recipe Lifecycle Test Matrix

Generated: Mon May 11 02:10:08 EDT 2026
Namespace: `dynamo-test` | Image: `nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.1.0` | PVC: `model-cache`

Runs the **entire** `./test/e2e/dgdr/` suite per recipe (Lifecycle + Lifecycle Scenarios + Profiling + Validation).
Per-recipe columns: passed / failed / skipped / total. Failure details are in `results/<name>.reason`.

| Recipe | Model | Backend | Mode | GPUs | Pass | Fail | Skip | Total | First failure |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| l3-70b-vllm-agg | nvidia/Llama-3.1-70B-Instruct-FP8 | vllm | agg | 4 | 17 | 0 | 14 | 31 | OK |
| l3-70b-vllm-dis | nvidia/Llama-3.1-70B-Instruct-FP8 | vllm | disagg-single-node | 8 | 17 | 0 | 14 | 31 | OK |
| q235b-tllm-agg | Qwen/Qwen3-235B-A22B-FP8 | trtllm | agg-hopper | 16 | 17 | 0 | 14 | 31 | OK |
| q235b-tllm-dis | Qwen/Qwen3-235B-A22B-FP8 | trtllm | disagg-hopper | 16 | 17 | 0 | 14 | 31 | OK |
| n3s-vllm-agg | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 | vllm | agg | 4 | 17 | 0 | 14 | 31 | OK |
| n3s-sglang-agg | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 | sglang | agg | 4 | 17 | 0 | 14 | 31 | OK |
| n3s-tllm-dis | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 | trtllm | disagg | 4 | 17 | 0 | 14 | 31 | OK |
| dsr1-sgl-d8 | deepseek-ai/DeepSeek-R1 | sglang | disagg-8gpu | 16 | - | - | - | - | known to fail (#8571) |
| dsr1-sgl-d16 | deepseek-ai/DeepSeek-R1 | sglang | disagg-16gpu | 32 | - | - | - | - | uses entire cluster |
| dsr1-vllm-dep16 | deepseek-ai/DeepSeek-R1 | vllm | disagg-dep16 | 32 | - | - | - | - | uses entire cluster |
| q32b-tllm-agg | Qwen/Qwen3-32B-FP8 | trtllm | agg | 2 | - | - | - | - | model not on PVC; needs download |
| q32b-tllm-dis | Qwen/Qwen3-32B-FP8 | trtllm | disagg | 8 | - | - | - | - | model not on PVC; needs download |
| q32b-vllm-router | Qwen/Qwen3-32B | vllm | disagg-kv-router | 16 | - | - | - | - | model not on PVC; needs download |
| l33-rh-vllm-agg | RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic | vllm | agg | 4 | - | - | - | - | AIC: compressed-tensors quant unsupported |
| gpt-oss-120b | openai/gpt-oss-120b | trtllm | agg | 4 | - | - | - | - | Blackwell-only |
| deepseek-v3.2-nvfp4 | deepseek-ai/DeepSeek-V3.2 | trtllm | disagg | 32 | - | - | - | - | Blackwell-only |
| kimi-k2.5-nvfp4 | moonshotai/Kimi-K2.5 | trtllm | disagg | 8 | - | - | - | - | Blackwell-only |
| glm-5-nvfp4 | zai-org/GLM-5 | trtllm | disagg | 20 | - | - | - | - | Blackwell-only |
| qwen3-vl-30b | Qwen/Qwen3-VL-30B | vllm | agg | 1 | - | - | - | - | Blackwell-only (multimodal) |
| nemotron-3-nano-omni | nvidia/Nemotron-3-Nano-Omni | custom | agg | 1 | - | - | - | - | Blackwell-only (custom container) |
| deepseek-v4-flash | deepseek-ai/DeepSeek-V4-Flash | trtllm | disagg | 8 | - | - | - | - | Blackwell-only |
| deepseek-v4-pro | deepseek-ai/DeepSeek-V4-Pro | trtllm | disagg | 16 | - | - | - | - | Blackwell-only |
| qwen3-235b-blackwell | Qwen/Qwen3-235B-A22B-FP8 | trtllm | agg-blackwell | 8 | - | - | - | - | Blackwell-only |
| q32b-vllm-a100 | Qwen/Qwen3-32B-FP8 | vllm | disagg | 8 | - | - | - | - | A100-only |
