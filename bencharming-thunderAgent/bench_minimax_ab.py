#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
MiniMax A/B Benchmark: ThunderAgent vs KvRouter
Assumes port-forwards are already set up:
  kubectl port-forward pod/<ta-frontend> 8200:8000
  kubectl port-forward pod/<kv-frontend> 8201:8000
"""
import json, time, asyncio, aiohttp, random, string, uuid, sys, os

MODEL_NAME = "MiniMaxAI/MiniMax-M2"
NUM_PROMPTS = int(os.environ.get("NUM_PROMPTS", "60"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "8"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))
TA_PORT = int(os.environ.get("TA_PORT", "8200"))
KV_PORT = int(os.environ.get("KV_PORT", "8201"))


async def send_request(session, url, payload, headers=None):
    start = time.perf_counter()
    tokens = 0
    ttft = None
    response_text = ""
    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            async for line in resp.content:
                decoded = line.decode().strip()
                if decoded.startswith("data: ") and decoded != "data: [DONE]":
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    tokens += 1
                    try:
                        chunk = json.loads(decoded[6:])
                        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        response_text += delta
                    except Exception:
                        pass
    except Exception as e:
        return {"error": str(e)}, ""
    total = time.perf_counter() - start
    return {"ttft": ttft, "total": total, "tokens": tokens,
            "tps": tokens / total if total > 0 else 0}, response_text


async def run_single_turn(base_url, label):
    url = f"{base_url}/v1/chat/completions"
    sem = asyncio.Semaphore(CONCURRENCY)
    prompts = []
    for i in range(NUM_PROMPTS):
        prefix = "".join(random.choices(string.ascii_letters, k=50))
        prompts.append({
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": f"{prefix} Write a short story about topic {i}. Be creative."}],
            "max_tokens": MAX_TOKENS,
            "stream": True,
        })

    async def bounded(p):
        async with sem:
            r, _ = await send_request(session, url, p)
            return r

    async with aiohttp.ClientSession() as session:
        start = time.perf_counter()
        results = await asyncio.gather(*[bounded(p) for p in prompts])
        wall_time = time.perf_counter() - start

    valid = [r for r in results if "error" not in r and r.get("ttft") is not None]
    if not valid:
        return {"label": label, "error": "no successful requests"}
    ttfts = sorted([r["ttft"] for r in valid])
    return {
        "label": label, "type": "single-turn",
        "successful": len(valid), "wall_time_s": round(wall_time, 2),
        "throughput_rps": round(len(valid) / wall_time, 2),
        "avg_ttft_ms": round(sum(ttfts) * 1000 / len(valid), 1),
        "p50_ttft_ms": round(ttfts[len(valid) // 2] * 1000, 1),
        "p99_ttft_ms": round(ttfts[int(len(valid) * 0.99)] * 1000, 1),
        "avg_tps": round(sum(r["tps"] for r in valid) / len(valid), 1),
    }


async def run_multi_turn(base_url, label):
    url = f"{base_url}/v1/chat/completions"
    sem = asyncio.Semaphore(CONCURRENCY)
    num_sessions = NUM_PROMPTS // 3
    turns_per_session = 3

    async def run_session(session_http):
        session_id = str(uuid.uuid4())
        prefix = "".join(random.choices(string.ascii_letters, k=30))
        messages = [{"role": "user", "content": f"{prefix} You are a helpful coding assistant. Help me write a Python web server."}]
        results = []
        follow_ups = [
            "Now add error handling to the server.",
            "Add logging and make it production-ready.",
        ]
        for turn in range(turns_per_session):
            payload = {
                "model": MODEL_NAME,
                "messages": messages,
                "max_tokens": MAX_TOKENS,
                "stream": True,
            }
            headers = {"x-dynamo-session-id": session_id}
            async with sem:
                result, response = await send_request(session_http, url, payload, headers=headers)
            result["turn"] = turn
            result["session_id"] = session_id
            results.append(result)
            if "error" in result:
                break
            messages.append({"role": "assistant", "content": response[:200]})
            if turn < len(follow_ups):
                messages.append({"role": "user", "content": follow_ups[turn]})
        return results

    async with aiohttp.ClientSession() as session_http:
        start = time.perf_counter()
        all_sessions = await asyncio.gather(*[run_session(session_http) for _ in range(num_sessions)])
        wall_time = time.perf_counter() - start

    all_results = [r for s in all_sessions for r in s]
    valid = [r for r in all_results if "error" not in r and r.get("ttft") is not None]
    if not valid:
        return {"label": label, "error": "no successful requests"}

    ttfts = sorted([r["ttft"] for r in valid])
    turn_stats = {}
    for turn in range(turns_per_session):
        turn_results = [r for r in valid if r.get("turn") == turn]
        if turn_results:
            t_ttfts = sorted([r["ttft"] for r in turn_results])
            turn_stats[f"turn_{turn}"] = {
                "count": len(turn_results),
                "avg_ttft_ms": round(sum(t_ttfts) * 1000 / len(turn_results), 1),
                "p50_ttft_ms": round(t_ttfts[len(turn_results) // 2] * 1000, 1),
                "p99_ttft_ms": round(t_ttfts[int(len(turn_results) * 0.99)] * 1000, 1),
                "avg_tps": round(sum(r["tps"] for r in turn_results) / len(turn_results), 1),
                "avg_e2e_s": round(sum(r["total"] for r in turn_results) / len(turn_results), 2),
            }

    # Per-session stats: total time for all turns in a session
    session_times = {}
    for s in all_sessions:
        valid_turns = [r for r in s if "error" not in r and r.get("ttft") is not None]
        if valid_turns:
            sid = valid_turns[0].get("session_id", "unknown")
            session_times[sid] = sum(r["total"] for r in valid_turns)
    session_durations = sorted(session_times.values()) if session_times else [0]
    session_stats = {
        "num_complete": sum(1 for s in all_sessions if len([r for r in s if "error" not in r and r.get("ttft") is not None]) == turns_per_session),
        "avg_session_s": round(sum(session_durations) / len(session_durations), 2),
        "p50_session_s": round(session_durations[len(session_durations) // 2], 2),
        "p99_session_s": round(session_durations[int(len(session_durations) * 0.99)], 2),
    }

    return {
        "label": label, "type": "multi-turn",
        "num_sessions": num_sessions, "turns_per_session": turns_per_session,
        "total_requests": len(all_results), "successful": len(valid),
        "failed": len(all_results) - len(valid),
        "success_rate": round(len(valid) / len(all_results) * 100, 1) if all_results else 0,
        "wall_time_s": round(wall_time, 2),
        "throughput_rps": round(len(valid) / wall_time, 2),
        "total_tokens": sum(r["tokens"] for r in valid),
        "cluster_tps": round(sum(r["tokens"] for r in valid) / wall_time, 1),
        "avg_ttft_ms": round(sum(ttfts) * 1000 / len(valid), 1),
        "p50_ttft_ms": round(ttfts[len(valid) // 2] * 1000, 1),
        "p99_ttft_ms": round(ttfts[int(len(valid) * 0.99)] * 1000, 1),
        "avg_e2e_s": round(sum(r["total"] for r in valid) / len(valid), 2),
        "avg_tps": round(sum(r["tps"] for r in valid) / len(valid), 1),
        "per_turn": turn_stats,
        "per_session": session_stats,
    }


def print_comparison(ta, kv, test_type):
    if "error" in ta or "error" in kv:
        print(f"  TA: {ta.get('error', 'ok')}  KV: {kv.get('error', 'ok')}")
        return
    keys = ["avg_ttft_ms", "p50_ttft_ms", "p99_ttft_ms", "avg_tps", "throughput_rps",
            "cluster_tps", "avg_e2e_s", "success_rate"]
    present = [k for k in keys if k in ta and k in kv]
    print(f"\n{'Metric':<20} {'ThunderAgent':>14} {'KvRouter':>14} {'Delta':>10}")
    print("-" * 62)
    for key in present:
        delta = ta[key] - kv[key]
        sign = "+" if delta > 0 else ""
        print(f"{key:<20} {ta[key]:>14} {kv[key]:>14} {sign}{delta:>9.1f}")
    if "per_turn" in ta and "per_turn" in kv:
        print(f"\n{'Per-turn TTFT':<20} {'ThunderAgent':>14} {'KvRouter':>14} {'Delta':>10}")
        print("-" * 62)
        for turn_key in sorted(set(list(ta["per_turn"].keys()) + list(kv["per_turn"].keys()))):
            ta_t = ta["per_turn"].get(turn_key, {}).get("avg_ttft_ms", 0)
            kv_t = kv["per_turn"].get(turn_key, {}).get("avg_ttft_ms", 0)
            delta = ta_t - kv_t
            sign = "+" if delta > 0 else ""
            print(f"{turn_key:<20} {ta_t:>12.1f}ms {kv_t:>12.1f}ms {sign}{delta:>9.1f}")
    if "per_session" in ta and "per_session" in kv:
        print(f"\n{'Session stats':<20} {'ThunderAgent':>14} {'KvRouter':>14} {'Delta':>10}")
        print("-" * 62)
        for key in ["num_complete", "avg_session_s", "p50_session_s", "p99_session_s"]:
            ta_v = ta["per_session"].get(key, 0)
            kv_v = kv["per_session"].get(key, 0)
            delta = ta_v - kv_v
            sign = "+" if delta > 0 else ""
            print(f"{key:<20} {ta_v:>14} {kv_v:>14} {sign}{delta:>9.1f}")


async def run_high_contention(base_url, label):
    """Test C: Many concurrent sessions to stress pause/resume and admission."""
    url = f"{base_url}/v1/chat/completions"
    num_sessions = max(NUM_PROMPTS // 2, 30)  # More sessions than workers
    turns_per_session = 3
    # Higher concurrency to create real contention
    contention_concurrency = max(CONCURRENCY * 2, 16)
    sem = asyncio.Semaphore(contention_concurrency)

    async def run_session(session_http):
        session_id = str(uuid.uuid4())
        prefix = "".join(random.choices(string.ascii_letters, k=30))
        messages = [{"role": "user", "content": f"{prefix} Implement a complete REST API with CRUD operations, authentication, and database integration."}]
        results = []
        follow_ups = [
            "Add comprehensive input validation, rate limiting, and error handling middleware.",
            "Add unit tests, integration tests, and API documentation with examples.",
        ]
        for turn in range(turns_per_session):
            payload = {
                "model": MODEL_NAME,
                "messages": messages,
                "max_tokens": MAX_TOKENS,
                "stream": True,
            }
            headers = {"x-dynamo-session-id": session_id}
            async with sem:
                result, response = await send_request(session_http, url, payload, headers=headers)
            result["turn"] = turn
            result["session_id"] = session_id
            results.append(result)
            if "error" in result:
                break
            messages.append({"role": "assistant", "content": response[:200]})
            if turn < len(follow_ups):
                messages.append({"role": "user", "content": follow_ups[turn]})
        return results

    async with aiohttp.ClientSession() as session_http:
        start = time.perf_counter()
        all_sessions = await asyncio.gather(*[run_session(session_http) for _ in range(num_sessions)])
        wall_time = time.perf_counter() - start

    all_results = [r for s in all_sessions for r in s]
    valid = [r for r in all_results if "error" not in r and r.get("ttft") is not None]
    if not valid:
        return {"label": label, "error": "no successful requests",
                "num_sessions": num_sessions, "concurrency": contention_concurrency}

    ttfts = sorted([r["ttft"] for r in valid])
    turn_stats = {}
    for turn in range(turns_per_session):
        turn_results = [r for r in valid if r.get("turn") == turn]
        if turn_results:
            t_ttfts = sorted([r["ttft"] for r in turn_results])
            turn_stats[f"turn_{turn}"] = {
                "count": len(turn_results),
                "avg_ttft_ms": round(sum(t_ttfts) * 1000 / len(turn_results), 1),
                "p50_ttft_ms": round(t_ttfts[len(turn_results) // 2] * 1000, 1),
                "p99_ttft_ms": round(t_ttfts[int(len(turn_results) * 0.99)] * 1000, 1),
                "avg_tps": round(sum(r["tps"] for r in turn_results) / len(turn_results), 1),
            }

    session_times = {}
    for s in all_sessions:
        valid_turns = [r for r in s if "error" not in r and r.get("ttft") is not None]
        if valid_turns:
            sid = valid_turns[0].get("session_id", "unknown")
            session_times[sid] = sum(r["total"] for r in valid_turns)
    session_durations = sorted(session_times.values()) if session_times else [0]
    complete_sessions = sum(1 for s in all_sessions
                           if len([r for r in s if "error" not in r and r.get("ttft") is not None]) == turns_per_session)

    return {
        "label": label, "type": "high-contention",
        "num_sessions": num_sessions, "turns_per_session": turns_per_session,
        "concurrency": contention_concurrency,
        "total_requests": len(all_results), "successful": len(valid),
        "failed": len(all_results) - len(valid),
        "success_rate": round(len(valid) / len(all_results) * 100, 1) if all_results else 0,
        "wall_time_s": round(wall_time, 2),
        "throughput_rps": round(len(valid) / wall_time, 2),
        "total_tokens": sum(r["tokens"] for r in valid),
        "cluster_tps": round(sum(r["tokens"] for r in valid) / wall_time, 1),
        "avg_ttft_ms": round(sum(ttfts) * 1000 / len(valid), 1),
        "p50_ttft_ms": round(ttfts[len(valid) // 2] * 1000, 1),
        "p99_ttft_ms": round(ttfts[int(len(valid) * 0.99)] * 1000, 1),
        "avg_e2e_s": round(sum(r["total"] for r in valid) / len(valid), 2),
        "avg_tps": round(sum(r["tps"] for r in valid) / len(valid), 1),
        "per_turn": turn_stats,
        "per_session": {
            "num_complete": complete_sessions,
            "completion_rate": round(complete_sessions / num_sessions * 100, 1),
            "avg_session_s": round(sum(session_durations) / len(session_durations), 2),
            "p50_session_s": round(session_durations[len(session_durations) // 2], 2),
            "p99_session_s": round(session_durations[int(len(session_durations) * 0.99)], 2),
        },
    }


async def main():
    ta_url = f"http://127.0.0.1:{TA_PORT}"
    kv_url = f"http://127.0.0.1:{KV_PORT}"

    print("=" * 62)
    print(f"MiniMax-M2 A/B Benchmark: ThunderAgent vs KvRouter")
    print(f"Prompts={NUM_PROMPTS}  Concurrency={CONCURRENCY}  MaxTokens={MAX_TOKENS}")
    print("=" * 62)

    # Single-turn
    print("\n" + "=" * 62)
    print("TEST A: Single-turn (no session affinity)")
    print("=" * 62)
    print("\nRunning ThunderAgent...")
    ta_st = await run_single_turn(ta_url, "ThunderAgent")
    print(json.dumps(ta_st, indent=2))
    print("\nRunning KvRouter...")
    kv_st = await run_single_turn(kv_url, "KvRouter")
    print(json.dumps(kv_st, indent=2))
    print_comparison(ta_st, kv_st, "single-turn")

    # Multi-turn
    print("\n" + "=" * 62)
    print("TEST B: Multi-turn (session affinity)")
    print("=" * 62)
    print("\nRunning ThunderAgent...")
    ta_mt = await run_multi_turn(ta_url, "ThunderAgent")
    print(json.dumps(ta_mt, indent=2))
    print("\nRunning KvRouter...")
    kv_mt = await run_multi_turn(kv_url, "KvRouter")
    print(json.dumps(kv_mt, indent=2))
    print_comparison(ta_mt, kv_mt, "multi-turn")

    # High-contention
    print("\n" + "=" * 62)
    print("TEST C: High-contention (many sessions, stress pause/resume)")
    print("=" * 62)
    print("\nRunning ThunderAgent...")
    ta_hc = await run_high_contention(ta_url, "ThunderAgent")
    print(json.dumps(ta_hc, indent=2))
    print("\nRunning KvRouter...")
    kv_hc = await run_high_contention(kv_url, "KvRouter")
    print(json.dumps(kv_hc, indent=2))
    print_comparison(ta_hc, kv_hc, "high-contention")

    # Save results
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": MODEL_NAME,
        "config": {"num_prompts": NUM_PROMPTS, "concurrency": CONCURRENCY, "max_tokens": MAX_TOKENS},
        "single_turn": {"thunderagent": ta_st, "kvrouter": kv_st},
        "multi_turn": {"thunderagent": ta_mt, "kvrouter": kv_mt},
        "high_contention": {"thunderagent": ta_hc, "kvrouter": kv_hc},
    }
    outfile = f"benchmark-results-{time.strftime('%Y%m%d-%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    asyncio.run(main())
