import concurrent.futures
import json
import sys
import time
from typing import Any

import requests

PORT = sys.argv[1]
LABEL = sys.argv[2]
MODEL = sys.argv[3] if len(sys.argv) > 3 else "dsv2lite"
URL = f"http://localhost:{PORT}/v1/completions"


def one_request(prompt, max_tokens):
    t0 = time.time()
    r = requests.post(
        URL,
        json={"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0},
        timeout=180,
    )
    dt = time.time() - t0
    d = r.json()
    ctoks = d["usage"]["completion_tokens"]
    ptoks = d["usage"]["prompt_tokens"]
    text = d["choices"][0]["text"]
    return dt, ctoks, ptoks, text


results: dict[str, Any] = {}

# 0) multiple deterministic prompts -- real answer-quality correctness check
qa_prompts = [
    "The capital of France is",
    "Explain what a large language model is, in three sentences.",
    "Write a Python function that returns the factorial of n.",
    "2 + 2 =",
    "The opposite of hot is",
]
qa = []
for p in qa_prompts:
    dt, ctoks, ptoks, text = one_request(p, 48)
    qa.append({"prompt": p, "text": text})
results["qa"] = qa

# 1) single-request short-decode
dt, ctoks, ptoks, _ = one_request("Write a short story about a robot learning to paint.", 256)
results["single_256tok"] = {"seconds": dt, "completion_tokens": ctoks, "tok_per_s": ctoks / dt}

# 2) concurrent throughput
N = 16
prompt = "Write a short story about a robot learning to paint."
t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=N) as ex:
    futs = [ex.submit(one_request, prompt, 200) for _ in range(N)]
    reqs = [f.result() for f in futs]
wall = time.time() - t0
total_ctoks = sum(r[1] for r in reqs)
results["concurrent_16x200tok"] = {
    "wall_seconds": wall,
    "total_completion_tokens": total_ctoks,
    "aggregate_tok_per_s": total_ctoks / wall,
}

# 3) long-context prefill, cold + warm
long_prompt = ("The history of artificial intelligence is a long and winding one. " * 130)[:9000]
dt_cold, ctoks, ptoks, _ = one_request(long_prompt, 32)
dt_warm, ctoks, ptoks, _ = one_request(long_prompt, 32)
results["long_prefill"] = {
    "seconds_cold": dt_cold,
    "seconds_warm": dt_warm,
    "prompt_tokens": ptoks,
}

print(f"=== {LABEL} (port {PORT}, model={MODEL}) ===")
print(json.dumps(results, indent=2))
