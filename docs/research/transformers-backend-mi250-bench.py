import concurrent.futures
import json
import sys
import time

import requests

PORT = sys.argv[1]
LABEL = sys.argv[2]
URL = f"http://localhost:{PORT}/v1/completions"
MODEL = "Qwen/Qwen3-4B"


def one_request(prompt, max_tokens):
    t0 = time.time()
    r = requests.post(
        URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        timeout=120,
    )
    dt = time.time() - t0
    d = r.json()
    ctoks = d["usage"]["completion_tokens"]
    ptoks = d["usage"]["prompt_tokens"]
    return dt, ctoks, ptoks


results = {}

# 1) single-request short-decode (already have from earlier manual curl, re-measure for
# consistency in one script)
dt, ctoks, ptoks = one_request("Write a short story about a robot learning to paint.", 256)
results["single_256tok"] = {"seconds": dt, "completion_tokens": ctoks, "tok_per_s": ctoks / dt}

# 2) concurrent throughput: 16 concurrent requests, 200 output tokens each
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
    "per_request_seconds": [round(r[0], 3) for r in reqs],
}

# 3) long-context prefill: ~2000-token prompt, short decode (32 tokens) to isolate prefill cost
long_prompt = ("The history of artificial intelligence is a long and winding one. " * 130)[:9000]
dt, ctoks, ptoks = one_request(long_prompt, 32)
results["long_prefill"] = {
    "seconds": dt,
    "prompt_tokens": ptoks,
    "completion_tokens": ctoks,
}

print(f"=== {LABEL} (port {PORT}) ===")
print(json.dumps(results, indent=2))
