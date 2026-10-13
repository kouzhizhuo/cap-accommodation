"""Shared helpers for the P3 harness (stdlib only, so it runs in any env)."""
import hashlib, json, os, platform, sys, time
from pathlib import Path


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def file_sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def env_info():
    info = {"python": sys.version.split()[0], "host": platform.node(), "time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        import vllm, torch, transformers
        info.update(vllm=vllm.__version__, torch=torch.__version__, transformers=transformers.__version__)
    except Exception:
        pass
    return info


def request_seed(seed, idx):
    """Deterministic per-request seed so a (seed, prompt index) pair always reproduces the same sample."""
    return int(seed) * 1_000_003 + int(idx)
