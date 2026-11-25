#!/usr/bin/env python3
"""pi_bench.py: unattended Pi coding-agent patch-and-test runner with hard tool-call & time budgets.

Runs Pi non-interactively on self-contained disposable Python bugfix repos (including security/defense
utilities) with an isolated PI_CODING_AGENT_DIR so user ~/.pi config is never modified.

Captures the exact failure modes HumanEval+ misses:
  - patch_passes_tests    : visible + hidden tests pass after agent finishes
  - final_answer_returned : agent produced a non-empty final assistant reply and stopped on its own
  - tool_calls            : exact count of tool executions
  - budget_exceeded       : killed because tool_calls > --max-tool-calls (catches infinite tool loops)
  - timed_out             : killed because wall-clock > --timeout
  - elapsed_s             : wall-clock seconds

Works against any OpenAI-compatible server (Maya runtime-ablation preset, standalone GGUF via
llama-server, or Prometheus LAN router), with optional --ssh-tunnel HOST:REMOTE_PORT.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import select
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    task_id: str
    prompt: str
    files: dict[str, str]
    hidden_test_path: str
    hidden_test_code: str


TASKS: list[Task] = [
    Task(
        task_id="cidr_acl",
        prompt=(
            "Fix the bug in `acl.py` so `tests/test_acl.py` passes. "
            "Run `python3 -m unittest discover -s tests` to verify, then summarize what you changed in one short paragraph."
        ),
        files={
            "acl.py": '''import ipaddress


def is_allowed(ip_str: str, allow_cidrs: list[str], deny_cidrs: list[str] | None = None) -> bool:
    """Return True if ip_str matches at least one allow CIDR and no deny CIDR.
    Deny rules always take precedence over allow rules. Invalid IPs return False.
    """
    try:
        addr = ipaddress.ip_address(ip_str.strip())
    except ValueError:
        return False

    for net_s in allow_cidrs:
        net = ipaddress.ip_network(net_s, strict=False)
        if addr.version == net.version and addr in net:
            return True

    for net_s in (deny_cidrs or []):
        net = ipaddress.ip_network(net_s, strict=False)
        if addr.version == net.version and addr in net:
            return False

    return False
''',
            "tests/test_acl.py": '''import unittest
from acl import is_allowed


class TestAcl(unittest.TestCase):
    def test_deny_overrides_allow(self):
        self.assertTrue(is_allowed("10.0.1.5", ["10.0.0.0/8"], ["10.0.2.0/24"]))
        self.assertFalse(is_allowed("10.0.2.5", ["10.0.0.0/8"], ["10.0.2.0/24"]))

    def test_invalid_ip(self):
        self.assertFalse(is_allowed("999.1.1.1", ["0.0.0.0/0"]))


if __name__ == "__main__":
    unittest.main()
''',
        },
        hidden_test_path="tests/test_hidden_acl.py",
        hidden_test_code='''import unittest
from acl import is_allowed


class TestHiddenAcl(unittest.TestCase):
    def test_ipv6_and_exact_host_deny(self):
        self.assertTrue(is_allowed("2001:db8::1", ["2001:db8::/32"], ["2001:db8::2/128"]))
        self.assertFalse(is_allowed("2001:db8::2", ["2001:db8::/32"], ["2001:db8::2/128"]))
        self.assertFalse(is_allowed("10.0.0.1", ["2001:db8::/32"], []))
        self.assertFalse(is_allowed("192.168.1.10", [], ["192.168.1.0/24"]))
''',
    ),
    Task(
        task_id="ssh_log_triage",
        prompt=(
            "Fix the bug in `triage.py` so `tests/test_triage.py` passes. "
            "Run `python3 -m unittest discover -s tests` to verify, then summarize what you changed in one short paragraph."
        ),
        files={
            "triage.py": '''import re
from collections import Counter

FAILED_RE = re.compile(r"Failed password for (?:invalid user )?\\S+ from (\\d+\\.\\d+\\.\\d+\\.\\d+)")


def top_bruteforce_ips(lines: list[str], threshold: int = 3) -> list[tuple[str, int]]:
    """Return (ip, count) pairs with count >= threshold, sorted by count descending then IP ascending."""
    counts = Counter()
    for line in lines:
        m = FAILED_RE.search(line)
        if m:
            counts[m.group(1)] += 1
    filtered = [(ip, n) for ip, n in counts.items() if n > threshold]
    filtered.sort(key=lambda item: (item[1], item[0]))
    return filtered
''',
            "tests/test_triage.py": '''import unittest
from triage import top_bruteforce_ips


class TestTriage(unittest.TestCase):
    def test_threshold_and_ordering(self):
        lines = [
            "Nov 24 10:00:01 sshd: Failed password for root from 203.0.113.9 port 22",
            "Nov 24 10:00:02 sshd: Failed password for root from 203.0.113.9 port 22",
            "Nov 24 10:00:03 sshd: Failed password for invalid user admin from 203.0.113.9 port 22",
            "Nov 24 10:00:04 sshd: Failed password for root from 198.51.100.2 port 22",
            "Nov 24 10:00:05 sshd: Failed password for root from 198.51.100.2 port 22",
            "Nov 24 10:00:06 sshd: Failed password for root from 198.51.100.2 port 22",
            "Nov 24 10:00:07 sshd: Failed password for root from 198.51.100.2 port 22",
            "Nov 24 10:00:08 sshd: Accepted publickey for ops from 198.51.100.2 port 22",
        ]
        self.assertEqual(
            top_bruteforce_ips(lines, threshold=3),
            [("198.51.100.2", 4), ("203.0.113.9", 3)],
        )


if __name__ == "__main__":
    unittest.main()
''',
        },
        hidden_test_path="tests/test_hidden_triage.py",
        hidden_test_code='''import unittest
from triage import top_bruteforce_ips


class TestHiddenTriage(unittest.TestCase):
    def test_tiebreak_ascending_ip(self):
        lines = [
            "Failed password for a from 10.0.0.2 port 1",
            "Failed password for a from 10.0.0.2 port 2",
            "Failed password for a from 10.0.0.1 port 1",
            "Failed password for a from 10.0.0.1 port 2",
            "Failed password for a from 10.0.0.9 port 1",
        ]
        self.assertEqual(
            top_bruteforce_ips(lines, threshold=2),
            [("10.0.0.1", 2), ("10.0.0.2", 2)],
        )
''',
    ),
    Task(
        task_id="sliding_rate_limit",
        prompt=(
            "Fix the bug in `limiter.py` so `tests/test_limiter.py` passes. "
            "Run `python3 -m unittest discover -s tests` to verify, then summarize what you changed in one short paragraph."
        ),
        files={
            "limiter.py": '''from collections import deque


class SlidingWindowLimiter:
    """Allow up to max_requests per key within a window_seconds sliding window.
    An event at timestamp t0 expires when t - t0 >= window_seconds.
    Rejected requests must NOT consume quota.
    """

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = {}

    def allow(self, key: str, now: float) -> bool:
        q = self._events.setdefault(key, deque())
        while q and (now - q[0]) > self.window_seconds:
            q.popleft()
        q.append(now)
        return len(q) <= self.max_requests
''',
            "tests/test_limiter.py": '''import unittest
from limiter import SlidingWindowLimiter


class TestLimiter(unittest.TestCase):
    def test_rejected_does_not_extend_block_and_exact_boundary_expires(self):
        lim = SlidingWindowLimiter(max_requests=2, window_seconds=10.0)
        self.assertTrue(lim.allow("k", 0.0))
        self.assertTrue(lim.allow("k", 1.0))
        self.assertFalse(lim.allow("k", 2.0))
        # At t=10.0, t=0.0 has expired (10.0 - 0.0 >= 10.0); t=2.0 was rejected and must not count.
        self.assertTrue(lim.allow("k", 10.0))


if __name__ == "__main__":
    unittest.main()
''',
        },
        hidden_test_path="tests/test_hidden_limiter.py",
        hidden_test_code='''import unittest
from limiter import SlidingWindowLimiter


class TestHiddenLimiter(unittest.TestCase):
    def test_independent_keys_and_burst_rejections(self):
        lim = SlidingWindowLimiter(max_requests=1, window_seconds=5.0)
        self.assertTrue(lim.allow("a", 0.0))
        for t in (1.0, 2.0, 3.0, 4.0):
            self.assertFalse(lim.allow("a", t))
        self.assertTrue(lim.allow("b", 2.0))
        self.assertTrue(lim.allow("a", 5.0))
''',
    ),
    Task(
        task_id="config_merge",
        prompt=(
            "Fix the bug in `cfg/merge.py` so `tests/test_merge.py` passes. "
            "Run `python3 -m unittest discover -s tests` to verify, then summarize what you changed in one short paragraph."
        ),
        files={
            "cfg/__init__.py": "",
            "cfg/merge.py": '''def merge_configs(base: dict, override: dict) -> dict:
    """Recursively merge override into base without mutating either input dict."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k].update(v)
        else:
            out[k] = v
    return out
''',
            "tests/test_merge.py": '''import unittest
from cfg.merge import merge_configs


class TestMerge(unittest.TestCase):
    def test_nested_merge_does_not_mutate_base(self):
        base = {"tls": {"enabled": True, "ciphers": ["a"]}}
        override = {"tls": {"min_version": "1.3"}}
        merged = merge_configs(base, override)
        self.assertEqual(merged["tls"]["enabled"], True)
        self.assertEqual(merged["tls"]["min_version"], "1.3")
        self.assertNotIn("min_version", base["tls"])


if __name__ == "__main__":
    unittest.main()
''',
        },
        hidden_test_path="tests/test_hidden_merge.py",
        hidden_test_code='''import unittest
from cfg.merge import merge_configs


class TestHiddenMerge(unittest.TestCase):
    def test_three_level_nested_no_mutation(self):
        base = {"a": {"b": {"x": 1, "y": 2}}, "list": [1, 2]}
        override = {"a": {"b": {"y": 9, "z": 3}}}
        merged = merge_configs(base, override)
        self.assertEqual(merged["a"]["b"], {"x": 1, "y": 9, "z": 3})
        self.assertEqual(base["a"]["b"], {"x": 1, "y": 2})
        merged["list"].append(3)
        self.assertEqual(base["list"], [1, 2])
''',
    ),
    Task(
        task_id="token_claims",
        prompt=(
            "Fix the bug in `claims.py` so `tests/test_claims.py` passes. "
            "Run `python3 -m unittest discover -s tests` to verify, then summarize what you changed in one short paragraph."
        ),
        files={
            "claims.py": '''def validate_claims(claims: dict, now: float, expected_aud: str, leeway: float = 0.0) -> tuple[bool, str]:
    """Validate exp, nbf, and aud claims.
    - exp is required; reject if now >= exp + leeway with reason 'expired'.
    - nbf is optional; if present, reject if now + leeway < nbf with reason 'not_yet_valid'.
    - aud may be a string or a list of strings; reject if expected_aud does not match with reason 'bad_aud'.
    Return (True, 'ok') if valid.
    """
    exp = claims.get("exp")
    if exp is None or now > exp + leeway:
        return False, "expired"

    aud = claims.get("aud")
    if aud != expected_aud:
        return False, "bad_aud"

    return True, "ok"
''',
            "tests/test_claims.py": '''import unittest
from claims import validate_claims


class TestClaims(unittest.TestCase):
    def test_nbf_and_list_aud(self):
        ok, reason = validate_claims({"exp": 200, "nbf": 150, "aud": ["api", "admin"]}, now=100, expected_aud="api")
        self.assertEqual((ok, reason), (False, "not_yet_valid"))
        ok2, reason2 = validate_claims({"exp": 200, "nbf": 150, "aud": ["api", "admin"]}, now=160, expected_aud="api")
        self.assertEqual((ok2, reason2), (True, "ok"))


if __name__ == "__main__":
    unittest.main()
''',
        },
        hidden_test_path="tests/test_hidden_claims.py",
        hidden_test_code='''import unittest
from claims import validate_claims


class TestHiddenClaims(unittest.TestCase):
    def test_exact_exp_and_leeway(self):
        self.assertEqual(validate_claims({"exp": 100, "aud": "svc"}, now=100, expected_aud="svc"), (False, "expired"))
        self.assertEqual(validate_claims({"exp": 100, "aud": "svc"}, now=104, expected_aud="svc", leeway=5), (True, "ok"))
        self.assertEqual(validate_claims({"exp": 200, "nbf": 105, "aud": "svc"}, now=100, expected_aud="svc", leeway=5), (True, "ok"))
        self.assertEqual(validate_claims({"exp": 200, "aud": ["other"]}, now=100, expected_aud="svc"), (False, "bad_aud"))
''',
    ),
]


def free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def start_ssh_tunnel(spec: str) -> tuple[subprocess.Popen, int]:
    host, port_s = spec.rsplit(":", 1)
    rport = int(port_s)
    lport = free_local_port()
    proc = subprocess.Popen(
        ["ssh", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes", "-N", "-L", f"127.0.0.1:{lport}:127.0.0.1:{rport}", host],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            err = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"ssh tunnel failed: {err.strip()}")
        try:
            with socket.create_connection(("127.0.0.1", lport), timeout=0.3):
                return proc, lport
        except OSError:
            time.sleep(0.1)
    proc.terminate()
    raise RuntimeError(f"timed out waiting for ssh tunnel to {spec}")


def make_agent_dir(base_url: str, model_id: str, max_tokens: int, ctx_window: int) -> pathlib.Path:
    d = pathlib.Path(tempfile.mkdtemp(prefix="pi-bench-agent-"))
    models = {
        "providers": {
            "bench": {
                "baseUrl": base_url.rstrip("/"),
                "api": "openai-completions",
                "apiKey": "local-not-a-secret",
                "models": [
                    {
                        "id": model_id,
                        "name": f"Bench ({model_id})",
                        "reasoning": True,
                        "input": ["text"],
                        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                        "contextWindow": ctx_window,
                        "maxTokens": max_tokens,
                        "samplingParams": {"temperature": 0.0},
                    }
                ],
            }
        }
    }
    (d / "models.json").write_text(json.dumps(models, indent=2) + "\n", encoding="utf-8")
    return d


def run_task(
    task: Task,
    agent_dir: pathlib.Path,
    model_id: str,
    thinking: str,
    max_tool_calls: int,
    timeout_s: int,
    pi_bin: str,
) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"pi-task-{task.task_id}-") as td:
        repo = pathlib.Path(td)
        for rel, content in task.files.items():
            p = repo / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")

        cmd = [
            pi_bin,
            "--offline",
            "--provider", "bench",
            "--model", f"bench/{model_id}",
            "--thinking", thinking,
            "--no-session",
            "--no-context-files",
            "--no-skills",
            "--no-extensions",
            "--no-prompt-templates",
            "--no-mcp",
            "--tools", "read,bash,edit,write",
            "--mode", "json",
            "-p", task.prompt,
        ]
        env = dict(os.environ)
        env["PI_CODING_AGENT_DIR"] = str(agent_dir)

        t0 = time.monotonic()
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        events: list[dict] = []
        tools: list[str] = []
        budget_exceeded = False
        timed_out = False
        assert proc.stdout is not None
        fd = proc.stdout.fileno()

        while True:
            rem = timeout_s - (time.monotonic() - t0)
            if rem <= 0:
                timed_out = True
                proc.kill()
                break
            ready, _, _ = select.select([fd], [], [], min(0.5, rem))
            if ready:
                line = proc.stdout.readline()
                if not line:
                    if proc.poll() is not None:
                        break
                    continue
                if line.startswith("{"):
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    events.append(ev)
                    if ev.get("type") == "tool_execution_end":
                        tools.append(str(ev.get("toolName") or "?"))
                        if len(tools) >= max_tool_calls:
                            budget_exceeded = True
                            proc.kill()
                            break
            elif proc.poll() is not None:
                # Drain any remaining buffered lines
                for line in proc.stdout:
                    if line.startswith("{"):
                        try:
                            ev = json.loads(line)
                            events.append(ev)
                            if ev.get("type") == "tool_execution_end":
                                tools.append(str(ev.get("toolName") or "?"))
                        except json.JSONDecodeError:
                            pass
                break

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        stderr = (proc.stderr.read() if proc.stderr else "")[-600:]
        elapsed = round(time.monotonic() - t0, 2)

        # Extract final assistant reply
        assistant_msgs = [
            e["message"]
            for e in events
            if e.get("type") == "message_end" and e.get("message", {}).get("role") == "assistant"
        ]
        final_text = ""
        if assistant_msgs:
            final_text = "".join(
                item.get("text", "")
                for item in assistant_msgs[-1].get("content", [])
                if item.get("type") == "text"
            ).strip()

        # Check which files changed
        changed_files = [
            rel for rel, orig in task.files.items()
            if not (repo / rel).exists() or (repo / rel).read_text(encoding="utf-8", errors="replace") != orig
        ]

        # Run visible + hidden unit tests after agent finishes
        hp = repo / task.hidden_test_path
        hp.parent.mkdir(parents=True, exist_ok=True)
        hp.write_text(task.hidden_test_code, encoding="utf-8")
        tproc = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=20,
        )
        tests_passed = tproc.returncode == 0
        final_answer_returned = bool(final_text) and not budget_exceeded and not timed_out and proc.returncode == 0

        return {
            "task_id": task.task_id,
            "passed": bool(tests_passed and final_answer_returned),
            "tests_passed": tests_passed,
            "final_answer_returned": final_answer_returned,
            "budget_exceeded": budget_exceeded,
            "timed_out": timed_out,
            "returncode": proc.returncode,
            "tool_calls": len(tools),
            "tools": tools,
            "changed_files": changed_files,
            "elapsed_s": elapsed,
            "final_answer_preview": final_text[:240],
            "test_tail": (tproc.stderr or tproc.stdout)[-300:],
            "stderr_tail": stderr,
        }


def main() -> int:
    ap = argparse.ArgumentParser(description="Unattended Pi coding-agent benchmark with tool-call budget.")
    ap.add_argument("--url", help="OpenAI-compatible base URL including /v1 (e.g. http://127.0.0.1:18087/v1)")
    ap.add_argument("--ssh-tunnel", help="Open temporary SSH tunnel HOST:REMOTE_PORT (e.g. 192.168.100.111:8087)")
    ap.add_argument("--model", default="inkling-2-mini", help="Model ID served by endpoint")
    ap.add_argument("--tag", default="candidate", help="Label for this run")
    ap.add_argument("--thinking", default="medium", help="Pi thinking level (off|minimal|low|medium|high)")
    ap.add_argument("--max-tool-calls", type=int, default=20, help="Kill task if tool calls reach this cap")
    ap.add_argument("--timeout", type=int, default=180, help="Per-task wall-clock timeout in seconds")
    ap.add_argument("--max-tokens", type=int, default=8192, help="Model maxTokens in temporary models.json")
    ap.add_argument("--ctx-window", type=int, default=32768, help="Model contextWindow in temporary models.json")
    ap.add_argument("--tasks", help="Comma-separated subset of task IDs (default: all 5)")
    ap.add_argument("--pi-bin", default=shutil.which("pi") or "pi", help="Path to pi executable")
    ap.add_argument("--out", type=pathlib.Path, help="Optional JSONL path to save per-task results + summary")
    args = ap.parse_args()

    if not args.url and not args.ssh_tunnel:
        ap.error("provide --url or --ssh-tunnel")

    selected = TASKS
    if args.tasks:
        wanted = {t.strip() for t in args.tasks.split(",") if t.strip()}
        selected = [t for t in TASKS if t.task_id in wanted]
        if not selected:
            ap.error(f"no matching tasks in {args.tasks}; valid: {[t.task_id for t in TASKS]}")

    tunnel_proc = None
    agent_dir = None
    try:
        base_url = args.url
        if args.ssh_tunnel:
            tunnel_proc, lport = start_ssh_tunnel(args.ssh_tunnel)
            base_url = f"http://127.0.0.1:{lport}/v1"
        assert base_url
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"

        agent_dir = make_agent_dir(base_url, args.model, args.max_tokens, args.ctx_window)
        results = []
        for task in selected:
            row = run_task(task, agent_dir, args.model, args.thinking, args.max_tool_calls, args.timeout, args.pi_bin)
            row["tag"] = args.tag
            results.append(row)
            print(json.dumps(row), flush=True)

        summary = {
            "type": "summary",
            "tag": args.tag,
            "model": args.model,
            "thinking": args.thinking,
            "tasks": len(results),
            "passed": sum(1 for r in results if r["passed"]),
            "tests_passed": sum(1 for r in results if r["tests_passed"]),
            "final_answer_returned": sum(1 for r in results if r["final_answer_returned"]),
            "budget_exceeded": sum(1 for r in results if r["budget_exceeded"]),
            "timed_out": sum(1 for r in results if r["timed_out"]),
            "total_tool_calls": sum(r["tool_calls"] for r in results),
            "max_tool_calls_used": max((r["tool_calls"] for r in results), default=0),
            "total_elapsed_s": round(sum(r["elapsed_s"] for r in results), 2),
        }
        print(json.dumps(summary), flush=True)

        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
                f.write(json.dumps(summary) + "\n")
    finally:
        if agent_dir and agent_dir.exists():
            shutil.rmtree(agent_dir, ignore_errors=True)
        if tunnel_proc is not None:
            tunnel_proc.terminate()
            try:
                tunnel_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tunnel_proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
