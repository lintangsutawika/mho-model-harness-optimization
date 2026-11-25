#!/usr/bin/env python3
"""labctl: the in-image toolkit of the harnessforge toy benchmark (stdlib only, Python >= 3.10).

Copied into the shared task image as /opt/lab/labctl.py (on PATH as `labctl`). The task generator
on the host imports this same file to compute each task's expected answers, so fixtures and
verifiers can never drift apart. Every fixture is a pure function of (family, seed).

    labctl setup <id>                create the task's workspace under /app (LAB_ROOT)
    labctl huge-log <seed>           print a ~400k-char diagnostic log
    labctl serve                     run the status service (foreground, forever)
    labctl crunch <seed>             a ~60 s computation that prints `result: <value>`
    labctl job {status,info,result}  the lab job queue
    labctl emit <seed>               print one line full of quotes, backslashes and brackets
"""
from __future__ import annotations

import hashlib
import http.server
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("LAB_ROOT", "/app"))

WORDS = ("amber basil cedar delta ember fjord garnet harbor indigo juniper kestrel lumen maple nectar "
         "onyx pepper quartz raven saffron tundra umber violet willow xenon yarrow zephyr anchor beacon "
         "cobalt dune echo falcon glacier heron iris jasper kelp lotus meadow nimbus orchid pine quill "
         "reef sierra thistle ursa vale wren").split()


def rng(*key) -> random.Random:
    return random.Random(":".join(str(k) for k in key))


def digest(*parts, n: int = 12) -> str:
    return hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()[:n]


def _write(path: Path, text: str, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if mode is not None:
        path.chmod(mode)


# ============================================================================ huge_log
def huge_log_data(seed: int) -> dict:
    r = rng("huge_log", seed)
    return {"run_id": digest("run", seed, n=8), "final_checksum": digest("final", seed),
            "lines": 3500 + r.randrange(1500)}


def huge_log_lines(seed: int):
    d = huge_log_data(seed)
    r = rng("huge_log_lines", seed)
    yield f"run_id={d['run_id']} starting diagnostics"
    for i in range(d["lines"]):
        lvl = r.choice(("INFO", "INFO", "INFO", "DEBUG", "WARN"))
        yield (f"2026-03-{1 + i % 28:02d}T{i % 24:02d}:{i % 60:02d}:{(i * 7) % 60:02d} {lvl} worker-{r.randrange(16)} "
               f"batch={r.randrange(10**6):06d} items={r.randrange(500)} took={r.randrange(900)}ms "
               f"partial_checksum={digest('p', seed, i)}")
    yield "==== SUMMARY ===="
    yield f"batches={d['lines']} status=OK"
    yield f"final_checksum={d['final_checksum']}"


# ============================================================================ background_service
def service_expected(secret: str) -> str:
    return "ok " + digest("health", secret.strip())


def crunch_result(seed: int) -> str:
    return digest("crunch", seed, n=10)


def setup_background_service(seed: int) -> None:
    _write(ROOT / "service" / "secret", os.urandom(8).hex() + "\n")


def serve() -> None:
    secret = (ROOT / "service" / "secret").read_text()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = (service_expected(secret) if self.path == "/health" else "not found").encode()
            self.send_response(200 if self.path == "/health" else 404)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    _write(ROOT / "service" / "port", f"{port}\n")
    print(f"lab status service listening on 127.0.0.1:{port} (port recorded in {ROOT / 'service' / 'port'})",
          flush=True)
    srv.serve_forever()


def crunch(seed: int, seconds: float | None = None) -> None:
    total = float(os.environ.get("LAB_CRUNCH_SECONDS", seconds if seconds is not None else 60))
    steps = 12
    for i in range(steps):
        print(f"crunching... {100 * i // steps}%", flush=True)
        time.sleep(total / steps)
    print(f"result: {crunch_result(seed)}", flush=True)


# ============================================================================ deep_dirs
def deep_dirs_data(seed: int) -> dict:
    r = rng("deep_dirs", seed)
    proj = f"proj_{r.choice(WORDS)}_{seed % 1000}"
    mods = r.sample(WORDS, 3)
    values = [r.randrange(10, 999) for _ in mods]
    return {"proj": proj, "modules": mods, "values": values,
            "artifacts": [digest("artifact", m, v) for m, v in zip(mods, values)]}


_CONFIGURE = """#!/bin/sh
# Must be run from inside the module directory.
if [ ! -f module.ini ]; then echo "configure: module.ini not found in $(pwd); run me from the module directory" >&2; exit 2; fi
echo "configured=$(sed -n 's/^name=//p' module.ini)" > .configured
echo "configure: ok"
"""

_BUILD = """#!/bin/sh
# Must be run from inside the module directory, after ./configure.
if [ ! -f module.ini ]; then echo "build: module.ini not found in $(pwd); run me from the module directory" >&2; exit 2; fi
if [ ! -f .configured ]; then echo "build: not configured; run ./configure first" >&2; exit 3; fi
name=$(sed -n 's/^name=//p' module.ini)
value=$(tr -d ' \\n' < params/value.txt)
mkdir -p out
python3 -c "import hashlib,sys; print(hashlib.sha256(('artifact:%s:%s' % (sys.argv[1], sys.argv[2])).encode()).hexdigest()[:12])" "$name" "$value" > out/artifact.txt
echo "build: wrote out/artifact.txt"
"""


def setup_deep_dirs(seed: int) -> None:
    d = deep_dirs_data(seed)
    base = ROOT / "workspace" / d["proj"] / "modules"
    for m in d["modules"]:
        _write(base / m / "module.ini", f"name={m}\n")
        _write(base / m / "params" / "value.txt", "0\n")
        _write(base / m / "configure", _CONFIGURE, 0o755)
        _write(base / m / "build", _BUILD, 0o755)


# ============================================================================ bulk_write
def bulk_write_data(seed: int) -> dict:
    r = rng("bulk_write", seed)
    n = 120 + r.randrange(60)
    lines = [f"{i + 1:03d} {r.choice(WORDS)}-{r.choice(WORDS)} {r.randrange(10**5):05d} {r.choice(WORDS).upper()}"
             for i in range(n)]
    return {"lines": lines}


# ============================================================================ stuck_poll
def stuck_poll_data(seed: int) -> dict:
    return {"job": digest("job", seed, n=6), "code": digest("approve", seed, n=8),
            "result": digest("jobresult", seed, n=10)}


def setup_stuck_poll(seed: int) -> None:
    d = stuck_poll_data(seed)
    q = ROOT / "queue"
    _write(q / "jobs" / f"{d['job']}.job", json.dumps({"job": d["job"], "seed": seed}) + "\n")
    _write(q / "README.md", (
        "# Lab job queue\n\nJobs are executed in submission order. Jobs submitted from a shared account are held "
        "until they are approved: write the job's approval code (shown by `labctl job info`) into the file "
        f"`{q}/approvals/<job id>`. Approved jobs finish within a few seconds.\n"))


def _job_state(seed: int) -> str:
    d = stuck_poll_data(seed)
    f = ROOT / "queue" / "approvals" / d["job"]
    if f.exists() and d["code"] in f.read_text():
        return "DONE"
    return "PENDING"


def job(argv: list[str]) -> int:
    jobs = sorted((ROOT / "queue" / "jobs").glob("*.job"))
    if not argv or argv[0] in ("-h", "--help", "help"):
        print("usage: labctl job {status,info,result}\n  status  show the state of the queued job\n"
              "  info    show job details, including its approval code\n  result  print the result of a finished job\n"
              f"See {ROOT / 'queue' / 'README.md'} for how the queue works.")
        return 0
    if not jobs:
        print("no jobs queued", file=sys.stderr)
        return 1
    seed = json.loads(jobs[0].read_text())["seed"]
    d = stuck_poll_data(seed)
    state = _job_state(seed)
    if argv[0] == "status":
        print(f"job {d['job']}: {state}" + (" (waiting)" if state == "PENDING" else ""))
    elif argv[0] == "info":
        print(f"job {d['job']}\n  submitted by: shared-account\n  approval code: {d['code']}\n  state: {state}")
    elif argv[0] == "result":
        if state != "DONE":
            print(f"job {d['job']} has not finished", file=sys.stderr)
            return 1
        print(d["result"])
    else:
        print(f"unknown job command {argv[0]!r}", file=sys.stderr)
        return 2
    return 0


# ============================================================================ json_escape
def json_escape_data(seed: int) -> dict:
    r = rng("json_escape", seed)
    w = lambda: r.choice(WORDS)  # noqa: E731
    pool = [
        ("windows_path", f"C:\\Users\\{w()}\\{w()}.txt"),
        ("quoted", f'she said "{w()}" and left'),
        ("regex", rf"^{w()}-\d+\.\d{{2}}$"),
        ("shell", f"echo $HOME/{w()} && `{w()}`"),
        ("multiline", f"{w()}\n{w()}\t{w()}"),
        ("unicode", f"caf\u00e9 {w()} \u2192 {w()}"),
        ("mixed", f"'{w()}' \\\"{w()}\\\" {{{w()}}}"),
        ("html", f"<a href=\"/{w()}?x=1&y=2\">{w()}</a>"),
    ]
    picked = r.sample(pool, 6)
    return {"config": {"name": f"{w()}-{seed % 1000}", **dict(picked), "version": r.randrange(1, 9)}}


# ============================================================================ verify_first
_VERIFY_TEMPLATES = {
    "median": {
        "code": ("def median(xs):\n    \"\"\"Median of a non-empty list of numbers.\"\"\"\n    n = len(xs)\n"
                 "    return xs[n // 2]\n"),
        "symptom": "`median([{a}, {b}, {c}])` returns {wrong} instead of {right}",
        "cases": [([5, 1, 3], 3), ([4, 1, 3, 2], 2.5), ([7], 7), ([10, 2, 8, 4], 6.0), ([3, 1, 2], 2)],
        "fn": "median",
    },
    "percent_change": {
        "code": ("def percent_change(old, new):\n    \"\"\"Percent change from old to new, rounded to 1 decimal.\"\"\"\n"
                 "    return round((new - old) / new * 100, 1)\n"),
        "symptom": "`percent_change({a}, {b})` returns {wrong} instead of {right}",
        "cases": [((50, 75), 50.0), ((200, 150), -25.0), ((0, 10), None), ((80, 80), 0.0), ((40, 50), 25.0)],
        "fn": "percent_change",
    },
    "chunk": {
        "code": ("def chunk(xs, size):\n    \"\"\"Split xs into consecutive lists of length `size` (the last may be shorter).\"\"\"\n"
                 "    return [xs[i:i + size] for i in range(0, len(xs) - size, size)]\n"),
        "symptom": "`chunk(list(range({a})), {b})` drops the last chunk",
        "cases": [((list(range(7)), 3), [[0, 1, 2], [3, 4, 5], [6]]), ((list(range(6)), 3), [[0, 1, 2], [3, 4, 5]]),
                  (([], 2), []), ((list(range(3)), 5), [[0, 1, 2]]), ((list(range(4)), -1), "ValueError")],
        "fn": "chunk",
    },
    "clamp": {
        "code": ("def clamp(x, lo, hi):\n    \"\"\"Clamp x into [lo, hi].\"\"\"\n    if x < lo:\n        return lo\n"
                 "    return x\n"),
        "symptom": "`clamp({a}, 0, 10)` returns {a} instead of 10",
        "cases": [((15, 0, 10), 10), ((-3, 0, 10), 0), ((5, 0, 10), 5), ((5, 10, 0), "ValueError"), ((10, 0, 10), 10)],
        "fn": "clamp",
    },
}
# Hidden second requirement per template (documented only in the self-check):
#   median: even-length lists average the two middle values; percent_change: old == 0 -> None;
#   chunk: size < 1 raises ValueError; clamp: lo > hi raises ValueError.


def verify_first_data(seed: int) -> dict:
    name = sorted(_VERIFY_TEMPLATES)[seed % len(_VERIFY_TEMPLATES)]
    t = _VERIFY_TEMPLATES[name]
    r = rng("verify_first", seed)
    if name == "median":
        a, b, c = r.sample(range(1, 50), 3)
        right = sorted([a, b, c])[1]
        symptom = t["symptom"].format(a=a, b=b, c=c, wrong=[a, b, c][1], right=right)
        if [a, b, c][1] == right:
            symptom = f"`median([{c}, {a}, {b}])` is wrong for unsorted input"
    elif name == "percent_change":
        a = r.randrange(10, 90)
        b = a + r.randrange(5, 60)
        symptom = t["symptom"].format(a=a, b=b, wrong=round((b - a) / b * 100, 1), right=round((b - a) / a * 100, 1))
    elif name == "chunk":
        size = r.randrange(2, 5)
        symptom = t["symptom"].format(a=size * 3 + 1, b=size)
    else:
        symptom = t["symptom"].format(a=r.randrange(11, 99))
    return {"template": name, "fn": t["fn"], "symptom": symptom}


def _verify_check_source(fn: str, cases) -> str:
    return (f"import sys\nsys.path.insert(0, {str(ROOT / 'calc')!r})\nfrom stats import {fn}\n\nfailed = 0\n"
            f"for args, want in {cases!r}:\n"
            "    args = args if isinstance(args, tuple) else (args,)\n"
            "    try:\n        got = " + fn + "(*args)\n    except Exception as e:\n        got = type(e).__name__\n"
            "    if got != want:\n        failed += 1\n"
            f"        print(f'FAIL {fn}{{args}}: expected {{want!r}}, got {{got!r}}')\n"
            "print('all checks passed' if not failed else f'{failed} check(s) failed')\nsys.exit(1 if failed else 0)\n")


def setup_verify_first(seed: int) -> None:
    d = verify_first_data(seed)
    t = _VERIFY_TEMPLATES[d["template"]]
    _write(ROOT / "calc" / "stats.py", t["code"])
    _write(ROOT / "calc" / "check.py", _verify_check_source(t["fn"], t["cases"]))


def verify_first_cases(seed: int):
    return _VERIFY_TEMPLATES[verify_first_data(seed)["template"]]["cases"]


# ============================================================================ many_files
def many_files_data(seed: int) -> dict:
    r = rng("many_files", seed)
    names = [f"note_{digest('n', seed, i, n=5)}.txt" for i in range(48)]
    chain = r.sample(names, 22)
    return {"names": names, "chain": chain, "final": digest("final_code", seed, n=8)}


def setup_many_files(seed: int) -> None:
    d = many_files_data(seed)
    r = rng("many_files_text", seed)
    nxt = {a: b for a, b in zip(d["chain"], d["chain"][1:])}
    for name in d["names"]:
        lines = [" ".join(r.choice(WORDS) for _ in range(12)) for _ in range(38)]
        pos = r.randrange(5, 33)
        if name in nxt:
            lines.insert(pos, f"next: {nxt[name]}")
        elif name == d["chain"][-1]:
            lines.insert(pos, f"final: {d['final']}")
        else:
            lines.insert(pos, f"see also: {r.choice(d['names'])} (unrelated)")
        _write(ROOT / "notes" / name, "\n".join(lines) + "\n")


# ============================================================================ exact_text
def exact_text_data(seed: int) -> dict:
    r = rng("exact_text", seed)
    parts = [r.choice(WORDS) for _ in range(6)]
    line = (f"<{parts[0]} id=\"{parts[1]}\"> C:\\{parts[2]}\\{parts[3]} & '{parts[4]}' "
            + "\\" * r.randrange(1, 4) + f"\"{parts[5]}\" <</>> " + "\"" * r.randrange(1, 3) + " &&")
    return {"line": line, "backslashes": line.count("\\"), "quotes": line.count("\""), "lt": line.count("<")}


def emit(seed: int) -> None:
    print(exact_text_data(seed)["line"])


# ============================================================================ control
def control_data(seed: int) -> dict:
    r = rng("control", seed)
    kind = ("sum", "count", "max_word")[seed % 3]
    nums = [r.randrange(-500, 1000) for _ in range(40)]
    words = [r.choice(WORDS) for _ in range(60)]
    target = r.choice(words)
    if kind == "sum":
        answer = str(sum(nums))
    elif kind == "count":
        answer = str(words.count(target))
    else:
        answer = max(words, key=lambda w: (len(w), w))
    return {"kind": kind, "nums": nums, "words": words, "target": target, "answer": answer}


def setup_control(seed: int) -> None:
    d = control_data(seed)
    _write(ROOT / "data" / "numbers.txt", "\n".join(map(str, d["nums"])) + "\n")
    _write(ROOT / "data" / "words.txt", "\n".join(d["words"]) + "\n")


# ============================================================================ CLI
# `labctl setup <code>-<seed>`: opaque one-letter codes so the instruction does not name the family.
FAMILY_CODES = {
    "huge_log": "h", "background_service": "s", "deep_dirs": "d", "bulk_write": "b", "stuck_poll": "q",
    "json_escape": "j", "verify_first": "v", "many_files": "n", "exact_text": "x", "control": "c",
}
SETUP = {
    "huge_log": None, "background_service": setup_background_service, "deep_dirs": setup_deep_dirs,
    "bulk_write": None, "stuck_poll": setup_stuck_poll, "json_escape": None,
    "verify_first": setup_verify_first, "many_files": setup_many_files, "exact_text": None,
    "control": setup_control,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.split("\n\n", 1)[1] if __doc__ else "labctl")
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "setup":
        code, _, seed = rest[0].partition("-")
        family = {v: k for k, v in FAMILY_CODES.items()}.get(code)
        if family is None or not seed.isdigit():
            print(f"labctl setup: bad workspace id {rest[0]!r}", file=sys.stderr)
            return 2
        fn = SETUP[family]
        ROOT.mkdir(parents=True, exist_ok=True)
        if fn:
            fn(int(seed))
        print("workspace ready")
        return 0
    if cmd == "huge-log":
        for line in huge_log_lines(int(rest[0])):
            print(line)
        return 0
    if cmd == "serve":
        serve()
        return 0
    if cmd == "crunch":
        crunch(int(rest[0]))
        return 0
    if cmd == "job":
        return job(rest)
    if cmd == "emit":
        emit(int(rest[0]))
        return 0
    print(f"labctl: unknown command {cmd!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
