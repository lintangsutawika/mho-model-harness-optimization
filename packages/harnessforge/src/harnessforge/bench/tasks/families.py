"""Task families: one per feature (two features share `json_escape`) plus a `control` family.

A family turns a seed into a `TaskSpec`: the instruction the agent sees and a verifier (Python run
by tests/test.sh inside the task container, stdlib only). Fixtures and answers come from
`labctl` (the same file ships in the image), so they cannot drift apart.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from harnessforge.bench.tasks import labctl


@dataclass(frozen=True)
class TaskSpec:
    family: str
    seed: int
    instruction: str
    check: str                      # Python source defining `check() -> bool`
    expected: dict = field(default_factory=dict)
    features: tuple[str, ...] = ()  # gold features this family is built to exercise
    agent_timeout_sec: float = 600.0


def _setup_line(family: str, seed: int) -> str:
    return (f"First set up your workspace by running `labctl setup {labctl.FAMILY_CODES[family]}-{seed}` "
            f"(it creates the files under /app that this task refers to).\n\n")


def _answer_check(path: str, key: str = "answer") -> str:
    return (f"def check():\n    got = Path({path!r}).read_text().strip()\n"
            f"    print('got', repr(got), 'expected', repr(EXPECTED[{key!r}]))\n"
            f"    return got == EXPECTED[{key!r}]\n")


# ----------------------------------------------------------------------------- families
def huge_log(seed: int) -> TaskSpec:
    d = labctl.huge_log_data(seed)
    instr = (_setup_line("huge_log", seed)
             + f"Run `labctl huge-log {seed}`. It prints a long diagnostic log. The first line of the log contains "
               "the run id (`run_id=...`) and the summary at the very end contains `final_checksum=...`.\n\n"
               "Write a single line `<run_id> <final_checksum>` (the two values separated by one space) to "
               "/app/answer.txt.\n")
    return TaskSpec("huge_log", seed, instr, _answer_check("/app/answer.txt"),
                    {"answer": f"{d['run_id']} {d['final_checksum']}"}, ("output_truncation",))


def background_service(seed: int) -> TaskSpec:
    if seed % 2 == 0:
        instr = (_setup_line("background_service", seed)
                 + "Start the lab status service with `labctl serve`. It must keep running after you finish: the "
                   "checker will query it. The service records the port it listens on in /app/service/port.\n\n"
                   "Then fetch `http://127.0.0.1:<port>/health` from it and save the response body to "
                   "/app/health.txt.\n")
        check = ("def check():\n    secret = Path('/app/service/secret').read_text()\n"
                 "    want = 'ok ' + hashlib.sha256(('health:' + secret.strip()).encode()).hexdigest()[:12]\n"
                 "    port = int(Path('/app/service/port').read_text().strip())\n"
                 "    body = None\n    for _ in range(10):\n        try:\n"
                 "            body = urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3).read().decode()\n"
                 "            break\n        except Exception as e:\n            print('service not reachable:', e)\n"
                 "            time.sleep(0.5)\n"
                 "    saved = Path('/app/health.txt').read_text().strip()\n"
                 "    print('service:', repr(body), 'saved:', repr(saved), 'want:', repr(want))\n"
                 "    return body == want and saved == want\n")
        return TaskSpec("background_service", seed, instr, check, {}, ("timeout_guidance",))
    instr = (_setup_line("background_service", seed)
             + f"Run `labctl crunch {seed}`. It takes about a minute and finally prints `result: <value>`.\n\n"
               "Write the value (without the `result: ` prefix) to /app/result.txt.\n")
    return TaskSpec("background_service", seed, instr, _answer_check("/app/result.txt"),
                    {"answer": labctl.crunch_result(seed)}, ("timeout_guidance",))


def deep_dirs(seed: int) -> TaskSpec:
    d = labctl.deep_dirs_data(seed)
    mods = ", ".join(f"`{m}` (value {v})" for m, v in zip(d["modules"], d["values"]))
    instr = (_setup_line("deep_dirs", seed)
             + f"The project /app/workspace/{d['proj']} has three modules under `modules/`: {mods}.\n\n"
               "For each module, working from inside that module's directory:\n"
               "1. replace the number in `params/value.txt` with the module's value given above;\n"
               "2. run `./configure`;\n"
               "3. run `./build`, which writes `out/artifact.txt`.\n\n"
               "Both scripts only work when started from the module directory.\n")
    check = ("def check():\n    ok = True\n"
             f"    for m, want in zip(EXPECTED['modules'], EXPECTED['artifacts']):\n"
             f"        p = Path('/app/workspace') / EXPECTED['proj'] / 'modules' / m / 'out' / 'artifact.txt'\n"
             "        got = p.read_text().strip() if p.exists() else None\n"
             "        print(m, repr(got), repr(want))\n        ok = ok and got == want\n    return ok\n")
    return TaskSpec("deep_dirs", seed, instr, check,
                    {"proj": d["proj"], "modules": d["modules"], "artifacts": d["artifacts"]}, ("persistent_cwd",))


def bulk_write(seed: int) -> TaskSpec:
    lines = labctl.bulk_write_data(seed)["lines"]
    instr = (_setup_line("bulk_write", seed)
             + f"Create the file /app/manifest.txt containing exactly the following {len(lines)} lines, in this "
               "order, with nothing else (a final newline is fine):\n\n```\n" + "\n".join(lines) + "\n```\n")
    check = ("def check():\n    got = Path('/app/manifest.txt').read_text().rstrip('\\n').split('\\n')\n"
             "    want = EXPECTED['lines']\n"
             "    bad = [i for i, (a, b) in enumerate(zip(got, want)) if a != b]\n"
             "    print('lines', len(got), 'expected', len(want), 'first mismatch', bad[:1])\n"
             "    return got == want\n")
    return TaskSpec("bulk_write", seed, instr, check, {"lines": lines}, ("length_recovery",))


def stuck_poll(seed: int) -> TaskSpec:
    d = labctl.stuck_poll_data(seed)
    instr = (_setup_line("stuck_poll", seed)
             + "A job has been submitted to the lab job queue (`labctl job status` shows it). Wait until it has "
               "finished, then write its result (printed by `labctl job result`) to /app/result.txt.\n")
    return TaskSpec("stuck_poll", seed, instr, _answer_check("/app/result.txt"), {"answer": d["result"]},
                    ("loop_warning",))


def json_escape(seed: int) -> TaskSpec:
    cfg = labctl.json_escape_data(seed)["config"]
    instr = (_setup_line("json_escape", seed)
             + "Create /app/config.json containing exactly this JSON object (same keys and values; whitespace "
               "and key order do not matter):\n\n```json\n" + json.dumps(cfg, indent=2, ensure_ascii=False)
             + "\n```\n")
    check = ("def check():\n    got = json.loads(Path('/app/config.json').read_text())\n"
             "    diff = {k for k in set(got) | set(EXPECTED['config']) if got.get(k) != EXPECTED['config'].get(k)}\n"
             "    print('differing keys:', sorted(diff))\n    return got == EXPECTED['config']\n")
    return TaskSpec("json_escape", seed, instr, check, {"config": cfg},
                    ("format_streak_trim", "missing_command_repair"))


def verify_first(seed: int) -> TaskSpec:
    d = labctl.verify_first_data(seed)
    instr = (_setup_line("verify_first", seed)
             + f"Users report a bug in /app/calc/stats.py: {d['symptom']}. Fix `{d['fn']}` so that it is correct.\n\n"
               "A self-check is available: `python3 /app/calc/check.py`.\n")
    cases = [[list(a) if isinstance(a, tuple) else a, w] for a, w in labctl.verify_first_cases(seed)]
    check = ("def check():\n    spec = importlib.util.spec_from_file_location('stats_under_test', '/app/calc/stats.py')\n"
             "    mod = importlib.util.module_from_spec(spec)\n    spec.loader.exec_module(mod)\n"
             "    fn = getattr(mod, EXPECTED['fn'])\n    ok = True\n"
             "    for args, want in EXPECTED['cases']:\n"
             "        args = tuple(args) if EXPECTED['fn'] != 'median' else (args,)\n"
             "        try:\n            got = fn(*args)\n        except Exception as e:\n            got = type(e).__name__\n"
             "        print(EXPECTED['fn'], args, '->', repr(got), 'want', repr(want))\n"
             "        ok = ok and got == want\n    return ok\n")
    return TaskSpec("verify_first", seed, instr, check, {"fn": d["fn"], "cases": cases}, ("submit_confirmation",))


def many_files(seed: int) -> TaskSpec:
    d = labctl.many_files_data(seed)
    instr = (_setup_line("many_files", seed)
             + f"Start at /app/notes/{d['chain'][0]}. Each note in the chain names the next note to read on a line "
               "`next: <file>` (other `see also:` references are unrelated). Follow the chain until you reach a "
               "note that contains a line `final: <code>`.\n\nWrite the code to /app/answer.txt, and the number of "
               "notes in the chain (including the first and the last) on a second line.\n")
    check = ("def check():\n    got = Path('/app/answer.txt').read_text().split()\n"
             "    print('got', got, 'expected', EXPECTED['answer'])\n    return got == EXPECTED['answer']\n")
    return TaskSpec("many_files", seed, instr, check, {"answer": [d["final"], str(len(d["chain"]))]},
                    ("context_trim",))


def exact_text(seed: int) -> TaskSpec:
    d = labctl.exact_text_data(seed)
    instr = (_setup_line("exact_text", seed)
             + f"Run `labctl emit {seed}`. It prints one line of text.\n\nWrite three numbers to /app/answer.txt, "
               "separated by single spaces: how many backslash characters (`\\`), how many double-quote "
               "characters (`\"`) and how many less-than characters (`<`) that line contains.\n")
    return TaskSpec("exact_text", seed, instr, _answer_check("/app/answer.txt"),
                    {"answer": f"{d['backslashes']} {d['quotes']} {d['lt']}"}, ("raw_observation",))


def control(seed: int) -> TaskSpec:
    d = labctl.control_data(seed)
    ask = {
        "sum": "the sum of all the numbers in /app/data/numbers.txt (one integer per line)",
        "count": f"how many lines of /app/data/words.txt are exactly the word `{d['target']}`",
        "max_word": "the longest word in /app/data/words.txt (if several are equally long, the "
                    "alphabetically last of them)",
    }[d["kind"]]
    instr = _setup_line("control", seed) + f"Write {ask} to /app/answer.txt.\n"
    return TaskSpec("control", seed, instr, _answer_check("/app/answer.txt"), {"answer": d["answer"]}, ())


FAMILIES: dict[str, Callable[[int], TaskSpec]] = {
    "huge_log": huge_log,
    "background_service": background_service,
    "deep_dirs": deep_dirs,
    "bulk_write": bulk_write,
    "stuck_poll": stuck_poll,
    "json_escape": json_escape,
    "verify_first": verify_first,
    "many_files": many_files,
    "exact_text": exact_text,
    "control": control,
}
assert set(FAMILIES) == set(labctl.FAMILY_CODES)
