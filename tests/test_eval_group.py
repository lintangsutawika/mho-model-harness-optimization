"""--eval-group: a round's eval_b jobs share ONE node job (one served model, several harnesses)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_loop import by_tag, make, summary  # noqa: E402


def grouped(loop, fake, g=4):
    groups = []

    def submit_eval_group(sched, members, *, model, data, n_trials, n_concurrent, group_file,
                          logs, job_name, base_model=None):
        for m in members:      # the fake writes each member's trials exactly like a lone eval
            fake.submit_eval(sched, "", model=model, harness=m["harness"], data=data,
                             n_trials=n_trials, n_concurrent=n_concurrent,
                             out_dir=m["out_dir"], logs=m["logs"], job_name=m["job_name"],
                             base_model=base_model)
        groups.append({"tags": [Path(m["harness"]).parent.name for m in members],
                       "conc": n_concurrent, "group_file": group_file, "logs": logs})
        return fake._job()

    loop._submit_eval_group = submit_eval_group
    loop.args.eval_group = g
    loop.args.group_concurrent = 48
    return groups


def test_round_of_four_is_one_group_job_with_the_same_outcome(tmp_path):
    loop, fake = make(tmp_path, by_tag({
        "val": [0.5, 0.6], "eval_a": [0.3],
        "b": {"candidate_1_0_1a": 0.4, "candidate_1_0_1b": 0.6,
              "candidate_1_0_1c": 0.2, "candidate_1_0_1d": 0.5}}),
        iterations=4, proposals=4)
    groups = grouped(loop, fake)
    loop.run()
    assert [g["tags"] for g in groups] == [[f"candidate_1_0_1{x}" for x in "abcd"]]
    assert groups[0]["conc"] == 48
    assert groups[0]["logs"].name == "candidate_1_0_1a-eval_after-group.out"
    stages = [c["stage"] for c in fake.calls]
    assert stages == ["val", "eval_a"] + ["propose"] * 4 + ["eval_b"] * 4 + ["val"]
    rows = {r["candidate"]: r["outcome"] for r in summary(loop)[1:]}
    assert rows == {"candidate_1_0_1a": "not_selected", "candidate_1_0_1b": "accepted",
                    "candidate_1_0_1c": "train_fail", "candidate_1_0_1d": "not_selected"}
    # each member still logs to its own file
    logs = {Path(c["log"]).name for c in fake.calls if c["stage"] == "eval_b"}
    assert logs == {f"candidate_1_0_1{x}-eval_after.out" for x in "abcd"}


def test_failed_proposal_is_left_out_of_the_group(tmp_path):
    loop, fake = make(tmp_path, by_tag({
        "val": [0.5, 0.6], "eval_a": [0.3, 0.3],
        "b": {"candidate_1_0_1a": 0.5, "candidate_1_0_1c": 0.4}}),
        iterations=3, proposals=3, proposes=lambda t: t != "candidate_1_0_1b")
    groups = grouped(loop, fake)
    loop.run()
    assert groups[0]["tags"] == ["candidate_1_0_1a", "candidate_1_0_1c"]
    out = {r["candidate"]: r["outcome"] for r in summary(loop)[1:4]}
    assert out == {"candidate_1_0_1b": "no_proposal", "candidate_1_0_1a": "accepted",
                   "candidate_1_0_1c": "not_selected"}


def test_member_without_trials_is_retried_alone(tmp_path):
    seen = {"candidate_1_0_1b": 0}

    def plan(stage, harness, n):
        if stage == "eval_b":
            tag = Path(harness).parent.name
            if tag == "candidate_1_0_1b":
                seen[tag] += 1
                return None if seen[tag] == 1 else 0.6      # first attempt: no trials
            return 0.1
        return {"val": [0.5, 0.6], "eval_a": [0.3]}[stage][n]

    loop, fake = make(tmp_path, plan, iterations=2, proposals=2)
    groups = grouped(loop, fake)
    loop.run()
    assert [g["tags"] for g in groups] == [["candidate_1_0_1a", "candidate_1_0_1b"],
                                           ["candidate_1_0_1b"]]
    assert groups[1]["logs"].name == "candidate_1_0_1b-eval_after-group-attempt2.out"
    rows = {r["candidate"]: r["outcome"] for r in summary(loop)[1:]}
    assert rows == {"candidate_1_0_1a": "train_fail", "candidate_1_0_1b": "accepted"}


def test_group_size_one_keeps_one_job_per_sibling(tmp_path):
    loop, fake = make(tmp_path, by_tag({
        "val": [0.5, 0.6], "eval_a": [0.3],
        "b": {"candidate_1_0_1a": 0.6, "candidate_1_0_1b": 0.2}}),
        iterations=2, proposals=2)
    groups = grouped(loop, fake, g=1)
    loop.run()
    assert groups == []
    assert [c["stage"] for c in fake.calls].count("eval_b") == 2
