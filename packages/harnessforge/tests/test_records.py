from harnessforge import (Candidate, EvalBackend, EvalRun, Launcher, ModelPool, Order,
                          TrialRecord, per_task, read_jsonl, write_jsonl)


def rec(cand, task, i, reward=1.0, **kw):
    return TrialRecord(candidate=cand, task=task, index=i, finished=True, reward=reward, **kw)


def test_jsonl_roundtrip(tmp_path):
    rs = [rec("a", "t1", 0, output_tokens=10), TrialRecord("a", "t1", 1, finished=False)]
    write_jsonl(tmp_path / "r.jsonl", rs)
    assert read_jsonl(tmp_path / "r.jsonl") == rs
    assert read_jsonl(tmp_path / "missing.jsonl") == []


def test_per_task_counts_only_scored_trials():
    rs = [rec("a", "t1", 0, 1.0, output_tokens=10), rec("a", "t1", 1, 0.0, output_tokens=30),
          rec("a", "t1", 2, 1.0, env_failed=True),                 # no sandbox: not evidence
          TrialRecord("a", "t2", 0, finished=False),               # in flight
          rec("a", "t3", 0, None),                                 # finished, no reward -> 0
          rec("b", "t1", 0, 0.0)]
    pt = per_task(rs, "a")
    assert set(pt) == {"t1", "t3"}
    assert (pt["t1"].reward, pt["t1"].n, pt["t1"].cost) == (0.5, 2, 20.0)
    assert pt["t3"].reward == 0.0 and pt["t3"].cost is None
    assert per_task(rs, "a", max_index=1)["t1"].n == 1
    assert set(per_task(rs, "a", tasks={"t3"})) == {"t3"}


def test_contract_types_import():
    assert Order.TASK_MAJOR.value == "task_major"
    for proto in (EvalBackend, EvalRun, Launcher, ModelPool):
        assert hasattr(proto, "__protocol_attrs__") or hasattr(proto, "_is_protocol")
    assert Candidate("c0", harness_dir=__import__("pathlib").Path("/x")).parent is None
