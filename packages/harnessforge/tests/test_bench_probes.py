from concurrent.futures import ThreadPoolExecutor

import pytest

from harnessforge.bench import FEATURE_NAMES, builder, probes


@pytest.fixture(scope="module")
def calibration(tmp_path_factory):
    roots = builder.build_all(tmp_path_factory.mktemp("variants"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = dict(zip(roots, pool.map(lambda r: probes.probe(r, workers=3), roots.values())))
    return results


def _absent(res):
    return set(res) - probes.present_set(res)


def test_gold_has_all_features(calibration):
    res = calibration["gold"]
    assert _absent(res) == set(), {f: r.evidence for f, r in res.items() if not r.present}


def test_start_has_none(calibration):
    res = calibration["start"]
    assert probes.present_set(res) == set(), {f: r.evidence for f, r in res.items() if r.present}


@pytest.mark.parametrize("feature", FEATURE_NAMES)
def test_ablation_misses_exactly_its_feature(calibration, feature):
    assert _absent(calibration[f"ablate_{feature}"]) == {feature}


def test_runtime_flag_turns_feature_off(tmp_path):
    gold = builder.build(FEATURE_NAMES, tmp_path / "gold")
    on = probes.probe_feature(gold, "loop_warning")
    off = probes.probe_feature(gold, "loop_warning", env={"MICRO_SWE_FEATURES_OFF": "loop_warning"})
    assert on.present and not off.present


def test_broken_harness_reports_absent_with_error(tmp_path):
    (tmp_path / "src" / "micro_swe").mkdir(parents=True)
    r = probes.probe_feature(tmp_path, "output_truncation")
    assert not r.present and r.error
