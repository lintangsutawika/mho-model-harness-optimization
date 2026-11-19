"""Gold-harness toy benchmark (workstream D).

A micro-swe-agent fork with ten known features (`features.FEATURES`). `builder` materializes the
gold harness (all features), the start harness (none: the code is absent) and one ablation per
feature; `probes` detects each feature's behaviour in any harness dir with a scripted model;
`tasks` generates Harbor tasks whose families exercise the features; `metrics` turns probe results
and evaluations into discovery reports and defines the planned experiments.
"""
from harnessforge.bench.features import BY_NAME, FEATURE_NAMES, FEATURES, Feature

__all__ = ["BY_NAME", "FEATURE_NAMES", "FEATURES", "Feature"]
