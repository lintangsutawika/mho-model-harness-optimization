# @feature-file: *
"""Runtime switches for the scaffold's optional behaviours.

Every behaviour is on by default. Turn one off with a config key
(`-c features.<name>=false`) or with `MICRO_SWE_FEATURES_OFF=name1,name2`.
"""

import os

NAMES = (
    # >>> feature:output_truncation
    "output_truncation",
    # <<< feature:output_truncation
    # >>> feature:timeout_guidance
    "timeout_guidance",
    # <<< feature:timeout_guidance
    # >>> feature:persistent_cwd
    "persistent_cwd",
    # <<< feature:persistent_cwd
    # >>> feature:length_recovery
    "length_recovery",
    # <<< feature:length_recovery
    # >>> feature:loop_warning
    "loop_warning",
    # <<< feature:loop_warning
    # >>> feature:format_streak_trim
    "format_streak_trim",
    # <<< feature:format_streak_trim
    # >>> feature:missing_command_repair
    "missing_command_repair",
    # <<< feature:missing_command_repair
    # >>> feature:submit_confirmation
    "submit_confirmation",
    # <<< feature:submit_confirmation
    # >>> feature:context_trim
    "context_trim",
    # <<< feature:context_trim
    # >>> feature:raw_observation
    "raw_observation",
    # <<< feature:raw_observation
)

_overrides: dict[str, bool] = {}


def configure(flags: dict | None) -> None:
    """Apply a config's `features:` mapping ({name: bool})."""
    _overrides.clear()
    for name, value in (flags or {}).items():
        _overrides[str(name)] = bool(value)


def on(name: str) -> bool:
    off = {n.strip() for n in os.environ.get("MICRO_SWE_FEATURES_OFF", "").split(",") if n.strip()}
    if name in off:
        return False
    return _overrides.get(name, True)


def flags() -> dict[str, bool]:
    return {name: on(name) for name in NAMES}
