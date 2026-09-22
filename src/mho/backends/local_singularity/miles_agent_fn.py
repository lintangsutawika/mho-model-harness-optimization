"""Agent-function shim for the Miles harbor connector.

Miles' in-process Harbor connector selects the sandbox purely by ``HARBOR_ENV_TYPE``
(a built-in ``EnvironmentType``); harbor's ``EnvironmentFactory`` then instantiates the
class from its registry. For ``HARBOR_ENV_TYPE=singularity`` that would be harbor's STOCK
``SingularityEnvironment``, which converts the docker image and launches apptainer
*in-process* -- i.e. apptainer-inside-the-Miles-container, which fails on this HPC
(``[Errno 2] No such file or directory``: no apptainer binary in the training image).

Our fix is the registry swap in ``remote_env`` (EnvironmentType.SINGULARITY ->
RemoteSingularityEnvironment, which POSTs to the host /sandbox service instead of nesting).
That swap arms on import -- but nothing in the rollout worker imports ``remote_env``.

This module is that import hook. Point ``--custom-agent-function-path`` at it: the worker
imports it to get ``run``, which (a) imports ``remote_env`` -> arms the swap in THIS
process before any trial builds its environment, then (b) re-exports miles' real
``harbor_agent_function.run`` unchanged, so the connector's agent<->generate wiring and
native token recording are untouched.
"""
import os
import sys as _sys
from pathlib import Path as _Path

# The repo ROOT has its own `harness/` package (harbor_run/scaffold/relay, NO math_verifier)
# that shadows src/harness/ whenever the repo root lands on the worker's sys.path (cwd, etc.).
# harbor imports the verifier by module path (`harness.math_verifier`) IN THIS worker process,
# so force src/ to the FRONT and drop any already-imported `harness` that resolved outside src,
# else the verifier import hits the shadow -> "No module named 'harness.math_verifier'".
_SRC = str(_Path(__file__).resolve().parents[3])  # .../src
while _SRC in _sys.path:
    _sys.path.remove(_SRC)
_sys.path.insert(0, _SRC)
for _m in [m for m in list(_sys.modules) if m == "harness" or m.startswith("harness.")]:
    _f = getattr(_sys.modules.get(_m), "__file__", "") or ""
    if not _f.startswith(_SRC):
        del _sys.modules[_m]

from mho.backends.local_singularity import remote_env as _remote_env  # noqa: E402,F401  (arms the registry swap on import)

# Miles' harbor connector module. We patch its trial-config builder below, then re-export
# its `run` (which calls the patched builder). Imported AFTER _remote_env so the env
# registry swap is armed before any trial is built.
import harbor_agent_function as _haf  # noqa: E402
from harbor.models.trial.config import VerifierConfig as _VerifierConfig  # noqa: E402

# --------------------------------------------------------------------------- #
# Inject our custom verifier. The connector's build_trial_config() sets the trial-level
# VerifierConfig only for a timeout override, never import_path -- so harbor falls back to
# its shared file verifier (runs tests/test.sh, reads verifier/reward.txt). Our tasks use a
# stub test.sh and grade via harness.math_verifier:MathVerifier instead (as SkyRL did via
# trial_config.yaml), so without this every trial hits RewardFileNotFoundError -> reward 0.
# We wrap build_trial_config to set verifier.import_path (preserving any timeout override).
# --------------------------------------------------------------------------- #
_orig_build_trial_config = _haf.build_trial_config


def _build_trial_config_with_verifier(*args, **kwargs):
    tc = _orig_build_trial_config(*args, **kwargs)
    existing = getattr(tc, "verifier", None)
    if existing is None or getattr(existing, "import_path", None) is None:
        tc = tc.model_copy(update={"verifier": _VerifierConfig(
            import_path=os.environ.get("MHO_VERIFIER", "harness.math_verifier:MathVerifier"),
            override_timeout_sec=getattr(existing, "override_timeout_sec", None),
        )})
    return tc


_haf.build_trial_config = _build_trial_config_with_verifier

run = _haf.run  # re-export miles' agent function (now uses the patched build_trial_config)

__all__ = ["run"]
