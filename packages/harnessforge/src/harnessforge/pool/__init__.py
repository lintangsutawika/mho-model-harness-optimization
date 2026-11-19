"""The persistent model pool: N nodes of SGLang engines behind one sglang-router endpoint.

    from harnessforge.pool import SGLangPool, endpoint_of
"""
from __future__ import annotations

from harnessforge.pool.state import DEFAULT_ROOT, PoolDir, endpoint_of
from harnessforge.pool.sglang import PoolError, SGLangPool, policy_from_dict, spec_from_dict

__all__ = ["DEFAULT_ROOT", "PoolDir", "PoolError", "SGLangPool", "endpoint_of",
           "from_config", "load_config", "policy_from_dict", "spec_from_dict"]


def load_config(path) -> dict:
    """A pool spec file (.json, or .yaml/.yml when PyYAML is installed)."""
    import json
    from pathlib import Path
    p = Path(path)
    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        import yaml                     # optional; the SGLang images ship it
        return yaml.safe_load(text)
    return json.loads(text)


def from_config(cfg: dict, launcher=None) -> SGLangPool:
    """SGLangPool from a spec-file dict:

        pool_id, state_root, sif, binds, router_args, engine_env, advertise,
        policy: {id, path, base_model}, spec: {PoolSpec fields...}
    """
    from pathlib import Path
    root = cfg.get("state_root")
    return SGLangPool(
        spec_from_dict(cfg.get("spec", {})),
        policy_from_dict(cfg["policy"]) if cfg.get("policy") else None,
        pool_id=cfg["pool_id"], state_root=Path(root).expanduser() if root else None,
        sif=cfg.get("sif"), launcher=launcher, binds=cfg.get("binds"),
        engine_env=cfg.get("engine_env"), router_args=cfg.get("router_args"),
        advertise=cfg.get("advertise", "auto"), engine_cmd=cfg.get("engine_cmd"),
        router_cmd=cfg.get("router_cmd"), python=cfg.get("python"),
        startup_timeout=float(cfg.get("startup_timeout", 1800)), job_env=cfg.get("job_env"),
        container_bin=cfg.get("container_bin"))
