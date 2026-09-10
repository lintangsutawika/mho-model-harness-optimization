"""
Main entrypoint for training on Harbor tasks.
"""

import os
import sys

import ray
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.config import SkyRLTrainConfig, GeneratorConfig, get_config_as_yaml_str
from skyrl.train.utils.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray
from skyrl.train.utils.rate_limiter import RateLimiterConfig


# mho lives under src/ (on PYTHONPATH), so import directly -- the old `_load_module`
# path-loading hack is gone now that these modules are `mho.*` (not the colliding
# `harbor.*`). HarborGenerator/HarborTaskDataset pickle by reference to Ray workers,
# which import `mho.*` the same way (src on PYTHONPATH), so no register_pickle_by_value.
from mho.generator import HarborGenerator
from mho.dataset import HarborTaskDataset

# NOTE (sumanthrh): We use a YAML to store the defaults for the Harbor trial configuration
# TODO: Convert to a dataclass
HARBOR_DEFAULT_CONFIG = Path(__file__).parent.parent / "tasks" / "dapo_math_17k" / "trial_config.yaml"


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Merge overrides into base dict recursively, modifying base in-place."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


@dataclass
class HarborGeneratorConfig(GeneratorConfig):
    """GeneratorConfig with Harbor-specific rate limiting."""

    rate_limit: RateLimiterConfig = field(default_factory=RateLimiterConfig)


@dataclass
class HarborSkyRLConfig(SkyRLTrainConfig):
    """SkyRLTrainConfig with Harbor trial configuration."""

    harbor_trial_config: Dict[str, Any] = field(default_factory=dict)
    generator: HarborGeneratorConfig = field(default_factory=HarborGeneratorConfig)


class HarborExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """
        Initializes the HarborGenerator.
        """
        return HarborGenerator(
            generator_cfg=cfg.generator,
            harbor_cfg=cfg.harbor_trial_config,  # Pass harbor config to the generator
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            max_seq_len=cfg.trainer.algorithm.max_seq_len,
        )

    def get_train_dataset(self):
        """Initializes the training dataset.

        Returns:
            HarborTaskDataset: The training dataset.
        """
        prompts_dataset = HarborTaskDataset(
            data_files=self.cfg.data.train_data,
        )
        assert (
            len(prompts_dataset) >= self.cfg.trainer.train_batch_size
        ), f"dataset should be atleast as large as `train_batch_size` {self.cfg.trainer.train_batch_size}, got size {len(prompts_dataset)}"
        return prompts_dataset

    def get_eval_dataset(self):
        """Initializes the evaluation dataset.

        Returns:
            HarborTaskDataset: The evaluation dataset.
        """
        if self.cfg.trainer.eval_interval > 0 and self.cfg.data.val_data:
            prompts_dataset = HarborTaskDataset(
                data_files=self.cfg.data.val_data,
            )
            return prompts_dataset
        return None


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    # make sure that the training loop is not run on the head node.
    exp = HarborExp(cfg)
    exp.run()


def main() -> None:
    cfg = HarborSkyRLConfig.from_cli_overrides(sys.argv[1:])

    # Load harbor defaults and merge CLI overrides on top
    with open(HARBOR_DEFAULT_CONFIG) as f:
        defaults = yaml.safe_load(f)
    cfg.harbor_trial_config = _deep_merge(defaults, cfg.harbor_trial_config)

    # Sandbox backend selection. HARBOR_ENV_TYPE (from .env) chooses where harbor
    # runs each trial sandbox: "modal" (Modal cloud) or "docker"/"local" (a Docker
    # daemon reachable from this process). The default.yaml pins `singularity` for
    # the nested-sandbox path, which is no longer used. Modal needs no image cache
    # dir or the /mnt bind (which the stripped nested path mounted).
    harbor_env_type = os.environ.get("HARBOR_ENV_TYPE", "modal")
    env_cfg = cfg.harbor_trial_config.setdefault("environment", {})
    if harbor_env_type == "modal":
        env_cfg["type"] = "modal"
        env_cfg.pop("mounts", None)
        env_cfg.setdefault("kwargs", {}).pop("singularity_image_cache_dir", None)
    else:
        env_cfg["type"] = harbor_env_type
        if harbor_env_type == "singularity":
            # External-executor (host) path: default.yaml's /mnt/* values were for the old
            # NESTED-singularity design and don't exist/writable on the host. Point the SIF
            # cache at a host-writable dir (SIF_IMAGE_CACHE_DIR from train_math_dapo.sh), and
            # drop the /opt bind -- the executor runs writable sandbox dirs, so /opt is
            # writable in-sandbox and bootstrap builds its server venv there.
            kw = env_cfg.setdefault("kwargs", {})
            _cache = os.environ.get("SIF_IMAGE_CACHE_DIR")
            if _cache:
                kw["singularity_image_cache_dir"] = _cache
            else:
                kw.pop("singularity_image_cache_dir", None)
            env_cfg.pop("mounts", None)

    # The agent (AgentHarness) runs in the EXTERNAL executor, which does not inherit this
    # training process's env, so MICRO_SCAFFOLD_DIR won't reach it. Carry the scaffold
    # snapshot into the trial config (agent.kwargs.mini_fork_local) so it travels over HTTP
    # to the executor. default.yaml leaves it null; the agent reads mini_fork_local or env.
    _scaffold = os.environ.get("MICRO_SCAFFOLD_DIR")
    if _scaffold:
        _agent_kwargs = cfg.harbor_trial_config.setdefault("agent", {}).setdefault("kwargs", {})
        if not _agent_kwargs.get("mini_fork_local"):
            _agent_kwargs["mini_fork_local"] = _scaffold

    validate_cfg(cfg)
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError(
            "trainer.algorithm.max_seq_len must be explicitly set for Harbor training; "
            "it is required to truncate responses to the maximum allowed length."
        )
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
