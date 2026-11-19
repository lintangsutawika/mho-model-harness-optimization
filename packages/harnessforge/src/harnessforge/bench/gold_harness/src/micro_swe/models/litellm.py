import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import litellm
from pydantic import BaseModel

# >>> feature:*
from micro_swe import features
# <<< feature:*
from micro_swe.exceptions import FormatError
from micro_swe.tools import (
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)
from micro_swe.utils.retry import retry

logger = logging.getLogger("litellm_model")


def _wants_token_ids() -> bool:
    """Whether to request per-token IDs/logprobs from the provider (for RL training).

    Enabled by the ``MICRO_RETURN_TOKEN_IDS`` env var, which harbor's AgentHarness sets
    in every sandbox exec when SkyRL asks for ``collect_rollout_details``. Off by default
    so ordinary micro-swe-agent use (non-vLLM providers) pays no cost and risks no 400.
    """
    return os.environ.get("MICRO_RETURN_TOKEN_IDS", "").strip().lower() in ("1", "true", "yes", "on")


def _logprob_of(token_data) -> float | None:
    """Read a logprob from a litellm logprobs-content entry (dict or pydantic object)."""
    if isinstance(token_data, dict):
        return token_data.get("logprob")
    return getattr(token_data, "logprob", None)


def _extract_rollout(response) -> dict | None:
    """Pull per-turn token IDs + logprobs from a vLLM response, via the same access path
    harbor's own LLM wrapper uses (harbor/llms/lite_llm.py:_extract_token_ids/_logprobs):

      - completion_token_ids: ``response.choices[0].provider_specific_fields["token_ids"]``
      - prompt_token_ids:     ``response.prompt_token_ids`` (top-level attr, set by vLLM)
      - logprobs:             ``response.choices[0].logprobs.content[*].logprob``

    Returns None if the provider didn't return token IDs (e.g. ``return_token_ids`` unset).
    """
    try:
        prompt_token_ids = getattr(response, "prompt_token_ids", None)
        choice = response.choices[0]
        completion_token_ids = None
        psf = getattr(choice, "provider_specific_fields", None)
        if psf:
            tid = psf.get("token_ids") if isinstance(psf, dict) else getattr(psf, "token_ids", None)
            if isinstance(tid, list):
                completion_token_ids = tid
        logprobs = None
        lp = getattr(choice, "logprobs", None)
        content = getattr(lp, "content", None) if lp is not None else None
        if content:
            logprobs = [lpv for lpv in (_logprob_of(t) for t in content) if lpv is not None]
        if completion_token_ids is None and prompt_token_ids is None:
            return None
        return {
            "prompt_token_ids": prompt_token_ids,
            "completion_token_ids": completion_token_ids,
            "logprobs": logprobs,
        }
    except Exception as e:  # never let rollout capture break a trajectory
        logger.debug(f"rollout token-id extraction failed: {e}")
        return None


class ModelConfig(BaseModel):
    model_name: str
    """Model name. Highly recommended to include the provider in the model name, e.g., `anthropic/claude-sonnet-4-5-20250929`."""
    model_kwargs: dict[str, Any] = {}
    """Additional arguments passed to the API."""
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")
    """Model registry for cost tracking and model metadata. See the local model guide (https://mini-swe-agent.com/latest/models/local_models/) for more details."""
    cost_tracking: Literal["default", "ignore_errors"] = os.getenv("MSWEA_COST_TRACKING", "default")
    """Cost tracking mode for this model. Can be "default" or "ignore_errors" (ignore errors/missing cost info)"""
    format_error_template: str = "{{ error }}"
    """Template used when the LM's output is not in the expected format."""
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    """Template used to render the observation after executing an action."""
    # >>> feature:length_recovery
    length_recovery_factor: float = 2.0
    """On a response cut off by the output-token limit, re-query once with max_tokens times this."""
    length_recovery_ceiling: int = 16384
    """Upper bound for the raised max_tokens."""
    # <<< feature:length_recovery
    # >>> feature:missing_command_repair
    command_repair_max_tokens: int = 1024
    """Output budget of the repair re-query after a bash call without a command."""
    # <<< feature:missing_command_repair


class Model:
    abort_exceptions: list[type[Exception]] = [
        litellm.exceptions.UnsupportedParamsError,
        litellm.exceptions.NotFoundError,
        litellm.exceptions.PermissionDeniedError,
        litellm.exceptions.ContextWindowExceededError,
        litellm.exceptions.AuthenticationError,
        KeyboardInterrupt,
    ]

    def __init__(self, *, config_class: Callable = ModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))

    def _query(self, messages: list[dict[str, str]], **kwargs):
        call_kwargs = self.config.model_kwargs | kwargs
        if _wants_token_ids():
            # Ask vLLM to return token IDs + logprobs for RL training. Deep-merge into any
            # existing extra_body (e.g. the cache_salt SkyRL injects) rather than clobber it.
            call_kwargs["logprobs"] = True
            base_extra_body = call_kwargs.get("extra_body") or {}
            call_kwargs["extra_body"] = {**base_extra_body, "return_token_ids": True}
        try:
            return litellm.completion(
                model=self.config.model_name,
                messages=messages,
                tools=[BASH_TOOL],
                **call_kwargs,
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        # Strip our internal `extra` bookkeeping before sending to the API. (No
        # Anthropic thinking-block reorder / cache-control: micro targets non-Claude
        # models -- see README.)
        return [{k: v for k, v in msg.items() if k != "extra"} for msg in messages]

    def _complete(self, messages: list[dict], **kwargs):
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                response = self._query(self._prepare_messages_for_api(messages), **kwargs)
        return response

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        response = self._complete(messages, **kwargs)
        # >>> feature:length_recovery
        if features.on("length_recovery"):
            response = self._recover_from_length_cutoff(messages, response, **kwargs)
        # <<< feature:length_recovery
        # >>> feature:missing_command_repair
        if features.on("missing_command_repair"):
            response = self._repair_missing_command(messages, response, **kwargs)
        # <<< feature:missing_command_repair
        cost_output = self._calculate_cost(response)
        # Note: all model.query() implementations must persist the response and cost on FormatError.
        try:
            actions = self._parse_actions(response)
        except FormatError as e:
            e.messages[0]["extra"].update(cost_output)
            try:
                e.messages[0]["extra"]["response"] = response.model_dump(mode="json")
            except Exception:
                # model_dump failed (e.g. unserializable object); fall back to repr
                # so the spec contract ("response MUST be persisted") holds unconditionally.
                e.messages[0]["extra"]["response"] = repr(response)
            raise
        message = response.choices[0].message.model_dump()
        message["extra"] = {
            "actions": actions,
            "response": response.model_dump(),
            **cost_output,
            "timestamp": time.time(),
        }
        # Capture per-turn token IDs/logprobs for RL training. `response.model_dump()` above
        # drops vLLM's dynamically-set `prompt_token_ids`/`provider_specific_fields`, so read
        # them off the live response object here and persist under a stable key.
        if _wants_token_ids():
            rollout = _extract_rollout(response)
            if rollout is not None:
                message["extra"]["rollout"] = rollout
        return message

    # >>> feature:length_recovery
    def _recover_from_length_cutoff(self, messages: list[dict], response, **kwargs):
        """A response cut off by max_tokens before any tool call is retried once with a larger budget."""
        choice = response.choices[0]
        if choice.finish_reason != "length" or choice.message.tool_calls:
            return response
        budget = (kwargs.get("max_tokens") or self.config.model_kwargs.get("max_tokens")
                  or self.config.model_kwargs.get("max_completion_tokens") or 4096)
        raised = min(int(budget * self.config.length_recovery_factor), self.config.length_recovery_ceiling)
        if raised <= budget:
            return response
        logger.info(f"response cut off at max_tokens={budget}; re-querying with {raised}")
        retried = self._complete(messages, **(kwargs | {"max_tokens": raised}))
        return retried if retried.choices[0].message.tool_calls else response

    # <<< feature:length_recovery
    # >>> feature:missing_command_repair
    def _repair_missing_command(self, messages: list[dict], response, **kwargs):
        """A bash call without a command is re-asked once, telling the model exactly what is missing.
        The repair instruction is not added to the history; only a successful repaired call is."""
        tool_calls = response.choices[0].message.tool_calls or []
        if not tool_calls or not any(_missing_command(tc) for tc in tool_calls):
            return response
        repair = {"role": "user", "content": COMMAND_REPAIR_PROMPT}
        retried = self._complete(
            list(messages) + [repair], **(kwargs | {"max_tokens": self.config.command_repair_max_tokens})
        )
        new_calls = retried.choices[0].message.tool_calls or []
        return retried if new_calls and not any(_missing_command(tc) for tc in new_calls) else response

    # <<< feature:missing_command_repair
    def _calculate_cost(self, response) -> dict[str, float]:
        try:
            cost = litellm.cost_calculator.completion_cost(response, model=self.config.model_name)
            if cost <= 0.0:
                raise ValueError(f"Cost must be > 0.0, got {cost}")
        except Exception as e:
            cost = 0.0
            if self.config.cost_tracking != "ignore_errors":
                msg = (
                    f"Error calculating cost for model {self.config.model_name}: {e}, perhaps it's not registered? "
                    "You can ignore this issue from your config file with cost_tracking: 'ignore_errors' or "
                    "globally with export MSWEA_COST_TRACKING='ignore_errors'. "
                    "Alternatively check the 'Cost tracking' section in the documentation at "
                    "https://klieret.short.gy/mini-local-models. "
                    " Still stuck? Please open a github issue at https://github.com/SWE-agent/mini-swe-agent/issues/new/choose!"
                )
                logger.critical(msg)
                raise RuntimeError(msg) from e
        return {"cost": cost}

    def _parse_actions(self, response) -> list[dict]:
        """Parse tool calls from the response. Raises FormatError if unknown tool."""
        tool_calls = response.choices[0].message.tool_calls or []
        return parse_toolcall_actions(
            tool_calls,
            format_error_template=self.config.format_error_template,
            template_kwargs={"finish_reason": response.choices[0].finish_reason},
        )

    def format_message(self, **kwargs) -> dict:
        return kwargs

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format execution outputs into tool result messages."""
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }
# >>> feature:missing_command_repair


COMMAND_REPAIR_PROMPT = (
    "Your last bash tool call was rejected because its `command` argument was missing or empty. "
    "Do not explain. Reply with exactly one bash tool call whose arguments are "
    '{"command": "<the shell command you want to run>"} with a non-empty command.'
)


def _missing_command(tool_call) -> bool:
    if getattr(tool_call.function, "name", None) != "bash":
        return False
    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except Exception:
        return False
    return isinstance(args, dict) and not str(args.get("command") or "").strip()
# <<< feature:missing_command_repair
