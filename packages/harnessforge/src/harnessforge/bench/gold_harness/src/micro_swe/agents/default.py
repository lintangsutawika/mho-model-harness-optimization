"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation
or https://minimal-agent.com for a tutorial on the basic building principles.
"""

import json
import logging
import time
import traceback
from pathlib import Path

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from micro_swe import __version__
# >>> feature:*
from micro_swe import features
# <<< feature:*
from micro_swe.protocols import Environment, Model
from micro_swe.exceptions import FormatError, InterruptAgentFlow, LimitsExceeded, TimeExceeded
# >>> feature:submit_confirmation
from micro_swe.exceptions import Submitted
# <<< feature:submit_confirmation
from micro_swe.utils.serialize import recursive_merge


class AgentConfig(BaseModel):
    """Check config/mini.yaml for example settings."""

    system_template: str
    """Template for the system message (the first message)."""
    instance_template: str
    """Template for the first user message specifying the task (the second message overall)."""
    step_limit: int = 0
    """Maximum number of steps the agent can take."""
    cost_limit: float = 3.0
    """Stop agent after exceeding (!) this cost."""
    wall_time_limit_seconds: int = 0
    """Stop agent after this many seconds of wall-clock time. 0 means no limit."""
    max_consecutive_format_errors: int = 3
    """Exit after this many format errors in a row (0 = no limit)."""
    output_path: Path | None = None
    """Save the trajectory to this path."""
    # >>> feature:loop_warning
    loop_warning_after: int = 3
    """Warn the model once it has run the same command with the same output this many times in a row."""
    # <<< feature:loop_warning
    # >>> feature:context_trim
    context_keep_recent: int = 6
    """Observations newer than this many command results are always shown in full."""
    context_trim_min_chars: int = 1200
    """Older observations longer than this are shortened to a stub before querying the model."""
    # <<< feature:context_trim


class Agent:
    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        """See the `AgentConfig` class for permitted keyword arguments."""
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.logger = logging.getLogger("agent")
        self.cost = 0.0
        self.n_calls = 0
        self.n_consecutive_format_errors = 0
        self._start_time = time.time()
        # >>> feature:loop_warning
        self._last_exec: tuple[str, str] | None = None
        self._repeat_count = 0
        # <<< feature:loop_warning
        # >>> feature:submit_confirmation
        self._submit_confirmed = False
        # <<< feature:submit_confirmation

    def get_template_vars(self, **kwargs) -> dict:
        return recursive_merge(
            self.config.model_dump(),
            self.env.get_template_vars(),
            self.model.get_template_vars(),
            {
                "n_model_calls": self.n_calls,
                "model_cost": self.cost,
                "elapsed_seconds": int(time.time() - self._start_time),
            },
            # >>> feature:*
            {"features": features.flags()},
            # <<< feature:*
            self.extra_template_vars,
            kwargs,
        )

    def _render_template(self, template: str) -> str:
        return Template(template, undefined=StrictUndefined).render(**self.get_template_vars())

    def add_messages(self, *messages: dict) -> list[dict]:
        self.logger.debug(messages)  # set log level to debug to see
        self.messages.extend(messages)
        return list(messages)

    def handle_uncaught_exception(self, e: Exception) -> list[dict]:
        return self.add_messages(
            self.model.format_message(
                role="exit",
                content=str(e),
                extra={
                    "exit_status": type(e).__name__,
                    "submission": "",
                    "exception_str": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
        )

    def run(self, task: str = "", **kwargs) -> dict:
        """Run step() until agent is finished. Returns dictionary with exit_status, submission keys."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_messages(
            self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
            self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
        )
        while True:
            try:
                self.step()
                self.n_consecutive_format_errors = 0  # reset on any clean step
            except FormatError as e:
                # The call was billed before parsing failed, so query() never got to charge it.
                self.cost += e.messages[0].get("extra", {}).get("cost", 0.0)
                self.n_consecutive_format_errors += 1
                if 0 < self.config.max_consecutive_format_errors <= self.n_consecutive_format_errors:
                    self.add_messages(
                        *e.messages,
                        {
                            "role": "exit",
                            "content": "RepeatedFormatError",
                            "extra": {"exit_status": "RepeatedFormatError", "submission": ""},
                        },
                    )
                else:
                    self.add_messages(*e.messages)
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                break
        return self.messages[-1].get("extra", {})

    def step(self) -> list[dict]:
        """Query the LM, execute actions."""
        return self.execute_actions(self.query())

    def query(self) -> dict:
        """Query the model and return model messages. Override to add hooks."""
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        if 0 < self.config.wall_time_limit_seconds <= int(time.time() - self._start_time):
            raise TimeExceeded(
                {
                    "role": "exit",
                    "content": "TimeExceeded",
                    "extra": {"exit_status": "TimeExceeded", "submission": ""},
                }
            )
        self.n_calls += 1
        # >>> feature:format_streak_trim,context_trim
        message = self.model.query(self._model_messages())
        # --- else
        message = self.model.query(self.messages)
        # <<< feature:format_streak_trim,context_trim
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)
        return message

    # >>> feature:format_streak_trim,context_trim
    def _model_messages(self) -> list[dict]:
        """The history as the model sees it (the saved trajectory keeps everything)."""
        messages = self.messages
        # >>> feature:format_streak_trim
        if features.on("format_streak_trim"):
            messages = _drop_stale_format_errors(messages)
        # <<< feature:format_streak_trim
        # >>> feature:context_trim
        if features.on("context_trim"):
            messages = _stub_old_observations(
                messages, self.config.context_keep_recent, self.config.context_trim_min_chars
            )
        # <<< feature:context_trim
        return messages

    # <<< feature:format_streak_trim,context_trim
    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions in message, add observation messages, return them."""
        # >>> feature:submit_confirmation
        if features.on("submit_confirmation"):
            outputs = [self._execute_confirming_submit(action) for action in message.get("extra", {}).get("actions", [])]
        else:
            outputs = [self.env.execute(action) for action in message.get("extra", {}).get("actions", [])]
        # --- else
        outputs = [self.env.execute(action) for action in message.get("extra", {}).get("actions", [])]
        # <<< feature:submit_confirmation
        # >>> feature:loop_warning
        observations = self.model.format_observation_messages(message, outputs, self.get_template_vars())
        if features.on("loop_warning"):
            self._warn_on_repeats(message, outputs, observations)
        return self.add_messages(*observations)
        # --- else
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))
        # <<< feature:loop_warning

    # >>> feature:submit_confirmation
    def _execute_confirming_submit(self, action: dict) -> dict:
        """The first submission is held back once and the model is asked to check its work."""
        try:
            return self.env.execute(action)
        except Submitted:
            if self._submit_confirmed:
                raise
            self._submit_confirmed = True
            return {"output": SUBMIT_CHECKLIST, "returncode": 0, "exception_info": ""}

    # <<< feature:submit_confirmation
    # >>> feature:loop_warning
    def _warn_on_repeats(self, message: dict, outputs: list[dict], observations: list[dict]) -> None:
        actions = message.get("extra", {}).get("actions", [])
        for action, output, observation in zip(actions, outputs, observations):
            key = (action.get("command", ""), output.get("output", ""))
            self._repeat_count = self._repeat_count + 1 if key == self._last_exec else 1
            self._last_exec = key
            if self._repeat_count >= self.config.loop_warning_after:
                observation["content"] += LOOP_WARNING.format(n=self._repeat_count)

    # <<< feature:loop_warning

    def serialize(self, *extra_dicts) -> dict:
        """Serialize agent state to a json-compatible nested dictionary for saving."""
        last_message = self.messages[-1] if self.messages else {}
        last_extra = last_message.get("extra", {})
        agent_data = {
            "info": {
                "model_stats": {
                    "instance_cost": self.cost,
                    "api_calls": self.n_calls,
                },
                "config": {
                    "agent": self.config.model_dump(mode="json"),
                    "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "mini_version": __version__,
                "exit_status": last_extra.get("exit_status", ""),
                "submission": last_extra.get("submission", ""),
            },
            "messages": self.messages,
            "trajectory_format": "mini-swe-agent-1.1",
        }
        return recursive_merge(agent_data, self.model.serialize(), self.env.serialize(), *extra_dicts)

    def save(self, path: Path | None, *extra_dicts) -> dict:
        """Save the trajectory of the agent to a file if path is given. Returns full serialized data.
        You can pass additional dictionaries with extra data to be (recursively) merged into the output data.
        """
        data = self.serialize(*extra_dicts)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
        return data
# >>> feature:submit_confirmation


SUBMIT_CHECKLIST = (
    "Submission paused for one final check. Before you submit, make sure that:\n"
    "1. every file or output the task asks for exists at the exact path and in the exact format requested;\n"
    "2. you actually ran your result (script, build, service, test) after your last edit and it worked;\n"
    "3. anything that must keep running (e.g. a service) is still running.\n"
    "Fix anything that is missing. When everything is verified, issue the submit command again."
)
# <<< feature:submit_confirmation
# >>> feature:loop_warning

LOOP_WARNING = (
    "\n\nNOTE: you have now run this exact command {n} times in a row and got the same output each time. "
    "Running it again will not give new information. Change your approach: inspect the error, read the "
    "relevant files or documentation, try a different command, or submit if the task is done."
)
# <<< feature:loop_warning
# >>> feature:format_streak_trim


def _drop_stale_format_errors(messages: list[dict]) -> list[dict]:
    """During a streak of format errors, show the model only the latest rejection."""
    tail = len(messages)
    while tail > 0 and messages[tail - 1].get("extra", {}).get("interrupt_type") == "FormatError":
        tail -= 1
    if len(messages) - tail <= 1:
        return messages
    return messages[:tail] + messages[-1:]
# <<< feature:format_streak_trim
# >>> feature:context_trim


def _stub_old_observations(messages: list[dict], keep_recent: int, min_chars: int) -> list[dict]:
    """Shorten long command results that are older than the `keep_recent` most recent ones."""
    observation_ids = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    old = set(observation_ids[:-keep_recent] if keep_recent > 0 else observation_ids)
    out = []
    for i, m in enumerate(messages):
        content = m.get("content")
        if i in old and isinstance(content, str) and len(content) > min_chars:
            head = content[:400]
            m = {**m, "content": f"{head}\n[... older output shortened: {len(content) - 400} more characters not shown; re-run the command if you need it ...]"}
        out.append(m)
    return out
# <<< feature:context_trim