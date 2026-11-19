"""MicroSweAgent.install: run the candidate from a mounted toolchain (no network), else the per-trial uv install."""
import asyncio
import io
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
pytest.importorskip("harbor")

from harbor.environments.base import ExecResult  # noqa: E402

from harness import micro_swe_agent as msa  # noqa: E402

PACKAGES = frozenset({"litellm", "jinja2", "pydantic", "tenacity", "pyyaml", "mcp", "orjson", "fastapi"})


def _candidate(root: Path, deps=("litellm", "jinja2", "pydantic", "tenacity", "pyyaml", "mcp"),
               entry='"micro_swe.__main__:main"', layout="src") -> Path:
    pkg = (root / "src" / "micro_swe") if layout == "src" else (root / "micro_swe")
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "__main__.py").write_text(
        "import sys\n"
        "def main(argv=None):\n"
        "    print('ARGV', sys.argv[0], sys.argv[1:])\n"
        "    if '--fail' in sys.argv: return 3\n")
    dep_list = ", ".join(f'"{d}"' for d in deps)
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "micro-swe-agent"\nversion = "0"\ndependencies = [{dep_list}]\n'
        f'[project.scripts]\nmini-swe-agent = {entry}\n')
    return root


def test_toolchain_plan(tmp_path):
    assert msa.toolchain_plan(_candidate(tmp_path / "a"), PACKAGES) == ("src", "micro_swe.__main__", "main")
    assert msa.toolchain_plan(_candidate(tmp_path / "b", layout="root"), PACKAGES) == (".", "micro_swe.__main__", "main")
    extra = msa.toolchain_plan(_candidate(tmp_path / "c", deps=("litellm>=1", "rich[jupyter]~=13")), PACKAGES)
    assert isinstance(extra, str) and "rich" in extra
    assert "entry point" in msa.toolchain_plan(_candidate(tmp_path / "d", entry='"bad entry"'), PACKAGES)
    assert "no packages" in msa.toolchain_plan(_candidate(tmp_path / "e"), frozenset())


def test_launcher_runs_entry_point_like_a_console_script(tmp_path):
    cand = _candidate(tmp_path / "cand")
    script = tmp_path / "mini-swe-agent"
    script.write_text(msa.launcher(sys.executable, str(cand / "src"), "micro_swe.__main__", "main"))
    script.chmod(0o755)
    out = subprocess.run([str(script), "--yolo", "--task=it's \"quoted\" $HOME", "-c", "mini"],
                         capture_output=True, text=True, env={"PYTHONPATH": "/nonexistent"}, cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ARGV mini-swe-agent ['--yolo', '--task=it\\'s \"quoted\" $HOME', '-c', 'mini']"
    assert subprocess.run([str(script), "--fail"], capture_output=True).returncode == 3


class FakeEnv:
    """Records uploads and commands; `fail` makes commands containing it exit 1."""

    default_user = None

    def __init__(self, toolchain=True, fail=None):
        self.agent_toolchain = ("/opt/mho/toolchain", PACKAGES) if toolchain else None
        self.commands, self.tar_members, self.fail = [], {}, fail

    async def upload_file(self, source, target):
        with tarfile.open(source) as tf:
            for m in tf.getmembers():
                if m.isfile():
                    self.tar_members[m.name] = (tf.extractfile(m).read().decode(), m.mode)

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.commands.append(command)
        code = 1 if self.fail and self.fail in command else 0
        return ExecResult(stdout="", stderr="boom" if code else "", return_code=code)


def _agent(tmp_path, cand):
    return msa.MicroSweAgent(logs_dir=tmp_path / "logs", model_name="litellm_proxy/policy", mini_fork_local=str(cand))


def _ran_uv(env):
    return [c for c in env.commands if "uv tool install" in c]


def test_install_from_toolchain_downloads_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("MHO_AGENT_INSTALL", raising=False)
    env = FakeEnv()
    asyncio.run(_agent(tmp_path, _candidate(tmp_path / "cand")).install(env))
    assert not _ran_uv(env) and not any("curl" in c and "astral" in c for c in env.commands)
    (launcher_name,) = [n for n in env.tar_members if n.endswith("/.mho-bin/mini-swe-agent")]
    text, mode = env.tar_members[launcher_name]
    leaf = launcher_name.split("/")[0]
    assert mode & 0o111 and "/opt/mho/toolchain/bin/python3 -I -c" in text and f"/tmp/{leaf}/src" in text
    (check,) = [c for c in env.commands if '$HOME/.local/bin/mini-swe-agent' in c]
    assert "mini-swe-agent --help" in check and '/opt/mho/toolchain/bin/python3 -I -c "import litellm' in check


@pytest.mark.parametrize("case", ["no_toolchain", "extra_dependency", "forced", "launcher_fails"])
def test_install_falls_back_to_uv(tmp_path, monkeypatch, case):
    monkeypatch.delenv("MHO_AGENT_INSTALL", raising=False)
    deps = ("litellm", "numpy") if case == "extra_dependency" else ("litellm",)
    if case == "forced":
        monkeypatch.setenv("MHO_AGENT_INSTALL", "uv")
    env = FakeEnv(toolchain=case != "no_toolchain", fail=".mho-bin" if case == "launcher_fails" else None)
    asyncio.run(_agent(tmp_path, _candidate(tmp_path / "cand", deps=deps)).install(env))
    (uv,) = _ran_uv(env)
    assert "--with litellm --with mcp --with orjson --with fastapi" in uv and uv.endswith("mini-swe-agent --help")
    assert ("--force" in uv) == (case == "launcher_fails")  # only to replace the failed launcher
    launched = any(n.endswith(".mho-bin/mini-swe-agent") for n in env.tar_members)
    assert launched == (case == "launcher_fails")
