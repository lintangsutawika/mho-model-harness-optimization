# mho tasks/

Harbor task generators for the mho pipeline. Each task is a self-contained CLI
package: `main.py` holds everything (CLI + HF data loading + template/adapter
generation) with a 1-line `__main__.py` for `-m`; `prepare.py` and
`task-template/` supply the data loader and the task template.

## Tasks

| Task | Dataset | Verifier |
|------|---------|----------|
| `dapo_math_17k` | `BytedTsinghua-SIA/DAPO-Math-17k` | self-grade: `tests/test.sh` → `reward.txt` (parse `Answer: $A`, normalized) |
| `hmmt_feb_2025` | `MathArena/hmmt_feb_2025` | self-grade: `tests/test.sh` → `reward.txt` (`math_verify` → sympy → string) |
| `swe_rebench` | `nebius/SWE-rebench-V2` | self-grade: `tests/test.sh` → `reward.txt` (apply the reference test patch, run the instance's `test_cmd`, resolve only if every `FAIL_TO_PASS` and `PASS_TO_PASS` passes) |
| `nemotron_terminal` | `nvidia/Nemotron-Terminal-Synthetic-Tasks` (`skill_based/mixed`) | self-grade: `tests/test.sh` → `reward.txt` (pytest on the task's `tests/test_outputs.py`) |

## Isolation

Each task is a **self-contained harbor task package**: its `tests/test.sh` grades the trial
and writes `verifier/reward.txt` (read by harbor's built-in file verifier). No task imports
`src/` or relies on a global in-process verifier (`MHO_VERIFIER` is only an optional override
for tasks needing an in-process verifier). New tasks (math or otherwise) add a self-contained
`tests/test.sh` + `tests/test.py` (e.g. `test_outputs.py` + `ground_truth.json`).

## How to prepare

From the repo root (CLI, builds the dataset on first use):

```bash
uv run --with datasets python -m tasks.dapo_math_17k --limit 10 --output-dir out/dapo
uv run --with datasets python -m tasks.hmmt_feb_2025 --limit 10 --output-dir out/hmmt
python -m tasks.nemotron_terminal --output-dir out/nemotron   # 50 per skill category, seed 1
```
