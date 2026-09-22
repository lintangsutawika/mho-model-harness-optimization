# mho tasks/

Harbor task generators for the mho pipeline. Each task is a self-contained CLI
package: `main.py` holds everything (CLI + HF data loading + template/adapter
generation) with a 1-line `__main__.py` for `-m`; `prepare.py` and
`task-template/` supply the data loader and the task template.

## Tasks

| Task | Dataset | Verifier |
|------|---------|----------|
| `dapo_math_17k` | `BytedTsinghua-SIA/DAPO-Math-17k` | answer-string (harness.math_verifier style: parse `Answer: $A`) |
| `hmmt_feb_2025` | `MathArena/hmmt_feb_2025` | `math_verify` → sympy → normalized string compare |

## How to prepare

From the repo root (CLI, builds the dataset on first use):

```bash
uv run --with datasets python -m tasks.dapo_math_17k --limit 10 --output-dir out/dapo
uv run --with datasets python -m tasks.hmmt_feb_2025 --limit 10 --output-dir out/hmmt
```
