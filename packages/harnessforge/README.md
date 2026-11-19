# harnessforge

Optimize agent harnesses, and the models that run them:

- sequentially tested harness search (paired tests with early stopping, candidates race);
- a persistent model pool (N nodes of SGLang) that serves evaluation and doubles as the
  rollout engine, swapping to newer policies as they are trained;
- RL (GRPO, DPPO) and RSFT training through [platoon](https://github.com/ApGa/platoon).

Status: early. See [docs/PLAN.md](docs/PLAN.md). Design docs: sequential testing
[docs/stats.md](docs/stats.md), the model pool and launchers [docs/pool.md](docs/pool.md), the
gold-harness toy benchmark [docs/bench.md](docs/bench.md).

```bash
uv run --directory packages/harnessforge --extra dev pytest
```
