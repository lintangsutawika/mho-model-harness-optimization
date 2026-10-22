{problem_statement}

---

You are in a checkout of `{repo}` at commit `{base_commit}`, located at `{workdir}`.
Fix the issue by editing the repository source there.

The verifier restores every file the reference tests touch, applies the reference test patch
and then runs `{test_cmd}` in `{workdir}`, so editing the tests cannot make the task pass.
