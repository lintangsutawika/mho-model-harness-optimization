"""Command line for an eval node job.

    python -m harnessforge.evals run REQUEST.json [--poll 30]
        Start PlatoonBackend on the request and write <out_dir>/<run_id>/records.jsonl every --poll seconds until
        every candidate is done. Creating <out_dir>/<run_id>/<candidate>.stop stops that candidate.

    python -m harnessforge.evals stage-images REQUEST.json --from SHARED_CACHE --to NODE_CACHE
        Copy the hpc-sandbox image cache entries of the request's tasks (by their docker_image, else their
        Dockerfile's FROM) from a shared cache to a node-local one; missing ones are pulled on first use.

REQUEST.json holds EvalRequest's fields (candidates as {id, harness_dir, parent}) plus "platoon" (the platoon
executable) and "site" (a platoon site profile), for PlatoonBackend.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import tomllib
from pathlib import Path

from harnessforge.contracts import Candidate, EvalRequest
from harnessforge.evals.backend import PlatoonBackend
from harnessforge.evals.dataset import dockerfile_info


def load_request(path: str | Path) -> tuple[EvalRequest, dict]:
    raw = json.loads(Path(path).read_text())
    candidates = [Candidate(c["id"], Path(c["harness_dir"]), c.get("parent"), c.get("meta", {}))
                  for c in raw["candidates"]]
    request = EvalRequest(run_id=raw["run_id"], candidates=candidates, tasks=list(raw["tasks"]),
                          dataset=raw["dataset"], trials=int(raw["trials"]), endpoint=raw["endpoint"],
                          policy=raw.get("policy"), concurrency=int(raw.get("concurrency", 32)),
                          out_dir=Path(raw["out_dir"]), options=raw.get("options", {}))
    return request, raw


def cmd_run(args: argparse.Namespace) -> int:
    request, raw = load_request(args.request)
    backend = PlatoonBackend(platoon=raw.get("platoon", "platoon"), site=raw.get("site"))
    run = backend.start(request)
    run_dir = Path(request.out_dir) / request.run_id
    print(json.dumps(run.status(), indent=1), flush=True)
    while True:
        for candidate in request.candidates:
            if (run_dir / f"{candidate.id}.stop").exists():
                run.stop(candidate.id)
        records = run.write_records(run_dir / "records.jsonl")
        done = run.done()
        finished = sum(r.finished for r in records)
        print(f"[{time.strftime('%H:%M:%S')}] {finished} finished, {len(records) - finished} in flight", flush=True)
        if done:
            break
        time.sleep(args.poll)
    run.write_records(run_dir / "records.jsonl")
    print(json.dumps(run.status(), indent=1), flush=True)
    return 0


def task_image(task_dir: Path) -> str | None:
    environment = tomllib.loads((task_dir / "task.toml").read_text()).get("environment", {})
    image = environment.get("docker_image") or dockerfile_info(task_dir / "environment" / "Dockerfile")[0]
    if image is None:
        return None
    return image if "://" in image or image.endswith((".sif", ".sqsh")) else f"docker://{image}"


def cmd_stage_images(args: argparse.Namespace) -> int:
    request, _ = load_request(args.request)
    shared, dest = Path(args.from_), Path(args.to)
    staged, missing = stage_images(request.dataset, request.tasks, shared, dest)
    print(f"staged {staged} image(s) into {dest}; {missing} not in {shared} (pulled on first use)")
    return 0


def stage_images(dataset: str | Path, tasks, shared: str | Path, dest: str | Path) -> tuple[int, int]:
    """Copy the shared hpc-sandbox image cache entries of `tasks` to `dest` (node-local).
    Returns (staged, missing)."""
    shared, dest = Path(shared), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    staged = missing = 0
    for task in tasks:
        uri = task_image(Path(dataset) / task)
        if uri is None or "://" not in uri:
            continue
        key = "uri-" + hashlib.sha256(uri.encode()).hexdigest()[:40]
        index = shared / f"{key}.sif.json"
        if not index.is_file():
            missing += 1
            continue
        sif = shared / f"sha256-{json.loads(index.read_text())['sha256']}.sif"
        if not sif.is_file():
            missing += 1
            continue
        for src in (sif, index):  # the index last: an entry is usable once both are there
            if not (dest / src.name).exists():
                shutil.copyfile(src, dest / (src.name + ".part"))
                (dest / (src.name + ".part")).replace(dest / src.name)
        staged += 1
    return staged, missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m harnessforge.evals")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run a request with PlatoonBackend")
    run.add_argument("request")
    run.add_argument("--poll", type=float, default=30.0)
    stage = sub.add_parser("stage-images", help="copy cached sandbox images to node-local disk")
    stage.add_argument("request")
    stage.add_argument("--from", dest="from_", required=True)
    stage.add_argument("--to", required=True)
    args = parser.parse_args(argv)
    return {"run": cmd_run, "stage-images": cmd_stage_images}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
