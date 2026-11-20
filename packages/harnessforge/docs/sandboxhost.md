# Sandbox hosts: platoon's pods on other batch jobs

`harnessforge.sandboxhost` lets a platoon client (`platoon train` on the trainer's head node, `platoon eval` in an eval
job) run its hpc-sandbox pods in other CPU jobs as well as on its own node. Rollout concurrency is then bounded by
the number of host jobs rather than by the client node's cores and memory.

```
 client node (platoon train / eval)                       sandbox host job (rt_HC, 32 cores)
 +---------------------------------------+   TCP 10.0.x  +------------------------------------------+
 | rollout worker                        |  (token auth) | python -m harnessforge.sandboxhost serve  |
 |  RemotePodProvider.start(spec) -------+--- start ---->|  PodSingularityProvider.start(spec')      |
 |  RemoteSandbox.exec/upload/download --+--- exec ----->|   pod: main service (task image, root)    |
 |                                       |               |   pod loopback :18080 -> host socket      |
 |  UpstreamRelay (TCP, per process) <---+--- gateway ---+-- <scratch>/up/<n>/gateway/gw.sock        |
 |   -> platoon's per-rollout gw.sock    |               |                                          |
 |  local slots (optional): local pods   |               |  slots, start cap, leases, mirrors        |
 +---------------------------------------+               +------------------------------------------+
```

## How platoon talks to a sandbox (what had to be bridged)

- **Control.** platoon drives a sandbox only through the Python provider interface (`platoon.sandbox.base`): `start`,
  `exec`, `upload`, `download`, `kill_processes` / `keep_processes`, `ensure_routes`, `stop`; reaping goes through
  `provider.reap(labels)` from the 0600 mirror file of live sandboxes (`platoon.sandbox.cleanup`). There is no
  sidecar in the sandbox for control: hpc-sandbox runs each command with `singularity exec` into the instance.
- **Data (model calls).** The harness in the sandbox reaches platoon's model gateway, a unix socket the rollout
  creates in a private 0700 directory on the client node (`gw.sock`, `SandboxSpec.upstreams["gateway"]`). A pod's
  holder relays it to the pod's loopback (`http://127.0.0.1:18080`, the route); a no-network sandbox gets a
  forwarder process instead. Either needs the socket on the sandbox's node.
- **Paths.** Mounts are host paths: the candidate harness ($HOME, shared) and platoon's staged toolchain (node-local
  `$PLATOON_NODE_SCRATCH/platoon-toolchains-<uid>/...`, ~500 MB). Images are `docker://` refs resolved through the
  image cache, or prepared SIFs (`image_map`, on /groups).

## Design

**Host daemon** (`daemon.SandboxHost`, one per job). Listens on TCP; every connection answers an HMAC-SHA256
challenge over the fleet token (`<state>/token`, 0600, never printed or sent). It runs requests against its own
`PodSingularityProvider`, so prepared images, the `max_starting` start cap, fakeroot pods and slirp egress behave as
locally. Per request:

- `start`: refuses with HostFull past `slots_per_host` (started + starting). Rewrites each unix upstream to a socket
  in the job's scratch; each connection to it opens a TCP connection to the client's relay with the upstream's key.
  The pod relays that socket to its loopback as usual, so routes are unchanged. The reply is acknowledged; a client
  that is gone before the ack (it gave the start up) gets its sandbox stopped.
- `exec` / `upload` / `download` / `kill_processes` / `ensure_routes`: run on the sandbox; tar payloads; a client
  disconnect cancels the op. Long ops send a heartbeat every 10 s.
- `stop`, `reap(labels)`, `renew(ids)` (leases), `status`, `mirror_begin` / `mirror_put`.
- Leases: a sandbox untouched by any op or renewal for `lease_s` (900 s) with nothing in flight is stopped.
- Shutdown (SIGTERM from qdel, `STOP` / `STOP.<i>`): stops every sandbox, writes `status: stopped`.

**Client provider** (`provider.RemotePodProvider`, a `PodSingularityProvider` subclass, so every refusal check,
`collect` and local behaviour is inherited):

- Placement: a free local slot (`local_slots`, a node-wide flock count held for the sandbox's life) starts the pod
  here; otherwise it asks every live host (`hosts/<i>.json` refreshed in the last 60 s) for its free slots and tries
  them in a random order weighted by free slots; HostFull or an unreachable host moves on; all full: wait and retry
  (up to `place_timeout_s`, 3600 s). An unreachable host is avoided for 60 s.
- The spec crosses with the process's cleanup labels, after the prepared-image map is applied. Mount sources, a .sif
  and the owner manifest outside `shared_roots` (/groups, /home, $HOME) are mirrored to the host once per
  (host, path, inode, size, mtime) — platoon's staged toolchain is named by its tree hash and never changed in
  place — and the spec points at the copy. One process uploads; the others wait for it.
- Sandbox options fakeroot, allow_network_none, pod_users, rootfs_layout, ... (and image_cache when shared) are
  forwarded; the host keeps its own scratch and start cap.
- `RemoteSandbox` implements the contract over one TCP connection per call. `kill_processes` is a no-op under
  `keep_agent_processes` (harbor parity), as locally. `collect` (harbor export) runs through exec. No snapshots;
  stacks and restores run locally.
- Failures: a refused connection, a broken one, or 40 s without a heartbeat raises `hpc_sandbox.SandboxLostError`
  on that call and every later one, which platoon reports as an infrastructure failure (an untrainable outcome in
  training; env_failed in evals). `stop` of a lost sandbox just forgets it. A keepalive renews each host's leases
  every 60 s and marks lost the sandboxes a host no longer has. `reap(labels)` reaps locally and on every live host.

## Use

Fleet spec (example `configs/sandboxhost/grpo.json`):

```json
{"fleet_id": "sbh-grpo", "state_root": "~/hf-dev/sandboxhosts/state", "hosts": 5, "slots_per_host": 96,
 "python": "~/hf-dev/evals/venv/bin/python", "harnessforge_src": "<harnessforge src with this package>",
 "provider_options": {"image_cache": "/groups/gai51740/aci18914wh/mho/hpc-sandbox-images", "max_starting": 16,
                      "fakeroot": true, "allow_network_none": true},
 "launcher": {"kind": "pbs", "queue": "R9920261300", "account": "gai51740", "resource_type": "rt_HC",
              "walltime": "72:00:00"}}
```

```
python -m harnessforge.sandboxhost start  --spec fleet.json [--hosts N] [--no-wait]   # (re)submits missing hosts
python -m harnessforge.sandboxhost status --spec fleet.json                           # jobs, files, live pings
python -m harnessforge.sandboxhost stop   --spec fleet.json [--host I]                # STOP, then qdel after grace
```

`start` is idempotent: it submits a job for each index below `hosts` that has none running, so it also replaces a
dead host or grows the fleet (`--hosts 6`). Clients see a new host within 3 s of its first status file.

Train run config: `"sandboxes": {"mode": "remote", "hosts_spec": "<fleet.json or state dir>", "local_slots": 160}`
(top level; `train.max_concurrent_rollouts` should not exceed `local_slots + hosts x slots_per_host`). Evals:
`options.sandboxes` with the same keys. `mode: local` (default) keeps the old behaviour.

## Cluster tests (ABCI, 2026-11-20, no GPUs)

Fleet: 2, then 4 rt_HC host jobs (96 slots, `max_starting` 16, shared image cache, TB2.1 prepared images); clients
in their own rt_HC jobs; venv = a copy of the evals venv's packages (platoon 0.6.0, hpc-sandbox 0.5.1).

**Provider bench** (`RemotePodProvider` directly; N sandboxes started at once, then 30 `exec(["true"])` each):

| | local, 8 | remote, 8 (cold host) | remote, 64 (2 hosts) | remote, 384 (4 hosts x 96) |
|---|---|---|---|---|
| start p50 / max | 64.5 / 64.7 s (cold images) | 45.6 / 47.5 s (incl. 484 MB toolchain mirror) | 9.9 / 18.9 s | 65.7 / 113.5 s (96 per host behind a cap of 16) |
| exec RTT, one sandbox, p50 | 2.7 ms | 3.9 ms | 3.9 ms | 4.2 ms |
| exec RTT, all at once, p50 | 7.7 ms | 9.4 ms | 43 ms | 228 ms (384 concurrent singularity execs on 4 x 32 cores) |
| 1 MB upload / download | 6 / 7 ms | 18 / 23 ms | 16 / 14 ms | 49 / 20 ms |

The relay costs about 1.2 ms per call. A burst of 96 starts on one host takes about 110 s with the start cap of 16,
as it does locally.

**platoon eval, fake model** (scripted 2-turn model on the client; every sandbox remote):
- 8 TB2.1 tasks x 2: 16/16 scored, 0 env failures. The agent's model calls went through the relay. The verifier
  ran. collect / harbor export wrote test-stdout and the agent trajectory.
- 89 TB2.1 tasks x 4 = 356 at once on 4 hosts (fake model 20 s per call): all 356 finished in 540 s. Start, from
  request to up and including placement and the cap: p50 46 s, p90 72 s. Per host: 90 / 87 / 91 / 88.
  8 env failures were the known `qemu-*` apt 404s.
- Host 3 was hard-killed (`qsig -s KILL`) with 85 active and 2 starting. Its 83 running trials ended 0.3-0.7 s
  later as `error:unexpected:SandboxLostError`: infra in platoon, so untrainable, and env_failed in the reader. The 2
  starting ones went to other hosts. No hang; later starts avoided the dead host. `start --hosts 4` replaced it.

**platoon eval, real model** (pool node 0, Qwen3.5-4B policy, 24 TB2.1 tasks x 1 at 24): trials ran normally through
remote sandboxes. At 25 min, 3/12 finished were solved, with trajectories up to 235k output tokens.
