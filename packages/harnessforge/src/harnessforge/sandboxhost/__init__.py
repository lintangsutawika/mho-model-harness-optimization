"""Sandbox hosts: run platoon's hpc-sandbox pods in other batch jobs than the platoon client's.

    python -m harnessforge.sandboxhost start  --spec fleet.json [--hosts N] [--no-wait]
    python -m harnessforge.sandboxhost status --spec fleet.json [--json]
    python -m harnessforge.sandboxhost stop   --spec fleet.json [--host I]

and, in a run's config, `sandboxes: {mode: remote, hosts_spec: fleet.json, local_slots: N}` (train) or
`options.sandboxes` (evals), which select `provider.RemotePodProvider`.

- `daemon`: the host (one per CPU job): starts pods on request, relays their gateway sockets back, leases.
- `provider`: the client side platoon builds (`RemotePodProvider`, `RemoteSandbox`, the upstream relay).
- `fleet`: the fleet spec, its shared state dir (host files, token), job submission.
- `wire`: framing, the token's challenge-response auth, payloads, errors.

See docs/sandboxhost.md.
"""
from __future__ import annotations

REMOTE_PROVIDER = "harnessforge.sandboxhost.provider:RemotePodProvider"


def provider_settings(sandboxes: dict | None) -> tuple[str | None, dict]:
    """(provider, extra provider_options) for a run's `sandboxes` section; (None, {}) for local sandboxes.

        {mode: local}                                         -> the pod provider on this node (default)
        {mode: remote, hosts_spec: F, local_slots: N, ...}    -> RemotePodProvider; other keys pass through
                                                                 (shared_roots, advertise, place_timeout_s)
    """
    s = dict(sandboxes or {})
    mode = s.pop("mode", "local")
    if mode == "local":
        return None, {}
    if mode != "remote":
        raise ValueError(f"sandboxes.mode must be local or remote, not {mode!r}")
    if not s.get("hosts_spec") and not int(s.get("local_slots", 0)):
        raise ValueError("sandboxes.mode remote needs hosts_spec (and/or local_slots)")
    allowed = {"hosts_spec", "local_slots", "shared_roots", "advertise", "place_timeout_s"}
    unknown = set(s) - allowed
    if unknown:
        raise ValueError(f"unknown sandboxes keys: {sorted(unknown)} (known: {sorted(allowed)})")
    return REMOTE_PROVIDER, s
