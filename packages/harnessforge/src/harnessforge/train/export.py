"""HF export of a miles-tinker run's weights from its disk-delta stream.

With `--update-weight-transfer-mode disk-delta` miles publishes every weight push as
`<disk_dir>/weight_v<NNNNNN>/` (`model-*.safetensors` holding zstd-compressed per-tensor deltas
of the changed tensors, plus `model.safetensors.index.json` with version, base_version, encoding
and checksums) against the previous version, starting from the `--hf-checkpoint` bytes
(version 0). It never prunes them. So any version is `base (+) delta_1 (+) ... (+) delta_N`,
which this module replays on CPU, away from the trainer:

    DeltaExporter(base="/hf/Qwen3.5-4B", deltas="<run>/weights/delta", work="<run>/export/work",
                  out="<run>/hf", every=10, policy_file="<run>/hf/latest.json").run()

* `work` is a full HF copy of the base (config, tokenizer, vision tower, MTP head: everything
  training does not touch is kept as is), patched in place version by version, each tensor
  checked against the publisher's checksum when the hash library is available;
* every `every`-th version (and on `snapshot()`) `work` is copied to `<out>/step_<N>` (atomically:
  a temp dir renamed), with `HF_EXPORT.json`, and `policy_file` is pointed at it
  ({"id", "path", "base_model"}: what `harnessforge.pool` loads, and the lent-node restart target);
* the applied version is recorded in `work/.hf_export_version`; an apply interrupted midway
  (`.hf_export_applying` left behind) rebuilds `work` from the base, since XOR deltas are not
  idempotent.

Version N is the weights after the N-th `save_weights_for_sampler` push (version 0 = the base, at
the gateway's warmup). platoon pushes once at launch (its launch checkpoint) and then once per
optimizer step (a permanent or recovery checkpoint replaces that step's weights-only push), and
once more for its final checkpoint, so step k = version k + `version_offset` (1). The launch push
is not empty with Qwen3.5 (it carries the trainer's BF16 rounding of the F32 norm weights); the
final push is (same weights as the last step), so the final export is named after the last version
that changed weights. Exports are `step_<k>`, with the version in HF_EXPORT.json.

Needs numpy and zstandard (the miles images have both); checksums need xxhash/blake3 for those
formats (adler32 is stdlib).
"""
from __future__ import annotations

import json
import os
import shutil
import struct
import time
import zlib
from pathlib import Path
from typing import Any, Callable

VERSION_FILE = ".hf_export_version"
APPLYING_FILE = ".hf_export_applying"
STREAM_FILE = ".hf_export_stream"         # which delta stream work was built from


# ----------------------------------------------------------------------------- safetensors I/O
def _header(path: Path) -> tuple[int, dict[str, Any]]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return n, json.loads(f.read(n))


def tensor_locations(ckpt: Path) -> dict[str, tuple[Path, int, int]]:
    """tensor name -> (file, byte offset, byte length) over every *.safetensors in ckpt."""
    out: dict[str, tuple[Path, int, int]] = {}
    for path in sorted(Path(ckpt).glob("*.safetensors")):
        n, header = _header(path)
        for name, info in header.items():
            if name == "__metadata__":
                continue
            begin, end = info["data_offsets"]
            out[name] = (path, 8 + n + begin, end - begin)
    return out


def read_delta_file(path: Path) -> tuple[dict[str, bytes], dict[str, str]]:
    """A delta shard: {tensor name: compressed delta bytes}, {tensor name: checksum}."""
    n, header = _header(path)
    meta = header.pop("__metadata__", {}) or {}
    data = {}
    with open(path, "rb") as f:
        for name, info in header.items():
            begin, end = info["data_offsets"]
            f.seek(8 + n + begin)
            data[name] = f.read(end - begin)
    return data, meta


# ----------------------------------------------------------------------------- deltas
def _checksum(algorithm: str, buf) -> str | None:
    if algorithm == "adler32":
        return f"{zlib.adler32(buf, 1):08x}"
    try:
        if algorithm == "xxh3-128":
            import xxhash
            return xxhash.xxh3_128(buf).hexdigest()
        if algorithm == "blake3":
            import blake3
            return blake3.blake3(buf).hexdigest()
    except ImportError:
        return None
    return None


def apply_delta(old, delta: bytes, encoding: str):
    """New tensor bytes (numpy uint8) from the old bytes and one decompressed delta."""
    import numpy as np
    d = np.frombuffer(delta, dtype=np.uint8)
    if encoding == "xor":
        if d.size != old.size:
            raise ValueError(f"xor delta of {d.size} bytes for a {old.size}-byte tensor")
        return old ^ d
    if encoding == "overwrite":
        (count,) = struct.unpack("<I", delta[:4])
        pos = np.frombuffer(delta, dtype="<u4", count=count, offset=4)
        vals = d[4 + 4 * count:]
        if vals.size != count:
            raise ValueError("malformed overwrite delta")
        new = old.copy()
        new[pos] = vals
        return new
    raise ValueError(f"unknown delta encoding {encoding!r}")


def versions(delta_dir: Path) -> list[int]:
    """Published versions (those whose index is written: the publisher writes it last)."""
    out = []
    for p in Path(delta_dir).glob("weight_v*"):
        if (p / "model.safetensors.index.json").exists():
            try:
                out.append(int(p.name[len("weight_v"):]))
            except ValueError:
                pass
    return sorted(out)


def _atomic_json(path: Path, data: Any) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n")
    os.replace(tmp, path)


class DeltaExporter:
    def __init__(self, *, base: str | Path, deltas: str | Path, work: str | Path,
                 out: str | Path, every: int = 10, policy_file: str | Path | None = None,
                 run_id: str = "run", base_model: str | None = None, version_offset: int = 1,
                 log: Callable[[str], None] = print):
        self.base = Path(base)
        self.deltas = Path(deltas)
        self.work = Path(work)
        self.out = Path(out)
        self.every = int(every)
        self.policy_file = Path(policy_file) if policy_file else None
        self.run_id = run_id
        self.base_model = base_model or str(self.base)
        self.version_offset = int(version_offset)
        self.log = log

    # -- state
    @property
    def version(self) -> int:
        try:
            return int((self.work / VERSION_FILE).read_text().strip())
        except (FileNotFoundError, ValueError):
            return -1

    def _set_version(self, v: int) -> None:
        _atomic_json(self.work / VERSION_FILE, v)

    def reset(self) -> None:
        """work := a fresh copy of the base (version 0)."""
        if self.work.exists():
            aside = self.work.with_name(f"{self.work.name}.broken-{int(time.time())}")
            os.replace(self.work, aside)
            shutil.rmtree(aside, ignore_errors=True)     # our own scratch copy, rebuilt below
        tmp = self.work.with_name(f".{self.work.name}.tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(self.base, tmp, symlinks=False,
                        ignore=shutil.ignore_patterns(".cache", ".git*"))
        os.replace(tmp, self.work)
        self._set_version(0)

    def stream_id(self) -> str:
        """miles recreates the delta dir at every gateway start (a new stream from version 0): the dir's inode, and
        the inode of its first version dir (an inode can be reused by a recreated dir; both together rarely are).
        Not the dir's ctime: that changes whenever a version is added or removed, which made every later poll see
        a "new stream" and rebuild the export from the base (a full replay, ~14 s per version)."""
        try:
            st = os.stat(self.deltas)
        except FileNotFoundError:
            return ""
        first = sorted(Path(self.deltas).glob("weight_v*"))
        try:
            first_ino = os.stat(first[0]).st_ino if first else 0
        except FileNotFoundError:
            first_ino = 0
        return f"{st.st_dev}-{st.st_ino}-{first_ino}"

    def ensure(self) -> None:
        try:
            built_from = (self.work / STREAM_FILE).read_text().strip()
        except FileNotFoundError:
            built_from = None
        if (self.work / APPLYING_FILE).exists() or self.version < 0 or built_from != self.stream_id():
            self.log(f"[export] (re)building {self.work} from {self.base}")
            self.reset()
            (self.work / STREAM_FILE).write_text(self.stream_id())

    # -- applying
    def apply(self, v: int) -> dict[str, Any]:
        vdir = self.deltas / f"weight_v{v:06d}"
        index = json.loads((vdir / "model.safetensors.index.json").read_text())
        meta = index.get("metadata", {})
        base_v = int(meta.get("base_version", v - 1))
        if base_v != self.version:
            raise RuntimeError(f"delta v{v} is against v{base_v}, work is at v{self.version}")
        encoding = meta.get("delta_encoding", "xor")
        algorithm = meta.get("checksum_format", "xxh3-128")
        import numpy as np
        import zstandard
        dctx = zstandard.ZstdDecompressor()
        locs = tensor_locations(self.work)
        (self.work / APPLYING_FILE).write_text(str(v))
        n = changed = unchecked = 0
        for fname in sorted(set(index.get("weight_map", {}).values())):
            data, sums = read_delta_file(vdir / fname)
            for name, blob in data.items():
                if name not in locs:
                    raise KeyError(f"delta v{v} names {name!r}, not in the checkpoint")
                path, off, nbytes = locs[name]
                with open(path, "r+b") as f:
                    f.seek(off)
                    old = np.frombuffer(f.read(nbytes), dtype=np.uint8)
                    new = apply_delta(old, dctx.decompress(blob, max_output_size=5 * nbytes + 8),
                                      encoding)
                    want = sums.get(name)
                    got = _checksum(algorithm, new.tobytes()) if want else None
                    if want and got is None:
                        unchecked += 1
                    elif want and got != want:
                        raise RuntimeError(f"checksum mismatch for {name!r} at v{v}")
                    f.seek(off)
                    f.write(new.tobytes())
                n += 1
                changed += nbytes
        self._set_version(v)
        (self.work / APPLYING_FILE).unlink()
        return {"version": v, "tensors": n, "bytes": changed, "unchecked": unchecked}

    def stream_base(self) -> int:
        """Steps before this stream (a resumed run: the checkpoint's batch). The driver records it
        in <deltas>/../stream_base.json for the stream it starts against."""
        f = self.deltas.parent / "stream_base.json"
        try:
            d = json.loads(f.read_text())
        except (FileNotFoundError, ValueError):
            return 0
        return int(d.get("batch", 0)) if d.get("stream") == self.stream_id() else 0

    def step_of(self, v: int) -> int:
        return self.stream_base() + v - self.version_offset

    def changed(self, v: int) -> bool:
        index = json.loads((self.deltas / f"weight_v{v:06d}" / "model.safetensors.index.json").read_text())
        return bool(index.get("weight_map"))

    def last_changed(self, upto: int) -> int:
        """The last version <= upto that changed weights (later ones re-push the same weights)."""
        last = 0
        for u in versions(self.deltas):
            if u <= upto and self.changed(u):
                last = u
        return last

    def snapshot(self, v: int | None = None) -> Path:
        """Copy work (at its current version) to <out>/step_<k> (k = v - version_offset) and point
        policy_file at it. v may be an earlier version when every version after it was empty."""
        v = self.version if v is None else v
        if v != self.version and self.last_changed(self.version) != v:
            raise RuntimeError(f"work is at v{self.version}, not v{v}")
        k = self.step_of(v)
        dest = self.out / f"step_{k}"
        if dest.exists():
            return dest
        self.out.mkdir(parents=True, exist_ok=True)
        tmp = self.out / f".step_{k}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(self.work, tmp, ignore=shutil.ignore_patterns(VERSION_FILE, APPLYING_FILE, STREAM_FILE))
        _atomic_json(tmp / "HF_EXPORT.json", {"version": v, "step": k, "run_id": self.run_id,
                                              "base": str(self.base), "deltas": str(self.deltas),
                                              "exported_at": time.time()})
        os.replace(tmp, dest)
        if self.policy_file is not None:
            self.policy_file.parent.mkdir(parents=True, exist_ok=True)
            _atomic_json(self.policy_file, {"id": f"{self.run_id}-step{k}", "path": str(dest),
                                            "base_model": self.base_model, "version": v, "step": k})
        self.log(f"[export] v{v} = step {k} -> {dest}")
        return dest

    def poll(self, *, final: bool = False) -> list[int]:
        """Apply every new published version; snapshot each multiple of `every` (and, final,
        the last). Returns the versions applied."""
        self.ensure()
        done = []
        for v in versions(self.deltas):
            if v <= self.version:
                continue
            if v != self.version + 1:
                raise RuntimeError(f"missing delta v{self.version + 1} (next is v{v})")
            t0 = time.time()
            info = self.apply(v)
            self.log(f"[export] applied v{v}: {info['tensors']} tensors in {time.time() - t0:.1f}s"
                     + (f" ({info['unchecked']} unchecked: no hash lib)" if info["unchecked"] else ""))
            done.append(v)
            k = self.step_of(v)
            if self.every and k > 0 and k % self.every == 0 and self.changed(v):
                self.snapshot(v)
        if final:
            last = self.last_changed(self.version)
            if self.step_of(last) > 0:
                self.snapshot(last)
        return done

    def run(self, *, stop: Callable[[], bool], interval: float = 30.0) -> None:
        while True:
            stopping = stop()
            self.poll(final=stopping)
            if stopping:
                return
            time.sleep(interval)
