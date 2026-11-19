import json
import struct
import zlib

import pytest

np = pytest.importorskip("numpy")
zstandard = pytest.importorskip("zstandard")

from harnessforge.train.export import (APPLYING_FILE, DeltaExporter, apply_delta,
                                       tensor_locations, versions)


def write_safetensors(path, tensors, metadata=None):
    header, blobs, off = {}, [], 0
    for name, arr in tensors.items():
        b = arr.tobytes()
        dt = {"float32": "F32", "uint8": "U8", "uint16": "BF16"}[str(arr.dtype)]
        header[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    if metadata:
        header["__metadata__"] = metadata
    h = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def adler(b):
    return f"{zlib.adler32(b, 1):08x}"


def publish(delta_dir, v, old, new, encoding="xor", corrupt=False):
    """What miles' DiskDeltaProtocol writes for one version (changed tensors only)."""
    vdir = delta_dir / f"weight_v{v:06d}"
    vdir.mkdir(parents=True)
    deltas, sums = {}, {}
    for name in new:
        o = old[name].view(np.uint8).ravel()
        n = new[name].view(np.uint8).ravel()
        if np.array_equal(o, n):
            continue
        if encoding == "xor":
            d = o ^ n
        else:
            mask = n != o
            pos = np.flatnonzero(mask).astype("<u4")
            d = np.concatenate([np.array([pos.size], "<u4").view(np.uint8), pos.view(np.uint8), n[mask]])
        deltas[name] = np.frombuffer(zstandard.ZstdCompressor(level=1).compress(d.tobytes()), np.uint8)
        sums[name] = adler(n.tobytes()) if not corrupt else "deadbeef"
    write_safetensors(vdir / "model-00000-of-00001.safetensors", deltas, sums)
    (vdir / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"version": f"{v:06d}", "base_version": f"{v - 1:06d}", "delta_encoding": encoding,
                     "compression_format": "zstd", "checksum_format": "adler32"},
        "weight_map": {k: "model-00000-of-00001.safetensors" for k in deltas}}))


def make_base(tmp_path):
    rng = np.random.default_rng(0)
    base = tmp_path / "base"
    base.mkdir()
    w = {"model.language_model.layers.0.w": rng.standard_normal((4, 8)).astype(np.float32),
         "model.language_model.embed": rng.standard_normal((16,)).astype(np.float32)}
    vis = {"model.visual.patch": rng.standard_normal((3,)).astype(np.float32)}
    write_safetensors(base / "model-00001-of-00002.safetensors", w)
    write_safetensors(base / "model-00002-of-00002.safetensors", vis)
    (base / "config.json").write_text("{}")
    return base, w, vis


def read(ckpt, name, dtype=np.float32):
    path, off, n = tensor_locations(ckpt)[name]
    with open(path, "rb") as f:
        f.seek(off)
        return np.frombuffer(f.read(n), dtype=dtype)


@pytest.mark.parametrize("encoding", ["xor", "overwrite"])
def test_replays_versions_and_snapshots(tmp_path, encoding):
    base, w, vis = make_base(tmp_path)
    deltas = tmp_path / "delta"
    deltas.mkdir()
    states = [w]
    rng = np.random.default_rng(1)
    for v in range(1, 6):
        prev = states[-1]
        nxt = {k: (a + rng.standard_normal(a.shape).astype(np.float32) * 1e-3) for k, a in prev.items()}
        if v == 3:                                    # a version that leaves one tensor unchanged
            nxt["model.language_model.embed"] = prev["model.language_model.embed"]
        publish(deltas, v, prev, nxt, encoding)
        states.append(nxt)
    (deltas / "weight_v000006").mkdir()                # being written: no index yet -> ignored
    assert versions(deltas) == [1, 2, 3, 4, 5]
    logs = []
    ex = DeltaExporter(base=base, deltas=deltas, work=tmp_path / "work", out=tmp_path / "hf",
                       every=2, policy_file=tmp_path / "hf" / "latest.json", run_id="r",
                       version_offset=0, log=logs.append)
    assert ex.poll() == [1, 2, 3, 4, 5]
    assert sorted(p.name for p in (tmp_path / "hf").glob("step_*")) == ["step_2", "step_4"]
    for v in (2, 4):
        for k, a in states[v].items():
            np.testing.assert_array_equal(read(tmp_path / "hf" / f"step_{v}", k), a.ravel())
    # untouched tensors and files are the base's
    np.testing.assert_array_equal(read(tmp_path / "hf" / "step_4", "model.visual.patch"),
                                  vis["model.visual.patch"])
    assert (tmp_path / "hf" / "step_4" / "config.json").exists()
    assert json.loads((tmp_path / "hf" / "step_4" / "HF_EXPORT.json").read_text())["version"] == 4
    assert json.loads((tmp_path / "hf" / "latest.json").read_text())["path"].endswith("step_4")
    assert ex.poll(final=True) == []                   # nothing new; the final snapshot is v5
    assert (tmp_path / "hf" / "step_5").exists()
    assert json.loads((tmp_path / "hf" / "latest.json").read_text())["id"] == "r-step5"
    # base untouched
    np.testing.assert_array_equal(read(base, "model.language_model.embed"), w["model.language_model.embed"])


def test_interrupted_apply_rebuilds_and_checksums(tmp_path):
    base, w, _ = make_base(tmp_path)
    deltas = tmp_path / "delta"
    new = {k: a + 1 for k, a in w.items()}
    publish(deltas, 1, w, new)
    ex = DeltaExporter(base=base, deltas=deltas, work=tmp_path / "work", out=tmp_path / "hf",
                       every=0, log=lambda m: None)
    ex.poll()
    assert ex.version == 1
    (tmp_path / "work" / APPLYING_FILE).write_text("2")          # crashed mid-apply
    ex.poll()
    assert ex.version == 1                                       # rebuilt from base and replayed
    for k, a in new.items():
        np.testing.assert_array_equal(read(tmp_path / "work", k), a.ravel())
    publish(deltas, 2, new, {k: a + 1 for k, a in new.items()}, corrupt=True)
    with pytest.raises(RuntimeError, match="checksum"):
        ex.poll()


def test_missing_version_and_bad_encoding(tmp_path):
    base, w, _ = make_base(tmp_path)
    deltas = tmp_path / "delta"
    publish(deltas, 2, w, {k: a + 1 for k, a in w.items()})
    ex = DeltaExporter(base=base, deltas=deltas, work=tmp_path / "work", out=tmp_path / "hf",
                       log=lambda m: None)
    with pytest.raises(RuntimeError, match="missing delta v1"):
        ex.poll()
    with pytest.raises(ValueError):
        apply_delta(np.zeros(4, np.uint8), b"\0" * 4, "nope")


def test_step_naming_launch_push_final_push_new_stream(tmp_path):
    """platoon pushes once at launch: version v holds step v - 1."""
    base, w, _ = make_base(tmp_path)
    deltas = tmp_path / "delta"
    states = [w]
    for v in range(1, 6):
        empty = v == 5                                            # final push re-sends the same weights
        nxt = {k: a + (0 if empty else 1) for k, a in states[-1].items()}
        if empty:
            (deltas / f"weight_v{v:06d}").mkdir(parents=True)
            (deltas / f"weight_v{v:06d}" / "model.safetensors.index.json").write_text(json.dumps(
                {"metadata": {"version": f"{v:06d}", "base_version": f"{v - 1:06d}", "delta_encoding": "xor",
                              "checksum_format": "adler32"}, "weight_map": {}}))
        else:
            publish(deltas, v, states[-1], nxt)
        states.append(nxt)
    ex = DeltaExporter(base=base, deltas=deltas, work=tmp_path / "work", out=tmp_path / "hf", every=2,
                       policy_file=tmp_path / "hf" / "latest.json", run_id="r", log=lambda m: None)
    ex.poll(final=True)
    # v1 = launch push (non-empty: rounding), v2..v4 = steps 1..3, v5 = final (empty)
    assert sorted(p.name for p in (tmp_path / "hf").glob("step_*")) == ["step_2", "step_3"]
    meta = json.loads((tmp_path / "hf" / "step_2" / "HF_EXPORT.json").read_text())
    assert (meta["version"], meta["step"]) == (3, 2)
    assert json.loads((tmp_path / "hf" / "step_3" / "HF_EXPORT.json").read_text())["version"] == 4
    assert json.loads((tmp_path / "hf" / "latest.json").read_text())["step"] == 3
    # a new stream (gateway restarted: miles recreates the dir) rebuilds the working copy
    import shutil
    shutil.move(str(deltas), str(tmp_path / "old-delta"))
    publish(deltas, 1, w, {k: a + 7 for k, a in w.items()})
    ex.poll()
    assert ex.version == 1
    np.testing.assert_array_equal(read(tmp_path / "work", "model.visual.patch"),
                                  read(base, "model.visual.patch"))
    for k, a in states[3].items():
        np.testing.assert_array_equal(read(tmp_path / "hf" / "step_2", k), a.ravel())


def test_resumed_stream_steps_continue(tmp_path):
    """A resumed run's new stream (checkpoint at batch 20): v1 = platoon's launch push of the
    checkpoint's weights = step 20, then step 21, 22."""
    base, w, _ = make_base(tmp_path)
    deltas = tmp_path / "weights" / "delta"
    states = [w]
    for v in range(1, 4):
        nxt = {k: a + 1 for k, a in states[-1].items()}
        publish(deltas, v, states[-1], nxt)
        states.append(nxt)
    ex = DeltaExporter(base=base, deltas=deltas, work=tmp_path / "work", out=tmp_path / "hf", every=1,
                       log=lambda m: None)
    (deltas.parent / "stream_base.json").write_text(json.dumps({"stream": ex.stream_id(), "batch": 20}))
    ex.poll()
    assert sorted(p.name for p in (tmp_path / "hf").glob("step_*")) == ["step_20", "step_21", "step_22"]
    (deltas.parent / "stream_base.json").write_text(json.dumps({"stream": "other", "batch": 20}))
    assert ex.step_of(3) == 2                          # a stale marker for another stream is ignored
