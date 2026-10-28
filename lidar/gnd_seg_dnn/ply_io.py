"""NumPy-only PLY loading shared by training and deployment utilities."""

from pathlib import Path

import numpy as np


_PLY_DTYPES = {
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
}


def read_ply(path):
    """Return vertex properties from a binary-little-endian PLY file."""
    path = Path(path)
    with open(path, "rb") as fh:
        if fh.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a PLY file")

        fmt = None
        n_vertex = 0
        props = []
        in_vertex = False

        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"{path}: unexpected EOF in header")
            line = line.strip()
            if line == b"end_header":
                break
            if line.startswith(b"format"):
                fmt = line.split()[1].decode()
            elif line.startswith(b"element"):
                parts = line.split()
                in_vertex = parts[1] == b"vertex"
                if in_vertex:
                    n_vertex = int(parts[2])
            elif line.startswith(b"property") and in_vertex:
                parts = line.split()
                if parts[1] == b"list":
                    raise ValueError(f"{path}: list properties unsupported")
                ptype, pname = parts[1].decode(), parts[2].decode()
                if ptype not in _PLY_DTYPES:
                    raise ValueError(f"{path}: unsupported type {ptype}")
                props.append((pname, _PLY_DTYPES[ptype]))

        if fmt != "binary_little_endian":
            raise ValueError(f"{path}: expected binary_little_endian, got {fmt}")
        if n_vertex == 0 or not props:
            return {name: np.zeros(0, dtype=dt) for name, dt in props}

        dtype = np.dtype([(name, dt) for name, dt in props])
        raw = fh.read(dtype.itemsize * n_vertex)
        if len(raw) < dtype.itemsize * n_vertex:
            raise ValueError(f"{path}: truncated vertex data")
        arr = np.frombuffer(raw, dtype=dtype, count=n_vertex)
        return {name: np.ascontiguousarray(arr[name]) for name, _ in props}


def load_cloud(path):
    """Load a PLY into ``(xyz float32 [N,3], intensity float32 [N])``."""
    data = read_ply(path)
    for key in ("x", "y", "z"):
        if key not in data:
            raise ValueError(f"{path}: missing '{key}' property")
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    intensity = (data["intensity"].astype(np.float32)
                 if "intensity" in data
                 else np.zeros(len(xyz), dtype=np.float32))
    return xyz, intensity
