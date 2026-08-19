import time
import gzip
import pickle
import h5py
import json
import numpy as np
from pathlib import Path

def write_json(obj, path):

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super().default(obj)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

def read_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_pkl(obj, path, protocol=pickle.HIGHEST_PROTOCOL, compress=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not compress:
        with path.open("wb") as f:
            pickle.dump(obj, f, protocol=protocol)
    else:
        if path.suffix != ".gz":
            raise ValueError("Compressed pickle should use .gz extension")
        with gzip.open(path, "wb") as f:
            pickle.dump(obj, f, protocol=protocol)


def read_pkl(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    else:
        if path.suffix != ".pkl":
            raise ValueError("Unsupported pickle format")
        else:
            with path.open("rb") as f:
                return pickle.load(f)

def write_h5(obj, path, mode="w"):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, mode) as f:
        if isinstance(obj, dict):
            for key, value in obj.items():
                f.create_dataset(key, data=value)
        else:
            f.create_dataset("data", data=obj)

def read_h5(path):
    path = Path(path)
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        if keys == ["data"]:
            return f["data"][()]
        else:
            return {k: f[k][()] for k in keys}

def read_probes(dir, probesfile='probes.dat', xyzfile='xyz.dat', out=None):
    dir = Path(dir)

    probesfile = dir / probesfile
    xyzfile = dir / xyzfile
    if out is None: out = dir / 'probes.pkl'

    dat = np.loadtxt(probesfile, dtype=np.float64)
    xyz = np.loadtxt(xyzfile, dtype=np.float64, skiprows=1, usecols=(0, 1, 2))

    n, m, step, _ = dat[0].astype(int)

    block_len = n + 1

    t = dat[::block_len, -1]

    dat_blocks = dat.reshape(dat.shape[0] // block_len, block_len, dat.shape[1])
    dat_blocks = dat_blocks[:, 1:, :]

    res = {'t': t, 'xyz': xyz, 'probes': dat_blocks}

    write_pkl(res, out)