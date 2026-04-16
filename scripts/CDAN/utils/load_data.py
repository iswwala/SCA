import h5py
import numpy as np

def load_ascad(path):
    f = h5py.File(path, "r")

    X_profiling = np.array(
        f["Profiling_traces/traces"],
        dtype=np.float32
    )
    Y_profiling = np.array(
        f["Profiling_traces/labels"]
    )

    X_attack = np.array(
        f["Attack_traces/traces"],
        dtype=np.float32
    )
    Y_attack = np.array(
        f["Attack_traces/labels"]
    )

    return (X_profiling, Y_profiling), (X_attack, Y_attack)