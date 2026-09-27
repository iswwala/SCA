# framework/utils/load_data.py

import os
import sys
import h5py
import numpy as np


def check_file_exists(file_path):

    file_path = os.path.normpath(file_path)

    if not os.path.exists(file_path):

        print(
            f"Error: file does not exist:\n{file_path}"
        )

        sys.exit(-1)


def load_ascad(
    ascad_database_file,
    load_metadata=False
):

    check_file_exists(ascad_database_file)

    try:

        in_file = h5py.File(
            ascad_database_file,
            "r"
        )

    except:

        print(
            f"Error opening HDF5:\n"
            f"{ascad_database_file}"
        )

        sys.exit(-1)

    # ======================================
    # Profiling traces
    # ======================================
    X_profiling = np.array(
        in_file['Profiling_traces/traces'],
        dtype=np.float32
    )

    Y_profiling = np.array(
        in_file['Profiling_traces/labels']
    )

    # ======================================
    # Attack traces
    # ======================================
    X_attack = np.array(
        in_file['Attack_traces/traces'],
        dtype=np.float32
    )

    Y_attack = np.array(
        in_file['Attack_traces/labels']
    )

    # ======================================
    # metadata
    # ======================================
    if load_metadata:

        Metadata_profiling = in_file[
            'Profiling_traces/metadata'
        ]

        Metadata_attack = in_file[
            'Attack_traces/metadata'
        ]

        return (
            (X_profiling, Y_profiling),
            (X_attack, Y_attack),
            (Metadata_profiling, Metadata_attack)
        )

    else:

        return (
            (X_profiling, Y_profiling),
            (X_attack, Y_attack)
        )