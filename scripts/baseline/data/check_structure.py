# check_structure.py
import h5py

print("Checking ASCAD variable structure...")
with h5py.File('D:/SCA_UDA/data/raw/ASCAD_variable/ascad-variable.h5', 'r') as f:
    print("Top-level keys:", list(f.keys()))
    
    print("\nProfiling_traces keys:", list(f['Profiling_traces'].keys()))
    for key in f['Profiling_traces'].keys():
        print(f"  - {key}: shape={f['Profiling_traces'][key].shape}")
    
    print("\nAttack_traces keys:", list(f['Attack_traces'].keys()))
    for key in f['Attack_traces'].keys():
        print(f"  - {key}: shape={f['Attack_traces'][key].shape}")