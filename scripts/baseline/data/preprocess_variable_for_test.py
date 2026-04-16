# preprocess_variable_for_test.py
import h5py
import numpy as np

print("Loading original variable data...")

with h5py.File('D:/SCA_UDA/data/raw/ASCAD_variable/ascad-variable.h5', 'r') as f:
    # 获取 metadata 的 dtype（关键：保持相同的数据类型）
    metadata_dtype = f['Profiling_traces']['metadata'].dtype
    print(f"Metadata dtype: {metadata_dtype}")
    
    # 加载攻击数据（只取前 700 个采样点）
    attack_traces = f['Attack_traces']['traces'][:10000, :700]
    attack_labels = f['Attack_traces']['labels'][:10000]
    attack_metadata = f['Attack_traces']['metadata'][:10000]
    
    # 加载 profiling 数据（取少量作为占位）
    profiling_traces = f['Profiling_traces']['traces'][:1, :700]
    profiling_labels = f['Profiling_traces']['labels'][:1]
    profiling_metadata = f['Profiling_traces']['metadata'][:1]

print(f"Attack traces shape: {attack_traces.shape}")
print(f"Attack metadata shape: {attack_metadata.shape}")

# 保存为新的 HDF5 文件
print("Saving to D:/SCA_UDA/data/processed/ascad-variable-700.h5")

with h5py.File('D:/SCA_UDA/data/processed/ascad-variable-700.h5', 'w') as f:
    # Profiling_traces 组
    f.create_group('Profiling_traces')
    f['Profiling_traces/traces'] = profiling_traces
    f['Profiling_traces/labels'] = profiling_labels
    f['Profiling_traces/metadata'] = profiling_metadata
    
    # Attack_traces 组
    f.create_group('Attack_traces')
    f['Attack_traces/traces'] = attack_traces
    f['Attack_traces/labels'] = attack_labels
    f['Attack_traces/metadata'] = attack_metadata

print("Done!")

# 验证保存的文件
print("\nVerifying saved file...")
with h5py.File('D:/SCA_UDA/data/processed/ascad-variable-700.h5', 'r') as f:
    print("Profiling_traces keys:", list(f['Profiling_traces'].keys()))
    print("Attack_traces keys:", list(f['Attack_traces'].keys()))
    print(f"Attack traces shape: {f['Attack_traces']['traces'].shape}")
    print(f"Attack metadata shape: {f['Attack_traces']['metadata'].shape}")
    print("✅ File is valid!")