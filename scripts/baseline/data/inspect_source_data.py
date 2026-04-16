# inspect_source_data.py
import h5py
import numpy as np
from collections import Counter

def inspect_ascad_fixed():
    """检查 ASCAD fixed 数据集的完整结构"""
    
    print("="*60)
    print("ASCAD FIXED KEY DATASET STRUCTURE INSPECTION")
    print("="*60)
    
    file_path = 'D:/SCA_UDA/data/raw/ASCAD_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5'
    
    with h5py.File(file_path, 'r') as f:
        
        # 1. 顶层结构
        print("\n[1] TOP-LEVEL STRUCTURE:")
        for key in f.keys():
            print(f"    - {key}")
        
        # 2. Profiling_traces 详细结构
        print("\n[2] PROFILING_TRACES STRUCTURE:")
        prof = f['Profiling_traces']
        for key in prof.keys():
            data = prof[key]
            if isinstance(data, h5py.Dataset):
                print(f"    - {key}: shape={data.shape}, dtype={data.dtype}")
            else:
                print(f"    - {key}: Group")
                for subkey in data.keys():
                    subdata = data[subkey]
                    print(f"        - {subkey}: shape={subdata.shape}, dtype={subdata.dtype}")
        
        # 3. Attack_traces 详细结构
        print("\n[3] ATTACK_TRACES STRUCTURE:")
        attack = f['Attack_traces']
        for key in attack.keys():
            data = attack[key]
            if isinstance(data, h5py.Dataset):
                print(f"    - {key}: shape={data.shape}, dtype={data.dtype}")
            else:
                print(f"    - {key}: Group")
                for subkey in data.keys():
                    subdata = data[subkey]
                    print(f"        - {subkey}: shape={subdata.shape}, dtype={subdata.dtype}")
        
        # 4. 标签统计
        print("\n[4] LABELS STATISTICS:")
        
        # Profiling labels
        prof_labels = f['Profiling_traces']['labels'][:]
        print(f"\n    Profiling_traces labels:")
        print(f"        Shape: {prof_labels.shape}")
        print(f"        Data type: {prof_labels.dtype}")
        print(f"        Min value: {prof_labels.min()}")
        print(f"        Max value: {prof_labels.max()}")
        print(f"        Number of unique labels: {len(np.unique(prof_labels))}")
        
        # 统计每个标签的出现次数
        prof_counter = Counter(prof_labels)
        print(f"        Label distribution (first 20):")
        for label, count in sorted(prof_counter.items())[:20]:
            print(f"            Label {label:3d}: {count:5d} times")
        
        # 检查是否所有256个标签都存在
        all_labels = set(range(256))
        prof_set = set(prof_labels)
        missing_prof = all_labels - prof_set
        print(f"        Missing labels in Profiling: {len(missing_prof)}")
        if missing_prof:
            print(f"            Missing: {sorted(missing_prof)[:20]}")
        
        # Attack labels
        attack_labels = f['Attack_traces']['labels'][:]
        print(f"\n    Attack_traces labels:")
        print(f"        Shape: {attack_labels.shape}")
        print(f"        Data type: {attack_labels.dtype}")
        print(f"        Min value: {attack_labels.min()}")
        print(f"        Max value: {attack_labels.max()}")
        print(f"        Number of unique labels: {len(np.unique(attack_labels))}")
        
        # 统计每个标签的出现次数
        attack_counter = Counter(attack_labels)
        print(f"        Label distribution (first 20):")
        for label, count in sorted(attack_counter.items())[:20]:
            print(f"            Label {label:3d}: {count:5d} times")
        
        # 检查是否所有256个标签都存在
        attack_set = set(attack_labels)
        missing_attack = all_labels - attack_set
        print(f"        Missing labels in Attack: {len(missing_attack)}")
        if missing_attack:
            print(f"            Missing: {sorted(missing_attack)[:20]}")
        
        # 5. 密钥统计
        print("\n[5] KEY STATISTICS:")
        
        # Profiling keys
        prof_metadata = f['Profiling_traces']['metadata']
        prof_keys = prof_metadata['key'][:, 0]  # 第一个密钥字节
        print(f"\n    Profiling_traces keys (first byte):")
        print(f"        Shape: {prof_keys.shape}")
        print(f"        Unique keys: {np.unique(prof_keys)}")
        print(f"        Number of unique keys: {len(np.unique(prof_keys))}")
        print(f"        All keys same: {len(np.unique(prof_keys)) == 1}")
        
        # Attack keys
        attack_metadata = f['Attack_traces']['metadata']
        attack_keys = attack_metadata['key'][:, 0]
        print(f"\n    Attack_traces keys (first byte):")
        print(f"        Shape: {attack_keys.shape}")
        print(f"        Unique keys: {np.unique(attack_keys)}")
        print(f"        Number of unique keys: {len(np.unique(attack_keys))}")
        print(f"        All keys same: {len(np.unique(attack_keys)) == 1}")
        
        # 6. 明文统计
        print("\n[6] PLAINTEXT STATISTICS:")
        
        prof_plaintext = prof_metadata['plaintext'][:, 0]
        print(f"\n    Profiling_traces plaintext (first byte):")
        print(f"        Unique plaintexts: {len(np.unique(prof_plaintext))}")
        print(f"        Min: {prof_plaintext.min()}, Max: {prof_plaintext.max()}")
        
        attack_plaintext = attack_metadata['plaintext'][:, 0]
        print(f"\n    Attack_traces plaintext (first byte):")
        print(f"        Unique plaintexts: {len(np.unique(attack_plaintext))}")
        print(f"        Min: {attack_plaintext.min()}, Max: {attack_plaintext.max()}")
        
        # 7. 痕迹统计
        print("\n[7] TRACES STATISTICS:")
        
        prof_traces = f['Profiling_traces']['traces']
        print(f"\n    Profiling_traces:")
        print(f"        Shape: {prof_traces.shape}")
        print(f"        Mean: {prof_traces[:].mean():.4f}")
        print(f"        Std: {prof_traces[:].std():.4f}")
        print(f"        Min: {prof_traces[:].min()}")
        print(f"        Max: {prof_traces[:].max()}")
        
        attack_traces = f['Attack_traces']['traces']
        print(f"\n    Attack_traces:")
        print(f"        Shape: {attack_traces.shape}")
        print(f"        Mean: {attack_traces[:].mean():.4f}")
        print(f"        Std: {attack_traces[:].std():.4f}")
        print(f"        Min: {attack_traces[:].min()}")
        print(f"        Max: {attack_traces[:].max()}")
        
        # 8. 检查 Profiling 和 Attack 的密钥是否相同
        print("\n[8] KEY COMPARISON BETWEEN PROFILING AND ATTACK:")
        prof_first_key = prof_keys[0]
        attack_first_key = attack_keys[0]
        print(f"    Profiling first key: {prof_first_key}")
        print(f"    Attack first key: {attack_first_key}")
        print(f"    Keys are the same: {prof_first_key == attack_first_key}")
        
        # 9. 标签重叠分析
        print("\n[9] LABEL OVERLAP ANALYSIS:")
        prof_label_set = set(prof_labels)
        attack_label_set = set(attack_labels)
        
        common_labels = prof_label_set & attack_label_set
        only_prof = prof_label_set - attack_label_set
        only_attack = attack_label_set - prof_label_set
        
        print(f"    Common labels: {len(common_labels)}")
        print(f"    Only in Profiling: {len(only_prof)}")
        print(f"    Only in Attack: {len(only_attack)}")
        
        if only_prof:
            print(f"        Labels only in Profiling: {sorted(only_prof)[:20]}")
        if only_attack:
            print(f"        Labels only in Attack: {sorted(only_attack)[:20]}")
        
        # 10. 检查标签是否均匀分布
        print("\n[10] LABEL DISTRIBUTION UNIFORMITY:")
        
        prof_counts = np.array([prof_counter.get(i, 0) for i in range(256)])
        attack_counts = np.array([attack_counter.get(i, 0) for i in range(256)])
        
        prof_mean = np.mean(prof_counts)
        prof_std = np.std(prof_counts)
        attack_mean = np.mean(attack_counts)
        attack_std = np.std(attack_counts)
        
        print(f"    Profiling: mean={prof_mean:.1f}, std={prof_std:.1f}, CV={prof_std/prof_mean:.3f}")
        print(f"    Attack: mean={attack_mean:.1f}, std={attack_std:.1f}, CV={attack_std/attack_mean:.3f}")
        
        if prof_std/prof_mean < 0.1:
            print(f"    Profiling labels are highly uniform")
        elif prof_std/prof_mean < 0.3:
            print(f"    Profiling labels are moderately uniform")
        else:
            print(f"    Profiling labels are non-uniform")
        
        if attack_std/attack_mean < 0.1:
            print(f"    Attack labels are highly uniform")
        elif attack_std/attack_mean < 0.3:
            print(f"    Attack labels are moderately uniform")
        else:
            print(f"    Attack labels are non-uniform")

if __name__ == "__main__":
    inspect_ascad_fixed()