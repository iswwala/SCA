import h5py
import numpy as np

def explore_ascad_fixed():
    """彻底探索 ASCAD fixed key 数据集的结构"""
    
    print("="*60)
    print("ASCAD FIXED KEY DATASET STRUCTURE EXPLORATION")
    print("="*60)
    
    with h5py.File('data/raw/ASCAD_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5', 'r') as f:
        
        # 1. 顶层结构
        print("\n1. TOP-LEVEL KEYS:")
        for key in f.keys():
            print(f"   - {key}")
        
        # 2. Profiling_traces 结构
        print("\n2. PROFILING_TRACES STRUCTURE:")
        for key in f['Profiling_traces'].keys():
            item = f['Profiling_traces'][key]
            if isinstance(item, h5py.Dataset):
                print(f"   - {key}: shape={item.shape}, dtype={item.dtype}")
            else:
                print(f"   - {key}: Group")
                for subkey in item.keys():
                    subitem = item[subkey]
                    print(f"       - {subkey}: shape={subitem.shape}, dtype={subitem.dtype}")
        
        # 3. Attack_traces 结构（如果有）
        if 'Attack_traces' in f:
            print("\n3. ATTACK_TRACES STRUCTURE:")
            for key in f['Attack_traces'].keys():
                item = f['Attack_traces'][key]
                if isinstance(item, h5py.Dataset):
                    print(f"   - {key}: shape={item.shape}, dtype={item.dtype}")
        
        # 4. 查看 metadata 的具体内容
        print("\n4. METADATA DETAILS:")
        metadata = f['Profiling_traces']['metadata']
        print(f"   Type: {type(metadata)}")
        print(f"   Shape: {metadata.shape}")
        print(f"   Dtype: {metadata.dtype}")
        
        # 查看 metadata 的字段名（如果是结构化数组）
        if metadata.dtype.names:
            print(f"   Fields: {metadata.dtype.names}")
            for field in metadata.dtype.names:
                print(f"     - {field}: {metadata[field][:5]}")
        else:
            # 如果不是结构化数组，查看前几行内容
            print(f"   First 3 rows of metadata:")
            for i in range(3):
                print(f"     Row {i}: {metadata[i]}")
        
        # 5. 查看 labels 的详细信息
        print("\n5. LABELS DETAILS:")
        labels = f['Profiling_traces']['labels']
        print(f"   Shape: {labels.shape}")
        print(f"   Dtype: {labels.dtype}")
        print(f"   Min value: {labels[:].min()}")
        print(f"   Max value: {labels[:].max()}")
        print(f"   Unique values count: {len(np.unique(labels[:]))}")
        print(f"   First 20 labels: {labels[:20]}")
        
        # 检查标签分布
        unique, counts = np.unique(labels[:10000], return_counts=True)
        print(f"   Label distribution (first 20 unique):")
        for u, c in zip(unique[:20], counts[:20]):
            print(f"     Label {u}: {c} samples")
        
        # 6. 查看 traces 的详细信息
        print("\n6. TRACES DETAILS:")
        traces = f['Profiling_traces']['traces']
        print(f"   Shape: {traces.shape}")
        print(f"   Dtype: {traces.dtype}")
        print(f"   Min value: {traces[:].min():.2f}")
        print(f"   Max value: {traces[:].max():.2f}")
        print(f"   Mean value: {traces[:].mean():.2f}")
        print(f"   Std value: {traces[:].std():.2f}")
        
        # 7. 检查是否有明文和密钥信息
        print("\n7. LOOKING FOR PLAINTEXT AND KEY:")
        
        # 尝试从 metadata 中提取
        if metadata.dtype.names:
            if 'plaintext' in metadata.dtype.names:
                plaintext = metadata['plaintext']
                print(f"   Plaintext found! shape={plaintext.shape}")
                print(f"   First plaintext: {plaintext[0]}")
            
            if 'key' in metadata.dtype.names:
                key = metadata['key']
                print(f"   Key found! shape={key.shape}")
                print(f"   Key value: {key[0] if key.shape[0]==1 else key[:5]}")
        else:
            # 尝试将 metadata 解释为 plaintext
            print(f"   Attempting to interpret metadata as plaintext...")
            print(f"   First metadata row as bytes: {metadata[0].tobytes()[:16]}")
        
        # 8. 检查 attack_traces 的 labels（如果有）
        if 'Attack_traces' in f and 'labels' in f['Attack_traces']:
            print("\n8. ATTACK TRACES LABELS:")
            attack_labels = f['Attack_traces']['labels']
            print(f"   Shape: {attack_labels.shape}")
            print(f"   First 10: {attack_labels[:10]}")
        
        return f

# 运行探索
file = explore_ascad_fixed()