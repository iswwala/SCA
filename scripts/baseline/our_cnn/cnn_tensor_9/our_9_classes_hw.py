import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("="*60)
print("Training 9-class HW classifier")
print("="*60)

# 1. 加载数据
with h5py.File('data/raw/ASCAD_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5', 'r') as f:
    train_traces = f['Profiling_traces']['traces'][:]  # (50000, 700)
    train_labels = f['Profiling_traces']['labels'][:]  # (50000,)
    
    test_traces = f['Attack_traces']['traces'][:]  # (10000, 700)
    test_labels = f['Attack_traces']['labels'][:]  # (10000,)

print(f"Train: {train_traces.shape}")
print(f"Test: {test_traces.shape}")

# 2. 预处理：直接使用原始标签，不做任何转换！
# 因为 ASCAD 的 labels 已经是 Sbox(plaintext XOR key) 的值
# 我们要做的是 HW 分类，所以需要计算 HW

def hw(x):
    return bin(x).count('1')

train_labels_hw = np.array([hw(l) for l in train_labels])
test_labels_hw = np.array([hw(l) for l in test_labels])

print(f"HW labels range: {train_labels_hw.min()} - {train_labels_hw.max()}")
print(f"HW distribution: {np.bincount(train_labels_hw)}")

# 3. 数据标准化：简单的全局标准化
train_traces = train_traces.astype(np.float32)
test_traces = test_traces.astype(np.float32)

# 全局标准化（不是逐条）
mean = train_traces.mean()
std = train_traces.std()
train_traces = (train_traces - mean) / (std + 1e-8)
test_traces = (test_traces - mean) / (std + 1e-8)

# 增加通道维度
train_traces = train_traces.reshape(-1, 700, 1)
test_traces = test_traces.reshape(-1, 700, 1)

# 4. 构建一个标准的 CNN
model = keras.Sequential([
    layers.Input(shape=(700, 1)),
    
    layers.Conv1D(32, 3, activation='relu', padding='same'),
    layers.MaxPooling1D(2),
    
    layers.Conv1D(64, 3, activation='relu', padding='same'),
    layers.MaxPooling1D(2),
    
    layers.Conv1D(128, 3, activation='relu', padding='same'),
    layers.GlobalAveragePooling1D(),
    
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(9, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 5. 训练
history = model.fit(
    train_traces, train_labels_hw,
    epochs=30,
    batch_size=256,
    validation_data=(test_traces, test_labels_hw),
    verbose=1
)

# 6. 评估
test_loss, test_acc = model.evaluate(test_traces, test_labels_hw)
print(f"\nTest accuracy: {test_acc:.4f}")

# 7. 检查预测分布
predictions = model.predict(test_traces[:1000])
predicted = np.argmax(predictions, axis=1)
unique, counts = np.unique(predicted, return_counts=True)
print("\nPredicted label distribution:")
for u, c in zip(unique, counts):
    print(f"  Label {u}: {c} times ({c/10:.1f}%)")