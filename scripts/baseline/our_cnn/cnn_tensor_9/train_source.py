import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

print("="*50)
print("Training 256-class classifier on ASCAD fixed key")
print("="*50)

# 1. 加载数据（直接使用官方 labels）
with h5py.File('data/raw/ASCAD_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5', 'r') as f:
    traces = f['Profiling_traces']['traces'][:50000]
    labels = f['Profiling_traces']['labels'][:50000]
    
    print(f"Traces shape: {traces.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label range: {labels.min()} - {labels.max()}")
    print(f"Unique labels: {len(np.unique(labels))}")  # 应该是 256
    
    # 数据标准化（int8 -> float32）
    traces = traces.astype(np.float32)
    traces = (traces - traces.mean(axis=1, keepdims=True)) / (traces.std(axis=1, keepdims=True) + 1e-8)
    traces = traces.reshape(-1, 700, 1)
    
    # 划分训练/验证集
    split = 40000
    train_x, train_y = traces[:split], labels[:split]
    val_x, val_y = traces[split:], labels[split:]

print(f"Train: {train_x.shape}, Val: {val_x.shape}")

# 2. 构建模型
def create_model():
    inputs = keras.Input(shape=(700, 1))
    
    # 卷积层
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    # 全连接层
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(256, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

model = create_model()
model.summary()

# 3. 编译
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. 回调
callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
    keras.callbacks.ModelCheckpoint('models/baseline/our_cnn_256class.h5', save_best_only=True)
]

# 5. 训练
history = model.fit(
    train_x, train_y,
    epochs=50,
    batch_size=256,
    validation_data=(val_x, val_y),
    callbacks=callbacks,
    verbose=1
)

# 6. 评估
val_loss, val_acc = model.evaluate(val_x, val_y)
print(f"\nValidation accuracy: {val_acc:.4f}")

# 7. 计算 GE
def compute_ge(model, traces, labels, max_traces=10000, step=100):
    predictions = model.predict(traces[:max_traces])
    ge_list = []
    for n in range(step, max_traces + 1, step):
        scores = np.sum(predictions[:n], axis=0)
        rank = np.argsort(scores)[::-1]
        true_label = labels[n-1]
        pos = np.where(rank == true_label)[0]
        ge_list.append(pos[0] if len(pos) > 0 else 255)
    return ge_list

ge = compute_ge(model, val_x, val_y, max_traces=5000)
print(f"GE after 5000 traces: {ge[-1]}")

# 8. 保存结果
model.save('models/baseline/our_cnn_256class_final.h5')
np.save('models/baseline/256class_history.npy', history.history)
np.save('models/baseline/256class_ge.npy', ge)

print("\n✅ Training complete!")