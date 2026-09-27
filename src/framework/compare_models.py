"""
模型对比脚本：官方 CNN vs 拆分模型（特征提取器 + 分类器）
对比指标：训练损失、验证准确率、测试准确率、参数量
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目路径到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.load_data import load_ascad
from models.feature_extractor import build_feature_extractor
from models.classifier import build_classifier

# 强制使用CPU避免GPU内存不足（可选）
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


# ==================== 官方 CNN 模型 ====================
def cnn_best(classes=256, input_dim=700):
    """ASCAD 官方 CNN 模型"""
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv1D, AveragePooling1D, Flatten, Dense
    from tensorflow.keras.optimizers import RMSprop
    
    input_shape = (input_dim, 1)
    img_input = Input(shape=input_shape)
    
    # Block 1
    x = Conv1D(64, 11, activation='relu', padding='same', name='block1_conv1')(img_input)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
    
    # Block 2
    x = Conv1D(128, 11, activation='relu', padding='same', name='block2_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
    
    # Block 3
    x = Conv1D(256, 11, activation='relu', padding='same', name='block3_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block3_pool')(x)
    
    # Block 4
    x = Conv1D(512, 11, activation='relu', padding='same', name='block4_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block4_pool')(x)
    
    # Block 5
    x = Conv1D(512, 11, activation='relu', padding='same', name='block5_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block5_pool')(x)
    
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)
    
    model = Model(img_input, x, name='cnn_best')
    
    # 使用官方相同的优化器配置
    optimizer = RMSprop(learning_rate=0.00001)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model


# ==================== 拆分模型（特征提取器 + 分类器）====================
def build_split_model(input_shape=(700, 1), feature_dim=4096, num_classes=256):
    """
    将特征提取器和分类器串联成一个端到端模型
    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # 特征提取器
    feature_extractor = build_feature_extractor(input_shape=input_shape)
    features = feature_extractor(inputs)
    
    # 分类器
    classifier = build_classifier(feature_dim=feature_dim, num_classes=num_classes)
    outputs = classifier(features)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='split_model')
    
    # 使用相同的优化器配置
    from tensorflow.keras.optimizers import RMSprop
    optimizer = RMSprop(learning_rate=0.00001)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model


# ==================== 训练和记录函数 ====================
def train_and_record(model, X_train, Y_train, X_val, Y_val, 
                     model_name, epochs=50, batch_size=64):
    """
    训练模型并记录历史
    """
    print(f"\n{'='*60}")
    print(f"训练模型: {model_name}")
    print(f"{'='*60}")
    print(f"训练样本数: {len(X_train)}")
    print(f"验证样本数: {len(X_val)}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}")
    print(f"{'='*60}\n")
    
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, Y_val),
        verbose=1,
        shuffle=True
    )
    
    return history


def plot_comparison(histories, save_dir):
    """
    绘制对比图
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'official': 'blue', 'split': 'red'}
    labels = {'official': 'Official CNN', 'split': 'Split Model'}
    
    # 损失曲线对比
    ax = axes[0]
    for name, history in histories.items():
        ax.plot(history.history['loss'], 
                color=colors[name], linestyle='-', 
                label=f'{labels[name]} (train)')
        ax.plot(history.history['val_loss'], 
                color=colors[name], linestyle='--', 
                label=f'{labels[name]} (val)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 准确率曲线对比
    ax = axes[1]
    for name, history in histories.items():
        ax.plot(history.history['accuracy'], 
                color=colors[name], linestyle='-', 
                label=f'{labels[name]} (train)')
        ax.plot(history.history['val_accuracy'], 
                color=colors[name], linestyle='--', 
                label=f'{labels[name]} (val)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f'model_comparison_{timestamp}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n对比图已保存: {save_path}")
    return save_path


def save_results(histories, model_params, test_results, save_dir):
    """
    保存对比结果到文件
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(save_dir, f'comparison_results_{timestamp}.txt')
    
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("模型对比实验报告\n")
        f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # 参数量对比
        f.write("1. 模型参数量对比\n")
        f.write("-"*40 + "\n")
        for name, params in model_params.items():
            f.write(f"   {name}: {params:,} 参数\n")
        f.write("\n")
        
        # 最终准确率对比
        f.write("2. 最终准确率对比\n")
        f.write("-"*40 + "\n")
        for name, results in test_results.items():
            f.write(f"   {name}:\n")
            f.write(f"      训练准确率: {results['train_acc']:.4f}\n")
            f.write(f"      验证准确率: {results['val_acc']:.4f}\n")
            f.write(f"      最终损失: {results['final_loss']:.4f}\n")
        f.write("\n")
        
        # 训练历史摘要
        f.write("3. 训练历史摘要（每10轮）\n")
        f.write("-"*40 + "\n")
        for name, history in histories.items():
            f.write(f"\n   {name}:\n")
            f.write(f"   {'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Train Acc':<12} {'Val Acc':<12}\n")
            for i in range(0, len(history.history['loss']), 10):
                f.write(f"   {i+1:<8} {history.history['loss'][i]:<12.4f} "
                       f"{history.history['val_loss'][i]:<12.4f} "
                       f"{history.history['accuracy'][i]:<12.4f} "
                       f"{history.history['val_accuracy'][i]:<12.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("实验结束\n")
    
    print(f"\n结果已保存: {result_file}")
    return result_file


def save_history_npy(history, model_name, save_dir):
    """
    保存训练历史为npy文件
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f'{model_name}_history_{timestamp}.npy')
    np.save(save_path, history.history)
    print(f"训练历史已保存: {save_path}")
    return save_path


# ==================== 主程序 ====================
def main():
    # 配置参数
    EPOCHS = 50
    BATCH_SIZE = 64
    VALIDATION_SPLIT = 0.1
    NUM_CLASSES = 256
    INPUT_DIM = 700
    
    # 结果保存目录
    RESULTS_DIR = "results/model_comparison"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("="*60)
    print("模型对比实验")
    print("="*60)
    
    # ==================== 1. 加载数据 ====================
    print("\n[1/5] 加载 ASCAD 数据...")
    
    # 使用固定密钥数据
    data_path = "D:/SCA_UDA/data/raw/ASCAD_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5"
    (X_profiling, Y_profiling), (X_attack, Y_attack) = load_ascad(data_path)
    
    print(f"   源域数据形状: {X_profiling.shape}")
    print(f"   源域标签形状: {Y_profiling.shape}")
    print(f"   目标域数据形状: {X_attack.shape}")
    print(f"   目标域标签形状: {Y_attack.shape}")
    
    # 重塑数据为 (n_samples, 700, 1)
    X_profiling = X_profiling.reshape(-1, INPUT_DIM, 1).astype(np.float32)
    X_attack = X_attack.reshape(-1, INPUT_DIM, 1).astype(np.float32)
    
    # 划分训练集和验证集（使用源域数据）
    from sklearn.model_selection import train_test_split
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_profiling, Y_profiling, 
        test_size=VALIDATION_SPLIT, 
        random_state=42,
        stratify=Y_profiling
    )
    
    print(f"\n   训练集: {X_train.shape}, 标签: {Y_train.shape}")
    print(f"   验证集: {X_val.shape}, 标签: {Y_val.shape}")
    
    # ==================== 2. 构建模型 ====================
    print("\n[2/5] 构建模型...")
    
    # 官方 CNN
    official_model = cnn_best(classes=NUM_CLASSES, input_dim=INPUT_DIM)
    official_params = official_model.count_params()
    print(f"   官方 CNN 参数量: {official_params:,}")
    
    # 拆分模型
    split_model = build_split_model(
        input_shape=(INPUT_DIM, 1),
        feature_dim=4096,
        num_classes=NUM_CLASSES
    )
    split_params = split_model.count_params()
    print(f"   拆分模型参数量: {split_params:,}")
    
    model_params = {
        'official': official_params,
        'split': split_params
    }
    
    # ==================== 3. 训练模型 ====================
    print("\n[3/5] 训练模型...")
    
    histories = {}
    
    # 训练官方 CNN
    official_history = train_and_record(
        official_model, X_train, Y_train, X_val, Y_val,
        model_name="official_cnn",
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    histories['official'] = official_history
    
    # 训练拆分模型
    split_history = train_and_record(
        split_model, X_train, Y_train, X_val, Y_val,
        model_name="split_model",
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    histories['split'] = split_history
    
    # ==================== 4. 评估模型 ====================
    print("\n[4/5] 评估模型...")
    
    test_results = {}
    
    for name, model in [('official', official_model), ('split', split_model)]:
        # 训练集评估
        train_loss, train_acc = model.evaluate(X_train, Y_train, verbose=0)
        # 验证集评估
        val_loss, val_acc = model.evaluate(X_val, Y_val, verbose=0)
        # 测试集评估（目标域）
        test_loss, test_acc = model.evaluate(X_attack, Y_attack, verbose=0)
        
        test_results[name] = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'final_loss': val_loss
        }
        
        print(f"\n   {name} 模型:")
        print(f"      训练准确率: {train_acc:.4f}")
        print(f"      验证准确率: {val_acc:.4f}")
        print(f"      测试准确率: {test_acc:.4f}")
    
    # ==================== 5. 保存结果 ====================
    print("\n[5/5] 保存结果...")
    
    # 绘制对比图
    plot_comparison(histories, RESULTS_DIR)
    
    # 保存训练历史
    for name, history in histories.items():
        save_history_npy(history, name, RESULTS_DIR)
    
    # 保存结果文本
    save_results(histories, model_params, test_results, RESULTS_DIR)
    
    # ==================== 结果总结 ====================
    print("\n" + "="*60)
    print("实验完成！")
    print("="*60)
    print(f"\n结果保存目录: {RESULTS_DIR}")
    
    # 输出最终对比
    print("\n最终对比结果:")
    print(f"  官方 CNN  - 验证准确率: {test_results['official']['val_acc']:.4f}")
    print(f"  拆分模型 - 验证准确率: {test_results['split']['val_acc']:.4f}")
    print(f"  准确率差距: {abs(test_results['official']['val_acc'] - test_results['split']['val_acc']):.4f}")
    
    if abs(test_results['official']['val_acc'] - test_results['split']['val_acc']) < 0.05:
        print("\n✅ 拆分模型与官方 CNN 性能接近，结构匹配成功！")
    else:
        print("\n⚠️ 拆分模型与官方 CNN 存在较大差异，需要进一步调整。")


if __name__ == "__main__":
    main()