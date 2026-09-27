import os
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    AveragePooling1D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# ==================== GPU配置 ====================
# 自动检测GPU并配置（不强制使用CPU）
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ 检测到 {len(gpus)} 个GPU:")
    for gpu in gpus:
        print(f"   - {gpu}")
    # 可选：设置GPU内存增长（避免显存占满）
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("⚠️ 未检测到GPU，将使用CPU训练")

# ==================== 配置参数（统一修改这里）====================
# 数据路径配置
DATA_CONFIG = {
    "source_file": "D:/SCA_UDA/data/raw/ASCAD_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5",           # 源域数据文件路径
    "target_file": "D:/SCA_UDA/data/raw/ASCAD_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5",  # 目标域数据文件路径
    "source_dataset_type": "Profiling_traces",      # 源域数据集类型
    "target_dataset_type": "Attack_traces",      # 目标域数据集类型
}

# 模型保存路径配置
SAVE_CONFIG = {
    "model_dir": "models/cdan/cdan_ascad",               # 模型保存目录
    "ge_curve_path": "results/cdan/cdan_ge_curve.png",    # GE曲线保存路径
    "history_path": "models/cdan/training_history.npy",  # 训练历史保存路径
}

# 训练参数配置
TRAIN_CONFIG = {
    "epochs": 50,
    "batch_size": 64,                             
    "warmup_epochs": 10,
    "max_lambda": 0.0,
    "learning_rate": 1e-5,
    "weight_decay": 5e-4,
    "feature_dim": 256,
    "num_classes": 256,
    "proj_dim": 1024,
    "eval_num_traces": 50000,
}

# ==================== 1. 梯度反转层（GRL）====================
@tf.custom_gradient
def grad_reverse(x, lmbda):
    y = tf.identity(x)
    def grad(dy):
        return -dy * lmbda, None
    return y, grad

class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self, lmbda=1.0, **kwargs):
        super(GradientReversalLayer, self).__init__(**kwargs)
        self.lmbda = lmbda
    def call(self, inputs):
        return grad_reverse(inputs, self.lmbda)
    def get_config(self):
        config = super().get_config()
        config.update({"lmbda": self.lmbda})
        return config

# ==================== 2. 随机多线性映射（条件特征）====================
class RandomMultilinearConditioning(tf.keras.layers.Layer):
    def __init__(self, feature_dim=256, num_classes=256, proj_dim=1024, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.proj_dim = proj_dim
        
        # 随机投影矩阵（固定，不训练）
        self.Rf = tf.random.normal((feature_dim, proj_dim)) / (feature_dim ** 0.25)
        self.Rg = tf.random.normal((num_classes, proj_dim)) / (num_classes ** 0.25)
    
    def call(self, inputs):
        f, g = inputs
        f_proj = tf.matmul(f, self.Rf)
        g_proj = tf.matmul(g, self.Rg)
        cond = f_proj * g_proj
        cond = cond / (self.proj_dim ** 0.5)
        return cond
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "feature_dim": self.feature_dim,
            "num_classes": self.num_classes,
            "proj_dim": self.proj_dim
        })
        return config

# ==================== 3. 熵加权 ====================
def entropy_weight(g):
    epsilon = 1e-8
    entropy = -tf.reduce_sum(g * tf.math.log(g + epsilon), axis=1)
    weight = 1.0 + tf.exp(-entropy)
    weight = weight / tf.reduce_mean(weight)
    return weight

# ==================== 4. 数据加载（使用配置参数）====================
def load_data(file_path, dataset_type='Profiling_traces'):
    """加载ASCAD数据集"""
    with h5py.File(file_path, 'r') as f:
        traces = np.array(f[dataset_type]['traces'])
        if 'labels' in f[dataset_type]:
            labels = np.array(f[dataset_type]['labels'])
        else:
            labels = None
    
    # 归一化
    traces = traces.astype(np.float32)
    mean = traces.mean(axis=1, keepdims=True)
    std = traces.std(axis=1, keepdims=True) + 1e-8
    traces = (traces - mean) / std
    traces = traces.reshape(-1, 700, 1)
    
    return traces, labels

# ==================== 5. CDAN模型组件 ====================
class CDANTrainer:
    def __init__(self, feature_dim=256, num_classes=256, proj_dim=1024, 
                 lr=0.001, weight_decay=5e-4):
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.proj_dim = proj_dim
        self.lr = lr
        self.weight_decay = weight_decay
        
        self._build_components()
        self.optimizer = Adam(learning_rate=lr)
    
    def _build_components(self):
        """构建模型组件"""
        # 特征提取器
        feature_input = Input(shape=(700, 1))

        # Block1
        x = Conv1D(64, 11, activation='relu', padding='same')(feature_input)
        x = AveragePooling1D(2)(x)

        # Block2
        x = Conv1D(128, 11, activation='relu', padding='same')(x)
        x = AveragePooling1D(2)(x)

        # Block3
        x = Conv1D(256, 11, activation='relu', padding='same')(x)
        x = AveragePooling1D(2)(x)

        # Block4
        x = Conv1D(512, 11, activation='relu', padding='same')(x)
        x = AveragePooling1D(2)(x)

        # Block5
        x = Conv1D(512, 11, activation='relu', padding='same')(x)
        x = AveragePooling1D(2)(x)

        x = Flatten()(x)

        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)

        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)

        features = Dense(self.feature_dim, activation='relu')(x)

        self.feature_extractor = Model(
            feature_input,
            features,
            name='feature_extractor'
        )
        
        # 分类器
        classifier_input = Input(shape=(self.feature_dim,))
        x = Dropout(0.5)(classifier_input)
        classifier_out = Dense(self.num_classes, activation='softmax')(x)
        self.classifier = Model(classifier_input, classifier_out, name='classifier')
        
        # 条件特征层
        self.cond_layer = RandomMultilinearConditioning(
            feature_dim=self.feature_dim,
            num_classes=self.num_classes,
            proj_dim=self.proj_dim
        )
        
        # 域判别器
        disc_input = Input(shape=(self.proj_dim,))
        x = Dense(512, activation='relu')(disc_input)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        disc_out = Dense(1, activation='sigmoid')(x)
        self.domain_discriminator = Model(disc_input, disc_out, name='domain_discriminator')
    
    def get_lambda(self, epoch, warmup_epochs=10, max_lambda=0.1, total_epochs=100):
        """渐进式λ调度"""
        if epoch < warmup_epochs:
            return 0.0
        p = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return max_lambda * (2. / (1. + np.exp(-10. * p)) - 1.)
    
    @tf.function
    def train_step(self, x_src, y_src, x_tgt, lambda_):
        # 修复：确保 lambda_ 是 float32 类型
        lambda_ = tf.cast(lambda_, tf.float32)
        
        with tf.GradientTape() as tape:
            # 源域前向
            f_src = self.feature_extractor(x_src, training=True)
            g_src_logits = self.classifier(f_src, training=True)
            g_src = tf.nn.softmax(g_src_logits)
            
            # 目标域前向
            f_tgt = self.feature_extractor(x_tgt, training=True)
            g_tgt_logits = self.classifier(f_tgt, training=True)
            g_tgt = tf.nn.softmax(g_tgt_logits)
            
            # 分类损失
            cls_loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(y_src, g_src_logits)
            )
            
            # 构建条件特征（带GRL）
            f_src_grl = grad_reverse(f_src, lambda_)
            f_tgt_grl = grad_reverse(f_tgt, lambda_)
            
            cond_src = self.cond_layer([f_src_grl, g_src])
            cond_tgt = self.cond_layer([f_tgt_grl, g_tgt])
            
            # 熵加权
            w_src = entropy_weight(g_src)
            w_tgt = entropy_weight(g_tgt)
            
            # 域判别损失
            d_src = self.domain_discriminator(cond_src, training=True)
            d_tgt = self.domain_discriminator(cond_tgt, training=True)
            
            domain_loss_src = tf.reduce_mean(w_src * tf.keras.losses.binary_crossentropy(
                tf.ones_like(d_src), d_src
            ))
            domain_loss_tgt = tf.reduce_mean(w_tgt * tf.keras.losses.binary_crossentropy(
                tf.zeros_like(d_tgt), d_tgt
            ))
            domain_loss = (domain_loss_src + domain_loss_tgt) / 2
            
            # 总损失
            total_loss = cls_loss + lambda_ * domain_loss
        
        # 梯度更新
        trainable_vars = (self.feature_extractor.trainable_variables +
                        self.classifier.trainable_variables +
                        self.domain_discriminator.trainable_variables)
        grads = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        
        # 计算准确率
        src_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(g_src_logits, axis=1), y_src), tf.float32))
        
        return cls_loss, domain_loss, src_acc

    def train(self, X_src, y_src, X_tgt, epochs=50, batch_size=128, 
              warmup_epochs=10, max_lambda=0.1):
        """完整训练循环"""
        n_src = len(X_src)
        n_tgt = len(X_tgt)
        
        history = {'cls_loss': [], 'domain_loss': [], 'src_acc': [], 'lambda': []}
        
        print("\n" + "="*60)
        print("开始训练CDAN模型")
        print("="*60 + "\n")
        
        for epoch in range(epochs):
            lambda_ = self.get_lambda(epoch, warmup_epochs, max_lambda, epochs)
            
            src_idx = np.random.permutation(n_src)
            tgt_idx = np.random.permutation(n_tgt)
            
            epoch_cls_loss = []
            epoch_domain_loss = []
            epoch_acc = []
            
            for step in range(0, min(n_src, n_tgt) - batch_size + 1, batch_size):
                src_batch_idx = src_idx[step:step+batch_size]
                tgt_batch_idx = tgt_idx[step:step+batch_size]
                
                x_src_batch = X_src[src_batch_idx]
                y_src_batch = y_src[src_batch_idx]
                x_tgt_batch = X_tgt[tgt_batch_idx]
                
                cls_loss, domain_loss, src_acc = self.train_step(
                    x_src_batch, y_src_batch, x_tgt_batch, lambda_
                )
                
                epoch_cls_loss.append(cls_loss.numpy())
                epoch_domain_loss.append(domain_loss.numpy())
                epoch_acc.append(src_acc.numpy())
            
            history['cls_loss'].append(np.mean(epoch_cls_loss))
            history['domain_loss'].append(np.mean(epoch_domain_loss))
            history['src_acc'].append(np.mean(epoch_acc))
            history['lambda'].append(lambda_)
            
            print(f"Epoch {epoch+1:03d}: cls_loss={history['cls_loss'][-1]:.4f}, "
                  f"domain_loss={history['domain_loss'][-1]:.4f}, "
                  f"src_acc={history['src_acc'][-1]:.4f}, lambda={lambda_:.4f}")
        
        return history
    
    def predict(self, X):
        """攻击阶段：只使用分类器"""
        features = self.feature_extractor(X, training=False)
        logits = self.classifier(features, training=False)
        return tf.nn.softmax(logits).numpy()
    
    def save(self, save_config):
        """保存模型（使用配置参数）"""
        os.makedirs(save_config["model_dir"], exist_ok=True)
        
        self.feature_extractor.save(os.path.join(save_config["model_dir"], 'feature_extractor.h5'))
        self.classifier.save(os.path.join(save_config["model_dir"], 'classifier.h5'))
        self.domain_discriminator.save(os.path.join(save_config["model_dir"], 'domain_discriminator.h5'))
        
        print(f"\n✅ 模型已保存到: {save_config['model_dir']}")

# ==================== 6. GE评估函数 ====================
def compute_ge(predictions, true_labels, max_traces=1000):
    """计算猜测熵"""
    ranks = []
    for i in range(min(len(predictions), max_traces)):
        prob = predictions[i]
        true_label = true_labels[i]
        sorted_indices = np.argsort(prob)[::-1]
        rank = np.where(sorted_indices == true_label)[0][0]
        ranks.append(rank)
    ge_curve = np.cumsum(ranks) / (np.arange(len(ranks)) + 1)
    return ge_curve

def evaluate_ge(trainer, X_tgt, y_tgt, num_traces=1000):
    """评估模型在目标域上的GE"""
    predictions = trainer.predict(X_tgt[:num_traces])
    ge_curve = compute_ge(predictions, y_tgt[:num_traces], num_traces)
    return ge_curve

def plot_ge_curve(ge_curve, save_path):
    """绘制并保存GE曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(ge_curve, label='CDAN', linewidth=2)
    plt.xlabel('Number of Traces')
    plt.ylabel('Guessing Entropy')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('CDAN - Guessing Entropy on Target Domain')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ GE曲线已保存到: {save_path}")

def save_history(history, save_path):
    """保存训练历史"""
    np.save(save_path, history)
    print(f"✅ 训练历史已保存到: {save_path}")

# ==================== 7. 主程序（使用配置参数）====================
if __name__ == "__main__":
    print("="*60)
    print("CDAN模型训练 - ASCAD数据集")
    print("="*60)
    
    # 显示GPU信息
    print("\n" + "="*30 + " 硬件信息 " + "="*30)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            print(f"✅ GPU: {gpu}")
    else:
        print("⚠️ 使用CPU训练")
    print("="*68)
    
    # 显示当前配置
    print("\n📁 数据路径配置:")
    print(f"   源域文件: {DATA_CONFIG['source_file']}")
    print(f"   目标域文件: {DATA_CONFIG['target_file']}")
    print(f"   源域类型: {DATA_CONFIG['source_dataset_type']}")
    print(f"   目标域类型: {DATA_CONFIG['target_dataset_type']}")
    
    print("\n💾 保存路径配置:")
    print(f"   模型目录: {SAVE_CONFIG['model_dir']}")
    print(f"   GE曲线: {SAVE_CONFIG['ge_curve_path']}")
    print(f"   训练历史: {SAVE_CONFIG['history_path']}")
    
    print("\n⚙️ 训练参数配置:")
    print(f"   轮数: {TRAIN_CONFIG['epochs']}")
    print(f"   批次大小: {TRAIN_CONFIG['batch_size']}")
    print(f"   预热轮数: {TRAIN_CONFIG['warmup_epochs']}")
    print(f"   最大λ: {TRAIN_CONFIG['max_lambda']}")
    
    # 加载数据
    print("\n正在加载源域数据...")
    X_src, y_src = load_data(
        DATA_CONFIG["source_file"], 
        DATA_CONFIG["source_dataset_type"]
    )
    print(f"源域: {X_src.shape}, 标签: {y_src.shape if y_src is not None else 'None'}")
    
    print("\n正在加载目标域数据...")
    X_tgt, y_tgt = load_data(
        DATA_CONFIG["target_file"], 
        DATA_CONFIG["target_dataset_type"]
    )
    print(f"目标域: {X_tgt.shape}, 标签: {y_tgt.shape if y_tgt is not None else 'None'}")
    
    # 创建训练器
    trainer = CDANTrainer(
        feature_dim=TRAIN_CONFIG["feature_dim"],
        num_classes=TRAIN_CONFIG["num_classes"],
        proj_dim=TRAIN_CONFIG["proj_dim"],
        lr=TRAIN_CONFIG["learning_rate"],
        weight_decay=TRAIN_CONFIG["weight_decay"]
    )
    
    # 训练CDAN
    history = trainer.train(
        X_src, y_src, X_tgt,
        epochs=TRAIN_CONFIG["epochs"],
        batch_size=TRAIN_CONFIG["batch_size"],
        warmup_epochs=TRAIN_CONFIG["warmup_epochs"],
        max_lambda=TRAIN_CONFIG["max_lambda"]
    )
    
    # 保存模型
    trainer.save(SAVE_CONFIG)
    
    # 保存训练历史
    save_history(history, SAVE_CONFIG["history_path"])
    
    # 评估GE
    print("\n" + "="*60)
    print("评估猜测熵...")
    print("="*60)
    
    ge_curve = evaluate_ge(
        trainer, X_tgt, y_tgt, 
        num_traces=TRAIN_CONFIG["eval_num_traces"]
    )
    
    # 绘制GE曲线
    plot_ge_curve(ge_curve, SAVE_CONFIG["ge_curve_path"])
    
    print(f"\n📊 最终GE ({TRAIN_CONFIG['eval_num_traces']} traces): {ge_curve[-1]:.2f}")
    print("\n✅ CDAN训练完成!")