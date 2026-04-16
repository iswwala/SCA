import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ASCADDataset
from cnn_baseline import ASCADCNN
import numpy as np
import random
from scipy.stats import norm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据路径
h5_path = "data/raw/ASCAD_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5"

# 加载数据
print("Loading datasets...")
train_set = ASCADDataset(h5_path, set_type='profiling')
test_set = ASCADDataset(h5_path, set_type='attack')

# 数据加载器
batch_size = 200
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"Training samples: {len(train_set)}")
print(f"Test samples: {len(test_set)}")

# 创建模型
model = ASCADCNN(num_classes=256, input_length=700).to(device)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# 优化器（官方使用RMSprop，学习率0.00001）
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.00001)
criterion = nn.CrossEntropyLoss()

# 训练函数
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        if batch_idx % 50 == 0:
            print(f'  Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# 计算Guessing Entropy的函数
def compute_ge(rank_array):
    """计算平均Guessing Entropy"""
    return np.mean(rank_array)

def compute_rank(scores, correct_label):
    """计算正确标签的排名（从0开始）"""
    # scores: (num_classes,)
    # 降序排序，得分越高排名越好
    sorted_indices = np.argsort(scores)[::-1]
    rank = np.where(sorted_indices == correct_label)[0][0]
    return rank

def evaluate_ge(model, loader, device, num_classes=256):
    """
    评估模型的Guessing Entropy
    返回：平均GE，以及每个样本的排名
    """
    model.eval()
    all_ranks = []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # 转换为numpy并应用softmax得到概率
            probs = torch.softmax(output, dim=1).cpu().numpy()
            targets = target.cpu().numpy()
            
            # 计算每个样本的排名
            for i in range(len(probs)):
                rank = compute_rank(probs[i], targets[i])
                all_ranks.append(rank)
    
    all_ranks = np.array(all_ranks)
    ge = np.mean(all_ranks)
    
    return ge, all_ranks

# 计算成功率（在特定排名阈值内）
def compute_success_rate(ranks, threshold=1):
    """计算排名<=threshold的成功率"""
    return np.mean(ranks < threshold) * 100

# 训练循环
num_epochs = 75
best_ge = float('inf')
best_epoch = 0

print("\nStarting training...")
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 50)
    
    # 训练
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    
    # 在测试集上评估GE
    ge, ranks = evaluate_ge(model, test_loader, device)
    sr_at_1 = compute_success_rate(ranks, threshold=1)
    sr_at_5 = compute_success_rate(ranks, threshold=5)
    sr_at_10 = compute_success_rate(ranks, threshold=10)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Test Guessing Entropy: {ge:.2f}")
    print(f"Success Rate (rank < 1): {sr_at_1:.2f}%")
    print(f"Success Rate (rank < 5): {sr_at_5:.2f}%")
    print(f"Success Rate (rank < 10): {sr_at_10:.2f}%")
    
    # 保存最佳模型（基于GE）
    if ge < best_ge:
        best_ge = ge
        best_epoch = epoch + 1
        torch.save(model.state_dict(), 'models/baseline/our_cnn_256_pytorch.pth')
        print(f"New best GE: {best_ge:.2f} at epoch {best_epoch}")

print(f"\nTraining completed!")
print(f"Best Guessing Entropy: {best_ge:.2f} at epoch {best_epoch}")

# 最终详细评估
print("\n" + "="*50)
print("FINAL EVALUATION")
print("="*50)

# 加载最佳模型
model.load_state_dict(torch.load('best_model_ge.pth'))
final_ge, final_ranks = evaluate_ge(model, test_loader, device)

print(f"Final Guessing Entropy: {final_ge:.2f}")

# 计算不同阈值下的成功率
thresholds = [1, 2, 3, 4, 5, 10, 20, 50, 100]
for thr in thresholds:
    sr = compute_success_rate(final_ranks, threshold=thr)
    print(f"Success Rate (rank < {thr}): {sr:.2f}%")

# 打印排名分布
print("\nRank Distribution:")
rank_counts = np.bincount(final_ranks, minlength=256)
for i in range(min(20, len(rank_counts))):
    if rank_counts[i] > 0:
        print(f"  Rank {i}: {rank_counts[i]} samples ({100.*rank_counts[i]/len(final_ranks):.2f}%)")

# 可选：绘制GE曲线
try:
    import matplotlib.pyplot as plt
    
    # 排序排名
    sorted_ranks = np.sort(final_ranks)
    cumulative_success = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks) * 100
    
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_ranks, cumulative_success, 'b-', linewidth=2)
    plt.xlabel('Rank')
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate vs Rank (Guessing Entropy Analysis)')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    
    # 标注GE
    plt.axvline(x=final_ge, color='r', linestyle='--', label=f'GE = {final_ge:.2f}')
    plt.legend()
    
    plt.savefig('results/baseline/our_cnn_256.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nGE curve saved as 'results/baseline/our_cnn_256.png'")
except:
    print("\nMatplotlib not available, skipping plot generation")