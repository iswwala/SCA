import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ASCADDataset
from cnn_baseline import ASCADCNN
import numpy as np
import argparse
import os

def compute_rank(scores, correct_label):
    """
    计算正确标签的排名（从0开始）
    scores: numpy array of shape (num_classes,)
    correct_label: 正确的标签
    """
    sorted_indices = np.argsort(scores)[::-1]
    rank = np.where(sorted_indices == correct_label)[0][0]
    return rank

def evaluate_ge(model, loader, device):
    """
    评估Guessing Entropy
    返回：平均GE，所有排名，以及每个样本的预测概率
    """
    model.eval()
    all_ranks = []
    all_probs = []
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # 应用softmax得到概率
            probs = torch.softmax(output, dim=1).cpu().numpy()
            targets = target.cpu().numpy()
            preds = output.argmax(dim=1).cpu().numpy()
            
            # 计算每个样本的排名
            for i in range(len(probs)):
                rank = compute_rank(probs[i], targets[i])
                all_ranks.append(rank)
            
            all_probs.extend(probs)
            all_labels.extend(targets)
            all_predictions.extend(preds)
    
    all_ranks = np.array(all_ranks)
    ge = np.mean(all_ranks)
    
    return ge, all_ranks, np.array(all_probs), np.array(all_labels), np.array(all_predictions)

def compute_success_rate(ranks, threshold=1):
    """计算排名<=threshold的成功率"""
    return np.mean(ranks < threshold) * 100

def compute_partial_ge(ranks, num_traces_list):
    """
    计算不同数量痕迹下的GE（用于攻击曲线）
    """
    results = {}
    for num_traces in num_traces_list:
        if num_traces <= len(ranks):
            partial_ranks = ranks[:num_traces]
            ge = np.mean(partial_ranks)
            results[num_traces] = ge
    return results

def main():
    parser = argparse.ArgumentParser(description='Test ASCAD CNN model')
    parser.add_argument('--model_path', type=str, default='best_model_ge.pth',
                        help='Path to the trained model')
    parser.add_argument('--h5_path', type=str, 
                        default='data/raw/ASCAD_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5',
                        help='Path to ASCAD HDF5 file')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='Batch size for testing')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found!")
        return
    
    # 加载测试数据
    print("Loading test data...")
    test_set = ASCADDataset(args.h5_path, set_type='attack')
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_set)}")
    
    # 创建模型并加载权重
    print(f"\nLoading model from {args.model_path}...")
    model = ASCADCNN(num_classes=256, input_length=700).to(device)
    
    # 加载权重
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    print("Model loaded successfully!")
    
    # 评估模型
    print("\n" + "="*60)
    print("Evaluating Guessing Entropy on Test Set")
    print("="*60)
    
    ge, ranks, probs, labels, predictions = evaluate_ge(model, test_loader, device)
    
    # 基本统计
    print(f"\n[1] Basic Statistics:")
    print(f"    Total test samples: {len(test_set)}")
    print(f"    Average Guessing Entropy: {ge:.2f}")
    print(f"    Min Rank: {ranks.min()}")
    print(f"    Max Rank: {ranks.max()}")
    print(f"    Std Rank: {ranks.std():.2f}")
    
    # 不同阈值下的成功率
    print(f"\n[2] Success Rate at Different Rank Thresholds:")
    thresholds = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200]
    for thr in thresholds:
        sr = compute_success_rate(ranks, threshold=thr)
        print(f"    Rank < {thr:3d}: {sr:6.2f}%")
    
    # 排名分布
    print(f"\n[3] Rank Distribution (Top 20 ranks):")
    rank_counts = np.bincount(ranks, minlength=256)
    cumulative = 0
    for i in range(min(20, len(rank_counts))):
        cumulative += rank_counts[i]
        percentage = 100. * cumulative / len(ranks)
        print(f"    Rank {i:3d}: {int(rank_counts[i]):5d} samples ({100.*rank_counts[i]/len(ranks):5.2f}%), Cumulative: {percentage:5.2f}%")
    
    # 不同数量痕迹下的GE（攻击曲线）
    print(f"\n[4] Guessing Entropy vs Number of Traces:")
    trace_counts = [100, 500, 1000, 2000, 5000, 10000]
    ge_by_traces = compute_partial_ge(ranks, trace_counts)
    for num_traces, ge_val in ge_by_traces.items():
        if num_traces <= len(ranks):
            print(f"    {num_traces:5d} traces: GE = {ge_val:.2f}")
    
    # 额外分析：正确预测和错误预测
    print(f"\n[5] Prediction Analysis:")
    accuracy = 100. * np.mean(predictions == labels)
    print(f"    Top-1 Accuracy: {accuracy:.2f}%")
    print(f"    Samples with rank=0 (correct prediction): {np.sum(ranks == 0)} ({100.*np.sum(ranks==0)/len(ranks):.2f}%)")
    print(f"    Samples with rank=1: {np.sum(ranks == 1)} ({100.*np.sum(ranks==1)/len(ranks):.2f}%)")
    print(f"    Samples with rank=2: {np.sum(ranks == 2)} ({100.*np.sum(ranks==2)/len(ranks):.2f}%)")
    
    # 概率分析
    print(f"\n[6] Probability Analysis:")
    # 正确预测的置信度
    correct_confidences = probs[np.arange(len(probs)), labels][predictions == labels]
    print(f"    Average confidence (correct predictions): {np.mean(correct_confidences):.4f}")
    
    # 错误预测的置信度
    wrong_confidences = probs[np.arange(len(probs)), labels][predictions != labels]
    if len(wrong_confidences) > 0:
        print(f"    Average confidence (wrong predictions): {np.mean(wrong_confidences):.4f}")
    
    # 排名与置信度的关系
    print(f"    Correlation between rank and confidence: {np.corrcoef(ranks, -probs[np.arange(len(probs)), labels])[0,1]:.4f}")
    
    # 保存结果
    print(f"\n[7] Saving Results:")
    results = {
        'ge': float(ge),
        'ranks': ranks.tolist(),
        'thresholds': thresholds,
        'success_rates': [float(compute_success_rate(ranks, thr)) for thr in thresholds],
        'accuracy': float(accuracy),
        'rank_distribution': rank_counts[:50].tolist()
    }
    
    import json
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("    Results saved to 'test_results.json'")
    
    # 可选：绘制GE曲线
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 5))
        
        # 子图1：成功率曲线
        plt.subplot(1, 2, 1)
        thresholds_plot = list(range(1, 51))
        success_rates = [compute_success_rate(ranks, thr) for thr in thresholds_plot]
        plt.plot(thresholds_plot, success_rates, 'b-', linewidth=2)
        plt.xlabel('Rank Threshold')
        plt.ylabel('Success Rate (%)')
        plt.title('Success Rate vs Rank Threshold')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=50, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=80, color='g', linestyle='--', alpha=0.5)
        
        # 子图2：排名直方图
        plt.subplot(1, 2, 2)
        plt.hist(ranks, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Rank')
        plt.ylabel('Frequency')
        plt.title(f'Rank Distribution (GE = {ge:.2f})')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/baseline/our_cnn_256_pytorch/test_results.png', dpi=150, bbox_inches='tight')
        print("    GE analysis plot saved to 'results/baseline/our_cnn_256_pytorch/test_results.png'")
        
        # 子图3：累积成功率曲线
        plt.figure(figsize=(10, 6))
        sorted_ranks = np.sort(ranks)
        cumulative_success = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks) * 100
        plt.plot(sorted_ranks, cumulative_success, 'b-', linewidth=2)
        plt.xlabel('Rank')
        plt.ylabel('Cumulative Success Rate (%)')
        plt.title('Cumulative Success Rate vs Rank')
        plt.grid(True, alpha=0.3)
        plt.axvline(x=ge, color='r', linestyle='--', label=f'GE = {ge:.2f}')
        plt.legend()
        plt.savefig('results/baseline/our_cnn_256_pytorch/cumulative_ge_curve.png', dpi=150, bbox_inches='tight')
        print("    Cumulative GE curve saved to 'results/baseline/our_cnn_256_pytorch/cumulative_ge_curve.png'")
        
    except ImportError:
        print("    Matplotlib not available, skipping plots")
    
    print("\n" + "="*60)
    print("Evaluation completed!")
    print("="*60)
    
    return ge, ranks

if __name__ == "__main__":
    main()