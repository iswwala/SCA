import sys
import importlib.util
import numpy as np

from utils.load_data import load_ascad
from models.feature_extractor import build_feature_extractor
from models.classifier import build_classifier
from trainers.trainer_cadn_supervised import train_supervised


def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.CONFIG


def normalize_traces(traces):
    """Z-score 归一化（可选，与官方保持一致可不归一化）"""
    traces = traces.astype(np.float32)
    mean = traces.mean(axis=1, keepdims=True)
    std = traces.std(axis=1, keepdims=True) + 1e-8
    return (traces - mean) / std


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python train_supervised.py configs/train/cdan_supervised.py")
        sys.exit(0)

    config_path = sys.argv[1]
    CONFIG = load_config(config_path)

    print("\nLoaded Config:")
    print(CONFIG)

    # =========================
    # 加载数据
    # =========================
    print("\nLoading ASCAD dataset...")
    (X_profiling, Y_profiling), _ = load_ascad(CONFIG["source_file"])

    print(f"X shape: {X_profiling.shape}")
    print(f"Y shape: {Y_profiling.shape}")
    print(f"X dtype: {X_profiling.dtype}")
    print(f"X range: [{X_profiling.min()}, {X_profiling.max()}]")

    # 可选：归一化（与官方保持一致可不做）
    # X_profiling = normalize_traces(X_profiling)

    # =========================
    # 创建模型
    # =========================
    print("\nBuilding models...")

    feature_extractor = build_feature_extractor()
    classifier = build_classifier(
        feature_dim=4096,
        num_classes=CONFIG["num_classes"]
    )

    # 验证维度
    test_x = np.random.randn(2, 700, 1).astype(np.float32)
    test_f = feature_extractor(test_x)
    test_p = classifier(test_f)
    print(f"Test feature shape: {test_f.shape}")
    print(f"Test prediction shape: {test_p.shape}")

    print("Models Built.")

    # =========================
    # 开始训练
    # =========================
    train_supervised(
        feature_extractor=feature_extractor,
        classifier=classifier,
        X_train=X_profiling,
        Y_train=Y_profiling,
        config=CONFIG
    )