import os
import sys
import importlib.util
import numpy as np
import h5py
import matplotlib as mpl

# ======================================
# 无显示环境处理
# ======================================
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
else:
    mpl.use('TkAgg')

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model


# ======================================
# AES Sbox
# ======================================
AES_Sbox = np.array([
    0x63,0x7C,0x77,0x7B,0xF2,0x6B,0x6F,0xC5,
    0x30,0x01,0x67,0x2B,0xFE,0xD7,0xAB,0x76,
    0xCA,0x82,0xC9,0x7D,0xFA,0x59,0x47,0xF0,
    0xAD,0xD4,0xA2,0xAF,0x9C,0xA4,0x72,0xC0,
    0xB7,0xFD,0x93,0x26,0x36,0x3F,0xF7,0xCC,
    0x34,0xA5,0xE5,0xF1,0x71,0xD8,0x31,0x15,
    0x04,0xC7,0x23,0xC3,0x18,0x96,0x05,0x9A,
    0x07,0x12,0x80,0xE2,0xEB,0x27,0xB2,0x75,
    0x09,0x83,0x2C,0x1A,0x1B,0x6E,0x5A,0xA0,
    0x52,0x3B,0xD6,0xB3,0x29,0xE3,0x2F,0x84,
    0x53,0xD1,0x00,0xED,0x20,0xFC,0xB1,0x5B,
    0x6A,0xCB,0xBE,0x39,0x4A,0x4C,0x58,0xCF,
    0xD0,0xEF,0xAA,0xFB,0x43,0x4D,0x33,0x85,
    0x45,0xF9,0x02,0x7F,0x50,0x3C,0x9F,0xA8,
    0x51,0xA3,0x40,0x8F,0x92,0x9D,0x38,0xF5,
    0xBC,0xB6,0xDA,0x21,0x10,0xFF,0xF3,0xD2,
    0xCD,0x0C,0x13,0xEC,0x5F,0x97,0x44,0x17,
    0xC4,0xA7,0x7E,0x3D,0x64,0x5D,0x19,0x73,
    0x60,0x81,0x4F,0xDC,0x22,0x2A,0x90,0x88,
    0x46,0xEE,0xB8,0x14,0xDE,0x5E,0x0B,0xDB,
    0xE0,0x32,0x3A,0x0A,0x49,0x06,0x24,0x5C,
    0xC2,0xD3,0xAC,0x62,0x91,0x95,0xE4,0x79,
    0xE7,0xC8,0x37,0x6D,0x8D,0xD5,0x4E,0xA9,
    0x6C,0x56,0xF4,0xEA,0x65,0x7A,0xAE,0x08,
    0xBA,0x78,0x25,0x2E,0x1C,0xA6,0xB4,0xC6,
    0xE8,0xDD,0x74,0x1F,0x4B,0xBD,0x8B,0x8A,
    0x70,0x3E,0xB5,0x66,0x48,0x03,0xF6,0x0E,
    0x61,0x35,0x57,0xB9,0x86,0xC1,0x1D,0x9E,
    0xE1,0xF8,0x98,0x11,0x69,0xD9,0x8E,0x94,
    0x9B,0x1E,0x87,0xE9,0xCE,0x55,0x28,0xDF,
    0x8C,0xA1,0x89,0x0D,0xBF,0xE6,0x42,0x68,
    0x41,0x99,0x2D,0x0F,0xB0,0x54,0xBB,0x16
])


# ======================================
# 配置读取
# ======================================
def load_config(config_path):

    spec = importlib.util.spec_from_file_location(
        "config",
        config_path
    )

    config_module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(config_module)

    return config_module.config


# ======================================
# 加载ASCAD
# ======================================
def load_ascad(database_file):

    in_file = h5py.File(database_file, "r")

    X_attack = np.array(
        in_file['Attack_traces/traces'],
        dtype=np.int8
    )

    metadata_attack = in_file[
        'Attack_traces/metadata'
    ]

    return X_attack, metadata_attack


# ======================================
# Rank计算
# ======================================
def rank(
    predictions,
    metadata,
    real_key,
    min_trace_idx,
    max_trace_idx,
    last_key_bytes_proba,
    target_byte
):

    if len(last_key_bytes_proba) == 0:
        key_bytes_proba = np.zeros(256)
    else:
        key_bytes_proba = last_key_bytes_proba

    for p in range(max_trace_idx - min_trace_idx):

        plaintext = metadata[
            min_trace_idx + p
        ]['plaintext'][target_byte]

        for i in range(256):

            proba = predictions[p][
                AES_Sbox[plaintext ^ i]
            ]

            if proba != 0:
                key_bytes_proba[i] += np.log(proba)
            else:
                key_bytes_proba[i] += np.log(1e-40)

    sorted_proba = np.array(
        list(
            map(
                lambda a: key_bytes_proba[a],
                key_bytes_proba.argsort()[::-1]
            )
        )
    )

    real_key_rank = np.where(
        sorted_proba == key_bytes_proba[real_key]
    )[0][0]

    return real_key_rank, key_bytes_proba


# ======================================
# Full ranks
# ======================================
def full_ranks(
    predictions,
    dataset,
    metadata,
    min_trace_idx,
    max_trace_idx,
    rank_step,
    target_byte
):

    real_key = metadata[0]['key'][target_byte]

    index = np.arange(
        min_trace_idx + rank_step,
        max_trace_idx,
        rank_step
    )

    f_ranks = np.zeros(
        (len(index), 2),
        dtype=np.uint32
    )

    key_bytes_proba = []

    for t, i in zip(index, range(len(index))):

        real_key_rank, key_bytes_proba = rank(
            predictions[t-rank_step:t],
            metadata,
            real_key,
            t-rank_step,
            t,
            key_bytes_proba,
            target_byte
        )

        f_ranks[i] = [
            t - min_trace_idx,
            real_key_rank
        ]

    return f_ranks


# ======================================
# 检测模型
# ======================================
def check_model(config):

    print("Loading attack traces...")

    X_attack, metadata_attack = load_ascad(
        config["ascad_database"]
    )

    # CNN输入 reshape
    input_data = X_attack[
        :config["num_traces"],
        :
    ]

    input_data = input_data.reshape(
        (
            input_data.shape[0],
            input_data.shape[1],
            1
        )
    )

    print("Loading feature extractor...")

    feature_extractor = load_model(
        config["feature_extractor_path"]
    )

    print("Loading classifier...")

    classifier = load_model(
        config["classifier_path"]
    )

    print("Rebuilding full model...")

    outputs = classifier(
        feature_extractor.output
    )

    model = Model(
        inputs=feature_extractor.input,
        outputs=outputs
    )

    print("Predicting...")

    predictions = model.predict(input_data)

    print("Computing GE...")

    ranks = full_ranks(
        predictions=predictions,
        dataset=X_attack,
        metadata=metadata_attack,
        min_trace_idx=0,
        max_trace_idx=config["num_traces"],
        rank_step=10,
        target_byte=config["target_byte"]
    )

    # 创建目录
    os.makedirs(
        os.path.dirname(config["ge_data_path"]),
        exist_ok=True
    )

    os.makedirs(
        os.path.dirname(config["save_file"]),
        exist_ok=True
    )

    # 保存GE数据
    np.save(
        config["ge_data_path"],
        ranks
    )

    print(
        f"GE data saved to: "
        f"{config['ge_data_path']}"
    )

    # 绘图
    x = [
        ranks[i][0]
        for i in range(ranks.shape[0])
    ]

    y = [
        ranks[i][1]
        for i in range(ranks.shape[0])
    ]
    model_name = config.get("model_name", None)
    if model_name is None:
        # 从特征提取器路径中提取文件名（不含扩展名）作为模型名
        model_name = os.path.splitext(os.path.basename(config["feature_extractor_path"]))[0]

    dataset_name = config.get("dataset_name", None)
    if dataset_name is None:
        # 从数据库路径中提取文件名（不含扩展名）作为数据集名
        dataset_name = os.path.splitext(os.path.basename(config["ascad_database"]))[0]

    target_byte = config["target_byte"]

    plt.figure(figsize=(12, 6))
    plt.title(f"Guessing Entropy\nModel: {model_name} | Dataset: {dataset_name} | Target Byte: {target_byte}")
    plt.xlabel("Number of Traces")
    plt.ylabel("GE")
    plt.grid(True)
    plt.plot(x, y)
    plt.savefig(config["save_file"])
    print(f"GE curve saved to: {config['save_file']}")
   


# ======================================
# Main
# ======================================
if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage: python test.py "
            "configs/test/supervised.py"
        )

        sys.exit(0)

    config_path = sys.argv[1]

    config = load_config(config_path)

    check_model(config)