#!/usr/bin/env python3
"""Unified smoke baseline runner for existing methods.

This script is intentionally small-scale by default. It verifies that the
Source Only and DANN baseline paths, output files, and target evaluation path
work before launching expensive full experiments.

CDAN-SCA is the proposed method and should be run in a separate proposed-method
or ablation experiment, not from this baseline runner.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random

import numpy as np

try:
    import h5py
    import tensorflow as tf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: {name}. Install with:\n"
        "  uv pip install tensorflow h5py scikit-learn matplotlib numpy\n"
        "or:\n"
        "  pip install tensorflow h5py scikit-learn matplotlib numpy".format(
            name=exc.name
        )
    ) from exc


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = ROOT / "data/raw/ASCAD_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5"
DEFAULT_TARGET = ROOT / "data/raw/ASCAD_fixed_key/ASCAD_data/ASCAD_databases/ASCAD_desync50.h5"
FALLBACK_TARGET = ROOT / "data/raw/ASCAD_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5"
OUTPUT_DIR = ROOT / "outputs/results/cdan-sca-baselines"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["source_only", "dann"],
        required=True,
    )
    parser.add_argument("--source-file", default=str(DEFAULT_SOURCE))
    parser.add_argument("--target-file", default=str(DEFAULT_TARGET))
    parser.add_argument("--source-group", default="Profiling_traces")
    parser.add_argument("--target-group", default="Attack_traces")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit-source", type=int, default=512)
    parser.add_argument("--limit-target", type=int, default=512)
    parser.add_argument("--limit-eval", type=int, default=256)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--lambda-adv", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prediction-path", default="")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def resolve_target(path: str) -> str:
    if os.path.exists(path):
        return path
    fallback = str(FALLBACK_TARGET)
    if os.path.exists(fallback):
        print(f"Target file not found: {path}")
        print(f"Using fallback target: {fallback}")
        return fallback
    return path


def load_group(path: str, group: str, limit: int) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        traces = np.array(handle[f"{group}/traces"][:limit], dtype=np.float32)
        labels = np.array(handle[f"{group}/labels"][:limit], dtype=np.int64)
    mean = traces.mean(axis=1, keepdims=True)
    std = traces.std(axis=1, keepdims=True) + 1e-8
    traces = (traces - mean) / std
    traces = traces.reshape((traces.shape[0], traces.shape[1], 1))
    return traces, labels


class GradientReversal(tf.keras.layers.Layer):
    def __init__(self, lambda_adv: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_adv = lambda_adv

    def call(self, inputs):
        lambda_adv = tf.cast(self.lambda_adv, tf.float32)

        @tf.custom_gradient
        def reverse(x):
            def grad(dy):
                return -lambda_adv * dy

            return tf.identity(x), grad

        return reverse(inputs)


def build_feature_extractor(input_len: int, feature_dim: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(input_len, 1))
    x = inputs
    for filters in [32, 64, 128]:
        x = tf.keras.layers.Conv1D(filters, 11, padding="same", activation="relu")(x)
        x = tf.keras.layers.AveragePooling1D(2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(feature_dim, activation="relu")(x)
    return tf.keras.Model(inputs, outputs, name="feature_extractor")


def build_classifier(feature_dim: int, num_classes: int = 256) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(feature_dim,))
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(inputs)
    return tf.keras.Model(inputs, outputs, name="classifier")


def build_domain_discriminator(input_dim: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="domain_discriminator")


def make_batches(x: np.ndarray, y: np.ndarray | None, batch_size: int):
    indices = np.random.permutation(len(x))
    for start in range(0, len(indices) - batch_size + 1, batch_size):
        batch_idx = indices[start : start + batch_size]
        if y is None:
            yield x[batch_idx]
        else:
            yield x[batch_idx], y[batch_idx]


def evaluate_accuracy(feature_extractor, classifier, x_eval, y_eval, batch_size):
    preds = []
    for x_batch in make_batches(x_eval, None, batch_size):
        probs = classifier(feature_extractor(x_batch, training=False), training=False)
        preds.append(probs.numpy())
    probs = np.concatenate(preds, axis=0)
    y_used = y_eval[: len(probs)]
    acc = float(np.mean(np.argmax(probs, axis=1) == y_used))
    ranks = []
    for prob, label in zip(probs, y_used):
        sorted_idx = np.argsort(prob)[::-1]
        ranks.append(int(np.where(sorted_idx == label)[0][0]))
    pseudo_ge = float(np.mean(ranks))
    return acc, pseudo_ge, probs


def train(args: argparse.Namespace) -> dict:
    source_file = args.source_file
    target_file = resolve_target(args.target_file)
    x_src, y_src = load_group(source_file, args.source_group, args.limit_source)
    x_tgt, y_tgt = load_group(target_file, args.target_group, args.limit_target)
    x_eval, y_eval = x_tgt[: args.limit_eval], y_tgt[: args.limit_eval]

    feature_extractor = build_feature_extractor(x_src.shape[1], args.feature_dim)
    classifier = build_classifier(args.feature_dim)
    use_domain = args.method == "dann"
    domain_input_dim = args.feature_dim
    domain_discriminator = build_domain_discriminator(domain_input_dim)
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)
    cls_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    domain_loss_fn = tf.keras.losses.BinaryCrossentropy(reduction="none")

    history = []
    final_probs = None
    for epoch in range(args.epochs):
        cls_losses = []
        domain_losses = []
        src_batches = list(make_batches(x_src, y_src, args.batch_size))
        tgt_batches = list(make_batches(x_tgt, None, args.batch_size))
        for (x_s, y_s), x_t in zip(src_batches, tgt_batches):
            with tf.GradientTape() as tape:
                f_s = feature_extractor(x_s, training=True)
                g_s = classifier(f_s, training=True)
                cls_loss = cls_loss_fn(y_s, g_s)

                domain_loss = tf.constant(0.0, dtype=tf.float32)
                if use_domain:
                    f_t = feature_extractor(x_t, training=True)
                    g_t = classifier(f_t, training=True)
                    d_s_in = f_s
                    d_t_in = f_t
                    grl = GradientReversal(args.lambda_adv)
                    d_s = domain_discriminator(grl(d_s_in), training=True)
                    d_t = domain_discriminator(grl(d_t_in), training=True)
                    y_ds = tf.ones_like(d_s)
                    y_dt = tf.zeros_like(d_t)
                    d_loss_s = domain_loss_fn(y_ds, d_s)
                    d_loss_t = domain_loss_fn(y_dt, d_t)
                    domain_loss = tf.reduce_mean(d_loss_s) + tf.reduce_mean(d_loss_t)
                    domain_loss = 0.5 * domain_loss

                total_loss = cls_loss + args.lambda_adv * domain_loss

            variables = feature_extractor.trainable_variables + classifier.trainable_variables
            if use_domain:
                variables += domain_discriminator.trainable_variables
            grads = tape.gradient(total_loss, variables)
            optimizer.apply_gradients(zip(grads, variables))
            cls_losses.append(float(cls_loss.numpy()))
            domain_losses.append(float(domain_loss.numpy()))

        target_acc, pseudo_ge, eval_probs = evaluate_accuracy(
            feature_extractor, classifier, x_eval, y_eval, args.batch_size
        )
        final_probs = eval_probs
        row = {
            "epoch": epoch + 1,
            "cls_loss": float(np.mean(cls_losses)),
            "domain_loss": float(np.mean(domain_losses)),
            "target_accuracy": target_acc,
            "label_rank_mean_proxy": pseudo_ge,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

    return {
        "method": args.method,
        "source_file": source_file,
        "target_file": target_file,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "limit_source": args.limit_source,
        "limit_target": args.limit_target,
        "limit_eval": args.limit_eval,
        "history": history,
        "prediction_path": args.prediction_path,
        "prediction_shape": list(final_probs.shape) if final_probs is not None else None,
        "_predictions": final_probs,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = train(args)
    if args.prediction_path and result.get("_predictions") is not None:
        Path(args.prediction_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.prediction_path, result.pop("_predictions"))
    out_path = OUTPUT_DIR / f"{args.method}_smoke_results.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
