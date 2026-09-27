import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def train_supervised(
    feature_extractor,
    classifier,
    X_train,
    Y_train,
    config
):

    # =========================
    # 配置读取
    # =========================
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    validation_split = config.get("validation_split", 0.1)
    early_stopping = config.get("early_stopping", False)
    patience = config.get("patience", 10)
    model_save_dir = config["model_save_dir"]

    os.makedirs(model_save_dir, exist_ok=True)

    # =========================
    # 数据shape处理
    # =========================
    X_train = X_train.reshape(
        (X_train.shape[0], X_train.shape[1], 1)
    ).astype("float32")

    # =========================
    # 标签转换为 one-hot（与官方一致）
    # =========================
    Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=256)
    print(f"Y_train shape (one-hot): {Y_train.shape}")

    # =========================
    # 划分训练集和验证集
    # =========================
    if validation_split > 0:
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_train, Y_train,
            test_size=validation_split,
            random_state=42
        )
        print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}")
    else:
        X_val, Y_val = None, None

    # =========================
    # tf.data
    # =========================
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X_train, Y_train)
    ).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    if X_val is not None:
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (X_val, Y_val)
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # =========================
    # optimizer & loss（与官方一致：RMSprop + categorical_crossentropy）
    # =========================
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00001)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    if X_val is not None:
        val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

    # =========================
    # early stopping
    # =========================
    best_val_loss = 1e9
    patience_counter = 0

    history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    for epoch in range(epochs):

        print("\n" + "="*50)
        print(f"Epoch {epoch + 1}/{epochs}")
        print("="*50)

        # ===== 训练阶段 =====
        epoch_loss = 0.0
        batch_count = 0
        train_acc_metric.reset_state()

        for step, (x_batch, y_batch) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                features = feature_extractor(x_batch, training=True)
                predictions = classifier(features, training=True)
                cls_loss = loss_fn(y_batch, predictions)

            train_vars = (
                feature_extractor.trainable_variables
                + classifier.trainable_variables
            )
            grads = tape.gradient(cls_loss, train_vars)
            optimizer.apply_gradients(zip(grads, train_vars))

            train_acc_metric.update_state(y_batch, predictions)
            epoch_loss += cls_loss.numpy()
            batch_count += 1

            if step % 50 == 0:
                print(f"Step {step:4d} | Loss: {cls_loss.numpy():.4f}")

        epoch_loss /= batch_count
        epoch_acc = train_acc_metric.result().numpy()
        history["loss"].append(epoch_loss)
        history["accuracy"].append(epoch_acc)

        print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")

        # ===== 验证阶段 =====
        if X_val is not None:
            val_acc_metric.reset_state()
            val_losses = []

            for x_val_batch, y_val_batch in val_dataset:
                features = feature_extractor(x_val_batch, training=False)
                predictions = classifier(features, training=False)
                v_loss = loss_fn(y_val_batch, predictions)
                val_losses.append(v_loss.numpy())
                val_acc_metric.update_state(y_val_batch, predictions)

            val_loss = np.mean(val_losses)
            val_acc = val_acc_metric.result().numpy()

            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)

            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # ===== Early Stopping =====
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # 保存最佳模型
                    feature_extractor.save(
                        os.path.join(model_save_dir, "best_feature_extractor.h5")
                    )
                    classifier.save(
                        os.path.join(model_save_dir, "best_classifier.h5")
                    )
                    print("\n✓ Best model saved")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"\n⚠️ Early stopping at epoch {epoch + 1}")
                        break
        else:
            # 无验证集时，按训练损失保存
            if epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                feature_extractor.save(
                    os.path.join(model_save_dir, "best_feature_extractor.h5")
                )
                classifier.save(
                    os.path.join(model_save_dir, "best_classifier.h5")
                )

    # =========================
    # 保存最终模型和历史
    # =========================
    feature_extractor.save(
        os.path.join(model_save_dir, "supervised_feature_extractor.h5")
    )
    classifier.save(
        os.path.join(model_save_dir, "supervised_classifier.h5")
    )
    np.save(
        os.path.join(model_save_dir, "supervised_history.npy"),
        history
    )

    print("\n✅ Training Finished.")