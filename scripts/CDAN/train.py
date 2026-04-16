import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from config import *
from utils.load_data import load_ascad
from utils.preprocess import normalize_source_target
from models.cdan_model import build_cdan_model

import os

os.makedirs("outputs/checkpoints", exist_ok=True)
os.makedirs("outputs/predictions", exist_ok=True)
os.makedirs("outputs/normalization", exist_ok=True)
os.makedirs("outputs/logs", exist_ok=True)


# source = ASCAD fixed
(Xs, Ys), _ = load_ascad(FIXED_PATH)

# target = ASCAD variable
(Xt, _), _ = load_ascad(VARIABLE_PATH)

"""
Xt shape:(1,700)!!!!!!!

"""
print("Xt shape:", Xt.shape)
n = min(len(Xs), len(Xt), TRAIN_SOURCE_NUM, TRAIN_TARGET_NUM)
Xs = Xs[:n]
Ys = Ys[:n]
Xt = Xt[:n]

TRACE_LEN = min(Xs.shape[1], Xt.shape[1])
Xs = Xs[:, :TRACE_LEN]
Xt = Xt[:, :TRACE_LEN]
Xs = Xs.reshape((-1, Xs.shape[1], 1))
Xt = Xt.reshape((-1, Xt.shape[1], 1))

Xs, Xt, mean, std = normalize_source_target(Xs, Xt)

np.save(MEAN_SAVE_PATH, mean)
np.save(STD_SAVE_PATH, std)

Ys_cat = to_categorical(Ys, NUM_CLASSES)
Yt_dummy = np.zeros((len(Xt), NUM_CLASSES))

Ds = np.tile([1, 0], (len(Xs), 1))
Dt = np.tile([0, 1], (len(Xt), 1))

X_all = np.concatenate([Xs, Xt], axis=0)
Y_all = np.concatenate([Ys_cat, Yt_dummy], axis=0)
D_all = np.concatenate([Ds, Dt], axis=0)

label_weights = np.concatenate([
    np.ones(len(Xs)),
    np.zeros(len(Xt))
])

domain_weights = np.ones(len(X_all))

model = build_cdan_model(
    input_dim=Xs.shape[1],
    lambda_grl=LAMBDA_GRL,
    domain_weight=DOMAIN_LOSS_WEIGHT
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_SAVE_PATH,
    save_best_only=True,
    monitor='val_classifier_accuracy',
    mode='max'
)

history = model.fit(
    X_all,
    {
        "classifier": Y_all,
        "domain_discriminator": D_all
    },
    sample_weight={
        "classifier": label_weights,
        "domain_discriminator": domain_weights
    },
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
    shuffle=True,
    callbacks=[checkpoint]
)



np.save(HISTORY_SAVE_PATH, history.history)