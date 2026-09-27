import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


def build_feature_extractor(input_shape=(700,1)):

    inputs = Input(shape=input_shape)

    # Block 1
    x = Conv1D(64, 11, activation='relu', padding='same')(inputs)
    x = AveragePooling1D(2, strides=2)(x)

    # Block 2
    x = Conv1D(128, 11, activation='relu', padding='same')(x)
    x = AveragePooling1D(2, strides=2)(x)

    # Block 3
    x = Conv1D(256, 11, activation='relu', padding='same')(x)
    x = AveragePooling1D(2, strides=2)(x)

    # Block 4
    x = Conv1D(512, 11, activation='relu', padding='same')(x)
    x = AveragePooling1D(2, strides=2)(x)

    # Block 5
    x = Conv1D(512, 11, activation='relu', padding='same')(x)
    x = AveragePooling1D(2, strides=2)(x)

    x = Flatten()(x)
    x = Dense(4096, activation='relu', name='fc1')(x)

    model = Model(inputs, x, name="feature_extractor")

    return model