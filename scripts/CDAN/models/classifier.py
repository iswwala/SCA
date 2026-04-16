from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def build_classifier():
    inputs = Input(shape=(1024,))

    x = Dense(512, activation='relu')(inputs)
    outputs = Dense(256, activation='softmax', name='label_output')(x)

    return Model(inputs, outputs, name="classifier")