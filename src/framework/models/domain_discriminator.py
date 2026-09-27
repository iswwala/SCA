from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def build_domain_discriminator():
    inputs = Input(shape=(128 * 128,))

    x = Dense(1024, activation='relu')(inputs)
    x = Dropout(0.5)(x)

    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(2, activation='softmax', name='domain_output')(x)

    return Model(inputs, outputs, name="domain_discriminator")