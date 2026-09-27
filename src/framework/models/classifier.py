from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


def build_classifier(feature_dim=4096, num_classes=256):

    inputs = Input(shape=(feature_dim,))

    x = Dense(4096, activation='relu')(inputs)
   
    
    outputs = Dense(
        num_classes,
        activation='softmax',
        name='label_output'
    )(x)

    return Model(inputs, outputs, name="classifier")