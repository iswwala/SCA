from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def build_feature_extractor(input_dim):
    inputs = Input(shape=(input_dim, 1))

    x = Conv1D(64, 11, activation='relu', padding='same')(inputs)
    x = AveragePooling1D(2)(x)

    x = Conv1D(128, 11, activation='relu', padding='same')(x)
    x = AveragePooling1D(2)(x)

    x = Conv1D(256, 11, activation='relu', padding='same')(x)
    x = AveragePooling1D(2)(x)

    x = Conv1D(512, 11, activation='relu', padding='same')(x)
    x = AveragePooling1D(2)(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    return Model(inputs, x, name="feature_extractor")