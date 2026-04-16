import numpy as np
from tensorflow.keras.models import load_model

from config import *
from utils.load_data import load_ascad
from utils.preprocess import normalize_test
from models.grl import GradientReversal
from models.conditional_feature import ConditionalFeature

custom_objects = {
    "GradientReversal": GradientReversal,
    "ConditionalFeature": ConditionalFeature
}

model = load_model(MODEL_SAVE_PATH, custom_objects=custom_objects)

feature_extractor = model.get_layer("feature_extractor")
classifier = model.get_layer("classifier")

(_, _), (X_test, Y_test) = load_ascad(VARIABLE_PATH)

X_test = X_test[:TEST_NUM]
Y_test = Y_test[:TEST_NUM]

X_test = X_test.reshape((-1, X_test.shape[1], 1))

mean = np.load(MEAN_SAVE_PATH)
std = np.load(STD_SAVE_PATH)

X_test = normalize_test(X_test, mean, std)

features = feature_extractor.predict(X_test, batch_size=256)
preds = classifier.predict(features, batch_size=256)

pred_labels = np.argmax(preds, axis=1)
acc = np.mean(pred_labels == Y_test)

print("Accuracy:", acc)

np.save(PRED_SAVE_PATH, preds)
print("Saved:", PRED_SAVE_PATH)