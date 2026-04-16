import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from models.feature_extractor import build_feature_extractor
from models.classifier import build_classifier
from models.domain_discriminator import build_domain_discriminator
from models.grl import GradientReversal
from models.conditional_feature import ConditionalFeature

def build_cdan_model(input_dim, lambda_grl=0.1, domain_weight=0.1):
    trace_input = Input(shape=(input_dim, 1))

    feature_extractor = build_feature_extractor(input_dim)
    classifier = build_classifier()
    domain_discriminator = build_domain_discriminator()

    features = feature_extractor(trace_input)
    class_pred = classifier(features)

    cond = ConditionalFeature()([features, class_pred])
    cond = GradientReversal(lambda_grl)(cond)

    domain_pred = domain_discriminator(cond)

    model = Model(
        inputs=trace_input,
        outputs=[class_pred, domain_pred]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss={
            "classifier": "categorical_crossentropy",
            "domain_discriminator": "categorical_crossentropy"
        },
        loss_weights={
            "classifier": 1.0,
            "domain_discriminator": 0.5
        },
        metrics={
            "classifier": ["accuracy"],
            "domain_discriminator": ["accuracy"]
        }
    )

    return model