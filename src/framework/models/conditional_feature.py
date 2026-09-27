import tensorflow as tf

class ConditionalFeature(tf.keras.layers.Layer):
    def __init__(self, feature_dim=1024, class_dim=256, proj_dim=128, **kwargs):
        super().__init__(**kwargs)

        self.feature_dim = feature_dim
        self.class_dim = class_dim
        self.proj_dim = proj_dim

        self.Wf = self.add_weight(
            shape=(feature_dim, proj_dim),
            initializer='random_normal',
            trainable=False,
            name='proj_feature'
        )

        self.Wg = self.add_weight(
            shape=(class_dim, proj_dim),
            initializer='random_normal',
            trainable=False,
            name='proj_class'
        )

    def call(self, inputs):
        f, g = inputs

        f = tf.matmul(f, self.Wf)
        g = tf.matmul(g, self.Wg)

        f = tf.expand_dims(f, axis=2)
        g = tf.expand_dims(g, axis=1)

        cond = tf.matmul(f, g)
        cond = tf.reshape(cond, (-1, self.proj_dim * self.proj_dim))

        return cond

    def get_config(self):
        config = super().get_config()
        config.update({
            "feature_dim": self.feature_dim,
            "class_dim": self.class_dim,
            "proj_dim": self.proj_dim
        })
        return config