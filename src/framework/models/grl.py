import tensorflow as tf

class GradientReversal(tf.keras.layers.Layer):
    def __init__(self, lambda_=1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_ = lambda_

    def call(self, x):
        @tf.custom_gradient
        def reverse_grad(x):
            def grad(dy):
                return -self.lambda_ * dy
            return x, grad
        return reverse_grad(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "lambda_": self.lambda_
        })
        return config