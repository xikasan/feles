# coding: utf-8

import numpy as np
import tensorflow as tf
import tensorflow.keras as tk


class FEL(tk.Model):

    def __init__(self, dim_act, max_act, lr=1e-3, l2_scale=1e-2, name="FEL", dtype=tf.float32):
        super().__init__(name=name, dtype=dtype)
        self.lout = tk.layers.Dense(
            dim_act,
            activation="tanh",
            kernel_initializer="zeros",
            kernel_regularizer=tk.regularizers.l2(l2_scale),
            use_bias=False,
            name="out",
            dtype=dtype
        )
        self.max_act = max_act
        self.optimizer = tk.optimizers.Adam(learning_rate=lr)

    def call(self, inputs):
        feature = self.lout(inputs)
        feature = feature * self.max_act
        return feature

    def reset(self, dim_input):
        dummy_input = tk.Input((dim_input,), dtype=self.dtype)
        self(dummy_input)

    def get_action(self, ref):
        act = self._get_action_body(ref)
        act = act.numpy()
        return act

    @tf.function
    def _get_action_body(self, ref):
        ref = tf.expand_dims(ref, axis=0)
        act = self(ref)
        act = tf.squeeze(act, axis=0)
        return act

    def update(self, ref, act):
        if len(ref.shape) == 1:
            ref = np.expand_dims(ref, axis=0)
        if len(act.shape) == 1:
            act = np.expand_dims(act, axis=0)
        loss = self._update_body(ref, act)
        loss = loss.numpy()
        return loss

    @tf.function
    def _update_body(self, ref, act):
        with tf.GradientTape() as tape:
            pre = self(ref)
            err = act - pre
            loss = tf.reduce_mean(tf.square(err))
        grad = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return loss


class DeepFEL(FEL):

    def __init__(
            self,
            units,
            dim_act,
            max_act,
            lr=1e-3,
            l2_scale=1e-2,
            name="DeepFEL",
            **kwargs
    ):
        super().__init__(dim_act, max_act, lr=lr, name=name, **kwargs)
        assert isinstance(units, (list, tuple))
        self.layers_ = []
        for l, unit in enumerate(units):
            layer = tk.layers.Dense(
                unit,
                activation="relu",
                kernel_regularizer=tk.regularizers.l2(l2_scale),
                name="L{}".format(l),
                dtype=self.dtype)
            self.layers_.append(layer)

    def call(self, inputs):
        feature = inputs
        for layer in self.layers_:
            feature = layer(feature)
        feature = self.lout(feature)
        feature = feature * self.max_act
        return feature
