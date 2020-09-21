# coding: utf-8

import numpy as np
import tensorflow as tf
import tensorflow.keras as tk


class SystemEstimator(tk.Model):

    def __init__(self, dim_state, lr=1e-3, l2_scale=1e-2, name="StateEstimator", dtype=tf.float32):
        super().__init__(name=name, dtype=dtype)
        self.lstate = tk.layers.Dense(
            dim_state,
            kernel_initializer="zeros",
            kernel_regularizer=tk.regularizers.l2(l2_scale),
            name="state",
            dtype=dtype
        )
        self.optimizer = tk.optimizers.Adam(learning_rate=lr)

    def call(self, inputs):
        feature = tf.concat(inputs, axis=1)
        feature = self.lstate(feature)
        return feature

    def reset(self, dim_state, dim_action):
        dummy_state = tk.Input((dim_state,), dtype=self.dtype)
        dummy_action = tk.Input((dim_action,), dtype=self.dtype)
        self([dummy_state, dummy_action])

    def predict(self, state, action):
        pre = self._predict_body(state, action)
        pre = pre.numpy()
        return pre

    def _predict_body(self, state, action):
        stt = tf.expand_dims(state, axis=0)
        act = tf.expand_dims(action, axis=0)
        pre = self([stt, act])
        pre = tf.squeeze(pre)
        return pre

    def update(self, state, action, next_state):
        stt = tf.expand_dims(state, axis=0) if len(state.shape) == 1 else state
        act = tf.expand_dims(action, axis=0) if len(action.shape) == 1 else action
        nst = tf.expand_dims(next_state, axis=0) if len(next_state.shape) == 1 else next_state
        loss = self._update_body(stt, act, nst)
        loss = loss.numpy()
        return loss

    @tf.function
    def _update_body(self, stt, act, nst):
        with tf.GradientTape() as tape:
            pre = self([stt, act])
            err = nst - pre
            loss = tf.reduce_mean(tf.square(err))
        grad = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return loss


class DeepSystemEstimator(SystemEstimator):

    def __init__(
            self,
            units,
            dim_state,
            lr=1e-3,
            l2_scale=1e-2,
            name="DeepSystemEstimator",
            **kwargs
    ):
        super().__init__(dim_state, lr=lr, l2_scale=l2_scale, name=name, **kwargs)
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
        feature = tf.concat(inputs, axis=1)
        for layer in self.layers_:
            feature = layer(feature)
        feature = self.lstate(feature)
        return feature
