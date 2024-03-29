# coding: utf-8

import xair
import gym
import xai
import xsim
import numpy as np
import pandas as pd
import xtools as xt
import tensorflow as tf
import tensorflow.keras as tk
import matplotlib.pyplot as plt
from tqdm import tqdm
from feles.controller import sfb
from feles.controller import tf_se


DTYPE = np.float32
DT = 0.01
DUE = 120

ENV_NAME = "LVAircraftPitch-v4"
ENV_TARGET_RANGE = xt.d2r([-1, 1])
ENV_TARGET_PERIOD = 10
ENV_FAIL_MODE = "GAIN_REDUCTION"
ENV_FAIL_RANGE = [0.5, 0.51]

# State Feedback Controller
Q = np.diag([100, 1]).astype(DTYPE)
R = np.diag([1]).astype(DTYPE)

# System Estimator
LR = 1e-4
UNITS = [8, 4]
L2_SCALE = 1e-1


def run():
    xt.info("run System Estimation")
    xai.please()

    env = gym.make(
        ENV_NAME, dt=DT,
        target_range=ENV_TARGET_RANGE,
        target_period=ENV_TARGET_PERIOD,
        fail_mode=ENV_FAIL_MODE,
        fail_range=ENV_FAIL_RANGE
    )
    xt.info("env", env)
    env.reset()
    max_act = env.action_space.high[env.IX_de]
    dim_act = 1  # xai.get_size(env.action_space)
    dim_obs = xai.size(env.observation_space)
    dim_stt = dim_obs - 1
    dim_ref = env.get_target().shape[0]

    # controller
    # SFB
    K = sfb.compute_sfb_gain(env, Q, R).T
    K = np.hstack([np.zeros_like(K), K]).astype(DTYPE)
    # SE
    sye = tf_se.DeepSystemEstimator(UNITS, dim_stt, lr=LR, l2_scale=L2_SCALE)
    sye.reset(dim_stt, dim_act)
    xt.info("sye", sye)

    # logger
    log = xsim.Logger()

    past_act = np.array([0.], dtype=DTYPE)
    past_stt = env.state
    pre = np.zeros_like(env.state)

    for time in tqdm(xsim.generate_step_time(DUE, DT)):
        xs = env.observation
        stt = env.state
        act = compute_sfb_action(env, K)
        next_pre = predict_next_state(env, sye, stt, act)
        loss = sye.update(past_stt, past_act, stt)
        # logging
        log.store(time=time, xs=xt.r2d(xs), act=xt.r2d(act), pre=xt.r2d(pre), loss=loss).flush()

        # simulation update
        # if time == 30:
        #     env.set_fail()
        env.step(act)
        pre = next_pre
        past_act = act[[env.IX_de]]
        past_stt = stt

    res = xsim.Retriever(log)
    res = pd.DataFrame({
        "time": res.time(),
        "pitch": res.xs(idx=env.IX_T),
        "reference": res.xs(idx=env.IX_C),
        "predict": res.pre(idx=env.IX_T),
        "loss": res.loss()
    })

    fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(DUE/10, 10))
    res.plot(x="time", y=["reference", "pitch", "predict"], ax=axes[0], xlim=[0, DUE])
    res.plot(x="time", y="loss", ax=axes[1])
    plt.show()


def compute_sfb_action(env, K):
    obs = env.observation
    x = env.state
    # r = np.array([obs[env.IX_C], 0.], dtype=env.dtype)
    r = env.target[:2]
    e = r - x
    u = e.dot(K)
    return u


def predict_next_state(env, sye, stt, act):
    act = act[[env.IX_de]]
    pre = sye.predict(stt, act)
    return pre


if __name__ == '__main__':
    run()
