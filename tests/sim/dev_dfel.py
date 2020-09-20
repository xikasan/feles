# coding: utf-8

import xair
import gym
import xai
import xsim
import numpy as np
import pandas as pd
import xtools as xt
import matplotlib.pyplot as plt
from feles.controller import sfb
from feles.controller.fel import DeepFEL
from tqdm import tqdm


DTYPE = np.float32
DT = 0.01
DUE = 600

ENV_NAME = "LVAircraftPitch-v1"
ENV_TARGET_RANGE = xt.d2r([-5, 5])
ENV_TARGET_PERIOD = 20

# State Feedback Controller
Q = np.diag([100, 1]).astype(DTYPE)
R = np.diag([1]).astype(DTYPE)
# FEL Controller
LR = 2e-4
UNITS = [32, 32]


def run():
    xt.info("test Feedback Error Learning control")
    xai.please()

    env = gym.make(
        ENV_NAME, dt=DT,
        target_range=ENV_TARGET_RANGE,
        target_period=ENV_TARGET_PERIOD
    )
    xt.info("env", env)
    env.reset()
    max_act = env.action_space.high[env.IX_de]
    dim_act = 1  # xai.get_size(env.action_space)
    dim_obs = xai.get_size(env.observation_space)
    dim_ref = env.get_target().shape[0]

    # controller
    # SFB
    K = sfb.compute_sfb_gain(env, Q, R).T
    K = np.hstack([np.zeros_like(K), K]).astype(DTYPE)
    # FEL
    fel = DeepFEL(UNITS, dim_act, max_act, lr=LR)
    fel.reset(dim_ref)
    xt.info("fel", fel)

    # logger
    log = xsim.Logger()

    for time in tqdm(xsim.generate_step_time(DUE, DT)):
        xs = env.observation
        act_sfb = compute_sfb_action(env, K)
        act_fel = compute_fel_action(env, fel)
        act = act_sfb + act_fel

        loss = fel.update(env.get_target(), act[[env.IX_de]])

        env.step(act)
        log.store(time=time, xs=xs, us=act, sfb=act_sfb, fel=act_fel, loss=loss).flush()

    # result plot
    res = xsim.Retriever(log)
    res = pd.DataFrame({
        "time": res.time(),
        "reference": res.xs(idx=env.IX_C),
        "pitch": res.xs(idx=env.IX_T),
        "elevator": res.us(idx=env.IX_de),
        "sfb": res.sfb(idx=env.IX_de),
        "fel": res.fel(idx=env.IX_de),
        "loss": res.loss()
    })

    fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(20, 10))
    res.plot(x="time", y=["reference", "pitch"], ax=axes[0])
    res.plot(x="time", y=["elevator", "sfb", "fel"], ax=axes[1])
    res.plot(x="time", y="loss", ax=axes[2])
    plt.show()


def comput_action(env, K, fel):
    act_sfb = compute_sfb_action(env, K)
    act_fel = compute_fel_action(env, fel)
    act = act_sfb + act_fel
    return act, act_sfb, act_fel


def compute_sfb_action(env, K):
    obs = env.observation
    x = obs[[env.IX_T, env.IX_q]]
    r = np.array([obs[env.IX_C], 0.], dtype=env.dtype)
    e = r - x
    u = e.dot(K)
    return u


def compute_fel_action(env, fel):
    ref = env.get_target()
    act = fel.get_action(ref)
    act = np.hstack([[0], act]).astype(DTYPE)
    return act


if __name__ == '__main__':
    run()
