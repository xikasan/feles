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
from feles.controller.tf_fel import FEL
from tqdm import tqdm


DTYPE = np.float32
DT = 0.01
DUE = 240

ENV_NAME = "LVAircraftPitch-v4"
ENV_TARGET_RANGE = xt.d2r([-1, 1])
ENV_TARGET_PERIOD = 20
ENV_FAIL_MODE = "GAIN_REDUCTION"
ENV_FAIL_RANGE = [0.5, 0.51]

# State Feedback Controller
Q = np.diag([100, 1]).astype(DTYPE)
R = np.diag([1]).astype(DTYPE)
# FEL Controller
LR = 3e-2


def run():
    xt.info("test Feedback Error Learning control")
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
    dim_ref = env.get_target().shape[0]

    # controller
    # SFB
    K = sfb.compute_sfb_gain(env, Q, R).T
    K = np.hstack([np.zeros_like(K), K]).astype(DTYPE)
    # FEL
    fel = FEL(dim_act, max_act, lr=LR)
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

        # simulation update
        # if time == 30:
        #     env.set_fail()
        env.step(act)
        log.store(
            time=time,
            xs=xt.r2d(xs),
            us=xt.r2d(act),
            sfb=xt.r2d(act_sfb),
            fel=xt.r2d(act_fel),
            loss=loss
        ).flush()

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

    fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(DUE/10, 10))
    res.plot(x="time", y=["reference", "pitch"], ax=axes[0], xlim=[0, DUE])
    res.plot(x="time", y=["elevator", "sfb", "fel"], ax=axes[1])
    res.plot(x="time", y="loss", ax=axes[2])
    plt.show()


def compute_action(env, K, fel):
    act_sfb = compute_sfb_action(env, K)
    act_fel = compute_fel_action(env, fel)
    act = act_sfb + act_fel
    return act, act_sfb, act_fel


def compute_sfb_action(env, K):
    x = env.state
    r = env.target[:2]
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
