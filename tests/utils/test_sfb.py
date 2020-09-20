# coding: utf-8

import xair
import gym
import xsim
import numpy as np
import pandas as pd
import xtools as xt
import matplotlib.pyplot as plt
from feles.utils import sfb


DTYPE = np.float32
DT = 0.01
DUE = 40

ENV_NAME = "LVAircraftPitch-v1"
ENV_TARGET_RANGE = xt.d2r([-5, 5])
ENV_TARGET_PERIOD = 20

Q = np.diag([100, 1]).astype(DTYPE)
R = np.diag([1]).astype(DTYPE)


def run():
    xt.info("test state feedback control")

    env = gym.make(
        ENV_NAME, dt=DT,
        target_range=ENV_TARGET_RANGE,
        target_period=ENV_TARGET_PERIOD
    )
    xt.info("env", env)
    env.reset()

    # controller
    K = sfb.compute_sfb_gain(env, Q, R).T
    K = np.hstack([np.zeros_like(K), K]).astype(DTYPE)
    xt.info("State Feedback Gain")
    print(K.T)

    # logger
    log = xsim.Logger()

    # simulation loop
    print("="*60)
    xt.info("start simulation")
    for time in xsim.generate_step_time(DUE, DT):
        xs = env.observation
        u = compute_action(env, K)

        env.step(u)
        log.store(time=time, xs=xs).flush()

    # result plot
    res = xsim.Retriever(log)
    res = pd.DataFrame({
        "time": res.time(),
        "reference": res.xs(idx=env.IX_C),
        "pitch": res.xs(idx=env.IX_T)
    })
    res.plot(x="time", y=["reference", "pitch"])
    plt.show()


def compute_action(env, K):
    obs = env.observation
    x = obs[[env.IX_T, env.IX_q]]
    r = np.array([obs[env.IX_C], 0.], dtype=env.dtype)
    e = r - x
    u = e.dot(K)
    return u


if __name__ == '__main__':
    run()
