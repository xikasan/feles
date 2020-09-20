# coding: utf-8

import xsim
import xair
import gym
import numpy as np
import xtools as xt
import pandas as pd
from matplotlib import pyplot as plt

dt = 1 / 50
due = 40

K = -4.0


def run():
    xt.info("test feedback")

    env = gym.make("LVAircraftPitch-v0", dt=dt, range_target=xt.d2r([-5, 5]), target_period=20)
    env.reset()
    xt.info("env", env)

    log = xsim.Logger()
    x = env.get_observation()
    u = [0, 0]
    log.store(time=0, x=xt.r2d(x), u=u).flush()

    for time in xsim.generate_step_time(due, dt):
        x = env.get_observation()
        T, q, Tc = x
        u = np.array([0, K * (Tc - T)])
        env.step(u)

        log.store(time=time, x=xt.r2d(x), u=u).flush()

    result = xsim.Retriever(log)
    result = pd.DataFrame({
        "time": result.time(),
        "pitch": result.x(idx=env.IX_T),
        "command": result.x(idx=env.IX_C)
    })
    result.plot(x="time", y=["command", "pitch"])
    plt.show()


if __name__ == '__main__':
    run()
