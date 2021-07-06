# coding: utf-8

import xtools as xt
import xsim
import xair
import gym
import pandas as pd
from matplotlib import pyplot as plt

dt = 1 / 50
due = 10


def run():
    xt.info("test env")

    env = gym.make("LVAircraftPitch-v0", dt=dt)
    xt.info("env", env)

    log = xsim.Logger()
    x = env.get_state()
    u = env.action_space.sample()
    log.store(time=0, x=xt.r2d(x), u=u).flush()

    for time in xsim.generate_step_time(due, dt):
        x = env.get_state()
        u = env.action_space.sample()
        env.step(u)

        log.store(time=time, x=xt.r2d(x), u=u).flush()

    result = xsim.Retriever(log)
    result = pd.DataFrame({
        "time": result.time(),
        "theta": result.x(idx=env.IX_T)
    })
    result.plot(x="time", y="theta")
    plt.show()


if __name__ == '__main__':
    run()
