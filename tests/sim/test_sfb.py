# coding: utf-8

import xsim
import xair
import gym
import numpy as np
import xtools as xt
import pandas as pd
import scipy.linalg as sl
from tqdm import tqdm
from matplotlib import pyplot as plt

DT = 0.01
DUE = 40

TARGET_RANGE = xt.d2r([-5, 5])
TARGET_PERIOD = 20

Q = np.diag([100, 1]).astype(np.float32)
R = np.diag([1]).astype(np.float32)


def run():
    xt.info("test sfb")

    # build env
    env = gym.make(
        "LVAircraftPitch-v1", dt=DT,
        range_target=TARGET_RANGE,
        target_period=TARGET_PERIOD
    )
    env.reset()

    # controller
    K = compute_lqr_gain(env).T
    K = np.hstack([np.zeros_like(K), K]).astype(env.dtype)

    # logger
    log = xsim.Logger()

    # simulation loop
    print("="*60)
    xt.info("start simulation")
    for time in tqdm(xsim.generate_step_time(DUE, DT)):
        xs = env.observation
        u = compute_action(env, K)

        env.step(u)
        log.store(time=time, xs=xt.r2d(xs), u=u).flush()

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



def build_short_term_model(env):
    """
    Build short term mode of pitching motion of aircraft.
    |dT| = |  0   1||T| + | 0|de
    |dq|   |-Mw -Mq||q|   |de|
    :param env:
    :return:
    """
    model = env._model
    Ao = model._A.T
    Bo = model._B.T
    Mw = Ao[model.IX_q, model.IX_w]
    Mq = Ao[model.IX_q, model.IX_q]
    Mde = Bo[model.IX_q, model.IX_de]
    A = np.array([[0, 1], [Mw, Mq]], dtype=model.dtype)
    B = np.array([[0, Mde]], dtype=model.dtype).T
    return A, B


def compute_lqr_gain(env):
    A, B = build_short_term_model(env)
    P = sl.solve_continuous_are(A, B, Q, R)
    K = sl.inv(R).dot(B.T).dot(P)
    return K


if __name__ == '__main__':
    run()
