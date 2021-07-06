# coding: utf-8

import xair
import gym
import xsim
import numpy as np
import xtools as xt
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass
from feles.controller import sfb
from feles.controller.se import SystemEstimator3
from feles.utils.buffer import ReplayBuffer, Experience

COLOR_ZERO = "black"
COLOR_DATA = ["gray", "forestgreen", "hotpink"]
WIDTH_ZERO = "1"


@dataclass
class Log:
    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray

    @staticmethod
    def format(buffer):
        state = np.vstack([exp.state for exp in buffer]).astype(np.float32)
        action = np.vstack([exp.action for exp in buffer]).astype(np.float32)
        next_state = np.vstack([exp.next_state for exp in buffer]).astype(np.float32)
        return Log(state, action, next_state)


def dev(args):
    xt.info("development mode of SystemEstimator2")
    cf = xt.Config(args.config)

    # plant
    ecf = cf.env
    env = build_env(cf)
    dim_state = 2
    dim_action = 1
    xt.info("env", env)

    # controller
    scf = cf.controller.sfb
    Q = np.diag(scf.Q).astype(float)
    R = np.diag(scf.R).astype(float)
    K = sfb.compute_sfb_gain(env, Q, R)
    K = np.squeeze(K)

    # estimator
    se3 = SystemEstimator3(dim_state, dim_action, cf.controller.se.lr)

    # logger
    logger = xsim.Logger()
    rb = ReplayBuffer(int(1.0 / ecf.dt))

    # prepare simulation
    loss = 0.0

    xt.info("run simulation")
    for time in tqdm(xsim.generate_step_time(ecf.due, ecf.dt)):
        obs = env.observation
        state = env.state
        action = compute_action(env, K)

        # simulation update
        env.step(action)

        rb.add(Log(state, action[1], env.state))
        if len(rb) >= 100:
            data = rb.buffer
            loss = se3.update(data.state, data.action, data.next_state)
        predict = se3.predict(state, action[1:])

        logger.store(time=time, obs=obs, act=action, pre=predict, loss=loss).flush()

    print(se3.A())
    exit()
    # retrieve result
    xt.info("make result")
    res = xsim.Retriever(logger)
    res = pd.DataFrame(dict(
        time=res.time(),
        pitch=xt.r2d(res.obs(env.IX_T)),
        reference=xt.r2d(res.obs(env.IX_C)),
        predict=xt.r2d(res.pre(env.IX_T)),
        elevator=xt.r2d(res.act(env.IX_de)),
        loss=res.loss()
    ))

    if not args.show_fig and not cf.result.fig.show:
        return args

    xt.info("plot result")
    delim = xt.round(xt.r2d([
        env.action_space.low[env.IX_de],
        env.action_space.high[env.IX_de],
    ]), 3)

    fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(8, 8))
    # draw zero-lines
    axes[0].axhline(0, color=COLOR_ZERO, linewidth=WIDTH_ZERO)
    axes[1].axhline(0, color=COLOR_ZERO, linewidth=WIDTH_ZERO)

    res.plot(
        x="time", y=["reference", "pitch", "predict"], ax=axes[0],
        style=["--", "-", "-"], color=COLOR_DATA,
        xlim=[0, ecf.due], ylim=ecf.reference.range, grid=True,
        yticks=np.arange(ecf.reference.range[0], ecf.reference.range[1]+1, 1)
    )
    res.plot(
        x="time", y="elevator", ax=axes[1],
        style="-", color="orange",
        ylim=delim, grid=True,
        yticks=np.arange(delim[0], delim[1]+1, 1)
    )
    res.plot(
        x="time", y="loss", ax=axes[2],
        style="-", color="orange",
        ylim=[0, np.max(res.loss) * 1.2], grid=True
    )

    plt.show()


def build_env(cf):
    ecf = cf.env
    env = gym.make(
        ecf.name, dt=ecf.dt,
        target_range=xt.d2r(ecf.reference.range),
        target_period=ecf.reference.interval
    )
    env.reset()
    return env


def compute_action(env, K):
    x = env.state
    r = env.target[:2]
    e = r - x
    u = e.dot(K)
    u = np.array([0, u], dtype=env.dtype)
    u = np.clip(u, env.action_space.low, env.action_space.high)
    return u


def run(args):
    xt.info("run to dev system estimation")
    cf = xt.Config(args.config)

    # prepare env
    ecf = cf.env
    env = gym.make(
        ecf.name, dt=ecf.dt,
        target_range=xt.d2r(ecf.reference.range),
        target_period=ecf.reference.interval
    )
    env.reset()
    xt.info("env", env)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # controllers
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # state F/B
    #
    scf = cf.controller.sfb
    Q = np.diag(scf.Q).astype(float)
    R = np.diag(scf.R).astype(float)
    K = sfb.compute_sfb_gain(env, Q, R)
    K = np.squeeze(K)

    # logger
    logger = xsim.Logger()
    rb = ReplayBuffer(int(ecf.due / ecf.dt + 1))

    xt.info("run simulation")
    for time in tqdm(xsim.generate_step_time(ecf.due, ecf.dt)):
        state = env.state
        action = compute_action(env, K)
        logger.store(time=time, x=state, u=action).flush()

        # simulation update
        env.step(action)

        rb.add(state, action, env.state)

    # retrieve result
    xt.info("make result")
    res = xsim.Retriever(logger)
    res = pd.DataFrame(dict(
        time=res.time(),
        pitch=xt.r2d(res.state(env.IX_T)),
        reference=xt.r2d(res.state(env.IX_C)),
        elevator=xt.r2d(res.action(env.IX_de))
    ))

    if not args.show_fig and not cf.result.fig.show:
        return args

    xt.info("plot result")
    delim = xt.round(xt.r2d([
        env.action_space.low[env.IX_de],
        env.action_space.high[env.IX_de],
    ]), 3)

    fig, axes = plt.subplots(nrows=2, sharex=True)
    # draw zero-lines
    axes[0].axhline(0, color=COLOR_ZERO, linewidth=WIDTH_ZERO)
    axes[1].axhline(0, color=COLOR_ZERO, linewidth=WIDTH_ZERO)

    res.plot(
        x="time", y=["reference", "pitch"], ax=axes[0],
        style=["--", "-"], color=COLOR_DATA,
        xlim=[0, ecf.due], ylim=ecf.reference.range, grid=True,
        yticks=np.arange(ecf.reference.range[0], ecf.reference.range[1]+1, 1)
    )
    res.plot(
        x="time", y="elevator", ax=axes[1],
        style="-", color="orange",
        ylim=delim, grid=True,
        yticks=np.arange(delim[0], delim[1]+1, 1)
    )

    plt.show()


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--config", default="configs/se_no_fail.yaml", type=str)
    # result export
    parser.add_argument("--show-fig", action="store_true")

    return parser.parse_args()


if __name__ == '__main__':
    args = prepare_args()
    if args.dev:
        dev(args)
        exit(0)
    run(args)
