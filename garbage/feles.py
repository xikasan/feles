# coding: utf-8

import xair
import gym
import xsim
import xtools as xt
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass
from feles.controller import sfb
from feles.controller import fel
from feles.controller import se
from feles.utils.buffer import ReplayBuffer

COLOR_ZERO = "black"
COLOR_DATA = ["gray", "forestgreen", "hotpink"]
WIDTH_ZERO = "1"


DTYPE = np.float32


@dataclass
class Sample:
    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray

    @staticmethod
    def format(buf):
        i = [e.i for e in buf]
        return Sample(i)


def run(args):
    xt.info("run feles simulation")
    cf = xt.Config(args.config)

    # plant
    env = build_env(cf.env)
    dim_state = 2
    dim_action = 1
    xt.info("env", env)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # controllers
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # state F/B
    #
    K = build_sfb(cf.controller.sfb, env)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # FEL
    #
    fel = build_fel(cf.controller.fel, env)
    xt.info("fel", fel)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # system estimator
    #
    sye = build_estimator(cf.controller.se, env)
    xt.info("se", sye)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # reference generator
    #
    ref = build_reference_queue(cf.env)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # run simulations
    #
    xt.info("simulation mode", cf.mode)
    res_fls = run_feles(cf, env, ref, fel, K, sye)

    plot(cf, env, res_fls)


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# simulation loop
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# feles fel with enhanced sampling
#
def run_feles(cf, env, ref, fel, K, sye):
    env.reset()
    fel.reset()
    sye.reset()
    log = xsim.Logger()
    rb = ReplayBuffer(int(1.0 / cf.env.dt + 1))

    # prepare
    loss_sye = 0.0

    past_x, past_u = None, None
    for time, rs in tqdm(ref.items()):
        act_sfb = compute_sfb_action(env, K)
        act = build_act(env, act_sfb)
        obs = env.observation
        state = env.state

        # simulation step
        env.step(act)

        rb.add(se.Log(state, act[env.IX_de], env.state))
        if len(rb) >= rb.max_size:
            data = rb.buffer
            loss_sye = sye.update(data.state, data.action, data.next_state)
        predict = np.zeros_like(state) if past_x is None else sye.predict(past_x, past_u)

        past_x = state
        past_u = act[env.IX_de:]
        # print(past_u)

        log.store(
            time=time,
            obs=obs,
            act=act,
            pre=predict,
            loss_sye=loss_sye
        ).flush()

    # retrieve result
    xt.info("make result")
    res = xsim.Retriever(log)
    res = pd.DataFrame(dict(
        time=res.time(),
        pitch=xt.r2d(res.obs(env.IX_T)),
        reference=xt.r2d(res.obs(env.IX_C)),
        predict=xt.r2d(res.pre(env.IX_T)),
        elevator=xt.r2d(res.act(env.IX_de)),
        loss=res.loss_sye()
    ))

    return res

def plot(cf, env, res):
    xt.info("plot result")
    delim = xt.round(xt.r2d([
        env.action_space.low[env.IX_de],
        env.action_space.high[env.IX_de],
    ]), 3)

    fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(8, 8))
    # draw zero-lines
    axes[0].axhline(0, color=COLOR_ZERO, linewidth=WIDTH_ZERO)
    axes[1].axhline(0, color=COLOR_ZERO, linewidth=WIDTH_ZERO)

    ecf = cf.env
    res.plot(
        x="time", y=["reference", "pitch", "predict"], ax=axes[0],
        style=["--", "-", "-"], color=COLOR_DATA,
        xlim=[0, ecf.due], ylim=ecf.reference.range, grid=True,
        yticks=np.arange(ecf.reference.range[0], ecf.reference.range[1] + 1, 1)
    )
    res.plot(
        x="time", y="elevator", ax=axes[1],
        style="-", color="orange",
        ylim=delim, grid=True,
        yticks=np.arange(delim[0], delim[1] + 1, 1)
    )
    res.plot(
        x="time", y="loss", ax=axes[2],
        style="-", color="orange",
        ylim=[0, np.max(res.loss) * 1.2], grid=True
    )

    plt.show()


def compute_sfb_action(env, K):
    x = env.state
    r = env.target[:2]
    e = r - x
    u = np.dot(K, e)
    return u


def build_act(env, act):
    act = np.array([0, act]).astype(np.float32)
    act = np.clip(act, env.action_space.low, env.action_space.high)
    return act


def build_env(cf):
    env = gym.make(
        cf.name, dt=cf.dt,
        target_range=xt.d2r(cf.reference.range),
        target_period=cf.reference.interval
    )
    env.reset()
    return env


def build_sfb(cf, env):
    Q = np.diag(cf.Q).astype(DTYPE)
    R = np.diag(cf.R).astype(DTYPE)
    K = sfb.compute_sfb_gain(env, Q, R)
    return np.squeeze(K)


def build_fel(cf, env):
    dim_act = 1
    dim_ref = env.reference.shape[0]
    max_act = env.action_space.high[env.IX_de]
    return fel.FEL(dim_ref, dim_act, max_act, lr=cf.lr)


def build_estimator(cf, env):
    dim_action = 1
    dim_state = env.state.shape[0]
    return se.SystemEstimator3(dim_state, dim_action, lr=cf.lr)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# reference
#
def build_reference_queue(cf, due=None):
    ecf = cf
    reference_width = (np.max(ecf.reference.range) - np.min(ecf.reference.range)) / 2
    reference_width = xt.d2r(reference_width) * 2
    ref = xsim.PoissonRectangularCommand(
        max_amplitude=reference_width,
        interval=ecf.reference.interval
    )
    ref.reset()
    ref_filter = xsim.Filter2nd(ecf.dt, ecf.reference.tau)
    ref_filter.reset()
    xt.info("reference generator", ref)

    def generate_full_state_reference(t):
        ref_filter(ref(t))
        return ref_filter.get_full_state()

    due = ecf.due if due is None else due
    ref = {
        time: generate_full_state_reference(time)
        for time in xsim.generate_step_time(due, ecf.dt)
    }
    return ref


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/feles_no_fail.yaml", type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = prepare_args()
    run(args)
