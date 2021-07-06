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
from feles.controller import sfb
from feles.controller import se
from feles.utils.builder import *
from feles.utils.buffer import ReplayBuffer

COLOR_ZERO = "black"
COLOR_DATA = ["gray", "forestgreen", "hotpink"]
WIDTH_ZERO = "1"


def run(args):
    xt.info("demo for system estimator")
    cf = xt.Config(args.config)

    # prepare env
    env = build_env(cf.env)
    # state feedback controller
    fbc = build_sfb(cf.controller.sfb, env)
    # system estimator
    sye = build_sye(cf.controller.sye, env, act_dim=1)
    # prepare reference
    ref = build_reference_queue(cf.env, at_random=False)

    res_sye = run_sye(cf, env, ref, fbc, sye)

    plot(cf, env, res_sye)


def run_sye(cf, env, ref, fbc, sye):
    env.reset()
    log = xsim.Logger()
    rb = ReplayBuffer(int(cf.controller.sye.window / cf.env.dt))

    past_x = None
    past_u = None
    loss_sye = 0.0
    predict = np.zeros_like(env.state)

    for time, r in tqdm(ref.items()):
        if hasattr(cf.env, "fail") and time == cf.env.fail.time:
            env.set_fail()
        x = env.state
        u = compute_fbc_action(fbc, r, x)
        act = build_action(env, u)

        if past_x is not None:
            rb.add(se.Log(past_x, past_u, x))
            predict = sye.predict(past_x, past_u)

        if len(rb) >= rb.max_size:
            data = rb.buffer
            loss_sye = sye.update(data.state, data.action, data.next_state)

        log.store(
            time=time,
            r=r,
            x=x,
            u=u,
            p=predict,
            loss=loss_sye,
            ws=sye.W.flatten()
        ).flush()

        # simulation update
        env.step(act)
        past_x = x
        past_u = u

    res = xsim.Retriever(log)
    res = pd.DataFrame(dict(
        time=res.time(),
        reference=res.r(env.IX_T, fn=xt.r2d),
        pitch=res.x(env.IX_T, fn=xt.r2d),
        predict=res.p(env.IX_T, fn=xt.r2d),
        elevator=res.u(fn=xt.r2d),
        loss=res.loss(),
        w1=res.ws(0),
        w2=res.ws(1),
        w3=res.ws(2),
        w4=res.ws(3),
        w5=res.ws(4),
        w6=res.ws(5),
    ))

    print(res)
    return res


def compute_fbc_action(fbc, r, x):
    r = r[:len(x)]
    e = r - x
    u = fbc(e)
    u = np.asanyarray(u)
    return u


def build_action(env, u):
    act = np.array([0., u]).astype(float)
    act = np.clip(act, env.action_space.low, env.action_space.high)
    return act


def plot(cf, env, res):
    xt.info("plot result")
    ecf = cf.env
    delim = xt.round(xt.r2d([
        env.action_space.low[env.IX_de],
        env.action_space.high[env.IX_de],
    ]), 3)

    fig, axes = plt.subplots(nrows=4, sharex=True, figsize=(8, 8))
    # draw zero-lines
    axes[0].axhline(0, color=COLOR_ZERO, linewidth=WIDTH_ZERO)
    axes[1].axhline(0, color=COLOR_ZERO, linewidth=WIDTH_ZERO)

    res.plot(
        x="time", y=["reference", "pitch", "predict"], ax=axes[0],
        style=["--", "-", "-"], color=COLOR_DATA,
        xlim=[0, ecf.due],
        grid=True,
        # ylim=ecf.reference.range,
        # yticks=np.arange(ecf.reference.range[0], ecf.reference.range[1]+1, 1)
    )
    res.plot(
        x="time", y="elevator", ax=axes[1],
        style="-", color="orange",
        # ylim=delim,
        grid=True,
        # yticks=np.arange(delim[0], delim[1]+1, 1)
    )
    res.plot(
        x="time", y="loss", ax=axes[2],
        style="-", color="orange", grid=True,
        ylim=[0, np.max(res.loss) * 1.2]
    )
    res.plot(
        x="time", y=["w1", "w2", "w3", "w4", "w5", "w6"],
        ax=axes[3], grid=True
    )

    plt.show()


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/old/se_no_fail.yaml", type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = prepare_args()
    run(args)
