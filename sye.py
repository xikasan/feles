# coding: utf-8

import xsim
import xtools as xt
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from feles.controller.se import Log
from feles.controller import compute_action as ca
from feles.utils.buffer import ReplayBuffer
from feles.utils.builder import *


def run(args):
    cf = xt.Config(args.config)
    if args.random:
        cf.env.reference.random = True

    # env
    env = build_env(cf.env)
    # LQR state feedback controller
    fbc = build_sfb(cf.controller.fbc, env)
    # system estimator
    sye = build_sye(cf.controller.sye, env)
    # reference queue
    ref = build_reference_queue(cf.env, at_random=cf.env.reference.random)

    ret = run_sye(cf, env, ref, fbc, sye)
    res = retrieve(ret)
    plot(cf, res)


def run_sye(cf, env, ref, fbc, sye):
    env.reset()
    sye.reset()

    log = xsim.Logger()
    step_in_window = int(cf.controller.sye.window / cf.env.dt)
    bf = ReplayBuffer(step_in_window)
    past_x = None
    past_u = None

    decay = build_decay_queue(step_in_window, cf.controller.sye.decay)
    predict = np.zeros_like(env.state)
    loss = 0

    for time, r in tqdm(ref.items()):
        x = env.state
        u = ca.fbc(fbc, r, x)
        act = ca.build_action(env, u)

        if past_x is not None:
            bf.add(Log(past_x, past_u, x))
            predict = sye.predict(past_x, past_u)

        if bf.is_full:
            data = bf.buffer
            loss = sye.update(data.state, data.action, data.next_state, factor=decay)

        log.store(
            t=time,
            x=x,
            u=u,
            r=r,
            p=predict,
            l=loss
        ).flush()

        env.step(act)
        past_x = x
        past_u = u

    return xsim.Retriever(log)


def plot(cf, res):
    fig, axes = plt.subplots(nrows=2, sharex=True)
    L_TIME = "time"

    res.plot(
        x=L_TIME, y=["reference"], ax=axes[0],
        style="--", color="gray",
        xlim=[0, cf.env.due], ylim=cf.env.reference.range
    )
    res.plot(
        x=L_TIME, y=["pitch", "predict"], ax=axes[0]
    )

    res.plot(
        x=L_TIME, y="loss", ax=axes[1]
    )

    plt.show()


def retrieve(ret):
    ixT = 0
    r2d = xt.r2d
    res = pd.DataFrame(dict(
        time=ret.t(),
        reference=ret.r(ixT, fn=r2d),
        pitch=ret.x(ixT, fn=r2d),
        predict=ret.p(ixT, fn=r2d),
        loss=ret.l(),
    ))
    return res


def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/sye.yaml")
    parser.add_argument("--random", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argument()
    run(args)
