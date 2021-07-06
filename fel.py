# coding: utf-8

import xsim
import xtools as xt
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from feles.controller import compute_action as ca
from feles.controller.fel import Log, NLCFEL
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
    # fel controller
    fel = build_fel(cf.controller.fel, env)
    # reference queue
    ref = build_reference_queue(cf.env, at_random=cf.env.reference.random)

    ret = run_fel(cf, env, ref, fbc, fel)
    res = retrieve(ret)

    # res_w = pd.DataFrame(dict(
    #     time=ret.t(),
    #     w1=ret.w(0),
    #     w2=ret.w(1),
    #     w3=ret.w(2),
    #     # w4=ret.w(3),
    # ))
    # fig2, axes2 = plt.subplots(nrows=3, sharex=True)
    # for i, ax in enumerate(axes2):
    #     res_w.plot(x="time", y="w{}".format(i+1), ax=ax)
    plot(cf, res)


def run_fel(cf, env, ref, fbc, fel):
    env.reset()
    fel.reset()

    log = xsim.Logger()
    step_in_window = int(cf.controller.fel.window / cf.env.dt)
    bf = ReplayBuffer(step_in_window)

    batch_train = hasattr(cf.controller.fel, "window")
    batch_train = False if not batch_train else cf.controller.fel.window > 0
    decay = build_decay_queue(step_in_window, cf.controller.fel.decay) * 0.15
    loss = 0

    for time, r in tqdm(ref.items()):
        if hasattr(cf.env, "fail") and time == cf.env.fail.time:
            env.set_fail()
        x = env.state
        u_fbc = ca.fbc(fbc, r, x)
        u_fel = ca.fel(fel, r)
        u = u_fbc + u_fel
        act = ca.build_action(env, u)

        if batch_train:
            bf.add(Log(r, u))

        if batch_train and bf.is_full:
            data = bf.buffer
            loss = fel.update(data.reference, data.action, factor=decay)
        else:
            loss = fel.update(r, u)

        if isinstance(fel, NLCFEL):
            loss, loss_lin, loss_nlc = loss

        log.store(
            t=time,
            r=r,
            x=x,
            u=u,
            u_fbc=u_fbc,
            u_fel=u_fel,
            l_fel=loss,
            # temp
            w=fel.weights
        ).flush()

        env.step(act)

    return xsim.Retriever(log)


def retrieve(ret):
    ixT = 0
    r2d = xt.r2d
    res = pd.DataFrame(dict(
        time=ret.t(),
        reference=ret.r(ixT, fn=r2d),
        pitch=ret.x(ixT, fn=r2d),
        elevator=ret.u(fn=r2d),
        u_fel=ret.u_fel(fn=r2d),
        u_fbc=ret.u_fbc(fn=r2d),
        loss=ret.l(),
        error=np.abs(ret.r(ixT, fn=r2d) - ret.x(ixT, fn=r2d))
    ))
    return res


def plot(cf, res):
    fig, axes = plt.subplots(nrows=4, sharex=True)
    L_TIME = "time"

    res.plot(
        x=L_TIME, y=["reference"], ax=axes[0],
        style="--", color="gray",
        xlim=[0, cf.env.due] #, ylim=np.array(cf.env.reference.range)*1.2
    )
    res.plot(
        x=L_TIME, y=["pitch"], ax=axes[0]
    )

    res.plot(
        x=L_TIME, y=["u_fbc", "u_fel", "elevator"], ax=axes[1]
    )

    res.plot(
        x=L_TIME, y="error", ax=axes[2]
    )

    res.plot(
        x=L_TIME, y="loss", ax=axes[3]
    )

    plt.show()


def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/fel.yaml")
    parser.add_argument("--random", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argument()
    run(args)

