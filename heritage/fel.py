# coding: utf-8

import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from feles.utils.builder import *
from feles.utils.buffer import ReplayBuffer
from feles.utils import retrieve
from feles.controller import compute_action as ca


def run(args):
    xt.info("run for fel")
    cf = xt.Config(args.config)

    env = build_env(cf.env)
    fbc = build_sfb(cf.controller.sfb, env)
    fel = build_fel(cf.controller.fel, env)
    ref = build_reference_queue(cf.env)

    res_fel = run_fel(cf, env, ref, fbc, fel)
    res_fbc = run_fbc(cf, env, ref, fbc)

    res_pitch = pd.DataFrame(dict(
        time=res_fel.t(),
        reference=res_fel.r(env.IX_T, fn=xt.r2d),
        fel=res_fel.x(env.IX_T, fn=xt.r2d),
        sfb=res_fbc.x(env.IX_T, fn=xt.r2d)
    ))
    res_elevator = pd.DataFrame(dict(
        time=res_fel.t(),
        sfb=res_fbc.u(fn=xt.r2d),
        fel=res_fel.u(0, fn=xt.r2d),
        u_fbc=res_fel.u(1, fn=xt.r2d),
        u_fel=res_fel.u(2, fn=xt.r2d)
    ))
    res_loss = pd.DataFrame(dict(
        time=res_fel.t(),
        fel=res_fel.loss(),
    ))
    res_weight = pd.DataFrame(dict(
        time=res_fel.t(),
        w1=res_fel.w(0),
        w2=res_fel.w(1),
        w3=res_fel.w(2),
    ))

    C_K = "black"
    C_FBC = "dodgerblue"
    C_FEL = "forestgreen"
    fig, axes = plt.subplots(nrows=4, sharex=True, figsize=(8, 8))

    res_pitch.plot(
        x="time", y=["reference", "sfb", "fel"], ax=axes[0],
        color=[C_K, C_FBC, C_FEL], style=["--", "-", "-"],
        xlim=[0, cf.env.due]
    )
    axes[0].set_ylabel("Pitch [deg.]")

    res_elevator.plot(
        x="time", y=["sfb", "fel"], ax=axes[1],
        color=[C_FBC, C_FEL], style=["-", "-"]
    )
    res_elevator.plot(
        x="time", y=["u_fbc", "u_fel"], ax=axes[1],
        style=["--", "--"], color=["blue", "orange"]
    )
    axes[1].set_ylabel("Elevator command [deg.]")

    res_loss.plot(
        x="time", y="fel", ax=axes[2],
        color=C_FEL, style="-"
    )
    axes[0].set_ylabel("FEL loss")

    res_weight.plot(
        x="time", y=["w1", "w2", "w3"], ax=axes[3],
    )

    plt.show()


def run_fel(cf, env, ref, fbc, fel):
    env.reset()
    fel.reset()

    log = xsim.Logger()

    for time, r in tqdm(ref.items()):
        if hasattr(cf.env, "fail") and time == cf.env.fail.time:
            env.set_fail()
        x = env.state
        u_fel = ca.fel(fel, r)
        u_fbc = ca.fbc(fbc, r, x)
        u = u_fbc + u_fel
        act = ca.build_action(env, u)

        loss_fel = np.mean([fel.update(r, u) for k in range(cf.controller.fel.repeat)])

        log.store(
            t=time,
            r=r,
            x=x,
            u=u,
            u_fbc=u_fbc,
            u_fel=u_fel,
            l_fel=loss_fel,
            w=np.squeeze(fel.weights)
        ).flush()

        env.step(act)

    return xsim.Retriever(log)


def run_fbc(cf, env, ref, fbc):
    env.reset()

    log = xsim.Logger()

    for time, r in tqdm(ref.items()):
        if hasattr(cf.env, "fail") and time == cf.env.fail.time:
            env.set_fail()
        x = env.state
        u_sfb = ca.fbc(fbc, r, x)
        u = u_sfb
        act = ca.build_action(env, u_sfb)

        log.store(
            t=time,
            r=r,
            x=x,
            u=u,
        ).flush()

        env.step(act)

    return xsim.Retriever(log)


def plot(cf, res_fel, res_sfb=None):
    rfc = cf.result.fig
    nrows = np.sum([int(f) for _, f in rfc._cf.items()])
    fig, axes = plt.subplots(nrows=nrows, sharex=True)
    axes_pointer = 0
    if not hasattr(axes, "__len__"):
        axes = [axes]

    res_params = lambda : xt.Config(dict(key=[], style=[], color=[]))

    if rfc.pitch:
        res_pitch_param = res_params()
        res_pitch = dict(
            time=res_fel
        )


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fel_normal.yaml")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arguments()
    run(args)
