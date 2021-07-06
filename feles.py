# coding: utf-8

import xsim
import xtools as xt
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from feles.controller import compute_action as ca
from feles.controller.se import Log as sLog
from feles.controller.fel import Log as fLog
from feles.controller.fel import NLCFEL
from feles.utils.buffer import ReplayBuffer, FELESLog
from feles.utils.builder import *
from fel import run_fel
from heritage.fel import run_fbc


def run(args):
    cf = xt.Config(args.config)
    if args.random:
        cf.env.reference.random = True
    if args.due is not None:
        cf.env.due = args.due

    # env
    env = build_env(cf.env)
    # LQR state feedback controller
    fbc = build_sfb(cf.controller.fbc, env)
    # fel controller
    fel = build_fel(cf.controller.fel, env)
    # system estimator
    sye = build_sye(cf.controller.sye, env)
    # reference queue
    ref = build_reference_queue(cf.env, at_random=cf.env.reference.random)

    ret_feles = run_feles(cf, env, ref, fbc, fel, sye)
    ret_fel = run_fel(cf, env, ref, fbc, fel)
    ret_fbc = run_fbc(cf, env, ref, fbc)

    # make directory
    if cf.result.fig.save or cf.result.data:
        cf.result.path = xt.make_dirs_current_time(cf.result.path)
        cf.dump(xt.join(cf.result.path, "config.yaml"))

    # make figure
    if cf.result.fig.save or cf.result.fig.show:
        plot(cf, env, ret_feles, ret_fel, ret_fbc)
    # save data sa csv file
    if cf.result.data:
        save(cf.result.path, ret_feles, ret_fel, ret_fbc)


def run_feles(cf, env, ref, fbc, fel, sye):
    env.reset()
    sye.reset()

    log = xsim.Logger()

    window_step = int(cf.controller.feles.window / cf.env.dt)
    buf_real = ReplayBuffer(window_step)
    past_x = None
    past_u = None
    past_r = None

    p = np.zeros(2)
    loss_sye = 0.0
    loss_fel = 0.0
    lr_factor_sim = 0.0
    decay = build_decay_queue(window_step, cf.controller.fel.decay) * 0.1

    for time, r in tqdm(ref.items()):
        if hasattr(cf.env, "fail") and time == cf.env.fail.time:
            env.set_fail()

        x = env.state
        u_fbc = ca.fbc(fbc, r, x)
        u_fel = ca.fel(fel, r)
        u = np.squeeze(np.asanyarray(u_fbc + u_fel))
        act = ca.build_action(env, u)

        if past_x is not None:
            buf_real.add(FELESLog(past_x, past_u, x, past_r))
            p = sye.predict(past_x, past_u)

        if buf_real.is_full:
            data = buf_real.buffer
            loss_sye = sye.update(data.state, data.action, data.next_state)
            lr_factor_sim = np.exp(-cf.controller.feles.temp * loss_sye)

        if buf_real.is_full:
            loss_fel = np.mean([fel.update(data.reference, data.action, factor=decay) for _ in range(cf.controller.feles.repeat)])

        log.store(
            t=time,
            x=x,
            u=u,
            u_fbc=u_fbc,
            u_fel=u_fel,
            r=r,
            p=p,
            l_sye=loss_sye,
            l_fel=loss_fel,
            f=lr_factor_sim
        ).flush()

        env.step(act)
        past_x = x
        past_u = u
        past_r = r

    return xsim.Retriever(log)


def plot(cf, env, res_feles, res_fel, res_fbc):
    xt.info("build result")
    # build results
    res_pitch = pd.DataFrame(dict(
        time=res_feles.t(),
        reference=res_feles.r(env.IX_T, fn=xt.r2d),
        feles=res_feles.x(env.IX_T, fn=xt.r2d),
        fel=res_fel.x(env.IX_T, fn=xt.r2d),
        fbc=res_fbc.x(env.IX_T, fn=xt.r2d)
    ))
    res_pred = pd.DataFrame(dict(
        time=res_feles.t(),
        pitch=res_feles.x(env.IX_T, fn=xt.r2d),
        speed=res_feles.x(env.IX_q, fn=xt.r2d),
        pitch_pred=res_feles.p(env.IX_T, fn=xt.r2d),
        speed_pred=res_feles.p(env.IX_q, fn=xt.r2d),
    ))

    res_error = pd.DataFrame(dict(
        time=res_feles.t(),
        feles=np.square(res_feles.r(env.IX_T, fn=xt.r2d) - res_feles.x(env.IX_T, fn=xt.r2d)),
        fel=np.square(res_fel.r(env.IX_T, fn=xt.r2d) - res_fel.x(env.IX_T, fn=xt.r2d)),
        fbc=np.square(res_fbc.r(env.IX_T, fn=xt.r2d) - res_fbc.x(env.IX_T, fn=xt.r2d))
    ))

    res_elevator = pd.DataFrame(dict(
        time=res_feles.t(),
        feles=res_feles.u(fn=xt.r2d),
        fel=res_fel.u(fn=xt.r2d),
        fbc=res_fbc.u(fn=xt.r2d),
    ))
    res_loss = pd.DataFrame(dict(
        time=res_feles.t(),
        loss_fel=res_feles.l_fel(),
        loss_sye=res_feles.l_sye(),
        factor=res_feles.f()
    ))


    xt.info("plot result")
    COLOR_ZERO = "black"
    WIDTH_ZERO = "1"

    C_REF = "gray"
    C_FBC = "dodgerblue"
    C_FEL = "forestgreen"
    C_FELES = "orangered"
    C_SET = [C_FBC, C_FEL, C_FELES]

    L_TIME = "time"
    L_SET = ["fbc", "fel", "feles"]

    fig, axes = plt.subplots(nrows=6, sharex=True, figsize=(15, 12))

    ecf = cf.env
    delim = xt.round(xt.r2d([
        env.action_space.low[env.IX_de],
        env.action_space.high[env.IX_de],
    ]), 3)

    # draw zero-lines
    axes[0].axhline(0, color=COLOR_ZERO, linewidth=WIDTH_ZERO)
    axes[1].axhline(0, color=COLOR_ZERO, linewidth=WIDTH_ZERO)

    res_pitch.plot(
        x=L_TIME, y="reference", ax=axes[0],
        style="--", color=C_REF,
        xlim=[0, ecf.due], ylim=ecf.reference.range, grid=True,
        yticks=np.arange(ecf.reference.range[0], ecf.reference.range[1] + 1, 1)
    )
    res_pitch.plot(
        x=L_TIME, y=L_SET, ax=axes[0],
        style="-", color=C_SET
    )
    axes[0].set_ylabel("pitch [deg]")

    res_elevator.plot(
        x=L_TIME, y=L_SET, ax=axes[1],
        style="-", color=C_SET,
        # ylim=delim,
        grid=True
    )
    axes[1].set_ylabel("elevator [deg]")

    res_error.plot(
        x=L_TIME, y=L_SET, ax=axes[2], grid=True,
        color=C_SET
    )
    axes[2].set_ylabel("error [deg]")

    res_pred.plot(
        x="time", y=["pitch", "pitch_pred", "speed", "speed_pred"], ax=axes[3],
        style=["--", "-", "--", "-"], color=["forestgreen", "hotpink", "forestgreen", "hotpink"],
        grid=True
    )
    axes[3].set_ylabel("predict [deg]")

    res_loss.plot(
        x=L_TIME, y="loss_sye", ax=axes[4],
        grid=True
    )
    res_loss.plot(
        x=L_TIME, y="loss_fel", ax=axes[4],
        grid=True, secondary_y=True
    )
    axes[4].set_ylabel("loss")

    res_loss.plot(
        x=L_TIME, y="factor", ax=axes[5],
        grid=True
    )
    axes[5].set_ylabel("factor")

    if cf.result.fig.show:
        plt.show()
    if cf.result.fig.save:
        fig.savefig(xt.join(cf.result.path, "summary.png"))


def save(path, res_feles=None, res_fel=None, res_fbc=None):
    if res_feles is not None:
        res_feles = build_feles_result(res_feles)
        res_feles.to_csv(xt.join(path, "feles.csv"))
    if res_feles is not None:
        res_fel = build_fel_result(res_fel)
        res_fel.to_csv(xt.join(path, "fel.csv"))
    if res_feles is not None:
        res_fbc = build_fbc_result(res_fbc)
        res_fbc.to_csv(xt.join(path, "fbc.csv"))


def build_fbc_result(ret):
    r2d = xt.r2d
    ixT = 0
    ixq = 1
    return pd.DataFrame(dict(
        time=ret.t(),
        reference=ret.r(ixT, fn=r2d),
        reference_speed=ret.r(ixq, fn=r2d),
        pitch=ret.x(ixT, fn=r2d),
        pitch_speed=ret.x(ixq, fn=r2d),
        error=np.abs(ret.r(ixT, fn=r2d) - ret.x(ixT, fn=r2d)),
        elevator=ret.u(fn=r2d)
    ))


def build_fel_result(ret):
    res = build_fbc_result(ret)
    res["loss_fel"] = ret.l_fel()
    res["u_fel"] = ret.u_fel(fn=xt.r2d)
    res["u_fbc"] = ret.u_fbc(fn=xt.r2d)
    return res


def build_feles_result(ret):
    res = build_fel_result(ret)
    res["predict_pitch"] = ret.p(0, fn=xt.r2d)
    res["predict_speed"] = ret.p(1, fn=xt.r2d)
    res["loss_sye"] = ret.l_sye()
    res["factor"] = ret.f()
    return res


def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/feles.yaml", type=str)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--due", default=None, type=int)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argument()
    run(args)
